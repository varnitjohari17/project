"""Document Ingestion Module (synopsis §6.1).

Walks every page of a PDF with PyMuPDF and extracts four kinds of content, each tagged with
source / page / type metadata:

* text paragraphs, with their section heading (from font size / boldness) and with repeating
  headers/footers removed;
* tables (PyMuPDF table finder -> Markdown);
* embedded raster images (rendered at their on-page position, tiny icons dropped);
* vector figures (charts and diagrams drawn as PDF paths, found by clustering drawings),
  which plain image extraction misses entirely.

Scanned pages that have no text layer are OCR'd. Failures are isolated per page so one bad
page never aborts the whole document (synopsis §10.4).
"""

from __future__ import annotations

import csv
import json
import logging
import re
import shutil
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf
from PIL import Image

log = logging.getLogger(__name__)

CAPTION_RE = re.compile(r"^\s*(fig(ure)?|table|chart|diagram|graph|plate|exhibit)\.?\s*\d+", re.I)
HEADING_NUM_RE = re.compile(r"^\s*(\d+(\.\d+)*|[IVX]+\.|[A-Z]\.)\s+\S")


@dataclass
class Element:
    type: str  # "text" | "table" | "image" | "video"
    page: int  # 1-based (or scene index for video)
    content: str
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    section: str = ""
    image_path: str | None = None
    caption: str = ""
    origin: str = ""  # raster | vector | scan | native | photo | video_scene | text_file
    rows: list[list[str]] | None = None  # table cells, for row linearisation
    video_path: str | None = None
    timestamp: str = ""  # e.g. "00:04 - 00:09" for video
    start_sec: float = 0.0
    end_sec: float = 0.0


@dataclass
class ParsedDocument:
    doc_id: str
    filename: str
    title: str
    n_pages: int
    doc_type: str = "pdf"  # "pdf" | "image" | "video" | "text" | "table"
    duration_s: float = 0.0
    elements: list[Element] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def _overlap_frac(a: pymupdf.Rect, b: pymupdf.Rect) -> float:
    """Fraction of `a`'s area that lies inside `b`."""
    inter = pymupdf.Rect(a) & b
    if inter.is_empty or a.get_area() == 0:
        return 0.0
    return inter.get_area() / a.get_area()


def _norm_margin(text: str) -> str:
    return re.sub(r"\d+", "#", text.strip().lower())


def _block_text(block: dict) -> tuple[str, float, bool]:
    """Return (text, max font size, all-bold) for a PyMuPDF text block."""
    lines, sizes, bold_chars, chars = [], [], 0, 0
    for line in block.get("lines", []):
        spans = line.get("spans", [])
        t = "".join(s.get("text", "") for s in spans)
        if t.strip():
            lines.append(t.strip())
        for s in spans:
            n = len(s.get("text", "").strip())
            if not n:
                continue
            sizes.append(s.get("size", 0))
            chars += n
            if s.get("flags", 0) & 16 or "bold" in s.get("font", "").lower():
                bold_chars += n
    text = " ".join(lines)
    text = re.sub(r"-\s+(?=[a-z])", "", text) if text.count("- ") > 2 else text
    text = re.sub(r"\s+", " ", text).strip()
    return text, max(sizes) if sizes else 0.0, chars > 0 and bold_chars / chars > 0.8


class PDFParser:
    def __init__(self, images_dir: Path, render_dpi: int = 150, min_image_px: int = 80, ocr=None):
        self.images_dir = Path(images_dir)
        self.dpi = render_dpi
        self.min_px = min_image_px
        self.ocr = ocr  # callable(path) -> str, used for scanned pages

    # ------------------------------------------------------------------------------------
    def parse(self, pdf_path: str | Path, doc_id: str, filename: str | None = None) -> ParsedDocument:
        pdf_path = Path(pdf_path)
        doc = pymupdf.open(pdf_path)
        filename = filename or pdf_path.name
        title = (doc.metadata or {}).get("title") or ""
        out = ParsedDocument(doc_id=doc_id, filename=filename, title=title, n_pages=doc.page_count)
        img_dir = self.images_dir / doc_id
        img_dir.mkdir(parents=True, exist_ok=True)

        page_blocks = []
        for page in doc:
            try:
                page_blocks.append(page.get_text("dict", sort=True)["blocks"])
            except Exception as e:  # pragma: no cover - defensive
                page_blocks.append([])
                out.warnings.append(f"page {page.number + 1}: text extraction failed ({e})")

        margin_repeats = self._repeating_margins(doc, page_blocks)
        body_size = self._body_font_size(page_blocks)
        section = ""

        for page, blocks in zip(doc, page_blocks):
            try:
                section = self._parse_page(page, blocks, out, img_dir, margin_repeats, body_size, section)
            except Exception as e:
                log.exception("page %s failed", page.number + 1)
                out.warnings.append(f"page {page.number + 1}: {type(e).__name__}: {e}")

        if not out.title and page_blocks and page_blocks[0]:
            cands = [_block_text(b) for b in page_blocks[0] if b.get("type") == 0]
            cands = [(size, t) for t, size, _ in cands if 3 < len(t) < 200]
            if cands:
                out.title = max(cands, key=lambda c: c[0])[1]
        out.title = out.title or Path(filename).stem
        doc.close()
        return out

    # ------------------------------------------------------------------------------------
    @staticmethod
    def _repeating_margins(doc, page_blocks) -> set[str]:
        if doc.page_count < 2:
            return set()
        counts: Counter[str] = Counter()
        for page, blocks in zip(doc, page_blocks):
            h = page.rect.height
            seen = set()
            for b in blocks:
                if b.get("type") != 0:
                    continue
                y0, y1 = b["bbox"][1], b["bbox"][3]
                if y1 < 0.09 * h or y0 > 0.91 * h:
                    t, _, _ = _block_text(b)
                    if t:
                        seen.add(_norm_margin(t))
            counts.update(seen)
        thresh = max(2, int(0.4 * doc.page_count))
        return {t for t, c in counts.items() if c >= thresh}

    @staticmethod
    def _body_font_size(page_blocks) -> float:
        sizes: Counter[float] = Counter()
        for blocks in page_blocks:
            for b in blocks:
                for line in b.get("lines", []):
                    for s in line.get("spans", []):
                        sizes[round(s.get("size", 0), 1)] += len(s.get("text", ""))
        return sizes.most_common(1)[0][0] if sizes else 11.0

    # ------------------------------------------------------------------------------------
    def _parse_page(self, page, blocks, out: ParsedDocument, img_dir: Path, margin_repeats, body_size, section):
        pno = page.number + 1
        prect = page.rect
        page_area = prect.get_area()

        # ---- tables ---------------------------------------------------------------------
        tables: list[tuple[pymupdf.Rect, str, list]] = []
        try:
            for t in page.find_tables().tables:
                if t.row_count < 2 or t.col_count < 2:
                    continue
                md = t.to_markdown(clean=False).strip()
                rows = [[re.sub(r"\s+", " ", str(c or "")).strip() for c in row] for row in (t.extract() or [])]
                if sum(1 for row in rows for c in row if c) < 4:
                    continue
                # Boxes drawn side by side in a diagram look like a 1-2 row table whose header
                # is mostly empty / "ColN" placeholders: that is a figure, not a table.
                hdr = rows[0] if rows else []
                blank = sum(1 for c in hdr if not c or re.fullmatch(r"col\d+", c, re.I))
                if len(rows) <= 2 and hdr and blank >= 0.4 * len(hdr):
                    continue
                tables.append((pymupdf.Rect(t.bbox), md, rows))
        except Exception as e:
            out.warnings.append(f"page {pno}: table detection failed ({e})")

        # ---- text blocks ----------------------------------------------------------------
        text_blocks = []
        native_chars = 0
        for b in blocks:
            if b.get("type") != 0:
                continue
            t, size, bold = _block_text(b)
            if not t:
                continue
            native_chars += len(t)
            if _norm_margin(t) in margin_repeats or re.fullmatch(r"(page\s*)?\d{1,4}", t.strip(), re.I):
                continue
            text_blocks.append((pymupdf.Rect(b["bbox"]), t, size, bold))

        # headings on this page, so figures/tables get the section they sit in
        headings = [(r.y0, t[:120]) for r, t, size, bold in text_blocks if self._is_heading(t, size, bold, body_size)]

        def section_at(y: float) -> str:
            cur = section
            for hy, ht in headings:
                if hy <= y:
                    cur = ht
            return cur

        # ---- raster images --------------------------------------------------------------
        figures: list[tuple[pymupdf.Rect, str]] = []
        for img in page.get_images(full=True):
            xref = img[0]
            try:
                rects = page.get_image_rects(xref)
            except Exception:
                rects = []
            for r in rects:
                r = pymupdf.Rect(r) & prect
                if r.is_empty:
                    continue
                if r.get_area() > 0.9 * page_area:
                    if native_chars < 50 and self.ocr is not None:
                        figures.append((r, "scan"))
                    continue
                figures.append((r, "raster"))

        # ---- vector figures (charts / diagrams drawn with paths) ------------------------
        figures += [(r, "vector") for r in self._vector_figures(page, tables, [f[0] for f in figures])]

        # ---- dedupe figures -------------------------------------------------------------
        kept: list[tuple[pymupdf.Rect, str]] = []
        for r, origin in sorted(figures, key=lambda f: -f[0].get_area()):
            if any(_overlap_frac(r, k[0]) > 0.6 for k in kept):
                continue
            kept.append((r, origin))

        # ---- emit figure elements -------------------------------------------------------
        fig_rects = []
        for i, (r, origin) in enumerate(sorted(kept, key=lambda f: (f[0].y0, f[0].x0))):
            w_px = r.width / 72 * self.dpi
            h_px = r.height / 72 * self.dpi
            if origin != "scan" and (w_px < self.min_px or h_px < self.min_px or r.get_area() < 0.01 * page_area):
                continue
            path = img_dir / f"p{pno}_{i}.png"
            page.get_pixmap(clip=r, dpi=self.dpi).save(path)
            if origin == "scan":
                text = self.ocr(str(path)) if self.ocr else ""
                if text.strip():
                    out.elements.append(Element("text", pno, text, tuple(r), section_at(r.y0), origin="scan"))
                continue
            native = page.get_text("text", clip=r).strip() if origin == "vector" else ""
            caption = self._find_caption(r, text_blocks)
            out.elements.append(
                Element("image", pno, re.sub(r"\s+", " ", native), tuple(r), section_at(r.y0), str(path), caption, origin)
            )
            fig_rects.append(r)

        for r, md, rows in tables:
            caption = self._find_caption(r, text_blocks)
            md = md.replace("<br>", " ")
            out.elements.append(
                Element("table", pno, md, tuple(r), section_at(r.y0), caption=caption, origin="native", rows=rows)
            )

        # ---- body text (excluding table cells and figure labels) ------------------------
        page_elems = []
        for r, t, size, bold in text_blocks:
            if any(_overlap_frac(r, tr) > 0.5 for tr, *_ in tables):
                continue
            if any(_overlap_frac(r, fr) > 0.8 for fr in fig_rects) and not CAPTION_RE.match(t):
                continue
            if self._is_heading(t, size, bold, body_size):
                section = t[:120]
            page_elems.append(Element("text", pno, t, tuple(r), section, origin="native"))
        out.elements.extend(page_elems)
        return section

    @staticmethod
    def _is_heading(t: str, size: float, bold: bool, body: float) -> bool:
        if len(t) > 120 or t.endswith((".", ",", ";", ":")) and not HEADING_NUM_RE.match(t):
            return False
        if size >= body * 1.15:
            return True
        return bold and len(t) < 90 and (HEADING_NUM_RE.match(t) is not None or t.istitle() or t.isupper())

    @staticmethod
    def _find_caption(r: pymupdf.Rect, text_blocks) -> str:
        best, best_d = "", 1e9
        for br, t, _, _ in text_blocks:
            if not CAPTION_RE.match(t):
                continue
            if min(br.x1, r.x1) - max(br.x0, r.x0) <= 0:  # no horizontal overlap
                continue
            d = br.y0 - r.y1 if br.y0 >= r.y1 - 5 else r.y0 - br.y1
            if -5 <= d < 80 and d < best_d:
                best, best_d = t, d
        return best

    def _vector_figures(self, page, tables, raster_rects) -> list[pymupdf.Rect]:
        try:
            drawings = page.get_drawings()
        except Exception:
            return []
        if len(drawings) < 3:
            return []
        try:
            clusters = page.cluster_drawings(drawings=drawings, x_tolerance=6, y_tolerance=6)
        except Exception:
            return []
        prect = page.rect
        out = []
        for c in clusters:
            r = pymupdf.Rect(c) & prect
            if r.is_empty or r.width < 0.18 * prect.width or r.height < 50:
                continue
            if r.get_area() > 0.85 * prect.get_area():
                continue
            if any(_overlap_frac(r, tr) > 0.5 or _overlap_frac(tr, r) > 0.5 for tr, *_ in tables):
                continue
            if any(_overlap_frac(r, rr) > 0.5 for rr in raster_rects):
                continue
            n_draw = sum(1 for d in drawings if _overlap_frac(pymupdf.Rect(d["rect"]), r) > 0.9)
            inner_text = page.get_text("text", clip=r)
            if n_draw < 4 and len(inner_text) > 300:  # a framed text box, not a figure
                continue
            out.append((r + (-8, -8, 8, 8)) & prect)
        return out


# ========================================================================================
# Image Parser (Standalone photos and graphics)
# ========================================================================================
class ImageParser:
    def __init__(self, images_dir: Path):
        self.images_dir = Path(images_dir)

    def parse(self, image_path: str | Path, doc_id: str, filename: str | None = None) -> ParsedDocument:
        image_path = Path(image_path)
        filename = filename or image_path.name
        img_dir = self.images_dir / doc_id
        img_dir.mkdir(parents=True, exist_ok=True)
        dest_path = img_dir / filename

        shutil.copy2(image_path, dest_path)

        with Image.open(dest_path) as img:
            w, h = img.size
            title = image_path.stem.replace("_", " ").replace("-", " ").title()

        elem = Element(
            type="image",
            page=1,
            content="",
            bbox=(0.0, 0.0, float(w), float(h)),
            section="Image",
            image_path=str(dest_path),
            caption=title,
            origin="photo",
        )
        return ParsedDocument(
            doc_id=doc_id,
            filename=filename,
            title=title,
            n_pages=1,
            doc_type="image",
            elements=[elem],
            metadata={"width": w, "height": h, "format": img.format},
        )


# ========================================================================================
# Video Parser (Scene extraction and keyframe sampling)
# ========================================================================================
class VideoParser:
    def __init__(
        self,
        videos_dir: Path,
        images_dir: Path,
        sample_fps: float = 0.5,
        max_frames: int = 30,
        min_interval_sec: float = 2.0,
    ):
        self.videos_dir = Path(videos_dir)
        self.images_dir = Path(images_dir)
        self.sample_fps = sample_fps
        self.max_frames = max_frames
        self.min_interval_sec = min_interval_sec

    def parse(self, video_path: str | Path, doc_id: str, filename: str | None = None) -> ParsedDocument:
        import cv2

        video_path = Path(video_path)
        filename = filename or video_path.name
        title = video_path.stem.replace("_", " ").replace("-", " ").title()

        v_dir = self.videos_dir / doc_id
        v_dir.mkdir(parents=True, exist_ok=True)
        dest_video = v_dir / filename
        shutil.copy2(video_path, dest_video)

        img_dir = self.images_dir / doc_id
        img_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(dest_video))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s = total_frames / fps if fps > 0 else 0.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Determine sampling interval
        if duration_s <= 0:
            step_sec = 2.0
            n_samples = 1
        else:
            step_sec = max(self.min_interval_sec, duration_s / float(self.max_frames))
            n_samples = min(self.max_frames, max(1, int(duration_s / step_sec)))

        elements: list[Element] = []
        warnings: list[str] = []

        for i in range(n_samples):
            t_start = i * step_sec
            t_end = min(duration_s, (i + 1) * step_sec) if duration_s > 0 else t_start + step_sec
            frame_num = int(t_start * fps)

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            frame_filename = f"frame_{i:03d}_{int(t_start)}s.jpg"
            frame_path = img_dir / frame_filename
            cv2.imwrite(str(frame_path), frame)

            m_s, s_s = divmod(int(t_start), 60)
            m_e, s_e = divmod(int(t_end), 60)
            ts_str = f"{m_s:02d}:{s_s:02d} - {m_e:02d}:{s_e:02d}"

            elements.append(
                Element(
                    type="video",
                    page=i + 1,
                    content="",
                    bbox=(0.0, 0.0, float(width), float(height)),
                    section=f"Scene {i + 1} ({ts_str})",
                    image_path=str(frame_path),
                    video_path=str(dest_video),
                    timestamp=ts_str,
                    start_sec=t_start,
                    end_sec=t_end,
                    caption=f"{title} ({ts_str})",
                    origin="video_scene",
                )
            )

        cap.release()

        if not elements:
            warnings.append(f"No frames could be extracted from video: {filename}")

        return ParsedDocument(
            doc_id=doc_id,
            filename=filename,
            title=title,
            n_pages=len(elements),
            doc_type="video",
            duration_s=round(duration_s, 2),
            elements=elements,
            warnings=warnings,
            metadata={
                "fps": round(fps, 2),
                "total_frames": total_frames,
                "duration_s": round(duration_s, 2),
                "width": width,
                "height": height,
            },
        )


# ========================================================================================
# Text & Structured Document Parser (.txt, .md, .csv, .json)
# ========================================================================================
class TextDocumentParser:
    def parse(self, file_path: str | Path, doc_id: str, filename: str | None = None) -> ParsedDocument:
        file_path = Path(file_path)
        filename = filename or file_path.name
        title = file_path.stem.replace("_", " ").replace("-", " ").title()
        ext = file_path.suffix.lower()

        elements: list[Element] = []
        raw_text = file_path.read_text(encoding="utf-8", errors="replace")

        if ext == ".csv":
            elements.extend(self._parse_csv(raw_text, title))
            doc_type = "table"
        elif ext in (".md", ".markdown"):
            elements.extend(self._parse_markdown(raw_text, title))
            doc_type = "text"
        elif ext == ".json":
            elements.extend(self._parse_json(raw_text, title))
            doc_type = "text"
        else:
            elements.extend(self._parse_plain(raw_text, title))
            doc_type = "text"

        return ParsedDocument(
            doc_id=doc_id,
            filename=filename,
            title=title,
            n_pages=1,
            doc_type=doc_type,
            elements=elements,
        )

    def _parse_csv(self, content: str, title: str) -> list[Element]:
        reader = list(csv.reader(content.splitlines()))
        if not reader:
            return []
        rows = [[c.strip() for c in r] for r in reader if any(c.strip() for c in r)]
        if len(rows) < 2:
            md = "\n".join(" | ".join(r) for r in rows)
        else:
            hdr = rows[0]
            sep = ["---"] * len(hdr)
            body = rows[1:]
            md_lines = ["| " + " | ".join(hdr) + " |", "| " + " | ".join(sep) + " |"]
            for r in body:
                padded = r + [""] * (len(hdr) - len(r))
                md_lines.append("| " + " | ".join(padded[: len(hdr)]) + " |")
            md = "\n".join(md_lines)

        return [
            Element(
                type="table",
                page=1,
                content=md,
                section="Table",
                caption=title,
                origin="text_file",
                rows=rows,
            )
        ]

    def _parse_markdown(self, content: str, title: str) -> list[Element]:
        elements: list[Element] = []
        lines = content.splitlines()
        cur_sec = title
        buf: list[str] = []

        def flush():
            if buf:
                text = "\n".join(buf).strip()
                if text:
                    elements.append(
                        Element(
                            type="text",
                            page=1,
                            content=text,
                            section=cur_sec,
                            origin="text_file",
                        )
                    )
                buf.clear()

        for line in lines:
            m = re.match(r"^(#{1,4})\s+(.+)$", line)
            if m:
                flush()
                cur_sec = m.group(2).strip()
            else:
                buf.append(line)
        flush()
        return elements

    def _parse_json(self, content: str, title: str) -> list[Element]:
        try:
            data = json.loads(content)
            pretty = json.dumps(data, indent=2)
            return [
                Element(
                    type="text",
                    page=1,
                    content=f"Structured JSON Data for {title}:\n```json\n{pretty[:4000]}\n```",
                    section="JSON Data",
                    origin="text_file",
                )
            ]
        except Exception:
            return self._parse_plain(content, title)

    def _parse_plain(self, content: str, title: str) -> list[Element]:
        paras = re.split(r"\n\s*\n", content.strip())
        return [
            Element(
                type="text",
                page=1,
                content=p.strip(),
                section=title,
                origin="text_file",
            )
            for p in paras
            if p.strip()
        ]


# ========================================================================================
# Universal Multimodal Parser (Routes by file extension)
# ========================================================================================
class UniversalParser:
    def __init__(
        self,
        images_dir: Path,
        videos_dir: Path,
        render_dpi: int = 150,
        min_image_px: int = 80,
        ocr=None,
        video_sample_fps: float = 0.5,
        video_max_frames: int = 30,
    ):
        self.pdf_parser = PDFParser(images_dir, render_dpi, min_image_px, ocr=ocr)
        self.image_parser = ImageParser(images_dir)
        self.video_parser = VideoParser(videos_dir, images_dir, sample_fps=video_sample_fps, max_frames=video_max_frames)
        self.text_parser = TextDocumentParser()

    def parse(self, file_path: str | Path, doc_id: str, filename: str | None = None) -> ParsedDocument:
        p = Path(file_path)
        ext = p.suffix.lower()
        fn = filename or p.name

        if ext == ".pdf":
            out = self.pdf_parser.parse(file_path, doc_id, fn)
            out.doc_type = "pdf"
            return out
        if ext in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
            return self.video_parser.parse(file_path, doc_id, fn)
        if ext in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"):
            return self.image_parser.parse(file_path, doc_id, fn)
        if ext in (".txt", ".md", ".markdown", ".csv", ".json", ".log"):
            return self.text_parser.parse(file_path, doc_id, fn)

        raise ValueError(
            f"Unsupported file format '{ext}' for file '{fn}'. "
            "Supported formats: PDF (.pdf), Videos (.mp4, .mov, .avi, .mkv, .webm), "
            "Photos (.png, .jpg, .jpeg, .webp, .gif, .bmp), Documents (.txt, .md, .csv, .json)"
        )
