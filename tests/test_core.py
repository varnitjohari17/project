import os
from pathlib import Path

import numpy as np
import pytest

from mmrag.chunking import build_chunks, linearize_table, tokenize
from mmrag.ingestion import PDFParser
from mmrag.pipeline import parse_citations
from mmrag.retriever import rrf
from mmrag.store import DocStore, LocalIndex

DATA = Path(__file__).resolve().parent.parent / "eval" / "data"
HELIX = DATA / "helix_r7_manual.pdf"
needs_data = pytest.mark.skipif(not HELIX.exists(), reason="run `python eval/build_dataset.py` first")


def test_rrf_rewards_agreement():
    s = rrf([(["a", "b", "c"], 1.0), (["b", "c", "a"], 1.0)], k=60)
    assert max(s, key=s.get) == "b"
    s = rrf([(["a"], 1.0), (["b"], 0.1)])
    assert s["a"] > s["b"]


def test_parse_citations():
    assert parse_citations("Payload is 7 kg [2]. Reach 912 mm [1, 3][9].", 3) == [2, 1, 3]
    assert parse_citations("no cites", 3) == []


def test_linearize_table_skips_placeholder_header():
    rows = [["Tech Stack", "Col2"], ["Component", "Technology"], ["Vector DB", "Pinecone"]]
    assert linearize_table(rows) == ["Tech Stack", "Component: Vector DB; Technology: Pinecone"]


def test_tokenize_keeps_compound_terms():
    toks = tokenize("Which model? all-MiniLM-L6-v2 and HX-OS 4.2.1")
    assert "all-minilm-l6-v2" in toks and "minilm" in toks and "4.2.1" in toks
    assert "which" not in toks


def test_local_index_roundtrip(tmp_path):
    idx = LocalIndex(tmp_path / "i", 3)
    v = np.eye(3, dtype=np.float32)
    idx.upsert(["d1#a", "d1#b", "d2#a"], v, [{"doc_id": "d1"}, {"doc_id": "d1"}, {"doc_id": "d2"}])
    assert idx.query(np.array([0, 1, 0], np.float32), 1)[0].id == "d1#b"
    assert idx.query(np.array([0, 1, 0], np.float32), 3, doc_ids=["d2"])[0].id == "d2#a"
    idx.delete_doc("d1")
    again = LocalIndex(tmp_path / "i", 3)  # persisted
    assert again.ids == ["d2#a"]


def test_docstore(tmp_path):
    ds = DocStore(tmp_path / "ds.json")
    ds.add({"doc_id": "x"}, [{"text": "hi", "metadata": {"id": "x#1", "doc_id": "x"}}])
    assert DocStore(tmp_path / "ds.json").get("x#1")["text"] == "hi"
    ds.delete("x")
    assert not ds.chunks and not ds.documents


@needs_data
def test_parser_finds_tables_raster_and_vector_figures(tmp_path):
    d = PDFParser(tmp_path).parse(HELIX, "helix")
    kinds = [(e.page, e.type, e.origin) for e in d.elements if e.type != "text"]
    assert (1, "table", "native") in kinds and (3, "table", "native") in kinds
    assert (2, "image", "raster") in kinds and (2, "image", "vector") in kinds
    vec = next(e for e in d.elements if e.origin == "vector")
    assert "180°/s" in vec.content and vec.caption.startswith("Figure 2")
    assert not any("confidential draft" in e.content for e in d.elements)  # footer removed
    chunks = build_chunks(d)
    ids = [c.metadata["id"] for c in chunks]
    assert len(ids) == len(set(ids)) and all(i.startswith("helix#") for i in ids)
    spec = next(c for c in chunks if c.metadata["type"] == "table" and c.metadata["page"] == 1)
    assert "Parameter: Maximum payload; Value: 7 kg" in spec.metadata["table_text"]


@needs_data
@pytest.mark.skipif(os.getenv("FAST") == "1", reason="loads models")
def test_end_to_end_extractive(tmp_path):
    from mmrag.config import Settings
    from mmrag.pipeline import RAGPipeline

    s = Settings(data_dir=tmp_path, llm_backend="extractive", vector_backend="local", enable_caption=False)
    p = RAGPipeline(s)
    info = p.ingest(HELIX)
    assert info["chunks"]["table"] == 2 and info["chunks"]["image"] == 2
    assert p.ingest(HELIX)["skipped"]
    out = p.answer("What does error code E204 mean?")
    top = out["sources"][0]
    assert (top["page"], top["type"]) == (3, "table")
    assert "Encoder" in out["answer"] and out["cited"]
    assert p.delete_document(info["doc_id"]) and not p.list_documents()


def test_image_parser(tmp_path):
    from PIL import Image
    from mmrag.ingestion import UniversalParser

    img_path = tmp_path / "sunset_beach.jpg"
    img = Image.new("RGB", (200, 150), color=(255, 120, 50))
    img.save(img_path)

    images_dir = tmp_path / "images"
    videos_dir = tmp_path / "videos"
    parser = UniversalParser(images_dir, videos_dir)
    parsed = parser.parse(img_path, "sunset1")

    assert parsed.doc_type == "image"
    assert parsed.title == "Sunset Beach"
    assert len(parsed.elements) == 1
    assert parsed.elements[0].type == "image"
    assert Path(parsed.elements[0].image_path).exists()


def test_video_parser(tmp_path):
    import cv2
    from mmrag.ingestion import UniversalParser

    # Create a 2-second synthetic video with cv2
    vid_path = tmp_path / "robot_demo.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(vid_path), fourcc, 10.0, (120, 80))
    for i in range(20):
        frame = np.full((80, 120, 3), (i * 10 % 255, 128, 200), dtype=np.uint8)
        out.write(frame)
    out.release()

    images_dir = tmp_path / "images"
    videos_dir = tmp_path / "videos"
    parser = UniversalParser(images_dir, videos_dir, video_sample_fps=1.0, video_max_frames=5)
    parsed = parser.parse(vid_path, "robot1")

    assert parsed.doc_type == "video"
    assert parsed.title == "Robot Demo"
    assert parsed.duration_s > 0
    assert len(parsed.elements) >= 1
    assert all(e.type == "video" for e in parsed.elements)
    assert parsed.elements[0].timestamp != ""
    assert Path(parsed.elements[0].image_path).exists()


def test_text_and_csv_parser(tmp_path):
    from mmrag.ingestion import UniversalParser

    # CSV
    csv_path = tmp_path / "prices.csv"
    csv_path.write_text("Item,Price,Stock\nLaptop,1200,5\nMouse,25,100\n", encoding="utf-8")
    parser = UniversalParser(tmp_path / "images", tmp_path / "videos")
    parsed_csv = parser.parse(csv_path, "csv1")
    assert parsed_csv.doc_type == "table"
    assert len(parsed_csv.elements) == 1
    assert parsed_csv.elements[0].type == "table"
    assert "Laptop" in parsed_csv.elements[0].content

    # Markdown
    md_path = tmp_path / "notes.md"
    md_path.write_text("# Overview\nThis is an introductory section.\n## Features\nMultimodal RAG with Gemini.\n", encoding="utf-8")
    parsed_md = parser.parse(md_path, "md1")
    assert parsed_md.doc_type == "text"
    assert len(parsed_md.elements) >= 2


def test_gemini_chat_model_structure():
    from mmrag.gemini import GeminiChatModel
    from langchain_core.messages import HumanMessage, SystemMessage

    model = GeminiChatModel(api_key="fake-key-for-test", model_name="gemini-2.5-flash")
    msgs = [
        SystemMessage("You are a helpful assistant."),
        HumanMessage("What is multimodal RAG?"),
    ]
    prompt, sys_inst, images = model._extract_inputs(msgs)
    assert sys_inst == "You are a helpful assistant."
    assert "What is multimodal RAG?" in prompt
    assert images == []


@pytest.mark.skipif(os.getenv("FAST") == "1", reason="loads models")
def test_pipeline_multimodal_image_and_text(tmp_path):
    from PIL import Image
    from mmrag.config import Settings
    from mmrag.pipeline import RAGPipeline

    img_path = tmp_path / "diagram.png"
    Image.new("RGB", (100, 100), color=(10, 150, 200)).save(img_path)

    doc_path = tmp_path / "guide.txt"
    doc_path.write_text("Multimodal RAG allows querying across documents, photos, and videos using Gemini.", encoding="utf-8")

    s = Settings(data_dir=tmp_path, llm_backend="extractive", vector_backend="local", enable_caption=False)
    p = RAGPipeline(s)

    info_img = p.ingest(img_path)
    assert info_img["doc_type"] == "image"
    assert info_img["chunks"]["image"] == 1

    info_doc = p.ingest(doc_path)
    assert info_doc["chunks"]["text"] >= 1

    docs = p.list_documents()
    assert len(docs) == 2

    res = p.answer("What does Multimodal RAG allow?")
    assert len(res["sources"]) >= 1
    assert "Multimodal RAG" in res["answer"]
