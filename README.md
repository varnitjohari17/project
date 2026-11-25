# Multimodal RAG AI System (Powered by Google Gemini)

An end-to-end **Multimodal Retrieval-Augmented Generation (RAG) system** that ingests and searches across **Videos, Photos/Images, PDFs, Tables, and Structured Documents**.

You can upload:
- 🎬 **Videos** (`.mp4, .mov, .avi, .mkv, .webm`): automatically sampled into keyframe scenes with timestamp ranges and multimodal descriptions.
- 🖼️ **Photos & Graphics** (`.png, .jpg, .jpeg, .webp, .gif, .bmp`): analyzed and described using Gemini Vision with OCR.
- 📄 **PDFs** (`.pdf`): pages, tables, vector diagrams, charts, and raster figures.
- 📊 **Structured Data & Documents** (`.csv, .md, .txt, .json`): tabular rows linearised and markdown sections preserved.

Ask questions in natural language and get answers grounded across all media types with **exact source citations, page numbers, and video timestamps**.

Stack: **Google Gemini (google.genai SDK) · OpenCV · FastAPI · LangChain · Streamlit · PyMuPDF · Sentence-Transformers · CLIP · Pinecone / Local NumPy Index**.

---

## Quick start

```bash
make setup                     # Python 3.12 venv + dependencies (uses uv)
cp .env.example .env           # Add GEMINI_API_KEY (or GOOGLE_API_KEY)
make api                       # FastAPI backend  -> http://localhost:8000/docs
make ui                        # Streamlit UI     -> http://localhost:8501
```

### Models & API Keys

| Key in `.env` | What switches on |
|---|---|
| `GEMINI_API_KEY` or `GOOGLE_API_KEY` | **Google Gemini** (default: `gemini-2.5-flash` or `gemini-2.5-pro` / `gemini-1.5-flash`) for generation, streaming, vision, and video scene comprehension |
| `PINECONE_API_KEY` | **Pinecone** serverless vector indexes (created automatically for text and CLIP image/video vectors) |
| `LLM_VISION=true` | Retrieved photos, diagrams, and video keyframes are sent directly to Gemini's multimodal vision input |
| `HF_TOKEN` | (Alternative fallback) Hugging Face Inference API |
| None (no keys) | Fully local offline mode: NumPy exact search index + extractive citation answers / local model |

---

## Multimodal Architecture

```
                 ┌──────────────────────── INGESTION ────────────────────────┐
  Videos ──────► OpenCV keyframe sampling ─┬─ video scenes + timestamps ────┐│
                 (adaptive scene intervals)├─ keyframe images ─────────────┼┼─► Gemini Vision / BLIP ─► text index
  Photos ──────► Image parser ─────────────┼─ visual details + OCR ────────┼┤   (+ contextual           (Pinecone/local)
  PDFs ────────► PyMuPDF parser ───────────┼─ text, tables, charts ────────┤│    headers)
  Text/CSV ────► Document parser ──────────┴─ tables & markdown sections ──┘│
                                           └──────────────► CLIP image encoder ─► image/video index
                                                                                      │
                 ┌───────────────────────── QUERY ───────────────────────────┐       │
  question ──► condense with chat history (Gemini) ─┬─ dense embedding ───────┐        │
                                                   ├─ BM25 keyword ──────────┼─► weighted RRF
                                                   └─ CLIP text→image/video ─┘        │
                         ──► cross-encoder re-rank ────────────────────────► top-k ──► multimodal context
                             (text + video frames + photos) ──► Gemini ──► grounded answer + citations [n]
```

| Synopsis module | Implementation |
|---|---|
| 6.1 Document Ingestion | [`mmrag/ingestion.py`](mmrag/ingestion.py): text with section headings, tables, raster images, **vector charts/diagrams** (clustered PDF drawings), OCR for scanned pages, caption linking ("Figure 2: …"), header/footer removal |
| 6.2 Multimodal Embedding | [`mmrag/models.py`](mmrag/models.py): Sentence-Transformers text embedder (a LangChain `Embeddings`), CLIP image/text towers, figure describer (VLM or BLIP + RapidOCR) |
| 6.3 Vector Storage & Retrieval | [`mmrag/store.py`](mmrag/store.py) (Pinecone / local) + [`mmrag/retriever.py`](mmrag/retriever.py) (hybrid search, RRF, re-ranking, a LangChain `BaseRetriever`) |
| 6.4 Query Processing | Query embedded with the same model used at ingestion (with the model's query instruction); follow-up questions rewritten into standalone ones |
| 6.5 Context Assembly & Generation | [`mmrag/pipeline.py`](mmrag/pipeline.py) + [`mmrag/llm.py`](mmrag/llm.py): LangChain prompts and chat models, token-budgeted numbered context, streaming |
| 6.6 Source Attribution | Each answer cites `[n]`; the API returns the document, page, type, section, snippet, and figure image for each source |
| 6.7 User Interface | [`ui/app.py`](ui/app.py): Streamlit chat with upload, per-document search scope, delete, streaming answers, cited sources with figure previews |
| Backend API (§8) | [`api/main.py`](api/main.py): FastAPI with `/documents`, `/query`, `/query/stream` (NDJSON), `/images`, `/health` |
| Testing & Evaluation (§7.6) | [`eval/`](eval/): labelled multimodal benchmark, ablations, end-to-end scoring, plus `tests/` |

### Optimisations beyond a basic RAG pipeline

Each item below was measured in `eval/`:

1. **Hybrid retrieval**: dense + BM25 keyword search fused with Reciprocal Rank Fusion. This catches exact tokens such as model names, error codes, and figures that embeddings blur. *(Listed as future scope in the synopsis; implemented here.)*
2. **Cross-encoder re-ranking** (`bge-reranker-base`). This is the single largest quality gain. *(Also listed as future scope.)*
3. **True cross-modal search**: CLIP's text tower searches figure pixels directly. In parallel, every figure also gets a text description (VLM or BLIP caption + OCR, or native PDF text for vector graphics), so a text-only LLM can reason over charts.
4. **Vector-figure extraction**: charts and diagrams drawn as PDF paths (most generated reports) are detected and rendered. Plain image extraction misses these entirely.
5. **Table linearisation**: rows become `header: value; header: value` lines for embedding and BM25, which match much better than Markdown pipes. Markdown is still what the LLM sees.
6. **Contextual chunk headers** (`title > section`) are prepended at embedding time. Chunks never cross pages, so citations stay exact.
7. **Conversational memory** via standalone-question rewriting *(future scope in the synopsis)*.
8. **Grounding prompt with abstention**: the model must cite or say it could not find the answer. Context is packed to a token budget (the context-window constraint from the literature review).
9. **Engineering**: idempotent ingestion (content hash), cached figure descriptions, per-page fault isolation, streaming responses, model singletons warmed at API startup, Pinecone id-prefix deletes (serverless-compatible), and API keys only from the environment (§10.1).

---

## Evaluation

`make eval` regenerates the benchmark and writes [`eval/results/report.md`](eval/results/report.md).

**Benchmark.** There are 45 questions over 4 PDFs: three generated documents (an annual report, a robot manual, a water-quality study) and the project synopsis itself. In the generated documents every fact lives in exactly one modality: body text, a table, a raster chart or diagram, or a vector drawing. That makes it possible to measure multimodal retrieval honestly. 3 questions are unanswerable, to test abstention.

**Retrieval** (42 answerable questions; Hit@k = the correct page and modality appears in the top k):

| Configuration (text embedding `bge-small-en-v1.5`) | Hit@1 | Hit@3 | Hit@5 | MRR@10 |
|---|---|---|---|---|
| Dense only (basic RAG baseline) | 0.690 | 0.881 | 0.905 | 0.792 |
| + BM25 hybrid (RRF) | 0.714 | 0.952 | 0.952 | 0.817 |
| + BM25 + CLIP, no re-ranker | 0.548 | 0.833 | 0.952 | 0.688 |
| + cross-encoder re-rank | 0.905 | 1.000 | 1.000 | 0.948 |
| **Full (default: + modality routing)** | **0.929** | **1.000** | **1.000** | **0.960** |
| Full with the smaller `ms-marco-MiniLM-L-6-v2` re-ranker | 0.881 | 0.976 | 1.000 | 0.931 |

`all-MiniLM-L6-v2` (the synopsis' model) reaches the same 0.960 MRR in the full configuration. On its own, though, it is weaker (dense-only 0.768), so `bge-small-en-v1.5` is the default and MiniLM is one env var away (`TEXT_EMBED_MODEL`).

What the measurements say:
- The re-ranker is the largest gain: +0.13 MRR over hybrid search.
- CLIP is useful only in combination with the re-ranker. It brings figures into the candidate pool, and the cross-encoder then orders them correctly.
- Keeping a fused-score prior after re-ranking hurt (−0.06 MRR), so it is set to 0.
- Tuned defaults are 20 candidates, CLIP weight 1.0, and modality boost 0.05. Retrieval with re-ranking takes about 0.6 s per query on an M1 laptop.

End-to-end answer accuracy, citation accuracy, abstention on unanswerable questions, and latency, for the local LLM and the extractive fallback, are in [`eval/results/report.md`](eval/results/report.md). Rerun `make eval` with `HF_TOKEN` set to score Llama-3.1-8B.

---

## API examples

```bash
curl -F "files=@samples/Multimodal_RAG_Synopsis_final.pdf" localhost:8000/documents
curl -s localhost:8000/query -H 'content-type: application/json' \
     -d '{"question": "Which vector database does the system use?"}' | jq '.answer, .sources[0]'
```

## Project layout

```
mmrag/       core library (config, ingestion, chunking, models, store, retriever, llm, pipeline)
api/         FastAPI backend
ui/          Streamlit front-end
eval/        benchmark generator, evaluation runner, results
tests/       unit + end-to-end tests  (make test)
samples/     the synopsis PDF
```
