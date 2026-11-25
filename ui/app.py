"""User Interface module (synopsis §6.7): Streamlit chat front-end over the FastAPI backend.

    streamlit run ui/app.py
"""

from __future__ import annotations

import json
import mimetypes
import os

import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
TYPE_ICON = {"text": "📄", "table": "📊", "image": "🖼️", "video": "🎬"}
DOC_ICON = {"pdf": "📄", "video": "🎬", "image": "🖼️", "table": "📊", "text": "📝"}

st.set_page_config(page_title="Multimodal RAG (Gemini)", page_icon="🔎", layout="wide")


# ---- helpers ----------------------------------------------------------------------------
def api(method: str, path: str, **kw):
    try:
        r = requests.request(method, f"{API}{path}", timeout=kw.pop("timeout", 30), **kw)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        st.session_state["api_error"] = str(e)
        return None


@st.cache_data(show_spinner=False, max_entries=256)
def image_bytes(rel: str) -> bytes | None:
    try:
        r = requests.get(f"{API}/images/{rel}", timeout=15)
        return r.content if r.ok else None
    except requests.RequestException:
        return None


def render_sources(sources: list[dict], cited: list[int]):
    if not sources:
        return
    used = [s for s in sources if s["n"] in cited] or sources[:1]
    other = [s for s in sources if s not in used]

    def _badge(s):
        loc = f"ts {s['timestamp']}" if s.get("timestamp") else f"p.{s.get('page', 1)}"
        return f"[{s['n']}] {TYPE_ICON.get(s['type'], '📄')} {s['source']} ({loc})"

    st.caption("Sources: " + " · ".join(_badge(s) for s in used))
    with st.expander(f"Show retrieved multimodal context ({len(sources)} chunks)"):
        for s in used + other:
            mark = "✅ cited" if s["n"] in cited else ""
            t = s.get("type", "text")
            loc_label = f"timestamp {s['timestamp']}" if s.get("timestamp") else f"page {s.get('page', 1)}"
            st.markdown(
                f"**[{s['n']}] {TYPE_ICON.get(t, '')} {s['source']} — {loc_label} · {t}** {mark}  \n"
                f"<small>{s.get('section') or ''} · relevance {s.get('score')}</small>",
                unsafe_allow_html=True,
            )
            # Video player or keyframe image
            if t == "video":
                c1, c2 = st.columns([1, 1])
                if s.get("image"):
                    b = image_bytes(s["image"])
                    if b:
                        c1.image(b, caption=f"Keyframe at {s.get('timestamp')}", use_container_width=True)
                if s.get("video"):
                    start = int(s.get("start_sec", 0))
                    c2.video(f"{API}/videos/{s['video']}", start_time=start)
                st.markdown(s["content"])
            elif t == "image":
                if s.get("image"):
                    b = image_bytes(s["image"])
                    if b:
                        st.image(b, caption=s.get("caption") or s.get("source"), width=480)
                st.text(s["content"][:1200])
            elif t == "table":
                st.markdown(s["content"])
            else:
                st.text(s["content"][:1200])
            st.divider()


# ---- sidebar ----------------------------------------------------------------------------
health = api("GET", "/health", timeout=5)
with st.sidebar:
    st.header("🔎 Multimodal RAG")
    if not health:
        st.error(f"Backend not reachable at {API}.\n\nStart it with:\n\n`uvicorn api.main:app --port 8000`")
        st.stop()
    st.caption(
        f"LLM: **{health['llm']}**  \nVectors: **{health['vector_backend']}**  \n"
        f"Text emb: `{health['text_embedding'].split('/')[-1]}` · Image emb: `{(health['image_embedding'] or '-').split('/')[-1]}`  \n"
        f"Description: **{health.get('multimodal_description', health.get('figure_description', 'gemini'))}**"
    )

    st.subheader("1. Upload Multimodal Data")
    st.caption("Upload PDFs, Videos (.mp4, .mov), Photos (.jpg, .png), or Documents (.txt, .md, .csv)")
    files = st.file_uploader(
        "Upload files",
        type=["pdf", "mp4", "mov", "avi", "mkv", "webm", "png", "jpg", "jpeg", "webp", "gif", "bmp", "txt", "md", "csv", "json"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if files and st.button("Ingest Files", type="primary", use_container_width=True):
        with st.spinner(f"Processing and indexing {len(files)} multimodal file(s)..."):
            upload_tuples = []
            for f in files:
                mtype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
                upload_tuples.append(("files", (f.name, f.getvalue(), mtype)))
            res = api("POST", "/documents", files=upload_tuples, timeout=1800)

        for r in res or []:
            if r.get("error"):
                st.error(f"{r['filename']}: {r['error']}")
            elif r.get("skipped"):
                st.info(f"{r['filename']} was already ingested.")
            else:
                c = r.get("chunks", {})
                parts = []
                if c.get("text"):
                    parts.append(f"{c['text']} text")
                if c.get("table"):
                    parts.append(f"{c['table']} tables")
                if c.get("image"):
                    parts.append(f"{c['image']} images")
                if c.get("video"):
                    parts.append(f"{c['video']} video scenes")
                breakdown = ", ".join(parts) if parts else "chunks indexed"
                st.success(f"{r['filename']}: {breakdown} in {r['timings']['total_s']} s")
                for w in r.get("warnings", []):
                    st.warning(w)

    st.subheader("2. Search Scope")
    docs = api("GET", "/documents") or []
    selected = []
    for d in docs:
        c1, c2 = st.columns([5, 1])
        c = d.get("chunks", {})
        dtype = d.get("doc_type", "pdf")
        icon = DOC_ICON.get(dtype, "📄")
        dur_or_pages = f"{d.get('duration_s')}s" if dtype == "video" else f"{d.get('pages', 1)}p"
        if c1.checkbox(
            f"{icon} {d['filename']}",
            value=True,
            key=f"sel-{d['doc_id']}",
            help=f"{dtype.upper()} · {dur_or_pages} · {c.get('text', 0)} text / {c.get('table', 0)} tables / {c.get('image', 0)} images / {c.get('video', 0)} video scenes",
        ):
            selected.append(d["doc_id"])
        if c2.button("🗑", key=f"del-{d['doc_id']}", help="Remove from index"):
            api("DELETE", f"/documents/{d['doc_id']}")
            st.rerun()
    if not docs:
        st.caption("No documents or media ingested yet.")

    st.subheader("3. Options")
    top_k = st.slider("Chunks given to LLM (top-k)", 1, 10, 5)
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---- chat -------------------------------------------------------------------------------
st.title("Multimodal RAG Search")
st.caption("Search across Videos, Photos, Diagrams, Tables, and Documents with Gemini-grounded answers and source citations.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            render_sources(m.get("sources", []), m.get("cited", []))
            if m.get("timings"):
                st.caption(f"⏱ {m['timings']['total_s']} s")

question = st.chat_input("Ask about your videos, photos, or documents…", disabled=not docs)
if question:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-6:]]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        state: dict = {"sources": [], "cited": [], "timings": None, "error": None}

        def tokens():
            body = {"question": question, "history": history, "doc_ids": selected or None, "top_k": top_k}
            try:
                with requests.post(f"{API}/query/stream", json=body, stream=True, timeout=600) as r:
                    r.raise_for_status()
                    for line in r.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        ev = json.loads(line)
                        if ev["type"] == "sources":
                            state["sources"] = ev["sources"]
                        elif ev["type"] == "token":
                            yield ev["text"]
                        elif ev["type"] == "done":
                            state["cited"], state["timings"] = ev["cited"], ev["timings"]
                        elif ev["type"] == "error":
                            state["error"] = ev["error"]
            except requests.RequestException as e:
                state["error"] = str(e)

        with st.spinner("Retrieving multimodal context…"):
            answer = st.write_stream(tokens())
        if state["error"]:
            st.error(state["error"])
        render_sources(state["sources"], state["cited"])
        if state["timings"]:
            st.caption(f"⏱ {state['timings']['total_s']} s")
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer if isinstance(answer, str) else "".join(answer or []),
            "sources": state["sources"],
            "cited": state["cited"],
            "timings": state["timings"],
        }
    )
