"""Run with: python -m streamlit run app.py"""
import json
import sqlite3
from pathlib import Path

import streamlit as st

from src.rag.workspace import list_pdfs, read_json, save_pdf, validate_url, write_json
from src.rag.engine import Engine, configured_sources
from src.rag.models import RagError
from src.rag.model_ui import render_models
from src.rag.catalog import canonical_tag

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config"
PDFS = ROOT / "data/pdfs"

st.set_page_config(page_title="Source Library · Local RAG", page_icon="📚", layout="wide")


def main():
    try:
        settings = read_json(CONFIG / "settings.json", CONFIG / "settings.example.json")
        sources = read_json(CONFIG / "sources.json", CONFIG / "sources.example.json")
    except (OSError, ValueError) as error:
        st.error(f"Cannot read configuration: {error}")
        st.stop()

    engine = Engine(ROOT, settings)
    inventory = {row["id"]: row for row in engine.store.summary()}
    current_sources = configured_sources(ROOT, sources)

    def source_status(source):
        row = inventory.get(source["id"])
        if not row or not row["chunks"]:
            return "Failed" if row and row["error"] else "Not indexed"
        if row["error"]:
            return "Previous version retained; update failed"
        return "Indexed" if row["active"] else "Inactive"

    with st.sidebar:
        st.title("📚 Source Library")
        st.caption("Your documents. Your local assistant.")
        page = st.radio("Workspace", ["Overview", "PDF library", "Web & YouTube", "Models", "Ask", "Index & activity", "Settings"])
        st.divider()
        st.caption("Local workspace · PDF-only by default")
        st.caption("Local RAG backend · Ollama models required")

    pdfs = list_pdfs(PDFS)
    urls = [(kind, url) for kind in ("websites", "youtube") for url in dict.fromkeys(sources.get(kind, []))]
    st.title(page)

    if page == "Overview":
        st.write("Build a collection, choose your evidence, and ask questions with citations.")
        cols = st.columns(3)
        cols[0].metric("PDF files", len(pdfs))
        cols[1].metric("Saved URLs", len(urls))
        cols[2].metric("Indexed sources", sum(bool(inventory.get(s["id"], {}).get("chunks")) and bool(inventory.get(s["id"], {}).get("active")) for s in current_sources))
        st.info("Sources are indexed only when you request it. Questions use the selected indexed sources and local Ollama models.")
        st.subheader("Get started")
        st.write("1. Upload files in PDF library.\n2. Optionally save links in Web & YouTube.\n3. Open Models to compare, download, and select local models.\n4. Build the index in Index & activity, then ask questions in Ask.")
        with st.expander("Read the architecture"):
            st.markdown((ROOT / "docs/architecture.md").read_text())

    elif page == "PDF library":
        st.caption(f"Files are saved in {PDFS}. Existing files are never overwritten.")
        uploads = st.file_uploader("Add PDFs", type=["pdf"], accept_multiple_files=True)
        if st.button("Save uploaded PDFs", disabled=not uploads, type="primary"):
            for uploaded in uploads:
                try:
                    save_pdf(PDFS, uploaded.name, uploaded.getvalue())
                    st.success(f"Saved {uploaded.name}. Pending indexing.")
                except (OSError, ValueError) as error:
                    st.error(f"{uploaded.name}: {error}")
            pdfs = list_pdfs(PDFS)
        st.subheader("Your files")
        if pdfs:
            statuses = {s["locator"]: source_status(s) for s in current_sources}
            st.dataframe([{"File": str(p.relative_to(PDFS)), "Size (KB)": round(p.stat().st_size / 1024, 1), "Status": statuses.get(str(p.relative_to(ROOT)), "Not indexed")} for p in pdfs], width="stretch", hide_index=True)
            selected = st.selectbox("Download a source", pdfs, format_func=lambda p: str(p.relative_to(PDFS)))
            st.download_button("Download PDF", selected.read_bytes(), file_name=selected.name, mime="application/pdf")
        else:
            st.info("No PDFs yet. Upload files above or copy them into data/pdfs/.")

    elif page == "Web & YouTube":
        st.write("Save individual page or video links. Saving a link does not fetch or index it.")
        with st.form("add_url", clear_on_submit=True):
            kind_label = st.selectbox("Source type", ["Website", "YouTube video"])
            value = st.text_input("HTTPS URL", placeholder="https://…")
            submitted = st.form_submit_button("Save link", type="primary")
        if submitted:
            kind = "websites" if kind_label == "Website" else "youtube"
            try:
                url = validate_url(value, kind)
                sources.setdefault(kind, [])
                if url not in sources[kind]:
                    sources[kind].append(url)
                    write_json(CONFIG / "sources.json", sources)
                    st.rerun()
                st.info("This link is already saved.")
            except (OSError, ValueError) as error:
                st.error(str(error))
        for kind, url in urls:
            left, right = st.columns([5, 1])
            left.text(f"{'YouTube' if kind == 'youtube' else 'Website'} · {url}")
            if right.button("Remove", key=f"remove:{kind}:{url}"):
                sources[kind] = [saved for saved in sources[kind] if saved != url]
                write_json(CONFIG / "sources.json", sources)
                st.rerun()
        st.caption("YouTube ingestion uses available captions; visuals are not extracted. After adding or removing sources, update the index. Removed links are excluded from queries immediately.")

    elif page == "Models":
        render_models(ROOT, settings)

    elif page == "Ask":
        scope = st.radio("Use evidence from", ["PDF only", "All sources", "Selected sources"], horizontal=True)
        choices = {s["id"]: f'{s["kind"]}: {s["locator"]}' for s in current_sources}
        selected = []
        if scope == "Selected sources":
            selected = st.multiselect("Choose sources", list(choices), format_func=choices.get)
        evidence_only = st.checkbox("Evidence only — retrieve passages without generated synthesis")
        question = st.text_area("Your question", placeholder="What do my documents say about …?")
        if st.button("Ask your sources", disabled=not question.strip(), type="primary"):
            try:
                with st.spinner("Searching selected sources…"):
                    result = engine.ask(question, sources, {"PDF only": "pdf", "All sources": "all", "Selected sources": "selected"}[scope], selected, evidence_only)
                for warning in result["warnings"]:
                    st.warning(warning)
                if result["status"] == "insufficient_evidence":
                    st.info("The selected indexed sources do not provide enough evidence to answer. Check source scope and indexing status.")
                elif result["status"] == "evidence_only":
                    st.success("Retrieved passages; no generated answer.")
                else:
                    for claim in result["claims"]:
                        # Model text is plain text: never render remote Markdown images/links.
                        st.text(claim["text"])
                        st.caption("Evidence: " + ", ".join(f'[{i}]' for i in claim["citations"]))
                for item in result["evidence"]:
                    location = item["location"]
                    label = item["locator"]
                    if "page" in location:
                        label += f' · physical page {location["page"]}'
                    if "start" in location:
                        label += f' · {location["start"]:.0f}–{location["end"]:.0f}s'
                    with st.expander(f'[{item["citation"]}] {label}'):
                        st.text(item["text"])
                        st.caption(f'Indexed {item["updated"]} · similarity {item["similarity"]:.3f}')
                        url = location.get("url")
                        if url and item["kind"] == "youtube":
                            url += f'&t={int(location.get("start", 0))}'
                        if url:
                            st.link_button("Open source", url)
            except RagError as error:
                st.error(str(error))
        st.caption("Citation IDs are validated, but a citation alone does not prove factual support. Inspect the evidence. Questions are independent; chat history is not used.")

    elif page == "Index & activity":
        st.write("Build or synchronize your index. Unchanged PDFs are skipped. Old versions survive failed updates. Removed sources are deactivated during synchronization.")
        include_external = st.checkbox("Also fetch saved website and YouTube sources (requires internet)")
        if st.button("Build / update index", type="primary"):
            try:
                bar = st.progress(0.0)
                def progress(number, total, name):
                    bar.progress((number - 1) / total, text=f"Processing {name}")
                with st.spinner("Extracting text and creating local embeddings…"):
                    results = engine.ingest(sources, include_external=include_external, progress=progress)
                bar.progress(1.0, text="Ingestion completed")
                for row in results:
                    if row["status"] == "failed":
                        st.error(f'{row["source"]}: {row["error"]}')
                    else:
                        st.success(f'{row["source"]}: {row["status"]}')
                        for warning in row.get("warnings", []):
                            st.warning(warning)
                if not results:
                    st.info("No sources selected for ingestion. Folder/manifest synchronization completed.")
            except RagError as error:
                st.error(str(error))
        rows = engine.store.summary()
        st.dataframe([{"Source": r["locator"], "Type": r["kind"], "Active": bool(r["active"]), "Passages": r["chunks"], "Indexed (UTC)": r["updated"], "Error": r["error"] or "", "Warnings": "; ".join(json.loads(r["warnings"]))} for r in rows], width="stretch")
        for run in engine.store.recent_runs():
            with st.expander(f'Run {run["id"]} · {run["finished"]}'):
                st.json(json.loads(run["results"]))

    elif page == "Settings":
        st.write("Settings are saved locally. Use the Models view to compare, download, and select pretrained models. Changed manual model names are checked with a synthetic compatibility sample before saving.")
        with st.form("settings"):
            generation = st.text_input("Ollama generation model", settings["generation_model"])
            embedding = st.text_input("Ollama embedding model", settings["embedding_model"])
            st.caption(f"Local model endpoint: {settings['ollama_base_url']}")
            chunks = st.number_input("Context passages", min_value=1, max_value=20, value=int(settings["context_chunks"]))
            budget = st.number_input("Context passage budget (UTF-8 bytes)", min_value=500, max_value=16000, value=int(settings["context_token_budget"]), step=100)
            tokens = st.number_input("Answer token budget", min_value=100, max_value=4000, value=int(settings["answer_token_budget"]), step=100)
            similarity = st.slider("Minimum retrieval similarity (calibrate on your documents)", 0.0, 1.0, float(settings.get("minimum_similarity", 0.35)), 0.01)
            if st.form_submit_button("Save settings", type="primary"):
                if not generation.strip() or not embedding.strip():
                    st.error("Both model names are required.")
                else:
                    try:
                        with st.spinner("Validating changed model names before saving…"):
                            if canonical_tag(generation.strip()) != canonical_tag(settings["generation_model"]):
                                engine.client.check_compatibility(generation.strip(), "generation")
                            if canonical_tag(embedding.strip()) != canonical_tag(settings["embedding_model"]):
                                engine.client.check_compatibility(embedding.strip(), "embedding")
                        settings.update(generation_model=generation.strip(), embedding_model=embedding.strip(), context_chunks=int(chunks), context_token_budget=int(budget), answer_token_budget=int(tokens), minimum_similarity=similarity)
                        write_json(CONFIG / "settings.json", settings)
                        st.success("Settings saved. Changing the embedding model or its installed digest requires reindexing.")
                    except RagError as error:
                        st.error(f"Settings unchanged: {error}")
        st.code(f'ollama pull {settings["generation_model"]}\nollama pull {settings["embedding_model"]}', language="bash")
        if st.button("Check local model setup"):
            try:
                for model in (settings["generation_model"], settings["embedding_model"]):
                    st.success(engine.client.identity(model))
            except RagError as error:
                st.error(str(error))
        st.download_button("Export settings", json.dumps(settings, indent=2), "rag-settings.json", "application/json")
        st.caption("PDF-only remains the default. Automatic web search and link crawling remain disabled.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, sqlite3.Error, RagError) as error:
        st.error(f"Workspace operation failed: {error}")
