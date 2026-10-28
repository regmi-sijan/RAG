"""Streamlit model library, kept separate from the retrieval pipeline."""
import streamlit as st

from .catalog import CATALOG, canonical_tag, choice_for, format_size, memory_label
from .model_server import PROJECT_URL, managed_server, start_server, stop_server, storage_directory
from .models import Ollama, RagError
from .workspace import write_json


def _installed_map(models):
    return {canonical_tag(item.get("name") or item["model"]): item for item in models
            if isinstance(item.get("name") or item.get("model"), str) and (item.get("name") or item.get("model"))}


def _notice(level, message):
    st.session_state["model_library_notice"] = (level, message)
    st.rerun()


def _use_model(root, settings, client, tag, role):
    with st.spinner("Checking the installed model before changing your selection…"):
        report = client.check_compatibility(tag, role)
    key = "embedding_model" if role == "embedding" else "generation_model"
    changed = canonical_tag(settings[key]) != canonical_tag(tag)
    saved = {**settings, key: tag}
    write_json(root / "config/settings.json", saved)
    message = f"Selected {tag} for {'retrieval embeddings' if role == 'embedding' else 'answer generation'}. {report['detail']}"
    if role == "embedding" and changed:
        _notice("warning", message + " Rebuild the index before asking questions.")
    _notice("success", message + (" No reindexing is needed for a generation-model change." if role == "generation" else ""))


def render_models(root, settings):
    st.write("Compare local models, download them, and choose which ones your RAG uses.")
    notice = st.session_state.pop("model_library_notice", None)
    if notice:
        getattr(st, notice[0])(notice[1])

    st.subheader("Shared download location")
    st.caption(f'Active Ollama server: {settings["ollama_base_url"]}')
    owned = managed_server(root)
    if owned and settings["ollama_base_url"].rstrip("/") == PROJECT_URL:
        st.success("Project model library is active. All downloads from this server use this folder:")
        st.code(str(storage_directory(root)), language=None)
    else:
        st.info("The current server controls model storage. Ollama's API does not report its folder. To keep GUI downloads together inside this project, start the project server below.")
        st.caption(f"Project library folder: {storage_directory(root)}")

    st.caption("Ollama must be installed first. Starting the project server does not download any model or move existing models. It uses a separate local port and keeps at most one model loaded at a time to limit memory use.")
    st.link_button("Install Ollama", "https://ollama.com/download")
    if st.button("Start project server and use this folder", type="primary"):
        try:
            with st.spinner("Starting the project model server…"):
                endpoint = start_server(root)
            write_json(root / "config/settings.json", {**settings, "ollama_base_url": endpoint})
            _notice("success", "Project server connected. Downloads will be kept in data/models/.")
        except (OSError, RagError) as error:
            st.error(str(error))
    if owned and st.button("Stop project model server"):
        stop_server(root)
        _notice("success", "Project server stopped. Downloaded files are preserved; restart it to use them.")

    with st.expander("Use an existing local Ollama server instead"):
        st.caption("Its storage folder is configured outside this GUI with OLLAMA_MODELS. Connecting never moves or deletes files. Switching servers may require reindexing if the installed embedding digest differs.")
        with st.form("existing_model_server"):
            endpoint = st.text_input("Local Ollama endpoint", "http://localhost:11434" if settings["ollama_base_url"] == PROJECT_URL else settings["ollama_base_url"])
            if st.form_submit_button("Connect existing server"):
                try:
                    endpoint = endpoint.strip().rstrip("/")
                    candidate = Ollama({**settings, "ollama_base_url": endpoint})
                    candidate.installed_models()
                    write_json(root / "config/settings.json", {**settings, "ollama_base_url": endpoint})
                    _notice("success", "Connected to the existing local model library. Its storage folder is managed by that server.")
                except (OSError, RagError) as error:
                    st.error(str(error))

    st.divider()
    st.subheader("Model catalog")
    st.info("Curated for an Apple M1 Mac with 16 GB unified memory: eight small answer models and two embedding models. These are hardware-size candidates, not a guarantee of answer quality. Downloading leaves the current selection unchanged; Use selected model runs a compatibility check before saving.")
    st.caption("For the lowest model-memory pair here, choose qwen2.5:0.5b for answers and all-minilm:22m for embeddings. Keep nomic-embed-text if you prefer a larger retrieval input window. Changing embeddings requires reindexing and may need similarity-threshold recalibration.")
    st.warning("RAM values below are planning estimates for one model at about 8K context, not measured requirements or guarantees. They exclude macOS, the GUI/browser, your index, and any other loaded model. On your Mac, CPU and GPU share unified memory. Longer context and concurrent work increase usage.")
    st.caption("Parameter counts, quantization and download sizes: Ollama library snapshot, 28 October 2026. Tags may change. Installed sizes come from your local server; shared layers mean these are not unique additional disk usage.")
    context = max(8192, int(settings["context_token_budget"]) + int(settings["answer_token_budget"]) + 2048)
    st.caption(f"Current RAG generation context allocation: {context:,} tokens. Catalog RAM ranges assume about 8,192 tokens; higher settings can exceed these estimates.")
    st.button("Refresh model status")
    client = Ollama(settings)
    connected, installed = True, {}
    try:
        installed = _installed_map(client.installed_models())
    except RagError as error:
        connected = False
        st.warning(str(error))

    st.dataframe([{
        "Model": choice.tag,
        "Role": "Answer generation" if choice.role == "generation" else "Embeddings / retrieval",
        "Parameters": choice.parameters,
        "Quantization": choice.quantization,
        "Expected disk size": format_size(choice.download_bytes),
        "Estimated model RAM": memory_label(choice),
        "Installed disk size": format_size(installed.get(choice.tag, {}).get("size")) if choice.tag in installed else "Not installed",
    } for choice in CATALOG], width="stretch", hide_index=True)

    role = st.selectbox("Model purpose", ["Answer generation", "Embeddings / retrieval"])
    role_key = "generation" if role == "Answer generation" else "embedding"
    low_memory = st.checkbox("Show only lower-memory choices (estimated upper range up to 3 GB per model)")
    options = [choice for choice in CATALOG if choice.role == role_key and (not low_memory or choice.memory_gb[1] <= 3)]
    tag = st.selectbox("Choose a pretrained model", [choice.tag for choice in options], key="catalog_model")
    choice = choice_for(tag)
    cols = st.columns(3)
    cols[0].metric("Parameters", choice.parameters)
    cols[1].metric("Expected disk size", format_size(choice.download_bytes))
    cols[2].metric("Estimated model RAM", memory_label(choice))
    st.caption(choice.note)
    st.link_button("Model details and license", choice.url)
    st.caption(f'Current answer model: {settings["generation_model"]} · Current embedding model: {settings["embedding_model"]}')

    if st.button("Download selected model", disabled=not connected or tag in installed, type="primary"):
        status = st.empty()
        bar = st.progress(0.0, text="Waiting for download progress…")
        def progress(event):
            status.text(str(event.get("status", "Downloading…"))[:300])
            total, completed = event.get("total"), event.get("completed", 0)
            if type(total) is int and type(completed) is int and total > 0 and completed >= 0:
                bar.progress(min(1.0, completed / total), text=f"Current layer: {format_size(completed)} / {format_size(total)}")
        try:
            client.pull(tag, progress=progress)
            bar.progress(1.0, text="Model download verified")
            _notice("success", f"Downloaded {tag}. Select Use selected model to make it active.")
        except RagError as error:
            st.error(str(error))
    if tag in installed:
        st.success(f'Installed: {tag} · {format_size(installed[tag].get("size"))} reported by Ollama')
    if st.button("Check selected model compatibility", disabled=tag not in installed):
        try:
            with st.spinner("Loading the model with a small synthetic sample…"):
                report = client.check_compatibility(tag, role_key)
            st.success(report["detail"])
        except RagError as error:
            st.error(str(error))
    if st.button("Use selected model", disabled=tag not in installed):
        try:
            _use_model(root, settings, client, tag, role_key)
        except (OSError, RagError) as error:
            st.error(str(error))
    st.caption("Download requires internet and can take several minutes. It does not load the weights into RAM. If interrupted, retry; Ollama may resume partial layers. Downloaded weights are stored in Ollama's blobs/manifests format, not one editable file per model.")
    st.caption("Compatibility checking and activation do load the selected model into memory. They test structured cited output or embedding dimensions with synthetic data. Failure keeps your current settings. A passing sample does not establish answer accuracy on your PDFs. Keep Ollama up to date for newer model families.")

    st.subheader("Installed and loaded models")
    running = {}
    if connected:
        try:
            running = _installed_map(client.running_models())
        except RagError as error:
            st.warning(str(error))
    if installed:
        st.dataframe([{
            "Model": name, "Parameters": (item["details"] if isinstance(item.get("details"), dict) else {}).get("parameter_size", "Unknown"),
            "Installed disk size": format_size(item.get("size")),
            "Loaded size (Ollama-reported)": format_size(running.get(name, {}).get("size")) if name in running else "Not loaded",
            "Loaded context": running.get(name, {}).get("context_length", "—"),
        } for name, item in sorted(installed.items())], width="stretch", hide_index=True)
        with st.expander("Use another installed model"):
            existing = st.selectbox("Installed model", sorted(installed))
            existing_role = st.selectbox("Use it for", ["Answer generation", "Embeddings / retrieval"])
            if st.button("Use installed model"):
                try:
                    _use_model(root, settings, client, existing, "generation" if existing_role == "Answer generation" else "embedding")
                except (OSError, RagError) as error:
                    st.error(str(error))
    else:
        st.info("No installed models reported. Connect a server and download a model above.")
    if running:
        loaded = st.selectbox("Loaded model to release from memory", sorted(running))
        if st.button("Unload selected model from RAM"):
            try:
                client.unload(loaded)
                _notice("success", "Unload requested. Downloaded model files remain on disk; refresh status to check completion.")
            except RagError as error:
                st.error(str(error))
    st.caption("Loaded size is what Ollama reports for its model allocation; it is not total process/system RAM. Do not add GPU memory again on an Apple Silicon Mac. Activity Monitor shows overall system memory pressure. Evidence-only mode avoids loading a generation model.")
