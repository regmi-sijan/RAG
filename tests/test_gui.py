"""Isolated GUI interaction test: real ingestion/storage, mocked model service."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest
from src.rag.models import RagError
from src.rag.model_server import PROJECT_URL
from src.rag.catalog import CATALOG
from test_rag import FakeModels, make_pdf


class GuiTests(unittest.TestCase):
    def test_navigation_index_and_ask(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "config").mkdir()
            (root / "docs").mkdir()
            (root / "data/pdfs").mkdir(parents=True)
            for name in ("settings.example.json", "sources.example.json"):
                shutil.copy(Path("config") / name, root / "config" / name)
            shutil.copy("docs/architecture.md", root / "docs/architecture.md")
            make_pdf(root / "data/pdfs/handbook.pdf", "Employees receive 15 days of annual leave.")
            source = Path("app.py").read_text().replace("ROOT = Path(__file__).resolve().parent", f"ROOT = Path({str(root)!r})")
            fake = FakeModels()
            with patch("src.rag.engine.Ollama", return_value=fake):
                app = AppTest.from_string(source, default_timeout=30).run()
                self.assertFalse(app.exception, app.exception)
                for page in ("PDF library", "Web & YouTube", "Ask", "Index & activity", "Settings"):
                    app.sidebar.radio[0].set_value(page).run()
                    self.assertFalse(app.exception, app.exception)
                app.sidebar.radio[0].set_value("Index & activity").run()
                next(b for b in app.button if b.label == "Build / update index").click().run()
                self.assertFalse(app.exception, app.exception)
                self.assertTrue(any("indexed" in item.value for item in app.success))
                app.sidebar.radio[0].set_value("Ask").run()
                app.text_area[0].set_value("How much annual leave?").run()
                next(b for b in app.button if b.label == "Ask your sources").click().run()
                self.assertFalse(app.exception, app.exception)
                self.assertTrue(any("15 days" in item.value for item in app.text))
                self.assertTrue(any("physical page 1" in item.label for item in app.expander))
                # Persist settings through the real GUI form, leaving defaults unchanged.
                app.sidebar.radio[0].set_value("Settings").run()
                app.text_input[0].set_value("another-local-model")
                next(b for b in app.button if b.label == "Save settings").click().run()
                self.assertFalse(app.exception, app.exception)
                self.assertEqual(json.loads((root / "config/settings.json").read_text())["generation_model"], "another-local-model")
                with patch.object(fake, "identity", side_effect=RagError("Start Ollama and install the configured models.")):
                    next(b for b in app.button if b.label == "Check local model setup").click().run()
                    self.assertFalse(app.exception, app.exception)
                    self.assertTrue(any("Start Ollama" in item.value for item in app.error))


class ModelLibraryGuiTests(unittest.TestCase):
    def test_duplicate_saved_urls_render_once_and_remove_together(self):
        path = self.root / "config/sources.json"
        path.write_text(json.dumps({"websites": ["https://example.com/page"] * 2, "youtube": []}))
        app = AppTest.from_string(self.source).run()
        app.sidebar.radio[0].set_value("Web & YouTube").run()
        self.assertFalse(app.exception, app.exception)
        remove = [button for button in app.button if button.label == "Remove"]
        self.assertEqual(len(remove), 1)
        remove[0].click().run()
        self.assertFalse(app.exception, app.exception)
        self.assertEqual(json.loads(path.read_text())["websites"], [])

    def test_null_installed_model_details_do_not_crash_catalog(self):
        self.client.installed_models.return_value = [{"name": "qwen2.5:0.5b", "size": 398000000, "details": None}]
        self.app()

    def test_invalid_url_does_not_create_manifest(self):
        app = AppTest.from_string(self.source).run()
        app.sidebar.radio[0].set_value("Web & YouTube").run()
        app.text_input[0].set_value("https://example.com:bad/page")
        next(button for button in app.button if button.label == "Save link").click().run()
        self.assertFalse(app.exception, app.exception)
        self.assertTrue(app.error)
        self.assertFalse((self.root / "config/sources.json").exists())

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "config").mkdir()
        (self.root / "docs").mkdir()
        for name in ("settings.example.json", "sources.example.json"):
            shutil.copy(Path("config") / name, self.root / "config" / name)
        shutil.copy("docs/architecture.md", self.root / "docs/architecture.md")
        self.source = Path("app.py").read_text().replace("ROOT = Path(__file__).resolve().parent", f"ROOT = Path({str(self.root)!r})")
        self.client = MagicMock()
        self.client.installed_models.return_value = []
        self.client.running_models.return_value = []
        self.client.validate_role.return_value = "model@digest"
        self.client.check_compatibility.return_value = {"detail": "Synthetic compatibility check passed."}
        self.client_patch = patch("src.rag.model_ui.Ollama", return_value=self.client)
        self.client_patch.start()
        self.addCleanup(self.client_patch.stop)
        self.owned_patch = patch("src.rag.model_ui.managed_server", return_value=None)
        self.owned = self.owned_patch.start()
        self.addCleanup(self.owned_patch.stop)

    def app(self):
        app = AppTest.from_string(self.source, default_timeout=30).run()
        app.sidebar.radio[0].set_value("Models").run()
        self.assertFalse(app.exception, app.exception)
        return app

    def test_download_then_select_generation_model(self):
        def pull(tag, progress=None):
            progress({"status": "pulling manifest"})
            progress({"status": "downloading", "total": 100, "completed": 50})
            progress({"status": "success"})
            self.client.installed_models.return_value = [{"name": tag, "size": 398_000_000,
                "details": {"parameter_size": "494M"}}]
            return "model@digest"
        self.client.pull.side_effect = pull
        app = self.app()
        self.assertTrue(any("494 million" in metric.value for metric in app.metric))
        self.assertTrue(any("not measured requirements" in item.value for item in app.warning))
        next(b for b in app.button if b.label == "Download selected model").click().run()
        self.assertFalse(app.exception, app.exception)
        self.client.pull.assert_called_once()
        self.assertTrue(any("Downloaded qwen2.5:0.5b" in item.value for item in app.success))
        next(b for b in app.button if b.label == "Use selected model").click().run()
        self.assertFalse(app.exception, app.exception)
        saved = json.loads((self.root / "config/settings.json").read_text())
        self.assertEqual(saved["generation_model"], "qwen2.5:0.5b")
        self.assertEqual(saved["embedding_model"], "nomic-embed-text")
        self.assertTrue(any("No reindexing" in item.value for item in app.success))

    def test_catalog_works_without_ollama_and_disables_download(self):
        self.client.installed_models.side_effect = RagError("Start Ollama")
        app = self.app()
        self.assertEqual(len(app.dataframe[0].value), len(CATALOG))
        self.assertTrue(next(b for b in app.button if b.label == "Download selected model").disabled)
        self.assertTrue(any("Start Ollama" in item.value for item in app.warning))
        self.client.pull.assert_not_called()

    def test_embedding_selection_shows_reindex_warning(self):
        settings = json.loads((self.root / "config/settings.example.json").read_text())
        settings["embedding_model"] = "old-embedding"
        (self.root / "config/settings.json").write_text(json.dumps(settings))
        self.client.installed_models.return_value = [{"name": "nomic-embed-text:latest", "size": 274_000_000}]
        app = self.app()
        next(w for w in app.selectbox if w.label == "Model purpose").set_value("Embeddings / retrieval").run()
        self.assertFalse(app.exception, app.exception)
        self.assertEqual(next(w for w in app.selectbox if w.label == "Choose a pretrained model").value, "nomic-embed-text:latest")
        next(b for b in app.button if b.label == "Use selected model").click().run()
        self.assertFalse(app.exception, app.exception)
        self.assertTrue(any("Rebuild the index" in item.value for item in app.warning))
        self.assertEqual(json.loads((self.root / "config/settings.json").read_text())["embedding_model"], "nomic-embed-text:latest")

    def test_project_server_control_saves_endpoint_and_displays_folder(self):
        def start(root):
            self.owned.return_value = {"directory": str(root / "data/models")}
            return PROJECT_URL
        app = self.app()
        with patch("src.rag.model_ui.start_server", side_effect=start) as start_server:
            next(b for b in app.button if b.label == "Start project server and use this folder").click().run()
        self.assertFalse(app.exception, app.exception)
        start_server.assert_called_once()
        self.assertEqual(json.loads((self.root / "config/settings.json").read_text())["ollama_base_url"], PROJECT_URL)
        self.assertTrue(any("data/models" in code.value for code in app.code))

    def test_download_error_is_displayed_without_changing_model(self):
        self.client.pull.side_effect = RagError("Model download failed: disk full")
        app = self.app()
        next(b for b in app.button if b.label == "Download selected model").click().run()
        self.assertFalse(app.exception, app.exception)
        self.assertTrue(any("disk full" in error.value for error in app.error))
        self.assertFalse((self.root / "config/settings.json").exists())

    def test_failed_compatibility_check_preserves_active_settings(self):
        original = json.loads((self.root / "config/settings.example.json").read_text())
        (self.root / "config/settings.json").write_text(json.dumps(original))
        self.client.installed_models.return_value = [{"name": "qwen2.5:0.5b", "size": 398_000_000}]
        self.client.check_compatibility.side_effect = RagError("Structured output unavailable")
        app = self.app()
        next(b for b in app.button if b.label == "Use selected model").click().run()
        self.assertFalse(app.exception, app.exception)
        self.assertTrue(any("Structured output unavailable" in item.value for item in app.error))
        self.assertEqual(json.loads((self.root / "config/settings.json").read_text()), original)

    def test_lower_memory_filter_limits_choices_for_both_roles(self):
        app = self.app()
        next(w for w in app.checkbox if w.label.startswith("Show only lower-memory")).check().run()
        choices = next(w for w in app.selectbox if w.label == "Choose a pretrained model").options
        self.assertEqual(set(choices), {c.tag for c in CATALOG if c.role == "generation" and c.memory_gb[1] <= 3})
        next(w for w in app.selectbox if w.label == "Model purpose").set_value("Embeddings / retrieval").run()
        self.assertFalse(app.exception, app.exception)
        choices = next(w for w in app.selectbox if w.label == "Choose a pretrained model").options
        self.assertEqual(set(choices), {c.tag for c in CATALOG if c.role == "embedding"})


if __name__ == "__main__":
    unittest.main()
