"""Regressions found by the frontend/backend audit; no external network or models."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.rag.engine import Engine, configured_sources
from src.rag.extraction import PublicHTTPS, extract_website, fetch_page
from src.rag.models import Ollama, RagError
from src.rag.workspace import read_json, save_pdf, validate_url
from test_rag import FakeModels, make_pdf


class InputAndExtractionRegressions(unittest.TestCase):
    def test_invalid_ports_and_brackets_rejected_before_saving(self):
        for url in ("https://example.com:bad/p", "https://example.com:99999/p", "https://example.com:8443/p", "https://[bad"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                validate_url(url, "websites")

    def test_plain_text_website_preserves_text_and_unicode(self):
        text = "Annual leave: 15 days. नमस्ते"
        with patch("src.rag.extraction.fetch_page", return_value=(text.encode(), "https://example.com/p.txt", "text/plain")):
            units, _ = extract_website("https://example.com/p.txt")
        self.assertEqual(units[0]["text"], text)

    def test_unicode_request_target_is_encoded(self):
        response = MagicMock()
        response.status = 200
        headers = {"Content-Type": "text/plain", "Content-Encoding": "identity"}
        response.getheader.side_effect = lambda key, default=None: headers.get(key, default)
        response.read.return_value = b"valid plain text"
        connection = MagicMock()
        connection.getresponse.return_value = response
        with patch("src.rag.extraction.PublicHTTPS", return_value=connection):
            self.assertEqual(fetch_page("https://example.com/नेपाल?q=नमस्ते")[0], b"valid plain text")
        target = connection.request.call_args.args[1]
        target.encode("ascii")
        self.assertIn("%E0", target)

    def test_unreachable_first_public_address_falls_back_safely(self):
        context = MagicMock()
        connection = PublicHTTPS("example.com", context=context)
        transport = MagicMock()
        with patch("src.rag.extraction.public_addresses", return_value=["2001:4860:4860::8888", "8.8.8.8"]), \
                patch("src.rag.extraction.socket.create_connection", side_effect=[OSError("no IPv6 route"), transport]) as connect:
            connection.connect()
        self.assertEqual(connect.call_count, 2)
        context.wrap_socket.assert_called_once_with(transport, server_hostname="example.com")

    def test_failed_upload_publish_leaves_no_partial_pdf(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            with patch("src.rag.workspace.os.link", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    save_pdf(directory, "incomplete.pdf", b"%PDF-1.4\n")
            self.assertEqual(list(directory.iterdir()), [])

    def test_non_object_configuration_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "settings.json"
            path.write_text("[]")
            with self.assertRaisesRegex(ValueError, "JSON object"):
                read_json(path, path)


class RetrievalRegressions(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "data/pdfs").mkdir(parents=True)
        make_pdf(self.root / "data/pdfs/policy.pdf", "Annual leave is 15 days.")
        self.settings = json.loads(Path("config/settings.example.json").read_text())
        self.client = FakeModels()
        self.engine = Engine(self.root, self.settings, self.client)
        self.manifest = {"websites": [], "youtube": []}
        self.engine.ingest(self.manifest)
        self.source = configured_sources(self.root, self.manifest)[0]

    def test_similar_but_conflicting_facts_are_not_deduplicated(self):
        common = "Annual leave policy for employees working regular hours with approval from their manager states that the annual leave entitlement is "
        passages = [{"text": common + "15 days.", "page": 1}, {"text": common + "20 days.", "page": 2}]
        old = self.engine.store.source(self.source["id"])
        self.engine.store.replace(self.source, old["content_hash"], old["fingerprint"], self.client.digest, passages, [[1, 0], [1, 0]], [])
        evidence, _ = self.engine.retrieve("Annual leave policy?", self.manifest)
        self.assertEqual(len(evidence), 2)
        self.assertTrue(any("15 days" in e["text"] for e in evidence))
        self.assertTrue(any("20 days" in e["text"] for e in evidence))

    def test_long_update_error_does_not_consume_evidence_budget(self):
        self.settings["context_token_budget"] = 500
        self.engine = Engine(self.root, self.settings, self.client)
        self.engine.store.failure(self.source, "x" * 2000)
        evidence, warnings = self.engine.retrieve("Annual leave?", self.manifest)
        self.assertEqual(len(evidence), 1)
        self.assertTrue(warnings)

    def test_corrupt_vectors_fail_closed_before_generation(self):
        with self.engine.store.connection() as db:
            db.execute("UPDATE chunks SET vector=?", ("[NaN, 1]",))
        with self.assertRaisesRegex(RagError, "Rebuild"):
            self.engine.ask("Annual leave?", self.manifest)
        self.assertEqual(self.client.generate_calls, 0)

    def test_question_type_is_validated(self):
        with self.assertRaisesRegex(RagError, "text"):
            self.engine.ask(None, self.manifest)


class ProtocolRegressions(unittest.TestCase):
    def setUp(self):
        self.settings = json.loads(Path("config/settings.example.json").read_text())
        self.client = Ollama(self.settings)

    def test_invalid_local_endpoint_is_actionable(self):
        for endpoint in ("http://[", "http://localhost:bad", "http://127.0.0.1:99999", "http://127.0.0.1:0"):
            with self.subTest(endpoint=endpoint), self.assertRaises(RagError):
                Ollama({**self.settings, "ollama_base_url": endpoint})

    def test_invalid_embedding_shapes_raise_rag_errors(self):
        for value in (None, "not vectors", {}, [None]):
            with self.subTest(value=value), patch.object(self.client, "request", return_value={"embeddings": value}):
                with self.assertRaises(RagError):
                    self.client.embed(["text"])

    def test_invalid_chat_message_does_not_crash_ui(self):
        for message in (None, [], {"content": None}):
            with self.subTest(message=message), patch.object(self.client, "request", return_value={"message": message}):
                with self.assertRaisesRegex(RagError, "invalid chat"):
                    self.client.generate([], {"type": "object"})

    def test_invalid_installed_model_list_is_actionable(self):
        with patch.object(self.client, "request", return_value={"models": [None]}):
            with self.assertRaises(RagError):
                self.client.identity("llama3.2:3b")


if __name__ == "__main__":
    unittest.main()
