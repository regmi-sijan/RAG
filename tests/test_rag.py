"""Pipeline integration tests use real PDFs/SQLite and a deterministic model double.

These verify routing, persistence and validation, not real-model answer quality.
"""
import json
import tempfile
import unittest
import requests
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.rag.chunking import chunks
from src.rag.engine import Engine, configured_sources, validate_answer
from src.rag.extraction import extract_pdf, extract_website, extract_youtube, fetch_page, public_addresses
from src.rag.models import Ollama, RagError, unit_vector


def make_pdf(path, text):
    from pypdf import PdfWriter
    from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                             NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"):
        DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode())
    page[NameObject("/Contents")] = writer._add_object(stream)
    with path.open("wb") as output:
        writer.write(output)


class FakeModels:
    def __init__(self):
        self.digest = "local-embedding@digest-one"
        self.embed_calls = 0
        self.generate_calls = 0
        self.fail = False
        self.output = json.dumps({"status": "answered", "claims": [
            {"text": "Employees receive 15 days of annual leave.", "citations": [1]}]})
        self.messages = []

    def identity(self, model):
        return self.digest

    def check_compatibility(self, model, role):
        return {"model": model, "role": role, "identity": self.digest, "detail": "Synthetic check passed."}

    def embed(self, texts, query=False):
        if self.fail:
            raise RagError("Embedding service unavailable")
        self.embed_calls += 1
        return [[0.0, 1.0] if "galaxy" in text.lower() else [1.0, 0.0] for text in texts]

    def generate(self, messages, schema):
        self.generate_calls += 1
        self.messages = messages
        return self.output


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "data/pdfs").mkdir(parents=True)
        self.pdf = self.root / "data/pdfs/handbook.pdf"
        make_pdf(self.pdf, "Employees receive 15 days of annual leave.")
        self.settings = json.loads(Path("config/settings.example.json").read_text())
        self.client = FakeModels()
        self.engine = Engine(self.root, self.settings, self.client)
        self.manifest = {"websites": [], "youtube": []}

    def ingest(self):
        result = self.engine.ingest(self.manifest)
        self.assertEqual(result[0]["status"], "indexed", result)
        return result

    def test_pdf_to_index_to_cited_answer_and_restart(self):
        self.ingest()
        engine = Engine(self.root, self.settings, self.client)
        answer = engine.ask("How much annual leave?", self.manifest)
        self.assertEqual(answer["status"], "answered")
        self.assertEqual(answer["claims"][0]["citations"], [1])
        self.assertEqual(answer["evidence"][0]["location"]["page"], 1)
        self.assertIn("15 days", answer["evidence"][0]["text"])

    def test_empty_index_abstains_without_model(self):
        self.assertEqual(self.engine.ask("What is the policy?", self.manifest)["status"], "insufficient_evidence")
        self.assertEqual(self.client.embed_calls, 0)

    def test_incremental_ingestion_and_replacement(self):
        self.ingest()
        calls = self.client.embed_calls
        self.assertEqual(self.engine.ingest(self.manifest)[0]["status"], "unchanged")
        self.assertEqual(self.client.embed_calls, calls)
        make_pdf(self.pdf, "Employees receive 20 days of annual leave.")
        self.ingest()
        evidence, _ = self.engine.retrieve("Annual leave", self.manifest)
        self.assertIn("20 days", evidence[0]["text"])
        self.assertNotIn("15 days", " ".join(e["text"] for e in evidence))

    def test_failed_update_preserves_previous_version(self):
        self.ingest()
        make_pdf(self.pdf, "New policy text")
        self.client.fail = True
        self.assertEqual(self.engine.ingest(self.manifest)[0]["status"], "failed")
        self.client.fail = False
        evidence, warnings = self.engine.retrieve("Annual leave", self.manifest)
        self.assertIn("15 days", evidence[0]["text"])
        self.assertTrue(any("previous version" in warning for warning in warnings))

    def test_removed_pdf_immediately_excluded_and_synchronized(self):
        self.ingest()
        self.pdf.unlink()
        self.assertEqual(self.engine.ask("Annual leave", self.manifest)["evidence"], [])
        self.engine.ingest(self.manifest)
        self.assertFalse(self.engine.store.summary()[0]["active"])

    def test_external_scope_never_leaks_into_pdf_context(self):
        self.manifest["websites"] = ["https://example.com/policy"]
        with patch("src.rag.engine.extract_website", return_value=([
                {"text": "WEBSITE_ONLY_SECRET: Annual leave is 99 days.", "url": "https://example.com/policy"}], [])):
            self.engine.ingest(self.manifest, include_external=True)
        answer = self.engine.ask("Annual leave", self.manifest, scope="pdf")
        self.assertTrue(all(e["kind"] == "pdf" for e in answer["evidence"]))
        self.assertNotIn("WEBSITE_ONLY_SECRET", json.dumps(self.client.messages))
        self.assertEqual(len(self.engine.retrieve("Annual leave", self.manifest, "all")[0]), 2)
        source = next(s for s in configured_sources(self.root, self.manifest) if s["kind"] == "website")
        evidence, _ = self.engine.retrieve("Annual leave", self.manifest, "selected", [source["id"]])
        self.assertTrue(all(e["kind"] == "website" for e in evidence))
        self.manifest["websites"] = []
        self.assertTrue(all(e["kind"] == "pdf" for e in self.engine.retrieve("Annual leave", self.manifest, "all")[0]))

    def test_default_ingestion_does_not_fetch_urls(self):
        self.manifest["websites"] = ["https://example.com"]
        with patch("src.rag.engine.extract_website") as fetch:
            self.ingest()
            fetch.assert_not_called()

    def test_fabricated_citations_rejected_after_retry(self):
        self.ingest()
        self.client.output = '{"status":"answered","claims":[{"text":"Invented","citations":[999]}]}'
        answer = self.engine.ask("Annual leave", self.manifest)
        self.assertEqual(answer["status"], "insufficient_evidence")
        self.assertEqual(answer["claims"], [])
        self.assertEqual(self.client.generate_calls, 2)

    def test_low_similarity_abstains_without_generation(self):
        self.ingest()
        answer = self.engine.ask("What is a galaxy?", self.manifest)
        self.assertEqual(answer["status"], "insufficient_evidence")
        self.assertEqual(self.client.generate_calls, 0)

    def test_evidence_only_skips_generation(self):
        self.ingest()
        self.assertEqual(self.engine.ask("Annual leave", self.manifest, evidence_only=True)["status"], "evidence_only")
        self.assertEqual(self.client.generate_calls, 0)

    def test_changed_model_digest_requires_reindexing(self):
        self.ingest()
        self.client.digest = "local-embedding@digest-two"
        with self.assertRaisesRegex(RagError, "digest"):
            self.engine.ask("Annual leave", self.manifest)
        self.ingest()
        self.assertTrue(self.engine.retrieve("Annual leave", self.manifest)[0])

    def test_scanned_pdf_reports_no_usable_text(self):
        from pypdf import PdfWriter
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        with self.pdf.open("wb") as stream:
            writer.write(stream)
        self.assertEqual(self.engine.ingest(self.manifest)[0]["status"], "failed")
        self.assertEqual(self.engine.store.summary()[0]["chunks"], 0)

    def test_malformed_output_abstains(self):
        self.ingest()
        self.client.output = "not JSON"
        self.assertEqual(self.engine.ask("Annual leave", self.manifest)["status"], "insufficient_evidence")

    def test_invalid_scope_is_rejected(self):
        with self.assertRaises(RagError):
            self.engine.ask("Annual leave", self.manifest, scope="unknown")

    def test_small_embedding_model_caps_chunks_and_warns_about_long_query(self):
        self.settings["embedding_model"] = "all-minilm:22m"
        make_pdf(self.pdf, "Employees receive 15 days of annual leave. " * 40)
        engine = Engine(self.root, self.settings, self.client)
        with patch.object(self.client, "embed", wraps=self.client.embed) as embed:
            results = engine.ingest(self.manifest)
            self.assertEqual(results[0]["status"], "indexed")
            self.assertTrue(any("224" in warning for warning in results[0]["warnings"]))
            passage_inputs = embed.call_args.args[0]
            self.assertGreater(len(passage_inputs), 1)
            self.assertTrue(all(len(text.encode()) <= 224 for text in passage_inputs))
        question = "Annual leave question? " * 40
        result = engine.ask(question, self.manifest)
        self.assertEqual(result["status"], "answered")
        self.assertTrue(any("shortened" in warning for warning in result["warnings"]))
        self.assertEqual(json.loads(self.client.messages[1]["content"])["question"], question.strip())


class ExtractionAndModelTests(unittest.TestCase):
    def test_public_address_check_blocks_loopback_and_private(self):
        for address in ("127.0.0.1", "10.0.0.2", "::1", "169.254.169.254"):
            with patch("socket.getaddrinfo", return_value=[(None, None, None, None, (address, 443))]):
                with self.assertRaises(RagError):
                    public_addresses("example.com", 443)

    def test_redirect_revalidates_scheme(self):
        response = SimpleNamespace(status=302, getheader=lambda name: "http://127.0.0.1/secret")
        connection = SimpleNamespace(request=lambda *a, **k: None, getresponse=lambda: response, close=lambda: None)
        with patch("src.rag.extraction.PublicHTTPS", return_value=connection):
            with self.assertRaises(ValueError):
                fetch_page("https://example.com")

    def test_website_main_text_extraction(self):
        html = b"<html><body><article><h1>Leave policy</h1><p>Employees receive 15 days of annual leave. Requests must be approved by a manager.</p></article></body></html>"
        with patch("src.rag.extraction.fetch_page", return_value=(html, "https://example.com/final", "text/html")):
            units, warnings = extract_website("https://example.com")
        self.assertIn("15 days", units[0]["text"])
        self.assertEqual(units[0]["url"], "https://example.com/final")

    def test_youtube_preserves_time_intervals(self):
        segments = [SimpleNamespace(text="Hello", start=0.0, duration=25.0),
                    SimpleNamespace(text="world", start=25.0, duration=25.0)]
        with patch("youtube_transcript_api.YouTubeTranscriptApi.fetch", return_value=segments):
            units, warnings = extract_youtube("https://youtu.be/dQw4w9WgXcQ", ["en"])
        self.assertEqual(units[0]["text"], "Hello world")
        self.assertEqual(units[0]["start"], 0)
        self.assertEqual(units[0]["end"], 50)

    def test_unicode_chunk_boundaries_and_size(self):
        text = "नमस्ते " * 100
        result = chunks([{"text": text, "page": 3}], 100, 20)
        self.assertGreater(len(result), 1)
        self.assertTrue(all(len(r["text"].encode()) <= 100 and r["page"] == 3 for r in result))

    def test_invalid_vectors_and_citations(self):
        for vector in ([0, 0], [float("nan"), 1], [True, 1], []):
            with self.assertRaises(RagError):
                unit_vector(vector)
        for citations in ([], [True], ["1"], [2]):
            with self.assertRaises(ValueError):
                validate_answer({"status": "answered", "claims": [{"text": "claim", "citations": citations}]}, [1])

    def test_nonlocal_model_endpoint_rejected(self):
        with self.assertRaises(RagError):
            Ollama({"ollama_base_url": "https://example.com"})

    def test_ollama_model_digest_and_cloud_rejection(self):
        client = Ollama({"ollama_base_url": "http://127.0.0.1:11434"})
        tags = {"models": [{"name": "test:latest", "digest": "abc"}]}
        with patch.object(client, "request", side_effect=[tags, {}]):
            self.assertEqual(client.identity("test"), "test:latest@abc")
        with patch.object(client, "request", side_effect=[tags, {"remote_host": "cloud"}]):
            with self.assertRaises(RagError):
                client.identity("test")

    def test_ollama_embedding_payload_and_normalization(self):
        client = Ollama({"ollama_base_url": "http://127.0.0.1:11434", "embedding_model": "nomic-embed-text"})
        response = SimpleNamespace(status_code=200, json=lambda: {"embeddings": [[3.0, 4.0]]})
        with patch.object(client.session, "request", return_value=response) as request:
            self.assertEqual(client.embed(["policy"]), [[0.6, 0.8]])
            self.assertEqual(request.call_args.kwargs["json"]["input"], ["search_document: policy"])
            self.assertFalse(request.call_args.kwargs["allow_redirects"])
            client.embed(["question"], query=True)
            self.assertEqual(request.call_args.kwargs["json"]["input"], ["search_query: question"])

    def test_ollama_connection_error_is_actionable(self):
        client = Ollama({"ollama_base_url": "http://127.0.0.1:11434"})
        with patch.object(client.session, "request", side_effect=requests.ConnectionError("offline")):
            with self.assertRaisesRegex(RagError, "Start Ollama"):
                client.request("/api/tags")

    def test_ollama_generation_uses_schema_and_local_chat(self):
        settings = {"ollama_base_url": "http://127.0.0.1:11434", "generation_model": "test",
                    "context_token_budget": 3500, "answer_token_budget": 700}
        client = Ollama(settings)
        response = SimpleNamespace(status_code=200, json=lambda: {"message": {"content": '{"status":"insufficient_evidence","claims":[]}'}})
        schema = {"type": "object"}
        with patch.object(client.session, "request", return_value=response) as request:
            self.assertIn("insufficient_evidence", client.generate([], schema))
            self.assertEqual(request.call_args.args[1], "http://127.0.0.1:11434/api/chat")
            self.assertEqual(request.call_args.kwargs["json"]["format"], schema)
            self.assertFalse(request.call_args.kwargs["json"]["stream"])


if __name__ == "__main__":
    unittest.main()
