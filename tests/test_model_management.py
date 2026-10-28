import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

from src.rag.catalog import CATALOG, canonical_tag, choice_for, format_size, embedding_byte_budget
from src.rag.models import Ollama, RagError
from src.rag import model_server


class ModelManagementTests(unittest.TestCase):
    def setUp(self):
        self.client = Ollama({"ollama_base_url": "http://127.0.0.1:11434"})

    def response(self, events, status=200):
        response = MagicMock()
        response.status_code = status
        response.__enter__.return_value = response
        response.iter_lines.return_value = iter(events)
        return response

    def test_catalog_separates_size_and_estimated_ram(self):
        self.assertEqual(canonical_tag("nomic-embed-text"), "nomic-embed-text:latest")
        self.assertEqual(choice_for("qwen2.5:0.5b").parameters, "494 million")
        self.assertEqual(format_size(398_000_000), "398 MB")
        self.assertEqual(format_size(None), "Unknown")
        self.assertEqual(sum(c.role == "embedding" for c in CATALOG), 2)
        self.assertEqual(sum(c.role == "generation" for c in CATALOG), 8)
        self.assertEqual(len({c.tag for c in CATALOG}), len(CATALOG))

    def test_pull_streams_layer_progress_and_verifies_install(self):
        response = self.response([b"", b'{"status":"pulling manifest"}',
            b'{"status":"downloading","digest":"sha256:abc","total":100,"completed":50}',
            b'{"status":"success"}'])
        events = []
        with patch.object(self.client.session, "post", return_value=response) as post, \
                patch.object(self.client, "identity", return_value="model@digest") as identity:
            self.assertEqual(self.client.pull("qwen2.5:0.5b", events.append), "model@digest")
        self.assertEqual(len(events), 3)
        self.assertEqual(events[1]["completed"], 50)
        self.assertEqual(post.call_args.kwargs["json"], {"model": "qwen2.5:0.5b", "stream": True})
        self.assertFalse(post.call_args.kwargs["allow_redirects"])
        identity.assert_called_once_with("qwen2.5:0.5b")
        response.__exit__.assert_called_once()

    def test_pull_requires_terminal_success(self):
        response = self.response([b'{"status":"pulling manifest"}'])
        with patch.object(self.client.session, "post", return_value=response):
            with self.assertRaisesRegex(RagError, "without a success"):
                self.client.pull("qwen2.5:0.5b")

    def test_pull_error_or_corrupt_stream_never_reports_success(self):
        for event in (b'{"error":"disk full"}', b'not json', b'[]'):
            response = self.response([event])
            with patch.object(self.client.session, "post", return_value=response), \
                    patch.object(self.client, "identity") as identity:
                with self.assertRaises(RagError):
                    self.client.pull("qwen2.5:0.5b")
                identity.assert_not_called()
                response.__exit__.assert_called_once()

    def test_pull_rejects_unknown_or_cloud_tags_before_network(self):
        with patch.object(self.client.session, "post") as post:
            for tag in ("cloud-model:cloud", "unknown", "https://example.com/model"):
                with self.assertRaises(RagError):
                    self.client.pull(tag)
            post.assert_not_called()

    def test_pull_http_error_and_connection_failure(self):
        with patch.object(self.client.session, "post", return_value=self.response([], 403)):
            with self.assertRaisesRegex(RagError, "403"):
                self.client.pull("qwen2.5:0.5b")
        with patch.object(self.client.session, "post", side_effect=requests.ConnectionError("offline")):
            with self.assertRaisesRegex(RagError, "interrupted"):
                self.client.pull("qwen2.5:0.5b")

    def test_success_progress_without_installed_tag_is_failure(self):
        with patch.object(self.client.session, "post", return_value=self.response([b'{"status":"success"}'])), \
                patch.object(self.client, "identity", side_effect=RagError("not installed")):
            with self.assertRaisesRegex(RagError, "not installed"):
                self.client.pull("qwen2.5:0.5b")

    def test_role_validation_and_unload_preserve_weights(self):
        with patch.object(self.client, "identity", return_value="model@digest"), \
                patch.object(self.client, "request", return_value={"capabilities": ["embedding"]}) as request:
            self.assertEqual(self.client.validate_role("nomic-embed-text", "embedding"), "model@digest")
            with self.assertRaisesRegex(RagError, "completion"):
                self.client.validate_role("nomic-embed-text", "generation")
            self.client.unload("nomic-embed-text")
            self.assertEqual(request.call_args.args, ("/api/embed", {"model": "nomic-embed-text", "input": [], "keep_alive": 0}))
            self.assertNotIn("/api/delete", [call.args[0] for call in request.call_args_list])

    def test_all_generation_choices_use_supported_adapter_and_citation_schema(self):
        settings = json.loads(Path("config/settings.example.json").read_text())
        for choice in (c for c in CATALOG if c.role == "generation"):
            with self.subTest(model=choice.tag):
                client = Ollama(settings)
                captured = []
                def request(path, payload=None):
                    if path == "/api/show":
                        return {"capabilities": ["completion"]}
                    self.assertEqual(path, "/api/chat")
                    captured.append(payload)
                    return {"message": {"content": json.dumps({"status": "answered", "claims": [
                        {"text": "The sample code is MAPLE.", "citations": [1]}]})}}
                with patch.object(client, "identity", return_value=f"{choice.tag}@digest"), \
                        patch.object(client, "request", side_effect=request):
                    result = client.check_compatibility(choice.tag, "generation")
                self.assertIn("passed", result["detail"])
                self.assertEqual(captured[0]["model"], choice.tag)
                self.assertEqual(captured[0]["options"]["num_ctx"], 8192)
                self.assertEqual(captured[0].get("think"), False if choice.disable_thinking else None)
                self.assertIn("claims", captured[0]["format"]["properties"])
                self.assertEqual(settings["generation_model"], "llama3.2:3b")

    def test_all_embedding_choices_probe_document_and_query_vectors(self):
        settings = json.loads(Path("config/settings.example.json").read_text())
        for choice in (c for c in CATALOG if c.role == "embedding"):
            with self.subTest(model=choice.tag):
                client = Ollama(settings)
                inputs = []
                def request(path, payload=None):
                    if path == "/api/show":
                        return {"capabilities": ["embedding"]}
                    self.assertEqual(path, "/api/embed")
                    self.assertEqual(payload["model"], choice.tag)
                    inputs.extend(payload["input"])
                    return {"embeddings": [[3, 4] for _ in payload["input"]]}
                with patch.object(client, "identity", return_value=f"{choice.tag}@digest"), \
                        patch.object(client, "request", side_effect=request):
                    result = client.check_compatibility(choice.tag, "embedding")
                self.assertEqual(len(inputs), 2)
                self.assertIn("2 dimensions", result["detail"])
                self.assertTrue(inputs[0].startswith("search_document:") if "nomic" in choice.tag else not inputs[0].startswith("search_document:"))

    def test_short_embedding_context_never_silently_truncates_documents(self):
        settings = json.loads(Path("config/settings.example.json").read_text())
        settings["embedding_model"] = "all-minilm:22m"
        client = Ollama(settings)
        with patch.object(client, "request", return_value={"embeddings": [[1, 0]]}) as request:
            with self.assertRaisesRegex(RagError, "Rebuild the index"):
                client.embed(["x" * 500])
            request.assert_not_called()
            client.embed(["नमस्ते " * 100], query=True)
            text = request.call_args.args[1]["input"][0]
            self.assertLessEqual(len(text.encode()), embedding_byte_budget("all-minilm:22m"))
            self.assertNotIn("\ufffd", text)

    def test_failed_generation_probe_or_digest_change_rejects_activation(self):
        settings = json.loads(Path("config/settings.example.json").read_text())
        client = Ollama(settings)
        with patch.object(client, "validate_role", return_value="model@before"), \
                patch.object(client, "generate", return_value='{"status":"answered","claims":[{"text":"MAPLE","citations":[999]}]}'):
            with self.assertRaisesRegex(RagError, "structured cited-answer"):
                client.check_compatibility("qwen2.5:0.5b", "generation")
        with patch.object(client, "validate_role", return_value="model@before"), \
                patch.object(client, "generate", return_value='{"status":"answered","claims":[{"text":"MAPLE","citations":[1]}]}'), \
                patch.object(client, "identity", return_value="model@after"):
            with self.assertRaisesRegex(RagError, "digest changed"):
                client.check_compatibility("qwen2.5:0.5b", "generation")


class ProjectServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.registry = patch.dict(model_server._servers, {}, clear=True)
        self.registry.start()
        self.addCleanup(self.registry.stop)

    def test_project_process_uses_one_shared_storage_folder(self):
        process = MagicMock()
        process.poll.return_value = None
        with patch.object(model_server, "_binary", return_value="/fake/ollama"), \
                patch.object(model_server.socket, "socket") as socket, \
                patch.object(model_server.subprocess, "Popen", return_value=process) as popen, \
                patch.object(model_server, "_ready", return_value=True):
            self.assertEqual(model_server.start_server(self.root), model_server.PROJECT_URL)
            self.assertEqual(model_server.start_server(self.root), model_server.PROJECT_URL)
            self.assertEqual(popen.call_count, 1)
            self.assertEqual(popen.call_args.args[0], ["/fake/ollama", "serve"])
            environment = popen.call_args.kwargs["env"]
            self.assertEqual(environment["OLLAMA_MODELS"], str(self.root / "data/models"))
            self.assertEqual(environment["OLLAMA_HOST"], "127.0.0.1:11435")
            self.assertEqual(environment["OLLAMA_MAX_LOADED_MODELS"], "1")
            (self.root / "data/models/sentinel").write_text("preserve")
            model_server.stop_server(self.root)
            process.terminate.assert_called_once()
            self.assertTrue((self.root / "data/models/sentinel").exists())

    def test_occupied_port_never_replaced(self):
        with patch.object(model_server, "_binary", return_value="/fake/ollama"), \
                patch.object(model_server.socket, "socket") as socket, \
                patch.object(model_server.subprocess, "Popen") as popen:
            socket.return_value.__enter__.return_value.bind.side_effect = OSError("occupied")
            with self.assertRaisesRegex(RagError, "does not own"):
                model_server.start_server(self.root)
            popen.assert_not_called()

    def test_failed_start_cleans_up_owned_child(self):
        process = MagicMock()
        process.poll.return_value = None
        with patch.object(model_server, "_binary", return_value="/fake/ollama"), \
                patch.object(model_server.socket, "socket"), \
                patch.object(model_server.subprocess, "Popen", return_value=process), \
                patch.object(model_server, "_ready", side_effect=RagError("failed probe")):
            with self.assertRaisesRegex(RagError, "failed probe"):
                model_server.start_server(self.root)
            process.terminate.assert_called_once()
            self.assertIsNone(model_server.managed_server(self.root))


if __name__ == "__main__":
    unittest.main()
