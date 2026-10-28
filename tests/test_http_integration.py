"""Opt-in real loopback HTTP transport test with synthetic model responses."""
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.rag.engine import Engine
from test_rag import make_pdf


@unittest.skipUnless(os.environ.get("RAG_HTTP_TESTS") == "1", "Set RAG_HTTP_TESTS=1 for loopback HTTP integration")
class HttpIntegrationTests(unittest.TestCase):
    def test_pdf_to_cited_answer_over_real_http(self):
        calls = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.respond({"models": [{"name": "nomic-embed-text:latest", "digest": "embed-one"},
                                         {"name": "llama3.2:3b", "digest": "answer-one"}]})

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                calls.append((self.path, payload))
                if self.path == "/api/show":
                    result = {"capabilities": ["embedding", "completion"]}
                elif self.path == "/api/embed":
                    result = {"embeddings": [[1, 0] for _ in payload["input"]]}
                elif self.path == "/api/chat":
                    result = {"message": {"content": json.dumps({"status": "answered", "claims": [
                        {"text": "Employees receive 15 days of annual leave.", "citations": [1]}]})}}
                else:
                    self.send_error(404)
                    return
                self.respond(result)

            def respond(self, result):
                body = json.dumps(result).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                (root / "data/pdfs").mkdir(parents=True)
                make_pdf(root / "data/pdfs/policy.pdf", "Employees receive 15 days of annual leave.")
                settings = json.loads(Path("config/settings.example.json").read_text())
                settings["ollama_base_url"] = f"http://127.0.0.1:{server.server_port}"
                engine = Engine(root, settings)
                manifest = {"websites": [], "youtube": []}
                engine.ingest(manifest)
                answer = engine.ask("How much annual leave?", manifest)
                self.assertEqual(answer["status"], "answered")
                self.assertEqual(answer["claims"][0]["citations"], [1])
                self.assertEqual(answer["evidence"][0]["location"]["page"], 1)
                chat = next(payload for path, payload in calls if path == "/api/chat")
                self.assertFalse(chat["stream"])
                self.assertIn("15 days", str(chat["messages"]))
                embeddings = [payload for path, payload in calls if path == "/api/embed"]
                self.assertTrue(all(payload["truncate"] is False for payload in embeddings))
                self.assertTrue(any(payload["input"][0].startswith("search_query:") for payload in embeddings))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
