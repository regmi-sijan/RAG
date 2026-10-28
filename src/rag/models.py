"""Local-only Ollama transport and validated embeddings."""
import ipaddress
import json
import math
from urllib.parse import urlsplit

import requests

from .catalog import canonical_tag, choice_for, embedding_byte_budget, shorten_embedding_query
from .answers import answer_schema, validate_answer


class RagError(RuntimeError):
    pass


def unit_vector(vector):
    if not isinstance(vector, list) or not vector:
        raise RagError("Embedding service returned an empty or invalid vector.")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in vector):
        raise RagError("Embedding contains non-finite or invalid values.")
    norm = math.sqrt(sum(x * x for x in vector))
    if not math.isfinite(norm) or not norm:
        raise RagError("Embedding has an invalid norm.")
    return [x / norm for x in vector]


class Ollama:
    def __init__(self, settings):
        self.settings = settings
        self.base = settings["ollama_base_url"].rstrip("/")
        try:
            parsed = urlsplit(self.base)
            port = parsed.port
        except ValueError as error:
            raise RagError("Enter a valid local Ollama HTTP(S) endpoint and port.") from error
        try:
            local = ipaddress.ip_address(parsed.hostname or "").is_loopback
        except ValueError:
            local = parsed.hostname == "localhost"
        if not local or parsed.scheme not in {"http", "https"} or port == 0 or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment:
            raise RagError("Ollama endpoint must be a local loopback HTTP(S) address.")
        self.session = requests.Session()
        self.session.trust_env = False

    def request(self, path, payload=None, timeout=(5, 180)):
        try:
            response = self.session.request("GET" if payload is None else "POST", self.base + path,
                                            json=payload, timeout=timeout, allow_redirects=False)
            if response.status_code != 200:
                raise RagError(f"Ollama returned HTTP {response.status_code}. Check installed models and server logs.")
            result = response.json()
            if not isinstance(result, dict) or result.get("error"):
                raise RagError("Ollama returned an invalid response or model error.")
            return result
        except (requests.RequestException, ValueError) as error:
            raise RagError("Cannot reach local Ollama or read its response. Start Ollama and install the configured models.") from error

    def installed_models(self):
        models = self.request("/api/tags", timeout=(2, 10)).get("models", [])
        if not isinstance(models, list) or any(not isinstance(item, dict) for item in models):
            raise RagError("Ollama returned an invalid installed-model list.")
        return models

    def running_models(self):
        models = self.request("/api/ps", timeout=(2, 10)).get("models", [])
        if not isinstance(models, list) or any(not isinstance(item, dict) for item in models):
            raise RagError("Ollama returned an invalid loaded-model list.")
        return models

    def pull(self, model, progress=None):
        """Pull only curated local tags. Progress describes a layer, not the whole download."""
        choice = choice_for(model)
        if not choice:
            raise RagError("Choose a downloadable local model from the catalog.")
        success = False
        try:
            with self.session.post(self.base + "/api/pull", json={"model": choice.tag, "stream": True},
                                   stream=True, timeout=(5, 90), allow_redirects=False) as response:
                if response.status_code != 200:
                    raise RagError(f"Model download failed: Ollama returned HTTP {response.status_code}.")
                for line in response.iter_lines():
                    if not line:
                        continue
                    if len(line) > 65536:
                        raise RagError("Ollama returned an oversized download progress record.")
                    event = json.loads(line)
                    if not isinstance(event, dict):
                        raise RagError("Ollama returned invalid download progress.")
                    if event.get("error"):
                        raise RagError(f"Model download failed: {event['error']}")
                    if progress:
                        progress(event)
                    if event.get("status") == "success":
                        success = True
                        break
        except (requests.RequestException, ValueError) as error:
            raise RagError("Model download was interrupted or its progress could not be read. Check Ollama and retry; partial layers may be resumed.") from error
        if not success:
            raise RagError("Download ended without a success response. Retry to resume if supported.")
        # A progress message alone is not evidence the tag is installed and usable.
        return self.identity(choice.tag)

    def validate_role(self, model, role):
        identity = self.identity(model)
        details = self.request("/api/show", {"model": model})
        capabilities = details.get("capabilities")
        required = "embedding" if role == "embedding" else "completion"
        if role not in {"generation", "embedding"}:
            raise RagError("Unknown model role.")
        if isinstance(capabilities, list):
            if required not in capabilities:
                raise RagError(f"The selected model does not support {required}.")
        else:
            known = choice_for(model)
            if not known or known.role != role:
                raise RagError("Cannot verify model capabilities. Update Ollama or choose a known catalog model.")
        return identity

    def unload(self, model):
        # The embed endpoint supports keep_alive and handles embedding-only models.
        choice = choice_for(model)
        self.identity(model)
        is_embedding = choice and choice.role == "embedding"
        if not choice:
            capabilities = self.request("/api/show", {"model": model}).get("capabilities", [])
            is_embedding = "embedding" in capabilities
        if is_embedding:
            self.request("/api/embed", {"model": model, "input": [], "keep_alive": 0})
        else:
            self.request("/api/generate", {"model": model, "keep_alive": 0, "stream": False})

    def check_compatibility(self, model, role):
        """Load the selected model with synthetic data before activating it.

        This checks the actual local API/model, not factual support on user PDFs.
        It may allocate memory, but never changes settings or the search index.
        """
        identity = self.validate_role(model, role)
        if role == "embedding":
            document = self.embed(["The sample code is MAPLE."], model=model)[0]
            query = self.embed(["What is the sample code?"], query=True, model=model)[0]
            if len(document) != len(query):
                raise RagError("Document and query embedding dimensions differ.")
            detail = f"Document/query embeddings passed ({len(document)} dimensions)."
        else:
            messages = [{"role": "system", "content": "Answer from the supplied evidence only. Return the required JSON with a claim citing evidence 1."},
                        {"role": "user", "content": 'Evidence 1: The sample code is MAPLE. Question: What is the sample code?'}]
            try:
                result = json.loads(self.generate(messages, answer_schema([1]), model=model))
                validate_answer(result, [1])
                if result["status"] != "answered" or not any("MAPLE" in claim["text"].upper() for claim in result["claims"]):
                    raise ValueError("The model did not answer the synthetic evidence question.")
            except (ValueError, TypeError, KeyError) as error:
                raise RagError("Model failed the structured cited-answer sample. Keep your current model, update Ollama, or try another catalog option.") from error
            detail = "Structured cited-answer sample passed. Real-document answer quality still needs evaluation."
        if self.identity(model) != identity:
            raise RagError("Model digest changed during compatibility checking. Retry with a stable installed model.")
        return {"model": model, "role": role, "identity": identity, "detail": detail}

    def identity(self, model):
        # Require an installed local model; never use an implicit cloud fallback.
        target = canonical_tag(model)
        for item in self.installed_models():
            if item.get("name") in {target, model} or item.get("model") in {target, model}:
                if item.get("remote_host") or item.get("remote_model") or "cloud" in target.lower():
                    raise RagError("Cloud-backed models are disabled. Choose a locally installed model.")
                details = self.request("/api/show", {"model": model})
                if details.get("remote_host") or details.get("remote_model"):
                    raise RagError("Cloud-backed models are disabled. Choose a locally installed model.")
                digest = item.get("digest")
                if not isinstance(digest, str) or not digest:
                    raise RagError("Cannot identify the installed model digest.")
                return f"{target}@{digest}"
        raise RagError(f"Model {model!r} is not installed. Run: ollama pull {model}")

    def embed(self, texts, query=False, model=None):
        model = model or self.settings["embedding_model"]
        budget = embedding_byte_budget(model)
        if query:
            texts = [shorten_embedding_query(model, text) for text in texts]
        elif budget and any(len(text.encode("utf-8")) > budget for text in texts):
            raise RagError("A passage exceeds this embedding model's safe input budget. Rebuild the index to apply model-aware chunking.")
        if model.split(":")[0] == "nomic-embed-text":
            prefix = "search_query: " if query else "search_document: "
            texts = [prefix + text for text in texts]
        vectors = []
        for start in range(0, len(texts), 16):
            batch = texts[start:start + 16]
            result = self.request("/api/embed", {"model": model, "input": batch, "truncate": False})
            values = result.get("embeddings", [])
            if not isinstance(values, list) or len(values) != len(batch):
                raise RagError("Embedding service returned the wrong number of vectors.")
            vectors.extend(unit_vector(value) for value in values)
        if len({len(v) for v in vectors}) > 1:
            raise RagError("Embedding dimensions are inconsistent.")
        return vectors

    def generate(self, messages, schema, model=None):
        model = model or self.settings["generation_model"]
        payload = {
            "model": model, "messages": messages,
            "format": schema, "stream": False,
            "options": {"temperature": 0, "num_predict": int(self.settings["answer_token_budget"]),
                        "num_ctx": max(8192, int(self.settings["context_token_budget"]) + int(self.settings["answer_token_budget"]) + 2048)},
        }
        choice = choice_for(model)
        if choice and choice.disable_thinking:
            payload["think"] = False
        result = self.request("/api/chat", payload)
        message = result.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise RagError("Ollama returned an invalid chat message. No generated answer is shown.")
        return message["content"]
