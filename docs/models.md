# Model choices for a 16 GB M1 Mac

The catalog contains eight answer models and two embedding models chosen as relatively small local candidates for an Apple M1 Mac with 16 GB unified memory. Downloading multiple models takes disk space; it does not load them all into memory. The project-managed server permits one loaded model at a time. Existing external servers keep their own settings.

These are not hardware benchmarks or a guarantee of factual accuracy. Published parameter counts, quantization, and disk sizes were checked on 28 October 2026; tags can change. Follow each model link for current metadata and license terms. RAM ranges are app planning estimates for a single model at about 8K generation context, excluding macOS, Python/browser, the index, and other processes. Embedding memory additionally depends on batch size and its own smaller context setting.

## Available options

| Model and official reference | Purpose | Parameters | Expected disk size | Estimated model RAM |
|---|---|---|---|---|
| [qwen2.5:0.5b](https://ollama.com/library/qwen2.5:0.5b) | Answers | 494 million | 398 MB | ~1–2.5 GB (estimate) |
| [qwen3:0.6b](https://ollama.com/library/qwen3:0.6b) | Answers | 752 million | 523 MB | ~1.2–3 GB (estimate) |
| [gemma3:1b](https://ollama.com/library/gemma3:1b) | Answers | 1.00 billion | 806 MB | ~1.5–3 GB (estimate) |
| [qwen2.5:1.5b](https://ollama.com/library/qwen2.5:1.5b) | Answers | 1.54 billion | 986 MB | ~2–4 GB (estimate) |
| [llama3.2:1b](https://ollama.com/library/llama3.2:1b) | Answers | 1.24 billion | 1.30 GB | ~2.5–4 GB (estimate) |
| [qwen3:1.7b](https://ollama.com/library/qwen3:1.7b) | Answers | 2.03 billion | 1.40 GB | ~2.5–4.5 GB (estimate) |
| [qwen2.5:3b](https://ollama.com/library/qwen2.5:3b) | Answers | 3.09 billion | 1.90 GB | ~3–5.5 GB (estimate) |
| [llama3.2:3b](https://ollama.com/library/llama3.2:3b) | Answers | 3.21 billion | 2.00 GB | ~3–5.5 GB (estimate) |
| [nomic-embed-text:latest](https://ollama.com/library/nomic-embed-text:latest) | Embeddings | 137 million | 274 MB | ~0.3–1 GB (estimate) |
| [all-minilm:22m](https://ollama.com/library/all-minilm:22m) | Embeddings | 22.6 million | 46 MB | ~0.15–0.6 GB (estimate) |

Qwen 3 tag names are nominal sizes: the current Ollama listings report 752M parameters for `qwen3:0.6b` and 2.03B for `qwen3:1.7b`. The Llama 1B tag uses Q8_0 weights, so a smaller parameter count is not automatically a smaller download.

## Starting combinations

- **Prioritize memory:** `qwen2.5:0.5b` + `all-minilm:22m`. MiniLM uses short passages and may sacrifice retrieval coverage; the small generation model may struggle with synthesis and reliable citations.
- **Larger-context retrieval:** `qwen2.5:0.5b` or `qwen2.5:1.5b` + `nomic-embed-text:latest`.
- **Compare larger answer candidates:** choose `qwen3:1.7b`, `qwen2.5:3b`, or `llama3.2:3b`, keeping the same embedding model and index. Assess results on your own questions before preferring any model.

The GUI's lower-memory checkbox narrows generation choices by their estimated upper RAM range. All listed models remain available with the checkbox off. No choice is automatically downloaded or selected for you.

## Compatibility checks before activation

**Use selected model** verifies the installed local model and its intended completion/embedding role, then runs a synthetic API sample before saving settings. The separate **Check selected model compatibility** button performs the check without selecting it. Changed manual model names in Settings are also checked before saving.

For answer models, the sample uses the same structured claim-and-citation schema as RAG and must produce a valid cited answer to an easy synthetic evidence question. For embedding models, document/query embeddings must be finite, nonzero, and dimensionally consistent. The installed digest must remain stable throughout the check. Synthetic checks may load the model and consume RAM. Failure leaves the previous settings untouched; downloaded files remain available for retries.

A passing sample checks API compatibility and a minimal output case. It does not establish PDF extraction quality, retrieval quality, general reasoning, prompt-injection resistance, or factual support. Automated tests simulate model responses; the catalog has not been benchmarked with real models on this Mac. Keep Ollama updated, especially when trying newer model families.

## Automatic model-specific handling

- **Qwen 3:** the app sends `think: false` so the answer budget is used for the structured response rather than an extra thinking phase.
- **Nomic:** the app applies the document/query task prefixes.
- **MiniLM:** the listed Ollama tag defaults to a 256-token embedding context. With conservative byte-based budgeting and headroom, passages are capped at 224 UTF-8 bytes. Longer documents are split into more passages; document text is not silently discarded. Overlap is adjusted to remain valid. A long semantic search question is shortened with a visible warning; generation still receives the full original question. Prefer concise questions, or Nomic for longer retrieval inputs.

Selecting a new embedding model requires rebuilding the index. Incompatible embedding identities are rejected instead of mixing vector spaces. Similarity scores are model-dependent: recalibrate the minimum similarity threshold on your own answerable/unanswerable questions after switching. Changing only the answer model does not require reindexing.

Use **Evidence only** to retrieve excerpts without a generation model. Inspect source passages regardless of the selected answer model.
