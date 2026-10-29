# Frontend and backend audit

## Follow-up review, October 28, 2026

The follow-up suite discovered 73 tests: 72 passed and the optional HTTP test was skipped. The HTTP test then passed separately, for 73 distinct passing tests. Dependency validation and Git whitespace checks passed.

Two additional issues were fixed locally:

- Unchanged PDF hashes previously caused ingestion to skip corrupt vectors or missing stored passages. Updates now validate reusable index data and rebuild damaged entries from the original source. Regression tests cover corrupt vectors and missing chunks.
- A missing/unavailable embedding model previously aborted ingestion without adding an activity record. Model-setup failures now appear in recent runs while preserving existing indexed content.

The Streamlit server was running during this review. The configured Ollama endpoint at port 11435 refused connections, so live model compatibility and real-document RAG could not be rechecked. Start the project model server from Models before using inference. This is a runtime setup condition, not evidence of a model-download failure.

## Original audit

Date: October 28, 2026. Scope: code, GUI, and simulated model tests, as requested. Ollama and model weights were not installed or downloaded.

## Results

The standard suite discovered 70 tests: **69 passed**, with the opt-in HTTP integration test skipped. That integration test was then run separately and **passed**, giving 70 distinct passing tests across both runs. Dependency validation (`pip check`) reported no broken requirements.

| Feature | Verification |
| --- | --- |
| All seven GUI views | Streamlit AppTest and running browser navigation |
| PDF storage | Filename, size/header, no-overwrite, and failed-write tests; browser upload control inspected |
| PDF ingestion and persistence | Synthetic PDF extraction, physical page citations, chunking, index restart, incremental updates |
| Website and YouTube ingestion | Mocked HTML/plain-text and timestamped caption responses; URL and redirect validation tests |
| Source isolation | PDF-only, selected scopes, removed sources, and failed-update preservation tests |
| Retrieval and answering | Synthetic embeddings, similarity gate, context budget, evidence-only, abstention, structured citation validation and retry |
| Local HTTP transport | Real localhost requests against a temporary synthetic model server, from PDF through cited answer |
| Model catalog and controls | Both roles, memory filter, simulated streamed downloads, compatibility checks, activation, unload and managed-server lifecycle tests |
| Settings and errors | Save behavior, malformed responses, unavailable runtime, invalid links, duplicate URLs, and corrupt stored vectors |

The running browser also confirmed empty-collection abstention, malformed-link feedback, corrected budget labels, and actionable missing-Ollama guidance. The real browser file chooser/upload and download dialogs were not exercised; storage operations were tested directly. The live collection remained empty and its model settings were unchanged.

## Problems corrected

- Plain-text websites now retain their content; Unicode URL request targets are encoded.
- Malformed URL ports and endpoint syntax produce clear errors. External sources use standard HTTPS port 443.
- HTTPS connections try alternate validated public addresses when the first address is unreachable, retaining hostname certificate verification.
- Uploads publish atomically without overwriting an existing file or leaving a partial PDF after failure.
- Similar passages with conflicting facts are retained; only exact duplicates are removed within each source.
- Error metadata no longer consumes generation passage space.
- Malformed embeddings, chat messages, model lists, configuration objects, and stored vectors fail with useful errors.
- Duplicate saved links no longer create duplicate widget keys; missing model details do not crash the catalog.
- The settings page correctly labels the context passage budget in UTF-8 bytes. RAM estimates remain explicitly estimates.
- Common environment files and Streamlit secrets are ignored by Git.

## Reproduce

Run from the repository root with the project virtual environment:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pip check
RAG_HTTP_TESTS=1 .venv/bin/python -m unittest discover -s tests -p test_http_integration.py -v
```

The optional integration test binds an ephemeral loopback port and serves only synthetic model responses. Restricted sandboxes may require permission to bind that port. It does not contact a model vendor or download weights. The normal suite uses isolated temporary workspaces and mocked external responses.

## Remaining checks and limitations

Actual model installation, downloaded-file sizes, compatibility with installed weights, answer quality, latency, and peak Mac RAM use require a later live-model evaluation. Simulated responses verify plumbing and failure handling, not a model's ability to answer correctly.

Live website access and YouTube caption availability were not tested against external services. Sites may block extraction; YouTube may restrict transcript access. Scanned PDFs require OCR, which is not implemented. JavaScript-only webpages and video visuals/audio are not extracted.

Citation IDs are validated, but this does not prove that claims follow from the cited text. The similarity threshold needs calibration using representative questions. Inspect evidence before relying on an answer. The lightweight MiniLM option requires short chunks and may reduce retrieval quality.

The app is a local prototype, without multi-user authentication or deployment hardening. A crashed or hot-reloaded application may leave its project Ollama child process running; an occupied project port is never forcibly replaced. Keep the default loopback binding.

## PDF removal feature verification

The PDF library now supports individual removal, confirmed bulk removal, and individual restoration. Removed files are moved to the Git-ignored data/removed-pdfs/ folder and immediately excluded from retrieval. Filename conflicts never overwrite existing PDFs. Existing indexed content remains as inactive history rather than being permanently erased.

The expanded standard suite discovered 77 tests: 76 passed and the optional localhost HTTP integration test was skipped. Tests cover individual and bulk GUI removal, restoration, nested paths, filename collisions, outside-path rejection, and immediate retrieval exclusion.
