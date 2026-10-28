# How Source Library works

The GUI and RAG backend are implemented. Install Ollama, then use the Models view to start/connect a local server, download, and select the configured models before indexing. Uploading and indexing remain separate actions. These Mermaid diagrams render directly on GitHub. Automated tests use mocked models; live model quality and external availability still require validation.

## 1. Available today: prepare your collection

```mermaid
flowchart TD
    Start[Launch GUI] --> Browser[Default browser opens local interface]
    Browser --> Choice{Choose a task}
    Choice --> PDFs[Upload PDFs]
    PDFs --> Check{Valid filename and PDF header?}
    Check -->|No| Error[Show error; do not save]
    Check -->|Yes| Exists{Filename already exists?}
    Exists -->|Yes| Rename[Preserve existing file; ask user to rename upload]
    Exists -->|No| Files[Save in data/pdfs]
    Choice --> URLs[Enter website or YouTube URL]
    URLs --> Validate{Valid HTTPS source URL?}
    Validate -->|No| Error
    Validate -->|Yes| Manifest[Save link in local sources JSON]
    Choice --> Settings[Edit model names and budgets]
    Settings --> Config[Save local settings JSON]
    Files --> Pending[Collection ready; start indexing explicitly]
    Manifest --> Pending
```

1. Start the app using `start_gui.command` on macOS, or `python -m streamlit run app.py` from the project root.
2. In **PDF library**, upload documents or inspect/download existing PDFs. Existing files are not overwritten.
3. In **Web & YouTube**, optionally save links. This does not download source content.
4. In **Models**, compare model sizes and RAM estimates. Optionally start the project server to store all downloads in `data/models/`, then download/select one answer model and one embedding model. Downloading does not load the model into memory. In **Settings**, adjust answer/context budgets separately.
5. In **Index & activity**, click **Build / update index**. Optionally enable external fetching. Then open **Ask**, select your source scope, and submit a question. **Evidence only** returns passages without generated synthesis.

Uploaded PDFs and local JSON persist after the app closes. Closing the browser does not stop the Python server; use Ctrl+C in its terminal.

## 2. Ingest sources into searchable knowledge

Ingestion is the preparation stage. It runs after sources are added or changed, rather than reading every document again for each question.

```mermaid
flowchart TD
    Build[User starts Build / update index] --> Sources[Read PDFs and explicitly supplied URLs]
    Sources --> PDF[PDF: extract text per page]
    Sources --> Web[Website: fetch and extract page text]
    Sources --> Video[YouTube: fetch available captions]
    PDF --> Quality{Usable text available?}
    Web --> Quality
    Video --> Quality
    Quality -->|No| Warning[Report failure or OCR / transcript warning]
    Quality -->|Yes| Provenance[Keep source identity, pages or timestamps]
    Provenance --> Split[Split into overlapping passages]
    Split --> Embed[Local embedding model converts passages to vectors]
    Embed --> Store[(Save passage text, provenance and vectors)]
    Store --> Ready[Sources available for questions]
```

- **Extraction** turns source content into text. Scanned PDFs need OCR; unavailable captions produce a visible failure. Video images are not covered by captions.
- **Provenance** records where each passage came from: PDF filename/page, webpage URL, or video timestamp.
- **Chunking** breaks long text into manageable passages while preserving source locations.
- **Embeddings** represent passages as numeric vectors for meaning-based search. This is indexing, not model training.
- **Storage** keeps the text and locations alongside the vectors. Unchanged sources can be reused; changed sources must replace their old indexed version safely.

The implementation uses local embeddings and SQLite storage. Website and YouTube acquisition need internet access. Automatic web search and following arbitrary page links remain disabled by default.

## 3. Answer a question

```mermaid
flowchart TD
    Ask[Enter question] --> Scope[Select PDF only, all sources, or specific sources]
    Scope --> Filter[Restrict search to allowed indexed sources]
    Filter --> Retrieve[Find relevant passages using semantic and keyword search]
    Retrieve --> Evidence{Enough relevant evidence?}
    Evidence -->|No| Abstain[Say the selected sources do not provide enough evidence]
    Evidence -->|Yes| Prompt[Give question and numbered passages to local SLM]
    Prompt --> Generate[Generate answer with passage citations]
    Generate --> Validate{Output and citation IDs valid?}
    Validate -->|First failure| Retry[Retry once]
    Validate -->|Yes| Display[Show answer, source locations and supporting excerpts]
    Retry --> Generate
    Validate -->|Second failure| Abstain
```

**PDF only** is the default. The source restriction is enforced before retrieval and context assembly, so a saved website cannot supply an answer in PDF-only mode. The answering stage has no browsing tools.

The local SLM synthesizes the retrieved evidence. It still has pretrained knowledge, so a prompt and valid citation IDs alone cannot guarantee factual support. Claim-support evaluation is required. Evidence-only mode returns retrieved excerpts without generated synthesis.

## Example of the intended experience

This is an illustrative example, not a measured result from a live model:

1. You upload `handbook.pdf` containing a passage on physical page 12: “Employees receive 15 days of annual leave.”
2. You build the index. That passage is stored with the file identity and page number.
3. You choose **PDF only** and ask: “How many annual leave days are provided?”
4. Retrieval finds the passage. The intended answer is: “Employees receive 15 days of annual leave. [handbook.pdf, page 12]”, with the passage available to inspect.
5. If you ask for a policy absent from the selected documents, the intended result is an insufficient-evidence response rather than an invented answer.

The model weights do not change when a PDF is added. The searchable index changes. This lets the collection grow without retraining a language model.

## What happens when a source changes?

| Action | Result |
|---|---|
| Add a PDF | Save/list the file; run ingestion to make its passages searchable |
| Add a URL | Save the link; enable external fetching during ingestion |
| Change a PDF on disk | Re-ingestion detects the change and replaces indexed passages; queries warn about the old version until then |
| Remove a URL or PDF | Queries exclude it immediately; synchronization marks its index inactive |
| Change embedding model or installed digest | Rebuild embeddings before querying selected incompatible sources |
| External fetch or embedding fails | Show the error and preserve the previous usable source version with a warning |

For technical details and acceptance tests, see [architecture.md](architecture.md). For installation and launch instructions, see the [README](../README.md).

## Model selection and download workflow

```mermaid
flowchart TD
    Open[Open Models view] --> Compare[Compare parameters, expected disk size and estimated RAM]
    Compare --> Server{Choose local model server}
    Server --> Project[Start project server: shared data/models folder]
    Server --> Existing[Connect existing Ollama library: storage managed externally]
    Project --> Pick[Choose a model and purpose]
    Existing --> Pick
    Pick --> Download[Download selected model with layer progress]
    Download --> Check{Success event and installed tag verified?}
    Check -->|No| Error[Show error; keep current selection unchanged]
    Check -->|Yes| Installed[Display installed size from Ollama]
    Installed --> Use[Use selected model: check role and run synthetic sample]
    Use --> Compatible{Compatibility check passed?}
    Compatible -->|No| Unchanged[Show failure; keep previous settings]
    Compatible -->|Yes| Save[Save selected model]
    Save --> Role{Model role changed?}
    Role -->|Generation| Ask[Ask with the existing compatible index]
    Role -->|Embedding| Index[Rebuild index if embedding model changed]
    Index --> Ask
    Ask --> Loaded[Inspect loaded allocation; optionally unload from memory]
```

RAM figures shown before download are estimates for about 8K context. Installed disk size and loaded allocation are separate server-reported values. Unloading preserves downloads. The project folder, model server logs, and index are excluded from Git.
