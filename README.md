# ChangeRadar

Local-first code change impact analysis using dependency-aware retrieval and RAG.

## Current scope

Phase 1 provides bounded, read-only intake for Python source, AST symbol and
line extraction, and a conservative local import/call graph. Phase 2 adds
symbol-level code chunks, lexical BM25-style ranking, optional local semantic
ranking, dependency-aware hybrid ranking, and fixture-based retrieval
evaluation. Phase 3 adds diff/change parsing, bounded evidence-grounded impact
reports, validated analysis tools, and an optional OpenAI-compatible chat
provider. Phase 4 provides a Streamlit interface for sample or ZIP repositories
and evidence-linked reports. Repository code is parsed but never executed.
External transmission is disabled by default.

Semantic retrieval uses Sentence Transformers with `all-MiniLM-L6-v2` on CPU.
It is optional and loaded lazily. Install the semantic extra only when needed:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python -m pip install -e ".[semantic]"
python scripts/smoke_semantic.py
```

The first command installs a CPU-only PyTorch build to avoid pulling GPU
libraries onto CPU-only machines. The smoke test checks that the model loaded,
semantic retrieval was used, and result paths and line ranges are valid. Its
first run may download model weights. Automated tests never download the model.
Without the optional package or usable model, semantic and hybrid requests
report a warning and return lexical fallback rankings.

## Development and tests

Python 3.10 or newer is required. Create and activate a virtual environment,
then install development tools and run quality checks and tests:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
pre-commit install
pre-commit run --all-files
ruff check .
ruff format --check .
python3 -m pytest
git diff --check
```

## Retrieval evaluation

The labeled fixture set has five representative questions and expected symbol
IDs. The script builds chunks and rankings from the fixture source and computes
mean Recall@5 and MRR from those ranked results. It reports the effective
retrieval method so an unavailable semantic model is visible as lexical
fallback, rather than being presented as semantic performance.

Run the offline evaluation (no model or API required):

```bash
python3 -m pip install -e ".[dev]"
python3 scripts/evaluate_retrieval.py
```

To evaluate with the actual local semantic model after installing the optional
extra (the weights may download on first use):

```bash
python3 -m pip install -e ".[semantic]"
python3 scripts/evaluate_retrieval.py --semantic
```

## Impact report evaluation

The impact evaluator runs the deterministic workflow against three labeled change
cases covering a direct email-validation target, an ambiguous zero-rate boundary,
and a dependency-linked token decoder caller. It computes direct-target,
dependency-candidate, and overall impact precision/recall/F1; reports whether
selected context-only negatives were mislabeled as impact; and checks expected
markers in test ideas, ambiguity disclosure, and source-reference validity.
These are fixture checks, not a broad estimate of production accuracy. Test-idea
marker coverage checks the labeled wording and does not execute generated tests.

Run the evaluation with lexical retrieval, no model, credentials, or network:

```bash
python scripts/evaluate_impact.py
```

Semantic and hybrid report evaluations are optional and use the local embedding
model:

```bash
python scripts/evaluate_impact.py --method hybrid
```

## Phase 3: impact analysis demonstration

After installing the development extra, run a deterministic local analysis on
the included sample repository:

```bash
python scripts/demo_impact.py tests/fixtures/retrieval_repo \
  "Change calculate_tax rounding for zero tax rates"
```

The JSON report includes impacted symbols, test ideas, evidence IDs, and
verified relative paths and line ranges. It uses the deterministic report when
no provider is enabled. Tool access is restricted to already scanned source;
no analyzed code is executed.

The optional OpenAI-compatible adapter uses the following environment
variables, with no provider or paid service selected by default:

```bash
export CHANGERADAR_ALLOW_EXTERNAL_CODE=true
export CHANGERADAR_LLM_ENDPOINT="https://your-provider.example/v1/chat/completions"
export CHANGERADAR_LLM_MODEL="your-model-name"
export CHANGERADAR_API_KEY="your-api-key"
python scripts/demo_impact.py tests/fixtures/retrieval_repo \
  "Change calculate_tax rounding for zero tax rates" --allow-external
```

Enabling the flag and environment opt-in permits transmission of the change
text and bounded, redacted evidence snippets. API credentials and any API
credits are separate from a ChatGPT subscription. If the provider is missing,
unavailable, rate-limited, returns malformed output, or cannot handle the tool
interface, the workflow returns its labeled deterministic report.

Credential redaction covers common key, token, password, bearer, and private-key
patterns. It is a best-effort filter and cannot guarantee detection of every
secret embedded in source; inspect the evidence and configure provider access
accordingly before opting in to external transmission.

## Phase 4: Streamlit application

Install the optional UI extra in the project environment:

```bash
python -m pip install -e ".[ui]"
```

Launch the local interface:

```bash
streamlit run app.py
```

For a quick demonstration, select **Included sample**, keep **Lexical
(offline)** selected, enter `Change calculate_tax rounding for zero tax rates`,
and click **Analyze change**. The report shows ranked symbols, heuristic
retrieval scores, source evidence with exact line ranges, test ideas, and
uncertainties. Use **Upload a ZIP** to analyze another repository; only Python
files are read, ZIP paths are validated, and the existing intake limits apply
(30 MB upload, 1,000 source files, 1 MB per file, 20 MB total source). ZIPs are
read in memory and never extracted or executed.

Lexical retrieval works offline and is the default. Semantic and hybrid choices
use the optional `all-MiniLM-L6-v2` model; first use may download model weights.
The model download does not contain repository source. A semantic failure falls
back to lexical retrieval and is identified in the report.

The LLM checkbox starts unchecked. External analysis requires both that
per-analysis consent and the provider configuration described above, including
`CHANGERADAR_ALLOW_EXTERNAL_CODE=true`. When enabled, only the proposed change
and bounded, redacted retrieved evidence are sent. Without this setup, the
interface remains fully usable and clearly labels its deterministic report.
Reports can be downloaded as JSON.

## Architecture and limitations

The UI calls `scan_directory` for the bundled sample and `scan_zip` for uploads,
then reuses the Python AST parser, dependency graph, code chunk builder,
retriever, `ImpactWorkflow`, and `ImpactReport`. The UI does not accept server
filesystem paths or execute analyzed code. Integration and failure tests
exercise the public helpers without requiring Streamlit. When the UI extra is
installed, an additional AppTest verifies the rendered sample-analysis flow.
Tests use no network access and do not download a model.

Static dependency analysis can miss dynamic dispatch, reflection, generated
code, and dependencies outside the included Python source. Ranking scores are
heuristics rather than probabilities. Test suggestions are not executed, and
the evaluation set is small. External provider behavior has not been verified
against a live service; deterministic local analysis remains available.
