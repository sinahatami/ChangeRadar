# ChangeRadar

Local-first code change impact analysis using dependency-aware retrieval and RAG.

## Current scope

Phase 1 provides bounded, read-only intake for Python source, AST symbol and
line extraction, and a conservative local import/call graph. Phase 2 adds
symbol-level code chunks, lexical BM25-style ranking, optional local semantic
ranking, dependency-aware hybrid ranking, and fixture-based retrieval
evaluation. Phase 3 adds diff/change parsing, bounded evidence-grounded impact
reports, validated analysis tools, and an optional OpenAI-compatible chat
provider. Repository code is parsed but never executed. External transmission
is disabled by default. No database, agent framework, or user interface is
included.

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
