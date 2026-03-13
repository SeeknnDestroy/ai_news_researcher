# AI News Researcher

A GenAI-powered CLI application designed to crawl news articles, summarize them, and synthesize high-signal weekly technical AI reports for executive and engineering teams.

Features advanced capabilities including:
- **LLM-as-a-Judge Evaluation:** Automated adherence grading to rubrics.
- **Self-Correction & Revision:** Self-healing pipelines when reports fail to meet quality thresholds.
- **Asynchronous Processing:** Concurrent LLM queries utilizing `asyncio`.
- **Token-Aware Truncations:** Accurate processing using `tiktoken`.

## Prerequisites

- **Python 3.10+** (Recommend **Python 3.13**)
- **uv** (or `pip`) for dependency management
- An active API key for **xAI** (or compatible LLM endpoints)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SeeknnDestroy/ai_news_researcher.git
   cd ai_news_researcher
   ```

2. **Install dependencies:**
   This project uses `pyproject.toml` to manage dependencies.

   Using `uv` (recommended):
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -e .
   ```

   *Alternatively, using `pip`:*
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install httpx tenacity tiktoken pydantic-settings typer pytest-asyncio
   ```

## Configuration

The application uses `pydantic-settings` to load configuration automatically from environment variables or a `.env` file.

1. Create a `.env` file at the root of the project:
   ```bash
   touch .env
   ```

2. Add your **xAI API Key** to the `.env` file:
   ```env
   XAI_API_KEY="xai-your-api-key-here"
   ```

### Optional Settings
You can override default evaluation thresholds by adding them to the `.env` file:
- `EVAL_REVISION_THRESHOLD` (default: 80.0)
- `EVAL_GROUNDEDNESS_THRESHOLD` (default: 70.0)
- `EVAL_COVERAGE_THRESHOLD` (default: 50.0)

## Usage

The main interface is powered by Typer and executed through `src/cli.py`.

### Generating a Report

Prepare an input YAML file (e.g., `inputs/links.yaml`) following the config schema and then run:

```bash
python -m src.cli --model grok-4-1-fast-reasoning --max-concurrency 5
```

By default it expects a `.yaml` configuration to govern the current run inputs in monthly folders (e.g. `inputs/2026-03/links_13-03-2026.yaml`). Existing legacy flat files under `inputs/` are still accepted as a fallback.

### Capturing Chrome Tabs Into Today's Input

On macOS, you can generate today's input YAML directly from your open Google Chrome tabs:

```bash
python -m src.capture_links_cli
```

This command:
- reads tabs from the current frontmost Chrome window
- writes to `inputs/YYYY-MM/links_DD-MM-YYYY.yaml`
- appends new URLs without duplicating ones already in the file
- creates the file with `evaluation: true` if it does not exist yet

Optional flags:

```bash
python -m src.capture_links_cli --all-windows
python -m src.capture_links_cli --replace
python -m src.capture_links_cli --output inputs/links_custom.yaml
python -m src.capture_links_cli --no-evaluation
```

If you install the package entrypoints, the same helper is also available as:

```bash
capture-links
```

Notes:
- Only `http://` and `https://` tabs are captured.
- `chrome://`, extension tabs, and blank tabs are ignored.
- On first use, macOS may ask for permission to let your terminal control Google Chrome.

**CLI Options:**
- `--model <model_name>`: Change the xAI model (default: `grok-4-1-fast-reasoning`)
- `--temperature <float>`: Sampling temperature (default: `0.2`)
- `--max-concurrency <int>`: Restrict maximum simultaneous crawls (default: `3`)

### Artifacts and Output
- **Reports:** The final summarized markdown document will be written to `reports/YYYY-MM/`.
- **Artifacts:** Debug logs, drafts, crawler text snapshots, and newsletter splits will be persisted in `artifacts/run_<timestamp>/`.

## Development & Testing

To run the automated tests and ensure the deterministic and LLM logic functions correctly:

```bash
pytest tests/ -v
```

The live crawl integration test now skips automatically unless a valid `CRAWL_TEST_INPUT` file is available, so a clean local `pytest` run does not depend on private input files.

### Architecture Notes

- `src/config.py` is settings-only.
- Canonical pipeline orchestration lives in `src/application/pipeline.py`.
- Canonical content-processing tasks live in `src/application/content_tasks.py`.
- Canonical report-generation tasks live in `src/application/report_tasks.py`.
- `src/application/ai_tasks.py`, `src/summarize.py`, `src/newsletter.py`, and `src/agents/*` remain as compatibility shims during the migration cycle and should not be used for new internal code.
