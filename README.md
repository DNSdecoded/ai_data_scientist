# AI Data Scientist Platform

A multi-agent data science pipeline powered by [CrewAI](https://github.com/crewAIInc/crewAI). Point it at a dataset and a crew of seven specialist agents loads, cleans, engineers features, runs statistics, trains models, visualizes results, and writes an executive report — end to end.

## What it does

A dataset flows through seven agents, each backed by real, deterministic tools (pandas / scikit-learn / scipy / matplotlib), running sequentially:

| # | Agent | Tools | Output |
|---|-------|-------|--------|
| 1 | Data Engineer | `data_loader`, `schema_inspector` | Dataset profile (types, missing %, memory) |
| 2 | Cleaning Specialist | missing-value analyze/impute, outlier detect, normalize | Cleaned CSV |
| 3 | Feature Engineer | encode, create, select | Feature-engineered CSV |
| 4 | Statistician | descriptive stats, correlation, hypothesis tests | Statistical findings |
| 5 | ML Engineer | `model_trainer`, `model_evaluator` | Trained models + metrics |
| 6 | Visualization Expert | `chart_generator`, `model_viz` | PNG charts |
| 7 | Business Analyst | `report_generator` | Executive report (Markdown) |

The LLM orchestrates and narrates; the tools do the actual computation, so results are reproducible and cheap on tokens.

## Requirements

- Python **3.11+**
- An LLM provider API key (Gemini, OpenAI, or Anthropic)

## Install

```bash
git clone https://github.com/DNSdecoded/ai_data_scientist.git
cd ai_data_scientist

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

## Configuration

Copy the example env file and fill in your key:

```bash
cp .env.example .env
```

`.env` is git-ignored — **never commit it**.

```ini
GEMINI_API_KEY=your-gemini-api-key
LLM_PROVIDER=gemini                       # gemini | openai | anthropic
LLM_MODEL=gemini/gemini-2.5-flash         # full litellm model id

# Rate / cost controls — tune to your provider tier to avoid 429s
MAX_RPM=10                                # requests per minute cap
AGENT_MAX_ITER=5                          # max reasoning loops per agent
ENABLE_MEMORY=false                       # crew memory (adds extra LLM/embedding calls)

TIMEOUT_SECONDS=1800
LOG_LEVEL=INFO
MAX_FILE_SIZE_MB=500
MAX_COLUMNS=5000
```

> Set `LLM_MODEL` to a model your key can actually call — check your provider's current model list. Lower `MAX_RPM` / `AGENT_MAX_ITER` if you hit rate limits.

## Usage

### CLI

```bash
# Analyze a dataset
python main.py --file data/titanic_sample.csv --run-name my_run

# Override the model for one run
python main.py --file data.csv --model gemini/gemini-2.5-flash

# Launch the Streamlit dashboard
python main.py --dashboard
```

### Dashboard

```bash
streamlit run src/dashboard/app.py
```

Upload a file (CSV / Excel / JSON / Parquet), run the analysis, browse results and experiment history in the browser.

### Quick demo

```bash
python run_demo.py        # runs the full pipeline on the bundled Titanic sample
```

### Docker

```bash
docker compose up
```

## Output

```
outputs/<run_name>/executive_report.md    # final report
outputs/<run_name>/crew_logs.txt          # agent execution log
outputs/*.png                             # charts
outputs/model_results.json                # model metrics
experiments/experiments.db                # experiment history (SQLite)
experiments/versions.db                   # dataset versions (SQLite)
data/processed/                           # intermediate cleaned/encoded CSVs
```

## Tests

```bash
pytest -q
```

## Project layout

```
src/
  agents/       one factory per specialist agent
  tools/        the actual data-science tools (pandas/sklearn/scipy/matplotlib)
  storage/      SQLite experiment + dataset-version stores
  models/       pydantic schemas
  dashboard/    Streamlit app
  orchestrator.py   builds the crew, wires tasks, runs with retry/backoff
  config.py     settings (env-driven) + LLM key export
main.py         CLI entry point
```

## Notes

- **Target column** — the bundled task prompts assume a Titanic-style dataset (`target_column='Survived'`, group `Sex`). For other datasets, adjust the task descriptions in `src/orchestrator.py`.
- **Rate limits** — the orchestrator retries on `429` with exponential backoff (60/120/180s). Keep `MAX_RPM` at or below your provider tier.
- **Security** — rotate any API key that has been shared or committed. Keys live in `.env` only.

## License

MIT
