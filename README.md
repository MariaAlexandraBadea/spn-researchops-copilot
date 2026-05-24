# SPN ResearchOps Copilot

An applied AI / agentic RAG project for probabilistic machine learning research workflows.

This project connects Python code, notebooks, Markdown documentation, LaTeX manuscripts, CSV experiment outputs and research notes into a grounded AI assistant. It was built around a CNN-GRU-SPN-Gamma probabilistic forecasting project for monthly temperature anomaly prediction.

The goal is not to build a generic "chat with PDF" application. The goal is to demonstrate how Retrieval-Augmented Generation can support technical research and engineering workflows: model understanding, experiment analysis, reviewer-response drafting, model-card generation and source-grounded explanation.

Default SPN knowledge-base repository used in the Colab workflow:

```text
https://github.com/MariaAlexandraBadea/cnn-gru-spn-gamma-temperature-forecast
```

---

## Project motivation

This project was built as a personal Applied AI project connecting probabilistic ML research with modern Generative AI / RAG workflows.

Instead of building a generic document chatbot, the assistant is designed around a real research codebase: a CNN-GRU-SPN-Gamma probabilistic forecasting model for monthly temperature anomaly prediction. The system indexes Python code, notebooks, Markdown documentation, experiment outputs and research notes, then uses retrieval-augmented generation to answer technical questions with source grounding.

The project supports research and engineering tasks such as:

- explaining model architecture from source code;
- identifying leakage controls in preprocessing;
- summarizing probabilistic evaluation metrics;
- generating model-card style documentation;
- drafting reviewer-response style explanations;
- tracing answers back to implementation files and documentation.

---

## Why this is not a generic RAG demo

Most simple RAG demos answer questions over one or more PDFs. This project is different because it is built over a real probabilistic ML research repository.

The assistant can reason over:

- model implementation code;
- Jupyter notebooks;
- technical documentation;
- experiment outputs;
- evaluation metrics;
- research notes;
- reviewer-response style tasks.

The indexed knowledge base is centered on a CNN-GRU-SPN-Gamma probabilistic forecasting model, including architecture, leakage controls, probabilistic inference and evaluation methodology.

---

## Demo example

The screenshot below shows the assistant answering a technical question about leakage controls in the preprocessing pipeline.

<img width="1920" height="2226" alt="image" src="https://github.com/user-attachments/assets/df3e48a5-8584-4022-9688-5bcf8e13e845" />

In this example, the assistant retrieves grounded context from the indexed SPN forecasting repository and identifies relevant preprocessing safeguards, including:

- train-only climatology;
- causal input-window imputation;
- no target imputation;
- disjoint city-based train/validation/test splits;
- temporal split rules based on target horizon boundaries.

The application can run in two modes:

- **LLM mode**, when an OpenAI API key is available;
- **extractive fallback mode**, when no LLM API key is configured or the API quota is unavailable.

The fallback mode is intentionally designed to remain useful during demos: it produces structured, source-grounded summaries instead of only dumping raw retrieved chunks.

---

## Key features

- Multi-source ingestion for `.py`, `.ipynb`, `.tex`, `.md`, `.txt`, `.csv`, `.json`, `.yaml` and `.pdf` files.
- Hybrid retrieval using:
  - sentence-transformer embeddings;
  - FAISS vector search;
  - BM25 keyword retrieval;
  - metadata-aware ranking.
- Source-grounded answers with citations, confidence scoring and retrieved-source inspection.
- Improved extractive fallback for demos without an active LLM API quota.
- Tool-style ResearchOps workflows for:
  - model-card generation;
  - reviewer-response drafting;
  - experiment summary;
  - metrics lookup;
  - GitHub issue draft generation;
  - research planning.
- Gradio interface for Google Colab demos.
- Optional FastAPI endpoint for local/API usage.
- Benchmark runner for retrieval and grounded-QA checks.
- Hallucination-reduction behavior through confidence thresholds, source tracing and fallback responses.

---

## System overview

The project follows a simple ResearchOps RAG flow:

```text
SPN forecasting repository
        |
        v
Document loader
(.py, .ipynb, .md, .tex, .csv, .json, .pdf)
        |
        v
Chunking + metadata extraction
        |
        v
Hybrid index
(FAISS embeddings + BM25 keyword retrieval)
        |
        v
Retriever + ranking
        |
        v
Agent router
(RAG answer / model card / reviewer response / metrics summary)
        |
        v
Gradio UI / optional FastAPI endpoint
```

---

## Repository structure

```text
spn-researchops-copilot/
  app/
    agent.py
    chunking.py
    config.py
    gradio_app.py
    llm.py
    loaders.py
    pretty.py
    retrieval.py
    schema.py
    tools.py

  scripts/
    clone_repo.py
    ingest.py
    evaluate.py

  data/
    benchmark_questions.json

  assets/
    demo-leakage-controls.png

  requirements-colab.txt
  README.md
```


## Run in Google Colab

### 1. Clone this repository

```python
%cd /content

!rm -rf spn-researchops-copilot

!git clone https://github.com/MariaAlexandraBadea/spn-researchops-copilot.git

%cd /content/spn-researchops-copilot
```

### 2. Install dependencies

```python
!pip -q install -r requirements-colab.txt
```

### 3. Clone the SPN forecasting repository

```python
!PYTHONPATH=. python scripts/clone_repo.py \
  --repo https://github.com/MariaAlexandraBadea/cnn-gru-spn-gamma-temperature-forecast \
  --out data/source_repo \
  --reset
```

### 4. Check the indexed source repository

```python
!ls -la data/source_repo
!find data/source_repo -maxdepth 2 -type f | head -50
```

You should see files such as:

```text
cnn_gru_spn_gamma_temperature_forecast.py
cnn_gru_spn_gamma_temperature_forecast.ipynb
README.md
model_overview.md
result.txt
experiments/
```

### 5. Build the retrieval index

```python
!PYTHONPATH=. TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 python scripts/ingest.py \
  --input data/source_repo \
  --index data/index \
  --reset
```

Expected output:

```text
[INFO] Loading documents from data/source_repo
[INFO] Loaded ... chunks
[OK] Index saved to data/index
```

### 6. Verify the index

```python
!ls -la data/index
```

You should see files such as:

```text
meta.json
chunks.json
index.faiss
bm25.pkl
```

### 7. Run the benchmark

```python
!PYTHONPATH=. TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 python scripts/evaluate.py \
  --index data/index \
  --benchmark data/benchmark_questions.json \
  --metrics-root data/source_repo
```

The benchmark checks whether the assistant can retrieve grounded information about:

- CNN-GRU-SPN-Gamma architecture;
- leakage controls;
- probabilistic metrics;
- model-card generation.

### 8. Start the Gradio UI

```python
!PYTHONPATH=. TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 python app/gradio_app.py \
  --index data/index \
  --metrics-root data/source_repo \
  --share
```

The command prints a public Gradio URL:

```text
Running on public URL: https://xxxxx.gradio.live
```

Keep the Colab cell running while using the interface. If the cell is stopped, the public Gradio link closes.

---

## Optional OpenAI usage

The app can run in extractive fallback mode without an API key. For LLM-generated answers, set an OpenAI API key before starting the Gradio UI:

```python
from getpass import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass("OpenAI API key: ")
```

Then start Gradio again:

```python
!PYTHONPATH=. TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 python app/gradio_app.py \
  --index data/index \
  --metrics-root data/source_repo \
  --share
```

If no valid API key is available, or if the API quota is unavailable, the project automatically falls back to the improved extractive mode.

---

## Extractive fallback mode

The extractive fallback mode is designed to make the demo usable even without an active LLM API quota.

Instead of returning only raw retrieved chunks, the fallback mode builds structured answers for common technical question types, including:

- architecture questions;
- leakage-control questions;
- metric and evaluation questions;
- MDN comparison questions;
- model-card style questions.

Example fallback answer style:

```text
The preprocessing pipeline implements the following leakage controls:

- Monthly aggregation is performed before window construction.
- Climatology is estimated only from training years.
- Input gaps are imputed only inside the historical input window.
- PCHIP imputation is causal and restricted to short gaps.
- Target values are never imputed.
- Train, validation and test splits are disjoint by city.
- Temporal split rules use target horizon boundaries.

Sources:
[1] model_overview.md, lines 58-90
[2] README.md, lines 34-61
```

---

## Example questions

### Architecture

```text
Explain the CNN-GRU-SPN-Gamma architecture.
```

```text
What is the role of the CNN1D skip-connection encoder?
```

```text
How does the conditional SPN-Gamma decoder model uncertainty?
```

### Leakage controls

```text
What leakage controls are implemented in the preprocessing pipeline?
```

```text
Why is train-only climatology important?
```

```text
Why are target values never imputed?
```

### Evaluation metrics

```text
Which probabilistic metrics are used?
```

```text
What is the role of CRPS, NLL, PIT diagnostics and Energy Score?
```

```text
How does the project evaluate joint multi-horizon forecast quality?
```

### ResearchOps workflows

```text
Generate a model card for the forecasting model.
```

```text
Draft a reviewer response explaining why the model is more than a standard Mixture Density Network.
```

```text
Create a GitHub issue for adding calibration plots to the project.
```

```text
Summarize experiment results from CSV files.
```

---

## Manual document upload option

The project can also index manually uploaded files in Colab.

```python
from google.colab import files
uploaded = files.upload()
```

Move uploaded files to a local folder:

```python
import os
import shutil

os.makedirs("data/manual_docs", exist_ok=True)

for name in uploaded.keys():
    shutil.move(name, f"data/manual_docs/{name}")
```

Build the index:

```python
!PYTHONPATH=. TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 python scripts/ingest.py \
  --input data/manual_docs \
  --index data/index \
  --reset
```

Start the UI:

```python
!PYTHONPATH=. TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 python app/gradio_app.py \
  --index data/index \
  --metrics-root data/manual_docs \
  --share
```

---

## Optional local FastAPI usage

The project also includes an optional FastAPI-style usage path for local/API experiments.

Install dependencies locally:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-colab.txt
```

Build the index:

```bash
PYTHONPATH=. python scripts/ingest.py \
  --input data/source_repo \
  --index data/index \
  --reset
```

Start the Gradio UI locally:

```bash
PYTHONPATH=. python app/gradio_app.py \
  --index data/index \
  --metrics-root data/source_repo
```

---

## Current limitations

- The Colab demo uses local FAISS indexing, so the index must be rebuilt after changing the source repository.
- Without an OpenAI API key, answers are extractive and less fluent than LLM-generated answers.
- Retrieval quality depends on the clarity of the indexed documentation.
- The project is designed for technical research assistance, not autonomous code execution.
- The assistant does not automatically deploy, train or modify the forecasting model.
