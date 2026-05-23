# SPN ResearchOps Copilot

An applied AI / agentic RAG project built around probabilistic ML research workflows.

The project is designed to work with an existing SPN forecasting repository, research papers, LaTeX manuscripts, Python code, CSV experiment outputs and notes. It is not a generic "chat with PDF" demo: it combines hybrid retrieval, source-grounded generation, tool/function style workflows and evaluation hooks.

Default SPN repository used in the Colab workflow:

```text
https://github.com/MariaAlexandraBadea/cnn-gru-spn-gamma-temperature-forecast
```

## What it does

- Ingests `.py`, `.tex`, `.md`, `.txt`, `.csv`, `.json` and `.pdf` files.
- Builds a hybrid retrieval index using:
  - sentence-transformer embeddings;
  - FAISS vector search;
  - BM25 keyword retrieval;
  - metadata-aware ranking.
- Answers questions with source citations and confidence scoring.
- Provides ResearchOps tools:
  - model card generation;
  - reviewer-response drafting;
  - experiment summary;
  - metrics lookup from CSV/JSON files;
  - GitHub issue draft generation;
  - research plan generation.
- Includes a Gradio UI for Google Colab.
- Includes an optional FastAPI endpoint for local/API usage.
- Includes a small benchmark runner for retrieval and grounded QA checks.

## Suggested CV description

**SPN ResearchOps Copilot - Personal Applied AI Project**  
Built an agentic RAG assistant for probabilistic ML research, connecting papers, Python code, LaTeX manuscripts and experiment outputs through grounded LLM responses and tool-based research workflows.

- Implemented hybrid retrieval with embeddings, BM25, FAISS vector search, metadata filtering and citation-based answer generation.
- Added tool workflows for experiment querying, model-card generation, reviewer-response drafting and GitHub issue planning.
- Built Gradio/FastAPI interfaces with confidence thresholds, source tracing and hallucination-reduction guardrails.

## Run in Google Colab

### Option A - From GitHub after you upload this project

1. Create a new GitHub repository, for example:

```text
spn-researchops-copilot
```

2. Upload the contents of this project to that repository.

3. In Google Colab, run:

```python
!git clone https://github.com/MariaAlexandraBadea/spn-researchops-copilot.git
%cd spn-researchops-copilot
!pip -q install -r requirements-colab.txt
```

4. Clone your existing SPN forecasting repository into the RAG project's data folder:

```python
!python scripts/clone_repo.py \
  --repo https://github.com/MariaAlexandraBadea/cnn-gru-spn-gamma-temperature-forecast \
  --out data/source_repo
```

5. Build the retrieval index:

```python
!python scripts/ingest.py \
  --input data/source_repo \
  --index data/index \
  --reset
```

6. Start the Gradio UI:

```python
!python app/gradio_app.py --index data/index --share
```

The command will print a public Gradio URL.

### Option B - Upload documents manually in Colab

```python
from google.colab import files
uploaded = files.upload()
```

Then move uploaded files to `data/manual_docs/` and ingest:

```python
import os, shutil
os.makedirs("data/manual_docs", exist_ok=True)
for name in uploaded.keys():
    shutil.move(name, f"data/manual_docs/{name}")

!python scripts/ingest.py --input data/manual_docs --index data/index --reset
!python app/gradio_app.py --index data/index --share
```

## Optional OpenAI usage

The app works in extractive fallback mode without an API key. For LLM-generated responses:

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."
```

or in Colab:

```python
from getpass import getpass
import os
os.environ["OPENAI_API_KEY"] = getpass("OpenAI API key: ")
```

## Example questions

- Explain the CNN-GRU-SPN-Gamma architecture and cite the source code.
- What leakage controls are implemented in the preprocessing pipeline?
- How does the conditional SPN-Gamma decoder differ from a standard MDN?
- What metrics are used for probabilistic and multivariate forecast evaluation?
- Generate a model card for the forecasting model.
- Draft a reviewer response explaining the role of uncertainty quantification.
- Create a GitHub issue for adding calibration plots to the project.
- Summarize experiment results from CSV files.
