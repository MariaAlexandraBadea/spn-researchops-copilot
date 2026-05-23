# Colab commands

After uploading this project to GitHub as `spn-researchops-copilot`, run in Colab:

```python
!git clone https://github.com/MariaAlexandraBadea/spn-researchops-copilot.git
%cd spn-researchops-copilot
!pip -q install -r requirements-colab.txt
```

Optional OpenAI key:

```python
from getpass import getpass
import os
os.environ["OPENAI_API_KEY"] = getpass("OpenAI API key: ")
```

Clone your existing SPN repository:

```python
!python scripts/clone_repo.py \
  --repo https://github.com/MariaAlexandraBadea/cnn-gru-spn-gamma-temperature-forecast \
  --out data/source_repo \
  --reset
```

Build index:

```python
!python scripts/ingest.py --input data/source_repo --index data/index --reset
```

Run UI:

```python
!python app/gradio_app.py --index data/index --metrics-root data/source_repo --share
```
