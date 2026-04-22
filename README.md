<div align="center">

# Financial Classification

<p>
  A practical, extensible foundation for financial transaction classification and modeling.
</p>

<p>
  <a href="#">
    <img alt="Status" src="https://img.shields.io/badge/status-active-2ea44f?style=for-the-badge" />
  </a>
  <a href="#">
    <img alt="Python" src="https://img.shields.io/badge/python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  </a>
  <a href="#">
    <img alt="Machine Learning" src="https://img.shields.io/badge/focus-classification-0A66C2?style=for-the-badge" />
  </a>
  <a href="#">
    <img alt="Finance" src="https://img.shields.io/badge/domain-finance-1f6feb?style=for-the-badge" />
  </a>
</p>

</div>

## Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Data Preprocessing](#data-preprocessing)
- [Example Workflow](#example-workflow)
- [Metrics to Track](#metrics-to-track)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

## Overview

`financial-classification` is a machine learning final project home base for classifying financial data (transactions, expense types, and merchant categories).

## Project Structure

```text
financial-classification/
├── README.md
├── data/                 # Raw and processed datasets
├── notebooks/            # Exploration and experiments
├── src/                  # Core source code
├── tests/                # Unit/integration tests
└── models/               # Saved model artifacts
```

## Quick Start

### 1) Clone the repository

```bash
git clone https://github.com/darielgu/financial-classification.git
cd financial-classification
```

### 2) Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Run your training or classification script

```bash
python -m src.main
```

## Data Preprocessing

Run one shared pipeline to combine both datasets and generate reusable splits for all models:

```bash
python src/data_preprocessing.py
```

This creates:
- `data/processed/transactions_preprocessed.csv`
- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`
- `data/processed/feature_columns.txt`

Use the same preprocessing object across models:

```python
from src.model_data import get_data_for_model

x_train, y_train, x_val, y_val, x_test, y_test, preprocessor = get_data_for_model()
```

## Example Workflow

```text
Ingest -> Clean -> Engineer Features -> Train -> Evaluate -> Predict
```

## Metrics to Track

- Accuracy
- Precision / Recall / F1
- Confusion Matrix
- Class imbalance performance

## Roadmap

- Add baseline model + benchmark script
- Add experiment tracking
- Add model versioning and reproducible pipelines

## Contributing

1. Create a feature branch
2. Commit focused changes
3. Open a pull request with clear context

## License

Add your preferred license (for example, MIT) and include a `LICENSE` file.

---

<p align="center">
  Built for practical, production-minded financial ML workflows.
</p>
