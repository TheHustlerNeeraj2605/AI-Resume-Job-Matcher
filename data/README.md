# Data Directory Structure & Management Guidelines

This directory contains datasets and intermediate artifacts for the **AI-Based Resume–Job Matching & Skill Gap Analysis** project.

---

## Directory Organization

```text
data/
├── raw/          # Unmodified source datasets (Kaggle, Hugging Face, or academic corpora)
├── processed/    # Cleaned, tokenized, normalized, and feature-engineered datasets
├── evaluation/   # Benchmarking sets, ground-truth label pairs, and manually curated test sets
└── README.md     # Documentation and data governance guidelines (this file)
```

---

## Data Governance & Version Control Principles

1. **No Raw Data in GitHub**:
   - In accordance with best ML practices and repository size limitations, raw datasets are excluded from Git tracking via `.gitignore`.
   - Never commit sensitive candidate information, uncleaned external dumps, or proprietary documents to version control.

2. **Source & License Documentation**:
   - Every dataset ingested into `data/raw/` must have its source URL, author/organization, release version, and license terms explicitly documented in this directory prior to model experimentation.
   - Licensing compatibility will be verified before incorporating any dataset into production pipelines.

3. **Reproducibility of Processed Data**:
   - All files inside `data/processed/` must be deterministically reproducible from `data/raw/` via scripts located in `src/`.
   - Data preprocessing pipelines will maintain fixed random seeds and structured transformations.

4. **Acquisition & Preparation Workflow**:
   - Clear, step-by-step download and ingestion instructions (or reproducible download/extraction scripts) will be provided once candidate datasets pass exploratory validation and licensing checks.
