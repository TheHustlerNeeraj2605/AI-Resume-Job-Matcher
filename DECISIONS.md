# Technical Decisions

This document records major technical decisions made during development.
Each decision should contain:
- Decision
- Context
- Alternatives considered
- Chosen approach
- Rationale
- Tradeoffs
- Consequences

---

## Decision 001: Dataset Strategy

- **Decision**: Curate and validate publicly available resume and job description datasets under strict provenance, licensing, and quality controls before model development, ensuring raw datasets are not automatically committed to GitHub.
- **Context**: Resume–job matching requires realistic paired text corpora. Public datasets vary widely in license permissiveness, cleanliness, annotation quality, and deduplication.
- **Alternatives Considered**:
  - *Synthetic dataset generation*: Quick to produce via LLMs, but lacks real-world candidate noise, domain variance, and organic formatting anomalies.
  - *Immediate ad-hoc downloading of arbitrary Kaggle datasets*: Fast startup, but risks licensing violations, unvetted distribution bias, cross-split leakage, and repository bloat.
  - *Systematic public dataset investigation with vetting*: Assess candidate public datasets against predefined criteria prior to model training.
- **Chosen Approach**: Systematically audit publicly available datasets (such as open Kaggle and Hugging Face corpora) for licensing, provenance, duplicates, class distribution, and data leakage before final selection. Raw data will be stored locally and ignored by Git.
- **Rationale**: Ensures the project remains compliant with open-source licensing, prevents repository bloat, and guarantees data integrity before spending computational resources on modeling.
- **Tradeoffs**: Requires front-loaded investigation time before modeling begins, rather than immediately loading the first available dataset.
- **Consequences**: Data exploration scripts and validation checklists will be documented in `data/` and `notebooks/`; raw artifacts will remain outside Git tracking.

---

## Decision 002: Evaluation Strategy

- **Decision**: Implement leakage-safe dataset splitting (incorporating resume-level grouping) alongside a manually reviewed ground-truth evaluation set.
- **Context**: A common pitfall in matching systems is data leakage across train, validation, and test splits (e.g., identical resumes or job postings appearing in both training and test sets), resulting in inflated performance metrics.
- **Alternatives Considered**:
  - *Random row-level train/test split*: Standard in generic ML pipelines, but susceptible to severe data leakage when candidate resumes or corporate job descriptions recur with slight variations.
  - *Resume-level grouped splitting*: Ensures that all instances, variations, or pairings of a single resume remain strictly isolated within one partition.
  - *Grouped splitting supplemented with an expert-reviewed benchmark set*: Combines grouped partitioning with a dedicated golden evaluation set.
- **Chosen Approach**: Use leakage-safe train/validation/test partitioning with resume-level grouping. Additionally, establish a separately curated, manually reviewed evaluation benchmark set to conduct final human-verifiable validation.
- **Rationale**: Group-aware splitting prevents optimistic bias, while the curated evaluation set provides realistic qualitative and quantitative benchmarks for semantic ranking and skill gap identification.
- **Tradeoffs**: Constructing and verifying a grouped split and manually reviewed set requires extra upfront data engineering and labeling effort.
- **Consequences**: Splitting scripts must enforce group isolation by candidate ID or resume hash; evaluation reports will clearly contrast automated metrics against the curated benchmark.

---

## Decision 003: Model Progression

- **Decision**: Follow a disciplined baseline-to-advanced progression starting with classical TF-IDF + cosine similarity, advancing to transformer-based semantic embeddings (Sentence-Transformers), and evaluating hybrid approaches only after establishing individual benchmarks.
- **Context**: Modern NLP applications often leap directly to heavy deep learning or large language models without establishing whether a simple, computationally lightweight baseline suffices or where dense representations provide tangible value.
- **Alternatives Considered**:
  - *Direct jump to transformer fine-tuning*: High initial complexity and computational overhead without an empirical point of comparison.
  - *Purely classical TF-IDF approach*: Fast and interpretable, but fails on semantic paraphrasing, synonymy, and contextual variations in job descriptions.
  - *Disciplined progression (TF-IDF Baseline -> Sentence-BERT -> Hybrid Evaluation)*: Implement and measure classical lexical matching first, followed by dense semantic encoders, then evaluate combinations.
- **Chosen Approach**: Establish a classical TF-IDF + cosine similarity baseline first. Next, implement dense semantic representations using pre-trained Sentence-Transformers (Sentence-BERT). Evaluate both independently before considering a weighted or cascading hybrid architecture.
- **Rationale**: Grounding the system in a lightweight baseline provides a solid reference for latency, interpretability, and accuracy gains, ensuring architectural complexity is justified by measurable improvements.
- **Tradeoffs**: Requires maintaining two distinct modeling pathways and evaluation harnesses during the experimental phase.
- **Consequences**: `src/` modules will cleanly decouple lexical extraction/matching from embedding generation, allowing clean ablation studies and modular benchmarking.

---

## Decision 004: Deployment Strategy

- **Decision**: Select Streamlit for the initial user interface.
- **Context**: The project aims to demonstrate an end-to-end ML/NLP pipeline with real-time text input, similarity scores, and visual skill gap analysis without diverting focus into extensive frontend engineering.
- **Alternatives Considered**:
  - *Decoupled FastAPI backend + React/Vue frontend*: Highly scalable for enterprise SaaS, but introduces substantial overhead (state management, CORS, routing, dual repositories) irrelevant to core ML validation.
  - *Jupyter Notebooks only*: Minimal engineering effort, but poor usability for non-technical stakeholders and awkward demonstration of interactive user flows.
  - *Streamlit UI*: Native Python dashboard framework offering rapid prototyping, native component rendering, and zero-boilerplate interactive UI.
- **Chosen Approach**: Use Streamlit to build an interactive, single-page application located in `app/`.
- **Rationale**: Streamlit allows end-to-end Python implementation, enabling rapid delivery of interactive candidate-job comparison, match scoring, and skill-gap visualization without complex frontend scaffolding.
- **Tradeoffs**: Less customizable than custom single-page web applications and not ideal for high-concurrency microservice architectures.
- **Consequences**: Application code will be contained in `app/` and consume core inference utilities exported from `src/`, keeping application logic decoupled from ML algorithms.

---

## Decision 005: GitHub and Reproducibility

- **Decision**: Maintain a clean, GitHub-friendly repository containing strictly source code, documentation, lightweight configuration, and reproducible recipes, while strictly barring secrets, virtual environments, caches, large datasets, and binary model artifacts.
- **Context**: Machine learning repositories frequently suffer from repository bloat, accidental commits of large datasets or model weights, and platform-specific path dependencies.
- **Alternatives Considered**:
  - *Commit model checkpoints and datasets directly to Git*: Causes repo bloat, slow clones, and potential violation of GitHub's file size limits (100MB).
  - *Git LFS for all data and weights*: Adds external dependency configuration and potential bandwidth quota constraints for collaborators.
  - *Strict .gitignore + reproducible artifact generation*: Keep the Git repository code-only, using documented download procedures, deterministic preprocessing pipelines, and pre-trained Hugging Face hubs for weight acquisition.
- **Chosen Approach**: Keep the Git repository lean and code-only. Exclude all virtual environments, credentials, system files, raw/processed datasets, and binary checkpoints via `.gitignore`.
- **Rationale**: Guarantees fast clone times, zero sensitive credential leaks, cross-platform compatibility, and compliance with GitHub standards.
- **Tradeoffs**: Users cloning the repository must execute data preparation and model download scripts rather than receiving pre-baked binaries out-of-the-box.
- **Consequences**: Every pipeline step must be fully reproducible from configuration and code; setup instructions and requirements must be documented in `README.md`.
