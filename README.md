# AI-Based Resume–Job Matching & Skill Gap Analysis

> **Project Status: Production Inference & Streamlit Application Complete**
> *Academic / Decision-Support Prototype — Complete Offline Reproducibility*
> *Verified Architecture: Model F (Ratio-Focused LightGBM) + Temperature Scaling ($T=1.4173$) + Decoupled Explainability + Interactive Streamlit UI*

---

## 1. Project Title
**AI-Based Resume–Job Matching & Skill Gap Analysis**

---

## 2. Project Overview
The **AI-Based Resume–Job Matching & Skill Gap Analysis** system is an end-to-end, machine learning–powered decision-support prototype designed to evaluate alignment between candidate resumes and job postings. Rather than relying on brittle keyword filtering or uncalibrated large language models, the system combines:

- **Lexical Overlap**: N-gram Term Frequency–Inverse Document Frequency (TF-IDF) representation and token Jaccard similarity.
- **Dense Semantic Embeddings**: Pre-trained Sentence-BERT (`sentence-transformers/all-MiniLM-L6-v2`) contextual embeddings.
- **Deterministic Skill Diagnostics**: Curated 55-skill taxonomy regex extraction identifying explicit overlaps and potential gaps.
- **Document Length & Ratio Signals**: Length-invariant relative ratios controlling for verbose corporate job posting boilerplate.
- **Non-Linear Tabular Classification**: A ratio-focused LightGBM decision forest (**Model F**, 18 features) trained with balanced class weights.
- **Post-Hoc Probability Calibration**: Monotonic Temperature Scaling ($T = 1.4173$) to ensure honest confidence metrics.
- **Decoupled Explainability Layer**: Purely functional, non-causal diagnostic feedback without prediction distortion ($\Delta = 0$).
- **Interactive Web Interface**: Streamlit application supporting direct PDF ingestion and live diagnostic visualization.

> **Important Role**: This system is designed as an **assistive decision-support prototype** for recruiters and applicants. It does **not** make autonomous hiring decisions, nor does it guarantee employment suitability or candidate competence.

---

## 3. Problem Statement
Modern talent acquisition workflows face substantial operational and technical bottlenecks:
1. **Screening Fatigue & High Applicant Volumes**: Recruiters review hundreds of profiles per requisition, risking inconsistent evaluations.
2. **Brittle Keyword Filtering (Traditional ATS)**: Exact-string filters unfairly reject qualified candidates who express identical proficiencies using alternative phrasing (e.g., *"statistical modeling"* vs. *"machine learning"*).
3. **Black-Box LLM / Heuristic Fragility**: Naive prompt engineering or uncalibrated scores produce unpredictable outputs, lack auditability, and frequently hallucinate qualifications.
4. **Vocabulary & Perspective Mismatch**: Resumes record past accomplishments in first-person narrative, whereas job postings outline aspirational requirements, corporate background, and non-technical boilerplate.
5. **Lack of Actionable Feedback**: Candidates rarely receive structured, explainable feedback identifying actionable technical gaps.

---

## 4. Key Objectives
- **Standardized Ingestion Pipeline**: Ingest unstructured text and digitally generated PDF resumes in-memory with Unicode normalization and technical token preservation.
- **Rigorous Leakage-Safe Evaluation**: Prevent data contamination via resume-level grouped partitioning and isolate the official unseen-job benchmark from candidate-level benchmarks.
- **Empirical Baseline-to-Advanced Progression**: Systematically benchmark TF-IDF baselines, Sentence-BERT bi-encoders, supervised linear models, transformer variants (MPNet), Cross-Encoders, and non-linear tree ensembles.
- **Length-Robust Modeling**: Prevent decision trees from overfitting corporate job-description formatting styles and lengths.
- **Honest Probability Calibration**: Apply leakage-safe temperature scaling so that confidence values reflect empirical accuracy rather than tree overconfidence.
- **Auditable & Non-Causal Explainability**: Deliver structured diagnostic feedback (*Matched Skills*, *Potential Skill Gaps*, *Model Confidence*) without altering model predictions.
- **Demonstration Interface**: Provide an executive-level Streamlit web dashboard for interactive exploration and auditability.

---

## 5. Final Architecture

```text
               +-----------------------------------+
               | Candidate PDF Resume / Plain Text |
               +-----------------+-----------------+
                                 |
                                 v
                     [ PDFResumeExtractor ]
                 (pypdf / NFKC / Token Safe)
                                 |
           +---------------------+---------------------+
           |                                           |
           v                                           v
[ Clean Resume Text ]                       [ Job Description Text ]
           |                                           |
           +---------------------+---------------------+
                                 |
                                 v
                    [ Feature Extraction Pipeline ]
          ┌─────────────────────────────────────────────────┐
          │ 1. TF-IDF Cosine Similarity (Fitted Train Vocab)│
          │ 2. MiniLM Contextual Semantic Cosine Sim (384d) │
          │ 3. Deterministic Skill Extraction (55 taxonomy) │
          │ 4. Set Overlaps: Matched Skills & Missing Skills│
          │ 5. Relative Document Ratios (Word & Char Ratios)│
          │ 6. Token Overlap & Jaccard Lexical Metrics      │
          │ 7. Explicit Boolean Signal & Match Indicators   │
          └──────────────────────┬──────────────────────────┘
                                 | (18 Selected Features)
                                 v
                 [ Model F: Ratio-Focused LightGBM ]
                   (300 Trees, Depth 5, lr 0.03)
                   (No Raw Absolute JD Counts)
                                 |
                                 v
                     [ Temperature Calibrator ]
                     (T = 1.4173, Monotonic)
                                 |
                                 v
                   [ InferenceResult (Locked) ]
                   - Predicted Label: No / Potential / Good Fit
                   - Calibrated Probabilities & Model Confidence
                   - Raw Model Probabilities
                   - 18 Model Features
                                 |
                                 v
                 [ Decoupled Explainability Layer ]
                 - build_explanation() (Δ = 0 Invariance)
                 - Matched Skills & Potential Skill Gaps
                 - Skill Coverage % (Explicit None vs 0.0%)
                 - Descriptive Alignment Summary
                                 |
                                 v
                [ Streamlit Web Application (app/) ]
                - Color-Coded Prediction Card
                - Horizontal Calibrated Probability Bars
                - 4-Metric Skill Overview & Styled Tags
                - Supporting Signals & Technical Audit Table
```

---

## 6. Dataset & Data Audit Findings

The system was evaluated using the open paired resume–job description corpus from Hugging Face: [`cnamuangtoun/resume-job-description-fit`](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit).
- **Source & Provenance**: Public community research dataset containing pairs of unstructured resume text, job posting descriptions, and 3-class fit labels (`No Fit`, `Potential Fit`, `Good Fit`).
- **License Status**: The upstream repository card lists license metadata as unspecified / open community access. No commercial warranty is asserted.

An initial audit in Phase 1 and Phase 2 uncovered critical structural properties that shaped the entire project:

1. **Repeated Resumes & Job Postings**:
   - Multiple candidate profiles appeared repeatedly across diverse job postings.
   - Individual corporate job postings were paired with numerous candidates.
2. **Duplicate & Conflicting Pairs**:
   - Exact duplicate pairs and near-duplicate variations were identified and audited (specifically, 7 exact duplicate resume–job pairs were identified in the raw training data and verified for label consistency, but retained to preserve total dataset integrity).
   - Multi-label instances (resumes paired with multiple jobs yielding conflicting ground-truth labels) were audited to prevent training ambiguity.
3. **Benchmark Disconnect (Official Test vs. Candidate Generalization)**:
   - **Official Test Set ($N=1,759$)**: Evaluates generalization to **unseen job descriptions**. Crucially, most resumes in this partition overlapped with profiles in the training set. It serves strictly as an **unseen-job benchmark**.
   - **Candidate-Level Grouped Split ($N=1,201$)**: To test whether models generalize to **unseen candidate profiles**, a leak-free grouped split was created (`candidate_train`, `candidate_validation`, `candidate_test`) enforcing strict resume-hash isolation.

---

## 7. Leakage-Safe Evaluation Strategy

To guarantee scientific validity:
- **Strict Grouped Partitioning**: All variations and applications of a candidate resume were restricted to exactly one partition (`train`, `val`, or `test`).
- **Frozen Preprocessing**:
  - `TfidfVectorizer` (27,762 terms) was fit strictly on `candidate_train`. Validation and test splits were transformed using the frozen vectorizer with zero out-of-fold vocabulary leakage.
  - Probability calibration was fitted strictly on `candidate_validation` ($N=1,282$). Test sets remained completely untouched.
- **Dual-Benchmark Evaluation**:
  - **Candidate Test Benchmark ($N=1,201$)**: Measures performance on completely unseen candidates applying to familiar job requisitions.
  - **Official Test Benchmark ($N=1,759$)**: Measures cross-domain robustness when candidates apply to completely unseen corporate job postings.

---

## 8. Baseline Models

1. **Baseline 1: TF-IDF Cosine Similarity (1D Thresholding)**:
   - Macro F1: 0.4393 on candidate test, 0.3691 on official test.
   - *Failure Mode*: Vocabulary mismatch. Good Fit recall was only 27.57% (64.9% of all errors were false negatives).
2. **Baseline 2: Sentence-BERT Cosine Similarity (MiniLM-L6-v2)**:
   - Improved semantic paraphrasing, but 1D scalar thresholding could not resolve the substantial overlap between `Potential Fit` and `Good Fit`.
3. **Decoupled Architecture**:
   - Error analysis confirmed that unsupervised 1D scalar thresholds cannot separate fine-grained classes. Relevance classification must be supervised over multi-faceted features, while skill extraction operates as an independent diagnostic engine.

---

## 9. Model Experimentation Summary

Across Phases 6 through 9, multiple model architectures and feature variations were empirically evaluated under controlled conditions:

| Phase | Model / Configuration | Features | Cand. Val F1 | Cand. Test F1 | Official Test F1 | Generalization Gap | Status / Decision |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Phase 3** | TF-IDF Cosine Baseline | 1 (1D) | 0.4335 | 0.4393 | 0.3691 | -0.0644 | Baseline reference |
| **Phase 4** | Sentence-BERT Baseline (MiniLM) | 1 (1D) | ~0.4350 | ~0.4400 | ~0.3750 | -0.0600 | Baseline reference |
| **Phase 6A** | **Model A**: Logistic Regression (Balanced) | 22 | 0.4603 | 0.4735 | 0.4220 | -0.0383 | Linear Reference Model |
| **Phase 6B** | **Model B**: Section-Aware Preprocessing (LR) | 22 | 0.4417 | 0.4465 | 0.4072 | -0.0345 | **Rejected**: Stripping JD context hurt F1 |
| **Phase 6C** | **Model B**: MPNet-Base-v2 (LR) | 22 | 0.4627 | 0.4811 | 0.4130 | -0.0497 | **Rejected**: 4.7x size, official test dropped |
| **Phase 7** | **Model C**: Cross-Encoder Augmented (LR) | 23 | 0.4590 | 0.4772 | 0.4211 | -0.0379 | **Rejected**: Validation dropped; 100x CPU latency |
| **Phase 8A** | Natural LightGBM | 22 | 0.5662 | 0.5188 | 0.3838 | -0.1824 | **Fragile**: Overfit raw absolute JD lengths |
| **Phase 8A** | Balanced LightGBM | 22 | 0.5824 | 0.5783 | 0.4164 | -0.1660 | High candidate F1; fragile on unseen jobs |
| **Phase 8B** | Model D: Natural Length-Ablated LightGBM | 19 | 0.5328 | 0.4902 | 0.3945 | -0.1383 | Ablating JD length improved robustness |
| **Phase 8B** | **Model E**: Balanced Length-Robust LightGBM | 19 | 0.5533 | **0.5468** | 0.4113 | -0.1420 | **Candidate Champion** (Specialized Ranker) |
| **Phase 8B** | **Model F: Ratio-Focused LightGBM** | **18** | **0.4952** | **0.4712** | **0.4360** | **-0.0592** | **LOCKED PRODUCTION CLASSIFIER** |
| **Phase 9** | Best Ensemble E+F (90:10 blend) | - | 0.5545 | 0.5389 | 0.4104 | -0.1441 | **Rejected**: Interpolates; Model F beats on jobs |
| **Phase 9** | Best Ensemble E+F+A (50:25:25 blend) | - | 0.5445 | 0.5275 | 0.4232 | -0.1212 | **Rejected**: Single models are Pareto optimal |
| **Phase 10C**| **Model F + Temperature Scaling ($T=1.42$)** | **18** | **0.4952** | **0.4712** | **0.4360** | **-0.0592** | **LOCKED PRODUCTION CALIBRATED STACK** |

---

## 10. Final Model Selection (Model F)

### Architectural Rationale
**Model F (Ratio-Focused LightGBM)** was selected as the locked primary production classifier:
- **18 Engineered Features**:
  1. `tfidf_cosine_similarity`
  2. `sbert_cosine_similarity`
  3. `skill_coverage`
  4. `num_job_skills`
  5. `num_resume_skills`
  6. `num_matched_skills`
  7. `num_missing_skills`
  8. `resume_word_count`
  9. `resume_char_count`
  10. `resume_to_job_word_ratio`
  11. `resume_to_job_char_ratio`
  12. `log_resume_word_count`
  13. `num_shared_unique_tokens`
  14. `jaccard_similarity`
  15. `resume_unique_token_count`
  16. `has_any_detected_job_skill`
  17. `has_any_detected_resume_skill`
  18. `has_any_matched_skill`

### Why Absolute Job-Description Counts Were Excluded
In Phase 8A, tree models over-relied on absolute job description features (`job_unique_token_count`, `job_word_count`, `job_char_count`), which accounted for 34.83% of total tree gain. When tested on external job descriptions with different writing formats, this caused catastrophic errors.

Model F strictly excludes all absolute JD length features. This regularizes the tree ensemble to split purely on **content alignment** and **relative ratios**, yielding:
- Peak official unseen-job Macro F1 of **0.4360** (surpassing Model A's 0.4220).
- The narrowest official generalization gap (**-0.0592**).
- The lowest rate of catastrophic `Good Fit` $\rightarrow$ `No Fit` false negatives on unseen jobs (142 cases vs. 182 for Model A and 281 for natural LightGBM).

*Generalization Trade-off Note*: While Model F is the champion on unseen job distributions, Model E achieves higher Macro F1 (0.5468) when evaluating unseen candidates within known job templates. Single-model Model F was locked for production because cross-domain job robustness is critical for a general-purpose screening tool.

---

## 11. Probability Calibration

Tree-boosting algorithms output softmax probabilities that optimize classification rank margins rather than posterior true log-loss calibration. On unseen job descriptions, uncalibrated Model F exhibited an Expected Calibration Error (ECE) of **0.0634**.

### Methodology & Guarantees
- **Method**: Post-hoc **Temperature Scaling** ($T = 1.4173$).
  $$\hat{p}_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$
- **Fitting Data**: Fitted strictly on `candidate_validation.csv` ($N = 1,282$) with zero access to test splits.
- **Invariance Guarantee**: Scalar temperature scaling is strictly monotonic:
  $$\Delta \text{predicted\_label} = 0$$
  Exactly **0 predictions were changed** across all 4,242 evaluated samples.
- **Calibration Impact**:
  - Official Unseen-Job Test ECE dropped from 0.0634 to **0.0292** (**-53.9% relative error reduction**).
  - Validation Log Loss improved from 1.0342 to **1.0235**.
  - Mean confidence softened from 0.5161 to 0.4654, aligning honestly with empirical accuracy (~45.5%).

> **Framing Disclaimer**: Model confidence reflects internal classification certainty among `[No Fit, Potential Fit, Good Fit]`. It is **NOT** a probability of hiring, interview likelihood, or candidate success.

---

## 12. Production Inference Pipeline

The production inference pipeline is packaged under `src/inference/`:

### Core Components
- `ResumeJobMatcher` (`src/inference/pipeline.py`): Core orchestrator initializing the frozen vectorizer, MiniLM bi-encoder, Model F tree classifier, and temperature calibrator.
- `PDFResumeExtractor` (`src/inference/pdf_extractor.py`): In-memory PDF text extractor using `pypdf` with Unicode NFKC normalization and technical symbol preservation (`C++`, `C#`, `.NET`, `AWS`).
- `FeatureExtractor` (`src/inference/features.py`): Deterministic 18-feature extractor matching Model F training order.
- `SkillExtractor` (`src/inference/skills.py`): Curated regex taxonomy extractor for 55 technical competencies.
- `build_explanation` (`src/inference/explainability.py`): Presentation layer mapping inference outputs to `ExplainabilityResult`.

### Programmatic Usage

```python
from src.inference import ResumeJobMatcher

# Initialize matcher (loads vectorizer, MiniLM, Model F, and Calibrator)
matcher = ResumeJobMatcher()

# Option A: Predict from raw text
result = matcher.predict(
    resume_text="Senior Machine Learning Engineer with 5 years experience in Python, PyTorch, Docker, and AWS.",
    job_description_text="Hiring ML Engineer proficient in Python, PyTorch, Kubernetes, and AWS."
)

# Option B: Predict directly from PDF file
result = matcher.predict_from_pdf(
    pdf_source="path/to/resume.pdf",
    job_description_text="Hiring ML Engineer proficient in Python, PyTorch, Kubernetes, and AWS."
)

# Generate non-destructive presentation explanation
explanation = matcher.explain(result)

print(f"Predicted Class: {explanation.predicted_label}")
print(f"Model Confidence: {explanation.confidence:.2%}")
print(f"Calibrated Probabilities: {explanation.calibrated_probabilities}")
print(f"Raw Probabilities: {explanation.raw_probabilities}")
print(f"Matched Skills: {explanation.matched_skills}")
print(f"Potential Skill Gaps: {explanation.potential_skill_gaps}")
print(f"Skill Coverage: {explanation.skill_coverage_percentage}%")
```

---

## 13. Explainability & Skill Gap Diagnostics

The explainability layer provides transparent, non-causal feedback:
- **Non-Destructive**: Purely functional; $\Delta \text{prediction} = 0$, $\Delta \text{probabilities} = 0$.
- **Strict Terminology**:
  - Uses **"Model Confidence"** (avoiding "hiring probability" or "selection odds").
  - Uses **"Matched Skills"** and **"Potential Skill Gaps"** (avoiding "verified competencies" or "confirmed deficiencies").
- **Skill Coverage Semantics**:
  - `skill_coverage_percentage = None`: When the job description contains **zero detected technical skills** in the taxonomy. The UI displays *"No technical skills detected in job description"* rather than a misleading `0%`.
  - `skill_coverage_percentage = 0.0%`: When job skills are detected, but the resume matches none of them.
- **Taxonomy Limitations**:
  - Relies on deterministic regex matching over a 55-skill technical dictionary.
  - Does **not** assess depth of experience, years of practice, project recency, or seniority.
  - Does not cover non-technical disciplines (e.g. healthcare, sales, legal).

---

## 14. Streamlit Application

The interactive web dashboard is located in `app/streamlit_app.py`.

```bash
streamlit run app/streamlit_app.py
```

### Application Features & Visual Hierarchy
1. **Executive Header**: Professional title, subtitle, and an explicit decision-support disclaimer.
2. **Input Section**: In-memory PDF upload and multi-line job description input with validation against empty, short (<3 words), or corrupted files.
3. **Prediction & Overall Fit**: Prominent color-coded banner:
   - 🟢 **Good Fit** | 🟡 **Potential Fit** | 🔴 **No Fit**
   - Displays **Model Confidence** (calibrated max probability).
4. **Calibrated Probability Distribution**: Horizontal progress bars showing calibrated probability for each class side-by-side.
5. **Match Explanation**: Non-causal narrative explaining document alignment based on learned model signals.
6. **Skill Analysis & Skill Gaps**:
   - 4-metric grid: *Job Skills Detected*, *Resume Skills Detected*, *Matched Skills*, *Potential Skill Gaps*.
   - Skill coverage bar and informative status captions.
   - High-contrast styled chips for *Matched Skills* (green) and *Potential Skill Gaps* (amber).
7. **Supporting Matching Signals**: Upfront view of Semantic Similarity, TF-IDF Similarity, Token Jaccard, and Word Ratio, with Boolean alignment flags (Semantic, Lexical, Skills).
8. **Technical Auditability Expander**: Side-by-side comparison of **Calibrated vs. Raw Probabilities** ($T = 1.4173$) and complete JSON metadata.

---

## 15. Evaluation Results Summary

| Evaluation Benchmark | Metric | TF-IDF Baseline | Model A (LR Reference) | Model F (Raw LightGBM) | Model F (Calibrated LightGBM) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Candidate Test** ($N=1,201$) | Macro F1 | 0.4393 | 0.4735 | 0.4712 | **0.4712** |
| (Unseen Candidates) | Accuracy | 52.37% | 51.54% | 50.29% | **50.29%** |
| | Good Fit Recall | 27.57% | 45.51% | 43.19% | **43.19%** |
| | ECE | - | - | 0.0333 | **0.0369** |
| **Official Test** ($N=1,759$) | Macro F1 | 0.3691 | 0.4220 | 0.4360 | **0.4360** |
| (Unseen Job Descriptions) | Accuracy | 42.98% | 45.08% | 45.48% | **45.48%** |
| | Expected Calibration Error | - | - | 0.0634 | **0.0292 (-53.9%)** |
| | Catastrophic Errors ($G \rightarrow N$) | 288 | 182 | 142 | **142 (Lowest)** |

---

## 16. Limitations

To ensure transparency and responsible usage, the following limitations must be noted:

1. **Dataset Bias & Paired Corpus Noise**: Training data originates from public web-scraped resumes and job descriptions. Annotation labels reflect historical human screening biases.
2. **Official Benchmark Structure**: In the official test set, job descriptions are unseen while candidate profiles overlap with training data.
3. **Generalization Trade-Off**: Model F excels on unseen job descriptions (0.4360 F1), while Model E excels on unseen candidates (0.5468 F1). No single model dominates both splits simultaneously.
4. **Deterministic Skill Taxonomy**: Skill extraction uses 55 predefined technical regex terms; it cannot detect unlisted emerging frameworks, niche tools, or domain-specific jargon.
5. **No Experience Depth or Recency**: The taxonomy detects keyword presence only; it cannot determine whether a skill was used for 5 years or learned yesterday.
6. **Text-Based PDF Constraint**: Ingestion requires text-encoded digital PDFs. Scanned image-only PDFs requiring OCR are rejected with an explicit prompt.
7. **Model Confidence Semantics**: Confidence represents multi-class softmax probability, not an absolute measure of candidate qualification or hiring viability.
8. **Domain Specificity**: The model is optimized for software, technology, and engineering roles; performance on healthcare, legal, or arts roles may be limited.
9. **Decision-Support Role**: The system is an assistive screening aid and must **never** be used for autonomous or unmonitored hiring decisions.
10. **Corpus Language**: All feature extractors and tokenizers assume English-language documentation.

---

## 17. Reproducibility & Installation

The project uses standard Python tooling and avoids proprietary cloud dependencies or vector database infrastructure.

### Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Resume_Matching_Project.git
cd Resume_Matching_Project
```

### 2. Create and Activate Virtual Environment
**Windows PowerShell:**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Model Artifacts & Dataset Availability Notice
In accordance with GitHub storage best practices and repository hygiene rules, large dataset files (`data/raw/`, `data/processed/`, `data/evaluation/`) and trained model binary weights (`models/*.joblib`, `models/*.pkl`) are excluded from Git tracking via `.gitignore`.
- **Required Model Artifacts**: Production inference and tests require three serialized artifacts in `models/`:
  1. `models/tfidf_vectorizer.joblib` (frozen training vocabulary vectorizer)
  2. `models/model_f_ratio_focused_lgbm.pkl` (Model F LightGBM classifier)
  3. `models/model_f_calibrator.joblib` (Temperature Scaling calibrator, $T = 1.4173$)
- **Local Reproduction & Current Scope**: The production artifacts are generated through the project's modular pipelines:
  - `models/tfidf_vectorizer.joblib`: Extracted and fitted on training vocabulary via `notebooks/01_dataset_audit.ipynb`.
  - `models/model_f_ratio_focused_lgbm.pkl`: Trained and exported via `run_phase8b_pipeline.py`.
  - `models/model_f_calibrator.joblib`: Fitted and exported via `run_phase10c_pipeline.py`.
  Note that public cloud hosting / external release bucket downloads for pre-baked binary weights are **not** currently bundled with this repository; users running from a fresh clone must train/export artifacts locally using the documented reproduction scripts or place their existing trained artifacts into `models/`.

---

## 18. How to Run

### Run Test Suite
Execute the complete test suite (regression, calibration, pipeline, and UI tests):
```bash
pytest -q
```
*Expected: 67 passed.*

### Run Sanity Check Script
Verify end-to-end inference against historical evaluation data:
```bash
python tests/sanity_check_inference.py
```

### Launch Interactive Streamlit Application
```bash
streamlit run app/streamlit_app.py
```
Open your web browser at `http://localhost:8501`.

---

## 19. Repository Structure

```text
Resume_Matching_Project/
├── .gitignore                      # Git exclusion rules (data, weights, caches)
├── DECISIONS.md                    # Architecture Decision Records (001 - 017)
├── README.md                       # Comprehensive documentation (this file)
├── requirements.txt                # Locked Python dependencies
│
├── app/
│   ├── __init__.py                 # Application package init
│   └── streamlit_app.py            # Streamlit interactive demonstration UI
│
├── data/
│   ├── raw/                        # Source datasets (ignored by Git)
│   ├── processed/                  # Normalized train/val/test splits (ignored)
│   ├── evaluation/                 # Extracted feature datasets (ignored)
│   └── README.md                   # Data governance & ingestion protocol
│
├── docs/                           # Architectural design documents & notes
│
├── models/                         # Model weights and artifacts (ignored by Git)
│   ├── .gitkeep
│   ├── tfidf_vectorizer.joblib     # Frozen TF-IDF vectorizer (27,762 terms)
│   ├── model_f_ratio_focused_lgbm.pkl # Model F LightGBM classifier
│   └── model_f_calibrator.joblib   # Temperature Scaling calibrator (T=1.4173)
│
├── notebooks/                      # Research notebooks (01 through 09)
│   ├── 01_dataset_audit.ipynb
│   ├── 02_error_analysis.ipynb
│   ├── 03_supervised_classifier.ipynb
│   ├── 04_section_aware_preprocessing.ipynb
│   ├── 05_semantic_model_comparison.ipynb
│   ├── 06_cross_encoder_evaluation.ipynb
│   ├── 07_nonlinear_classifier_lightgbm.ipynb
│   ├── 08_length_robust_lightgbm.ipynb
│   └── 09_ensemble_robustness.ipynb
│
├── src/
│   ├── __init__.py
│   └── inference/                  # Production inference package
│       ├── __init__.py             # Public API exports
│       ├── schemas.py              # Strongly-typed InferenceResult & ExplainabilityResult
│       ├── pdf_extractor.py        # PDF text extraction & normalization
│       ├── skills.py               # 55-skill deterministic taxonomy & matcher
│       ├── features.py             # 18-feature extraction pipeline
│       ├── calibration.py          # TemperatureScalingCalibrator
│       ├── explainability.py       # Non-destructive explainability layer
│       └── pipeline.py             # ResumeJobMatcher main facade
│
└── tests/                          # Automated Pytest suite (67 tests)
    ├── sanity_check_inference.py   # Historical sample verification script
    ├── test_calibration.py         # Calibration invariant tests
    ├── test_explainability.py      # Non-destructive explainability tests
    ├── test_inference_pipeline.py  # 18-feature inference pipeline tests
    ├── test_pdf_extractor.py       # PDF parsing & validation tests
    └── test_streamlit_app.py       # Streamlit UI & invariant tests
```

---

## 20. Future Improvements
- **Taxonomy Expansion**: Expand the deterministic taxonomy from 55 technical skills to broader domain ontologies (e.g., O*NET, ESCO, finance, medical).
- **OCR Support**: Incorporate a lightweight, privacy-preserving local OCR engine (e.g. Tesseract) to support scanned image PDFs.
- **Hierarchical Section Parsing**: Explore soft section-weighting models rather than hard text stripping.
- **Top-K Cross-Encoder Reranking**: For candidate retrieval across thousands of applicants, explore two-stage filtering (LightGBM fast-filter $\rightarrow$ Cross-Encoder reranker on top-10 candidates).
- **Docker Containerization**: Build an official multi-platform Dockerfile for simplified cloud container deployment.

---

## 21. Responsible-Use & Decision-Support Disclaimer

> **IMPORTANT ETHICAL & LEGAL NOTICE**
> This software is an experimental, academic research prototype designed solely to assist human decision-makers by highlighting text patterns, semantic similarity, and explicit skill overlap.
>
> - **No Autonomous Decisions**: This tool must never be used to make automated or autonomous employment decisions, rejections, or hiring determinations.
> - **Human-in-the-Loop Requirement**: All recruitment outcomes must be reviewed and decided by qualified human talent professionals.
> - **No Fairness or Competence Warranty**: The model was trained on historical web data and may reflect underlying domain distributions. It does not measure candidate character, integrity, intellectual capability, or practical work competence.
