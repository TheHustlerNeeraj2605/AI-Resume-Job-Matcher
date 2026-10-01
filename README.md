# AI-Based Resume–Job Matching & Skill Gap Analysis

> **Project Status: Phase 1 — Project Foundation & Architecture Setup**  
> *Note: This project is currently in the initial design and foundational setup stage. No models have been trained or evaluated yet.*

---

## 1. Project Title
**AI-Based Resume–Job Matching & Skill Gap Analysis**

---

## 2. Project Overview
The **AI-Based Resume–Job Matching & Skill Gap Analysis** system is designed to provide an automated, objective, and semantically grounded evaluation of candidate resumes against specific job descriptions. 

Rather than relying purely on brittle keyword lookups, the system compares resume content with role requirements across lexical and dense semantic representations. The core outputs provided to the user include:
- **Resume–Job Relevance / Match Score**: An aggregate compatibility indicator reflecting overall fit.
- **Semantic Similarity**: Dense vector-based alignment measuring contextual relevance beyond exact phrasing.
- **Matched Skills**: Technical and domain competencies shared by both the resume and the job posting.
- **Missing Skills**: Critical prerequisites and keywords identified in the job description that are absent or underrepresented in the resume.
- **Skill-Gap Analysis**: Structured feedback outlining actionable areas where the candidate’s profile diverges from job expectations.

---

## 3. Problem Statement
Traditional talent acquisition and resume screening workflows face significant challenges:
1. **High Volume & Manual Bottlenecks**: Recruitment teams receive hundreds of applications per role, leading to screening fatigue and prolonged hiring cycles.
2. **Brittle Keyword Filtering**: Traditional Applicant Tracking Systems (ATS) often rely on exact word matching, unfairly rejecting qualified candidates who express identical concepts using synonyms or alternative terminology (e.g., *"machine learning"* vs. *"statistical modeling"*).
3. **Lack of Actionable Feedback**: Job seekers rarely receive concrete insights into why their profile failed automated filters or what specific technical proficiencies are missing.
4. **Vocabulary Mismatch**: Resumes and job descriptions are written from different perspectives—resumes focus on past accomplishments, while job descriptions specify forward-looking demands.

There is a clear need for an intelligent, dual-faceted matching engine that combines the transparency of lexical matching with the contextual depth of semantic embeddings to yield fair, interpretable, and actionable gap analyses.

---

## 4. Planned Objectives
- **Standardized Text Processing**: Construct a robust NLP preprocessing pipeline to ingest, clean, tokenize, and normalize unstructured resume and job description text.
- **Classical Baseline Formulation**: Establish a transparent benchmark using term frequency-inverse document frequency (TF-IDF) and cosine similarity to quantify lexical overlap.
- **Dense Semantic Modeling**: Implement transformer-based sentence representations using pre-trained Sentence-Transformers (Sentence-BERT) to capture deep contextual semantic similarity.
- **Skill Extraction & Taxonomy Alignment**: Develop rule-based and entity-aware extraction modules to isolate skills and perform precise set-theoretic gap analysis (matched vs. missing skills).
- **Leakage-Safe Benchmarking**: Build a leak-free evaluation harness with resume-level grouping and a curated ground-truth test set to prevent optimistic bias.
- **Interactive Visual Interface**: Build a lightweight, intuitive Streamlit web application enabling recruiters and candidates to inspect matching scores and visual skill breakdowns in real time.

---

## 5. Planned Technology Stack

| Domain | Technology / Library | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Core programming runtime |
| **Data Manipulation** | Pandas, NumPy, SciPy | Structured data handling, vector calculations, matrix operations |
| **Classical ML & NLP** | Scikit-learn, NLTK | Tokenization, text normalization, vectorizers, TF-IDF baseline, metrics |
| **Deep Learning & Embeddings** | PyTorch, Sentence-Transformers | Sentence-BERT contextual embeddings and dense similarity computation |
| **User Interface** | Streamlit | Rapid, interactive web dashboard for user testing and demonstration |
| **Testing & Quality** | Pytest | Unit and integration testing across extraction and scoring modules |
| **Version Control** | Git, GitHub | Distributed version control and reproducible collaboration |

*Note: In accordance with core engineering principles, complex orchestration frameworks, vector databases, and external LLM APIs are intentionally excluded at this foundational stage to maintain simplicity, interpretability, and complete offline reproducibility.*

---

## 6. Planned System Pipeline

```text
+-----------------------+     +--------------------------+
|      Resume Text      |     |  Job Description Text    |
+-----------+-----------+     +------------+-------------+
            |                              |
            +--------------+---------------+
                           |
                           v
        [ Text Normalization & Cleaning ]
        (Lowercasing, Tokenization, Noise Removal)
                           |
            +--------------+--------------+
            |                             |
            v                             v
   [ Track A: Lexical ]          [ Track B: Semantic ]
   - TF-IDF Vectorization        - Sentence-BERT Encoding
   - Lexical Cosine Similarity   - Dense Vector Embeddings
            |                    - Dense Cosine Similarity
            +--------------+--------------+
                           |
                           v
              [ Skill Extraction Engine ]
              (Entity Matching / Taxonomy Lookup)
                           |
                           v
             [ Skill-Gap Analysis Module ]
             - Matched Skills Intersection
             - Missing Skills Difference
             - Weighted Match Scoring
                           |
                           v
            [ Streamlit Web Application ]
            - Match Percentage & Metrics
            - Side-by-Side Skill Breakdown
```

1. **Ingestion & Text Cleaning**: Clean raw text inputs, strip formatting noise, expand common acronyms, and preserve technical identifiers.
2. **Dual-Track Similarity Engine**:
   - **Track A (Lexical Baseline)**: TF-IDF matrix generation and cosine similarity calculation.
   - **Track B (Dense Semantic)**: Sentence-Transformers embedding inference and dense vector cosine similarity.
3. **Skill Extraction & Set Operations**: Extract technical skills from both documents using curated taxonomies and calculate:
   $$\text{Matched Skills} = S_{\text{resume}} \cap S_{\text{job}}$$
   $$\text{Missing Skills} = S_{\text{job}} \setminus S_{\text{resume}}$$
4. **Composite Scoring & Gap Reporting**: Produce a calibrated match score that synthesizes semantic alignment with explicit skill coverage.
5. **Interactive Presentation**: Display results interactively in Streamlit with visual indicators and clear gap diagnostics.

---

## 7. Planned Evaluation Strategy

To ensure scientific rigor and avoid premature or inflated performance claims, the evaluation plan consists of:

1. **Leakage-Safe Partitioning**:
   - Resumes will be split using **resume-level grouping**. No resume or candidate profile will ever appear across both train and test partitions.
   - Cross-validation will account for potential duplicate or near-duplicate job descriptions.
2. **Manually Reviewed Benchmark Set**:
   - A dedicated evaluation subset (`data/evaluation/`) will be manually reviewed and annotated with ground-truth relevance rankings (e.g., Highly Relevant, Moderately Relevant, Irrelevant).
3. **Quantitative Metrics**:
   - **Ranking & Correlation**: Spearman rank correlation between model match scores and human annotations; Precision@K and Mean Reciprocal Rank (MRR) for candidate retrieval.
   - **Skill Extraction Accuracy**: Precision, Recall, and F1-score on manually annotated skill tokens from unseen resumes.
   - **Ablation Studies**: Quantitative comparison of the TF-IDF baseline against Sentence-BERT embeddings to measure true semantic gain.

---

## 8. Future Deployment Plan

The planned roadmap for operationalizing the application includes:
- **Phase 1 (Current)**: Foundation setup, architecture decisions, repository structure, and data governance.
- **Phase 2**: Dataset ingestion, exploratory data analysis, and leakage-safe preprocessing.
- **Phase 3**: Implementation and benchmarking of the TF-IDF baseline.
- **Phase 4**: Integration of Sentence-Transformers and skill-gap analysis engine.
- **Phase 5**: Interactive Streamlit UI development (`app/`).
- **Phase 6**: Containerization via Docker for deterministic local or cloud execution.
- **Phase 7**: Lightweight cloud hosting (e.g., Streamlit Community Cloud or Hugging Face Spaces) backed by automated CI testing on GitHub Actions.

---

## 9. Repository Structure

```text
resume-job-matcher/
│
├── data/
│   ├── raw/                  # Unmodified source datasets (excluded from Git)
│   ├── processed/            # Cleaned and normalized datasets (reproducible)
│   ├── evaluation/           # Curated evaluation benchmarks
│   └── README.md             # Data governance and ingestion documentation
│
├── notebooks/                # Exploratory analysis and prototyping notebooks
│
├── src/                      # Production modular source code (preprocessing, models, metrics)
│
├── tests/                    # Automated unit and integration test suites
│
├── app/                      # Streamlit application scripts and UI components
│
├── docs/                     # Extended documentation and architectural specifications
│
├── models/                   # Local model weights and embeddings (excluded from Git)
│
├── DECISIONS.md              # Architectural decision log (ADRs)
├── README.md                 # Primary project documentation (this file)
├── requirements.txt          # Python dependencies
└── .gitignore                # Git exclusion rules for ML artifacts and temporary files
```
