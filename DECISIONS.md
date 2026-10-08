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

---

## Decision 006: Diagnostic Error Analysis & Decoupled Architecture

- **Decision**: Adopt a decoupled dual-track architecture separating relevance classification from explainable skill diagnostics, driven by empirical error analysis demonstrating that 1D unsupervised similarity thresholds cannot resolve severe class overlap and lexical mismatch.
- **Context**: Across Baselines 1 (TF-IDF), 2 (Sentence-BERT), Phase 3 (Skill Extraction), and Phase 4 (Hybrid linear combination), models struggled to surpass ~0.44 Macro F1 on unseen candidates. Rigorous error analysis on `candidate_test` ($N = 1,201$) was required to isolate root failure modes.
- **Key Empirical Findings**:
  - *Severe False Negatives (64.86% of errors)*: Good Fit recall is only 27.57% (135/301 Good Fits predicted No Fit) and Potential Fit recall is only 29.30% (153/273 predicted No Fit), driven by vocabulary mismatch and synonymy.
  - *No Fit Stability*: No Fit precision is 61.80% and recall is 74.32% (F1 = 0.6749), showing that TF-IDF successfully isolates distinct, non-overlapping career domains.
  - *Job Description Boilerplate Noise*: Error cases exhibit 23% longer job descriptions (median 369 words vs 300 words for correct cases) due to non-technical company overviews, benefits, and EEO statements that dilute cosine similarity.
  - *Skill Coverage Limits as a Global Metric*: While skill coverage offers high explainability, 67.1% of true Good Fits have 0.0 skill coverage because the corpus includes non-technical or specialized roles outside the 55-skill technical taxonomy.
- **Chosen Approach**:
  - Lock a **Decoupled Architecture**: Maintain TF-IDF as the primary baseline classifier while running Skill Extraction as an independent explainability and feedback engine.
  - Plan next phase around **Supervised Tabular Classification** (combining similarity, length ratios, and skill coverage with non-linear decision boundaries) and **Boilerplate Stripping / Section-Aware Parsing**.
- **Rationale**: Linear hybrid scoring forces conflicting trade-offs between technical and non-technical job descriptions. Decoupling ensures robust classification while providing actionable diagnostic feedback to recruiters and applicants.
- **Consequences**: Prevents wasting effort on arbitrary manual hybrid weight tuning; focuses future efforts on supervised classifiers and section-aware text preprocessing.

---

## Decision 007: Supervised Multi-Signal Classification

- **Decision**: Adopt a supervised multi-signal linear classifier (`LogisticRegression(C=0.1, class_weight='balanced')` with `StandardScaler`) over 1D scalar thresholding as the primary match prediction engine.
- **Context**: In Phase 5, error analysis demonstrated that 1D unsupervised similarity thresholds failed due to class overlap (Good Fit median 0.0746 vs $t_2 = 0.1164$) and severe false negatives (Good Fit recall 27.57%, Potential Fit recall 29.30%). Phase 6A evaluated whether a supervised classifier trained on training data with balanced class weights can learn a superior decision boundary across 22 engineered multi-signal features.
- **Alternatives Considered**:
  - *Retain 1D TF-IDF cosine thresholding*: Simple and requires zero parameters, but intrinsically bottlenecked at 0.4393 Macro F1 with severe false negatives (64.9% of all errors).
  - *Linear SVM (`LinearSVC`)*: Evaluated across $C \in [0.1, 1.0, 10.0]$; achieved strong validation accuracy (0.5016) but slightly lower validation Macro F1 (0.4510) compared to Logistic Regression.
  - *Supervised Logistic Regression with StandardScaler*: Regularized linear model with balanced class weighting across similarity, skill extraction, length ratios, and lexical overlap features.
- **Chosen Approach**: Select **Logistic Regression ($C=0.1$, class_weight='balanced')** trained with `StandardScaler` strictly on `candidate_train`.
- **Measured Empirical Results**:
  - *Validation Macro F1*: Improved from 0.4335 (TF-IDF) to **0.4603** ($+0.0268$).
  - *Candidate Test Macro F1 (Unseen Candidates)*: Improved from 0.4393 (TF-IDF) to **0.4735** ($+0.0342$ or $+7.78\%$ relative lift).
  - *Good Fit Recall*: Surged from **27.57%** (83 pairs) to **45.51%** (137 pairs) — a **+65.1% relative recovery in true positive matches**!
  - *Potential Fit Recall*: Rose from **29.30%** (80 pairs) to **35.90%** (98 pairs).
  - *Total False Negatives (predicting No Fit on positive/potential pairs)*: Slashed from **288 down to 217** (a reduction of 71 missed fits, $-24.65\%$).
  - *Official Test Macro F1 (Unseen JDs)*: Improved from 0.3691 to **0.4220** ($+14.33\%$ relative lift) and accuracy improved from 0.4298 to **0.4508**.
- **Rationale**: The supervised classifier learns complementary weights across multiple signals: positive coefficients on `jaccard_similarity` (+0.255) and `sbert_cosine_similarity` (+0.184) recover synonym/lexical mismatches, while length normalization features (`job_unique_token_count`, `resume_to_job_char_ratio`) control for verbose corporate boilerplate.
- **Tradeoffs**: Requires maintaining a 22-feature preprocessing pipeline and storing a lightweight model checkpoint (`models/supervised_logistic_regression.joblib`, ~4 KB) rather than a simple 2-threshold rule.
- **Consequences**: Supervised multi-signal learning is locked as the core predictive framework. Arbitrary manual thresholding is abandoned in favor of learned probabilistic boundaries.

---

## Decision 008: Section-Aware Job Description Preprocessing

- **Decision**: Retain Model A (trained on raw, unedited job descriptions) as the official Reference Model; reject rule-based boilerplate stripping and hard section-aware filtering as an upstream transformation for the primary classifier based on empirical negative evaluation.
- **Context**: In Phase 5 (Error Analysis) and Phase 6A (Supervised Classifier), job descriptions in misclassified pairs were substantially longer than in correct pairs (mean ~371 words, median 313 words), with verbose corporate boilerplate (EEO statements, company background, benefits, application disclaimers). Furthermore, length features (`job_unique_token_count`, `resume_to_job_char_ratio`) had high feature coefficients. Phase 6B tested the hypothesis that conservative boilerplate removal and section-aware extraction (`job_core_text`) would reduce noise and improve macro F1.
- **Alternatives Considered**:
  - *Adopt Section-Aware Preprocessing (Model B)*: Clean all job descriptions using deterministic inline regex boilerplate removal, heading-based segmentation (`CORE_SUMMARY`, `CORE_RESPONSIBILITIES`, `CORE_REQUIREMENTS`), and fallback safeguards, then retrain the supervised classifier.
  - *Retain Raw Job Descriptions (Model A)*: Keep full original job postings for feature extraction (TF-IDF, SBERT, skill extraction, length features).
  - *Aggressive Truncation / Head-Only Extraction*: Truncate job descriptions at the first occurrence of non-core headers.
- **Chosen Approach**: Retain **Model A (Raw Job Descriptions)** as the locked reference model. Document Model B's empirical results transparently in the evaluation repository.
- **Measured Empirical Results**:
  - *Corpus Cleaning Effect*: Across 351 unique JDs, 268 (76.35%) were modified, reducing mean word count from 371.1 to 309.3 words (16.6% reduction; median text retained 99.50%, mean text retained 86.62%). Fallback triggered on 3 JDs.
  - *Validation Macro F1*: Dropped from **0.4603** (Model A) to **0.4417** (Model B) ($\Delta = -0.0186$).
  - *Candidate Test Macro F1 (Unseen Candidates)*: Dropped from **0.4735** (Model A) to **0.4465** (Model B) ($\Delta = -0.0270$).
  - *Candidate Test Accuracy*: Dropped from **0.5154** (Model A) to **0.4929** (Model B) ($\Delta = -0.0225$).
  - *Official Test Macro F1 (Unseen JDs)*: Dropped from **0.4220** (Model A) to **0.4072** (Model B) ($\Delta = -0.0148$).
  - *Official Test Accuracy*: Dropped from **0.4508** (Model A) to **0.4406** (Model B) ($\Delta = -0.0102$).
  - *Error Transitions on Candidate Test*: While Model B recovered 62 previously misclassified pairs, it caused 89 previously correct pairs to regress (net negative change of $-27$ cases).
  - *Per-Class Degradation*: The largest drop occurred in `Potential Fit` F1, which fell from 0.3616 to 0.3160 ($-4.56\%$), with precision falling from 0.3643 to 0.3208.
- **Rationale / Why Cleaning Degraded Performance**:
  - *Loss of implicit domain context*: Broad company overview text ("leading healthcare provider", "fintech trading firm") provides domain-level cues that Sentence-BERT dense embeddings utilize to determine industry alignment.
  - *Destabilization of length calibration*: In Model A, the classifier learned balanced weights for length ratios to calibrate raw text lengths. Truncating JDs shifted the input distributions away from the linear classifier's optimal weights.
  - *Erosion of borderline lexical overlap*: Trimming non-core text reduced peripheral word co-occurrences that helped distinguish `Potential Fit` from `No Fit`.
- **Consequences**: Model A remains the active reference model. Upstream text deletion will not be used for classification; any future section-aware modeling should explore soft section weighting or attention rather than hard heuristic text removal.

---

## Decision 009: Stronger Semantic Representation (MiniLM vs MPNet)

- **Decision**: Retain Model A (`all-MiniLM-L6-v2`) as the standard production reference model; archive Model B (`all-mpnet-base-v2`) as an evaluated high-capacity variant due to inconsistent cross-dataset gains and significant latency/memory overhead.
- **Context**: In Phase 6A, semantic similarity was represented by `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions). Phase 6C evaluated whether substituting a larger, 12-layer pretrained dense model (`sentence-transformers/all-mpnet-base-v2`, 768 dimensions) improves classification performance across candidate and official test splits under controlled A/B conditions.
- **Alternatives Considered**:
  - *Adopt MPNet as the Default Embedding Engine (Model B)*: Replace MiniLM similarity with MPNet cosine similarity across all 22 features.
  - *Dual-Embedding Feature Concatenation (Model C)*: Include both MiniLM and MPNet similarities (23 features).
  - *Retain MiniLM Reference Model (Model A)*: Maintain the lightweight 384-dimensional MiniLM pipeline as the primary matching engine.
- **Chosen Approach**: Retain **Model A (`all-MiniLM-L6-v2`)** as the active Reference Model for production deployment. Document Model B's empirical performance in the benchmark registry.
- **Measured Empirical Results**:
  - *Resource & Latency Comparison*: MPNet requires 4.7x more disk storage (~420 MB vs ~90 MB) and 3.5x-4.0x slower CPU encoding latency (~28ms vs ~8ms per document; 617s vs ~140s total encoding time).
  - *Validation Selection*: Model B yielded a marginal improvement on `candidate_validation` Macro F1 (**0.4627** vs **0.4603** for Model A; $\Delta = +0.0024$). Model C lagged at **0.4537**.
  - *Candidate Test (Unseen Candidates)*: Model B achieved **0.4811 Macro F1** ($+0.0076$ lift, $+1.61\%$) and **0.5237 Accuracy** ($+0.0083$).
  - *Official Test (Unseen JDs)*: Model B declined to **0.4130 Macro F1** ($-0.0090$ drop, $-2.13\%$), while accuracy was essentially flat (**0.4520** vs **0.4508**).
  - *Per-Class Tradeoffs*: On `candidate_test`, MPNet improved `Potential Fit` F1 (0.3616 $\rightarrow$ **0.3848**, $+2.32\%$) and `No Fit` F1 (0.6254 $\rightarrow$ **0.6359**), but slightly reduced `Good Fit` recall (45.51% $\rightarrow$ **43.19%**).
  - *Error Transitions on Candidate Test*: 103 pairs were recovered (8.6%), 93 pairs regressed (7.7%), resulting in a modest net gain of $+10$ pairs.
- **Rationale / Tradeoff Evaluation**:
  - The performance improvement of MPNet is modest (+0.0076 F1 on unseen candidates) and inconsistent across domains ($-0.0090$ F1 on unseen job descriptions).
  - A 4.7x model footprint and 3.5x inference cost increase does not justify replacing the nimble MiniLM model in latency-sensitive matching scenarios.
  - Furthermore, the correlation between MiniLM and MPNet similarities is high ($r = 0.7341$), confirming that scaling bi-encoder size alone does not overcome the fundamental architectural bottleneck: independent vector compression of multi-faceted resumes and complex job specifications.
- **Consequences**: Model A remains the production reference model. Future enhancements should explore non-linear classifiers (e.g., LightGBM) or fine-grained Cross-Encoder architectures that compute cross-attention between resume and job tokens, rather than scaling bi-encoder parameter count.

---

## Decision 010: Cross-Encoder Signal Evaluation

- **Decision**: Retain Model A (22 features, MiniLM bi-encoder, regularized Logistic Regression) as the standard production Reference Model; do NOT adopt zero-shot Cross-Encoder score augmentation for the primary classification pipeline due to high computational latency ($O(N \times M)$) and lack of validation improvement.
- **Context**: In Phase 6C, we established that bi-encoder cosine similarity compresses documents into a single dense vector, limiting multi-skill match resolution. Phase 7 evaluated whether an off-the-shelf Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) providing token-level cross-attention could provide complementary signal when added as feature 23 to the supervised classifier.
- **Alternatives Considered**:
  - *Standalone Cross-Encoder Classifier (Model B)*: Predict 3-class fit exclusively using the Cross-Encoder score. Result: Failed to separate classes (Validation Macro F1 = 0.3561, `Potential Fit` F1 collapsed to 0.1084 due to binary pretraining).
  - *Cross-Encoder Augmented Feature (Model C)*: Add Cross-Encoder score to the 22 existing features. Result: Validation Macro F1 = 0.4590 (vs 0.4603 for Model A); Candidate Test Macro F1 = 0.4772 (+0.0037); Official Test Macro F1 = 0.4211 (-0.0009).
  - *Retain Model A (Reference)*: Keep the 22-feature bi-encoder + tabular classifier.
- **Chosen Approach**: Retain **Model A** as the active Reference Model. Document Model B and Model C benchmarks in the project registry.
- **Measured Empirical Results**:
  - *Complementarity*: Cross-Encoder showed moderate correlation with MiniLM ($r = 0.3722$) and TF-IDF ($r = 0.3634$), confirming genuinely distinct token-level interaction signal.
  - *Validation Selection*: Model C scored **0.4590** vs Model A's **0.4603** ($\Delta = -0.0013$), failing the validation selection criterion.
  - *Candidate Test (Unseen Candidates)*: Model C achieved a marginal gain to **0.4772 Macro F1** ($+0.0037$) and **0.5187 Accuracy** ($+0.0033$).
  - *Official Test (Unseen JDs)*: Model C scored **0.4211 Macro F1** ($-0.0009$) and **0.4503 Accuracy** ($-0.0005$).
  - *Per-Class*: Model C slightly improved `Good Fit` recall (45.51% $\rightarrow$ **46.51%**, $+1.00\%$) and F1 (0.4335 $\rightarrow$ **0.4416**), while recovering 12 candidate test pairs against 8 regressions (net $+4$ pairs).
  - *Computational Latency*: Bi-encoder inference requires milliseconds over cached embeddings, whereas Cross-Encoder requires joint forward passes (~190 ms/pair on CPU; 1,524 seconds for the dataset).
- **Rationale / Tradeoff Evaluation**:
  - Outcome B / Outcome C applies: A negligible +0.0037 F1 gain on unseen candidates and negative validation/official-test deltas do not justify a 100x–1000x latency and memory expansion at inference time.
  - The off-the-shelf MS-MARCO Cross-Encoder was pretrained on binary search passage retrieval, not fine-grained 3-way resume fit.
- **Consequences**: Model A remains the production reference model. Zero-shot Cross-Encoder scores will not be added to the tabular feature set. Future work (Phase 8) should prioritize non-linear gradient-boosted tree classifiers (LightGBM/XGBoost) over the 22 features, or explore task-specific Cross-Encoder fine-tuning only for late-stage reranking (top-10 candidates).

---

## Decision 011: Non-Linear Gradient-Boosted Classification

- **Decision**: Validate the non-linear interaction hypothesis by proving massive performance gains on unseen candidates; retain Model A as the conservative baseline for unseen job formats while designating LightGBM as the candidate matching champion for in-distribution job roles.
- **Context**: In Phase 7, zero-shot Cross-Encoder scores were rejected due to inference latency and lack of validation gains. Phase 8A tested the central hypothesis: "Linear Logistic Regression may be unable to capture non-linear interactions among semantic similarity, lexical overlap, skill coverage, and document-length signals." We trained LightGBM (`LGBMClassifier`, 300 trees, depth=5, lr=0.03) over the exact 22 features without adding any new data or features.
- **Alternatives Considered**:
  - *Retain Linear Logistic Regression (Model A)*: Stable global hyperplane, high robustness against shifted job formatting, but unable to model non-linear feature gating (e.g., SBERT $\ge 0.45$ AND Skill Coverage $\ge 0.20$).
  - *Adopt LightGBM (Natural Distribution)*: Shallow tree boosting on the natural class distribution.
  - *Adopt LightGBM (Balanced Diagnostic)*: Tree boosting with balanced class weights.
- **Chosen Approach**: Document the empirical findings under **Outcome B** (generalization divergence between unseen candidates and unseen job formats):
  - LightGBM conclusively proves that non-linear feature interactions exist and yield immense predictive power on unseen candidates.
  - Retain Model A as the conservative reference for arbitrary unseen job structures, while recording LightGBM's superior candidate-level separation.
- **Measured Empirical Results**:
  - *Validation Macro F1*: Surged from **0.4603** (LR) to **0.5662** (LightGBM) ($\mathbf{+0.1059}$ or $\mathbf{+23.0\%}$ relative gain). Validation Accuracy jumped from **0.4743** to **0.6139** ($\mathbf{+13.96\%}$).
  - *Candidate Test (Unseen Candidates, $N=1,201$)*:
    - Macro F1 rose from **0.4735** (LR) to **0.5188** (LightGBM) ($\mathbf{+0.0453}$).
    - Accuracy surged from **0.5154** to **0.5895** ($\mathbf{+7.41\%}$ absolute gain).
    - `Good Fit` Precision jumped from **41.39%** to **62.20%** ($\mathbf{+20.81\%}$).
    - `Good Fit` Recall rose from **45.51%** to **50.83%** ($\mathbf{+5.32\%}$).
    - `Good Fit` F1 jumped from **0.4335** to **0.5594** ($\mathbf{+12.59\%}$).
    - `No Fit` F1 rose from **0.6254** to **0.6796** ($\mathbf{+5.42\%}$).
    - Net error transitions: **222 pairs recovered** vs **133 regressed** (net gain of **+89 correctly classified pairs**).
  - *Official Test (Unseen Job Descriptions, $N=1,759$)*:
    - Accuracy improved from **0.4508** to **0.4781** ($+2.73\%$), and `No Fit` F1 rose from **0.5572** to **0.6309**.
    - However, Macro F1 dropped from **0.4220** to **0.3838** ($-0.0382$) because `Good Fit` recall declined from 38.65% to 15.50% on unseen job formats.
  - *Feature Importance (Top Gain %)*: `job_unique_token_count` (13.13%), `tfidf_cosine_similarity` (12.05%), `job_word_count` (11.24%), `job_char_count` (10.46%), `sbert_cosine_similarity` (9.97%), `num_job_skills` (6.42%).
- **Rationale / Tradeoff Evaluation**:
  - The non-linear hypothesis is confirmed: tree ensembles easily carve out rectangular decision boxes (e.g. high semantic similarity + high lexical overlap) that linear hyperplanes blur.
  - The degradation on official test is specifically driven by tree over-reliance on job length thresholds (`job_unique_token_count`, `job_word_count`), which shift when job postings originate from external companies with different formatting styles.
  - Computational cost is negligible: 0.72s training time, 0.02ms prediction latency per pair, and a 1.45 MB model footprint.
- **Consequences**: LightGBM establishes a new performance peak on unseen candidates (58.95% accuracy, 55.94% Good Fit F1). Future work should focus on regularizing length feature splits (or length-invariant boosting) to preserve this non-linear lift across unseen job description distributions.

---

## Decision 012: Length-Robust Non-Linear Classification

- **Decision**: Adopt the Length-Robust regularized architecture as the superior non-linear modeling framework; specifically, adopt **Model E (Balanced Length-Robust LightGBM, 19 features)** for unseen candidate matching and **Model F (Ratio-Focused LightGBM, 18 features)** for robust cross-distribution generalization (surpassing the Logistic Regression reference on unseen job formats).
- **Context**: Phase 8A revealed that natural LightGBM achieved major gains on unseen candidates (Candidate Test Macro F1 0.5188 vs 0.4735 LR), but suffered official-test degradation (0.3838 Macro F1) due to heavy reliance (34.83% of tree split gain) on absolute job description length features (`job_unique_token_count`, `job_word_count`, `job_char_count`). Phase 8B systematically evaluated length ablation (removing raw JD counts), balanced learning (`class_weight="balanced"`), and ratio-focused learning (removing all absolute JD lengths including `log_job_word_count`).
- **Alternatives Considered**:
  - *Model A (Reference Logistic Regression, 22 feats)*: Validation Macro F1 = 0.4603, Cand Test Macro F1 = 0.4735, Official Test Macro F1 = 0.4220.
  - *Model B (Natural LightGBM, 22 feats)*: High candidate performance (0.5188 F1) but severe official degradation (0.3838 F1, Gen Gap = -0.1824).
  - *Model C (Balanced LightGBM, 22 feats)*: Superb candidate test (0.5783 Macro F1, Gen Gap = -0.0041) but fragile on official test (0.4164 F1).
  - *Model D (Natural Ablated LightGBM, 19 feats)*: Removed raw JD length features; official Macro F1 improved to 0.3945 (Official Gen Gap narrowed from -0.1824 to -0.1383).
  - *Model E (Balanced Length-Robust LightGBM, 19 feats)*: Preserved strong candidate generalization (0.5468 Macro F1, 56.37% accuracy) while slashing catastrophic Good Fit false negatives on official test by 112 cases.
  - *Model F (Ratio-Focused LightGBM, 18 feats)*: Removed all absolute JD lengths; achieved **0.4360 Official Test Macro F1** (surpassing Model A's 0.4220 by +0.0140) with the narrowest official generalization gap (-0.0592).
- **Chosen Approach**:
  - Designate **Model F (Ratio-Focused LightGBM)** as the primary cross-domain classifier: it establishes that non-linear tree ensembles can outperform linear Logistic Regression even across unseen job formats when stripped of absolute JD length artifacts.
  - Retain **Model E / Model C** as the candidate matching benchmarks for settings where job templates follow consistent training conventions.
- **Measured Empirical Results**:
  - *Balanced Weighting Impact*: Balanced learning (Model C & E) virtually eliminated candidate-level generalization drop (Model C Gen Gap = -0.0041 vs -0.0474 for Natural), lifting Candidate Test Macro F1 to **0.5783** (+0.1048 over Reference LR).
  - *Length Ablation Robustness*: Removing raw JD length features reduced the official generalization gap from -0.1824 (Model B) to -0.1383 (Model D), lifting Official Test Accuracy to **48.15%**.
  - *Catastrophic Error Recovery*: On official test, `Good Fit` $\rightarrow$ `No Fit` misclassifications collapsed from 281 (Model B) to 169 (Model E) — recovering **112 missed Good Fit matches**.
  - *Pure Relative Features*: Model F forced tree splits onto semantic & lexical interaction features: `tfidf_cosine_similarity` (15.50%), `sbert_cosine_similarity` (13.07%), `num_job_skills` (9.91%), and `jaccard_similarity` (8.51%). This yielded **0.4360 Official Test Macro F1** and **45.48% Accuracy**, beating Model A across both metrics.
  - *Computational Efficiency*: Inference latency remained ultra-low (0.017–0.020 ms per pair), with training completing in <0.70 seconds.
- **Rationale / Tradeoff Evaluation**:
  - Absolute job length is an artifact of posting style, not job-candidate compatibility. Ablating absolute JD length regularizes the decision trees to split on invariant content alignment features (TF-IDF, SBERT, skill overlap, word ratio).
  - Model F provides the best worst-case guarantee against out-of-distribution job posting formats while beating linear models.
- **Consequences**: Phase 8B successfully resolves the tree-boosting fragility identified in Phase 8A. For Phase 9, investigate combining Model E and Model F in an ensemble/stacking framework, or deploying Model F as the production multi-signal matching engine.

---

## Decision 013: Ensemble Robustness Evaluation

- **Decision**: Conclude model exploration; retain **Model F (Ratio-Focused LightGBM, 18 features)** as the primary cross-domain production classifier, retain **Model E (Balanced Length-Robust LightGBM, 19 features)** as the specialized candidate-ranking engine, and retain **Model A** as the linear reference baseline. Do NOT deploy probability blending in production because single-model architectures achieve superior Pareto efficiency with lower operational complexity.
- **Context**: In Phase 8B, Model E established peak unseen-candidate performance (0.5468 Macro F1), while Model F established peak unseen-job robustness (0.4360 Macro F1). Phase 9 tested whether probability-level ensembling between Model E, Model F, and Model A could eliminate the trade-off between candidate-level precision and cross-domain generalization without adding meta-models or violating leakage constraints.
- **Alternatives Considered**:
  - *Best E+F Ensemble (90:10)*: Selected strictly on candidate validation ($w_E=0.90, w_F=0.10$).
  - *Best E+F+A Ensemble (50:25:25)*: Selected from 6 predefined tri-model blends ($w_E=0.50, w_F=0.25, w_A=0.25$).
  - *Single Model F (Ratio-Focused LightGBM)*: 18 features, length-invariant, single-model inference.
  - *Single Model E (Balanced Length-Robust LightGBM)*: 19 features, single-model inference.
  - *Model A (Reference Logistic Regression)*: 22 features, linear hyperplane.
- **Chosen Approach**:
  - Reject production probability ensembling. The empirical results show that blending interpolates between the two models rather than dominating both: no ensemble configuration surpasses Model F on unseen jobs (0.4360 F1) or Model E on unseen candidates (0.5468 F1).
  - Adopt **Model F** as the standard primary matching classifier: it provides the strongest defense against arbitrary job description formatting shifts and the lowest rate of catastrophic `Good Fit` $\rightarrow$ `No Fit` false negatives (142 cases vs 169 for Model E, 182 for Model A).
  - Adopt **Model E** for applicant ranking within fixed job requisition templates.
- **Measured Empirical Results**:
  - *Model Complementarity*: Model E and Model F exhibited a 21.06% prediction disagreement rate on validation ($r = 0.8782$). Model A disagreed with tree models on ~34.6% of samples ($r \approx 0.63 - 0.67$), confirming distinct error distributions.
  - *Validation Weight Search*:
    - E:F grid peaked at 90:10 (Validation Macro F1 = **0.5545**, $+0.0012$ over Model E).
    - E:F:A grid peaked at 50:25:25 (Validation Macro F1 = **0.5445**).
  - *Candidate Test (Unseen Candidates, $N=1,201$)*:
    - Model E: **0.5468 Macro F1**, **56.37% Accuracy** (Champion).
    - Best E+F (90:10): 0.5389 Macro F1, 55.70% Accuracy.
    - Best E+F+A (50:25:25): 0.5275 Macro F1, 55.04% Accuracy.
    - Model A: 0.4735 Macro F1, 51.54% Accuracy.
    - Model F: 0.4712 Macro F1, 50.29% Accuracy.
  - *Official Test (Unseen Job Descriptions, $N=1,759$)*:
    - Model F: **0.4360 Macro F1**, **45.48% Accuracy** (Champion).
    - Best E+F+A (50:25:25): 0.4232 Macro F1, 44.68% Accuracy.
    - Model A: 0.4220 Macro F1, 45.08% Accuracy.
    - Model E: 0.4113 Macro F1, 43.77% Accuracy.
    - Best E+F (90:10): 0.4104 Macro F1, 43.72% Accuracy.
  - *Generalization Gaps*:
    - Candidate gaps: Model E ($-0.0065$), Best E+F ($-0.0156$), Best E+F+A ($-0.0169$), Model F ($-0.0240$), Model A ($+0.0132$).
    - Official gaps: Model F (**-0.0592**, narrowest tree gap), Best E+F+A ($-0.1212$), Model E ($-0.1420$), Best E+F ($-0.1441$).
  - *Catastrophic Errors (Good Fit $\rightarrow$ No Fit on Official Test)*:
    - Model F: **142** (fewest missed Good Fits).
    - Best E+F+A: 166.
    - Model E: 169.
    - Best E+F: 170.
    - Model A: 182.
  - *Inference Latency*: Single models require ~0.018 ms/pair; ensembles require ~0.034–0.036 ms/pair.
- **Rationale / Tradeoff Evaluation**:
  - Validation tuning is naturally biased toward candidate-level discriminative features because validation pairs share the same job requisition distribution as the training set. Consequently, validation-driven weights heavily favor Model E, preventing the ensemble from capturing Model F's cross-domain resilience on official test.
  - The tri-model blend (50:25:25) successfully beats Model A on both benchmarks, demonstrating viable linear shrinkage. However, it does not beat Model F on official test or Model E on candidate test.
  - Adding architectural and pipeline complexity for an interpolating blend is unprincipled when clean domain-specific single models already exist.
- **Consequences**: Conclude all model architecture experimentation. Progress to Phase 10: Building the deployment-ready inference engine, probability calibration, skill extraction visualization, and interactive application.

---

## Decision 014: Post-Hoc Probability Calibration via Temperature Scaling

- **Decision**: Calibrate production Model F probabilities using **Temperature Scaling ($T = 1.4173$)** fitted strictly on `candidate_validation.csv`, and expose both calibrated `class_probabilities` and uncalibrated `raw_class_probabilities` in the production inference API.
- **Context**: In Phase 9, Model F (Ratio-Focused LightGBM, 18 features) was locked as the primary cross-domain production model. While Model F demonstrates strong class separation, raw tree boosting probabilities can exhibit miscalibration, especially on out-of-domain job descriptions. Phase 10C evaluated whether probability calibration improves probability quality without compromising classification accuracy or decision boundaries.
- **Alternatives Considered**:
  - *Retain Uncalibrated Raw Probabilities*: Raw LightGBM softmax probabilities. Baseline validation calibration was reasonable ($\text{ECE} = 0.0339$), but calibration degraded sharply on out-of-distribution unseen job descriptions ($\text{ECE} = 0.0634$, Log Loss = 1.0542).
  - *Vector Scaling / Platt Multinomial Scaling*: Evaluated on validation. Disrupted class decision boundaries, changed 412 predictions (32.14%), and degraded validation Macro F1 to 0.4544 ($-0.0408$). **Strictly rejected**.
  - *Isotonic Regression (1-vs-Rest)*: Evaluated on validation. Non-parametric step function overfit validation artifacts, changed 304 validation predictions (23.71%), 287 on candidate test, and 457 on official test, degrading official test Macro F1 from 0.4360 to 0.4095. **Strictly rejected**.
  - *Temperature Scaling ($T > 0$)*: A single global temperature parameter dividing logits prior to softmax ($\sigma(z / T)$). By mathematical guarantee, scalar temperature scaling is strictly monotonic and **preserves the exact argmax class ranking ($\Delta \text{predictions} = 0$)** while softening overconfident margins.
- **Chosen Approach**: Temperature Scaling with learned parameter $T = 1.4173$ (persisted in `models/model_f_calibrator.joblib`).
- **Measured Empirical Results**:
  - *Candidate Validation ($N=1,282$ - Calibration Fit Split)*:
    - Log Loss: Improved from 1.0342 to **1.0235** ($-0.0107$).
    - Brier Score: Improved from 0.6180 to **0.6146** ($-0.0034$).
    - ECE: 0.0339 to **0.0336**.
    - Mean Confidence: Softened from 0.5291 to 0.4755.
    - Predictions Changed: **0 / 1,282 (0.00%)**. Macro F1: 0.4952 (exact match).
  - *Candidate Test ($N=1,201$ - Isolated Unseen Candidates)*:
    - Log Loss: 0.9901 $\rightarrow$ 0.9918 ($+0.0017$).
    - Brier Score: 0.5925 $\rightarrow$ 0.5938 ($+0.0013$).
    - ECE: 0.0333 $\rightarrow$ 0.0369 ($+0.0036$).
    - Predictions Changed: **0 / 1,201 (0.00%)**. Macro F1: 0.4712 (exact match).
  - *Official Test ($N=1,759$ - Isolated Unseen Job Descriptions)*:
    - Log Loss: Improved from 1.0542 to **1.0412** ($-0.0130$).
    - Brier Score: Improved from 0.6345 to **0.6273** ($-0.0072$).
    - ECE: Slashed from 0.0634 to **0.0292** (**-53.9% relative reduction in calibration error**).
    - Mean Confidence: Softened from 0.5161 to 0.4654 (better aligned with ~45.5% empirical accuracy).
    - Predictions Changed: **0 / 1,759 (0.00%)**. Macro F1: 0.4360 (exact match).
- **Rationale & Tradeoffs**:
  - Strictly leak-free: Calibrator was fit solely on candidate validation data without touching test sets.
  - Zero classification risk: Guarantees that neither Candidate Test Macro F1 (0.4712) nor Official Test Macro F1 (0.4360) are affected in any way.
  - Slashes overconfidence on cross-domain postings, delivering reliable, honest confidence metrics for downstream human recruiters.
- **Consequences**: Updated `src/inference/pipeline.py` and `schemas.py` to seamlessly apply `TemperatureScalingCalibrator`. `InferenceResult.class_probabilities` contains calibrated probabilities, while `InferenceResult.raw_class_probabilities` retains raw model outputs for transparency.

---

## Decision 015: Decoupled Explainability & Presentation Diagnostic Layer

- **Decision**: Introduce a decoupled, non-destructive Explainability & Presentation Layer (`build_explanation`, `ExplainabilityResult`) that transforms `InferenceResult` into presentation-ready diagnostics without altering underlying model predictions or probabilities.
- **Context**: While Model F produces probabilities and classification labels, presentation in recruiter and candidate interfaces requires structured, interpretable diagnostics, explicit separation of raw and calibrated probabilities, non-causal summary narratives, distinct representation of "no detected skills" vs "0% skill coverage", and strict avoidance of misleading terminology such as "hiring probability" or "confirmed deficiencies".
- **Alternatives Considered**:
  - *Coupled In-Model Interpretation / Direct UI Binding*: Expose raw feature dictionaries and loose strings directly to the frontend. Prone to brittle UI state management, scattered presentation heuristics, and accidental mutation of prediction state.
  - *Causal Explanations via Post-Hoc Feature Attribution (e.g. SHAP / LIME)*: Computationally expensive per query (~0.5–2.0s per inference on trees), unstable across correlated linguistic features, and introduces high liability by claiming unverified causal grounds for rejecting/accepting applicants.
  - *Decoupled Presentation Layer with Deterministic Diagnostics*: Purely functional transformation of existing pipeline outputs into a strongly typed, immutable `ExplainabilityResult` schema with non-causal descriptions, explicit confidence framing, and distinct "None vs 0.0%" skill coverage representation.
- **Chosen Approach**: Decoupled Explainability Layer implemented in `src/inference/explainability.py` returning `ExplainabilityResult`.
- **Key Architectural Principles**:
  - *Zero Prediction Distortion*: Mathematical guarantee that predicted classes, class probabilities, and confidence values are 100% invariant ($\Delta = 0$).
  - *Strict Non-Causal Framing*: Descriptive summaries ("Strong overall alignment based on learned combination...") rather than causal attribution ("Candidate rejected because X").
  - *Honest Terminology*: Explicit use of "Model Confidence", "Matched Skills", and "Potential Skill Gaps" (avoiding "hiring probability", "verified skills", or "confirmed deficiencies").
  - *Coverage Semantics*: If zero skills are detected in the job posting, `skill_coverage_percentage` evaluates to `None` rather than a misleading `0.0%`.
- **Tradeoffs**: Requires maintaining an additional presentation schema (`ExplainabilityResult`) and helper methods alongside `InferenceResult`.
- **Consequences**: Enables immediate, robust UI construction in Streamlit without embedding business logic into Streamlit widget scripts.

---

## Decision 016: Streamlit Presentation Layer & Visual Hierarchy Design

- **Decision**: Implement a native Streamlit user interface (`app/streamlit_app.py`) as a strictly decoupled presentation consumer of `src/inference/`, enforcing mathematical prediction invariance ($\Delta \text{prediction} = 0$, $\Delta \text{probabilities} = 0$), non-causal feedback framing, and distinct coverage states (`None` vs `0.0%`).
- **Context**: In Phases 11A, 11B, and 11C, an interactive demonstration interface was required to allow users to upload PDF resumes, input job descriptions, inspect calibrated match predictions, analyze skill gaps, and audit model features without adding heavy frontend frameworks (e.g. React/FastAPI) or violating backend ML invariants.
- **Alternatives Considered**:
  - *Full-stack REST API + Single-Page React App*: Overkill for research demonstration; requires CORS management, multi-process orchestration, and substantial maintenance overhead.
  - *CLI-only inference*: High developer utility, but lacks visual appeal for non-technical stakeholders and recruitment demonstrations.
  - *Decoupled Streamlit Single-Page App*: High velocity, pure Python, native component rendering, and zero frontend boilerplate.
- **Chosen Approach**: Built `app/streamlit_app.py` consuming `ResumeJobMatcher` and `matcher.explain()`. Structured UI into clear visual hierarchy:
  1. Input Section: In-memory PDF upload and multi-line job description text area with explicit validation against short inputs (<3 words) and corrupted files.
  2. Prediction Card: Color-coded classification banner (Good Fit, Potential Fit, No Fit) with strictly labeled **"Model Confidence"**.
  3. Calibrated Probabilities: Side-by-side horizontal progress bars for all three classes derived from Temperature Scaling ($T = 1.4173$).
  4. Match Explanation: Non-causal descriptive overview of profile alignment.
  5. Skill Analysis: 4-metric grid (*Job Skills*, *Resume Skills*, *Matched Skills*, *Potential Skill Gaps*), accurate coverage semantics (`None` when 0 job skills detected, `0.0%` when skills exist but none matched), and styled badge chips.
  6. Supporting Signals: Upfront primary indicators with a 10-feature audit table tucked in an expander.
  7. Technical Audit: Collapsed expander showing Calibrated vs Raw Probabilities and raw metadata.
- **Measured Invariance Results**:
  - Programmatic verification via `test_explainability_invariance_guarantee` confirmed $\Delta \text{prediction} = 0$, $\Delta \text{calibrated\_probabilities} = 0.0$, and $\Delta \text{raw\_probabilities} = 0.0$.
  - 17 dedicated Streamlit integration and UI tests passed deterministically.
- **Consequences**: Delivers a demonstration-ready application for technical reviews and academic presentations while preserving complete separation between ML inference and UI presentation.

---

## Decision 017: Final Engineering Status & Conclusion of Model Experimentation

- **Decision**: Formally conclude all machine learning model experimentation, lock the production inference pipeline (Model F + Temperature Scaling $T=1.4173$ + PDF Ingestion + Explainability Layer), and transition the repository into a verified, documentation-complete, reproducibility-tested demonstration prototype.
- **Context**: Following extensive baseline, supervised, non-linear, length-robustness, ensemble, and calibration evaluations across Phases 1 through 11C, the modeling architecture has reached Pareto-optimal performance across both candidate-level and out-of-domain job benchmarks. Model exploration must be formally halted to prevent scope creep, unprincipled hyperparameter searching, or unnecessary complex frameworks.
- **Locked Production Stack**:
  - *Primary Classifier*: Model F (Ratio-Focused LightGBM, 18 features, balanced class weights, zero absolute job description counts).
  - *Post-Hoc Calibrator*: Temperature Scaling ($T = 1.4173$, fitted strictly on `candidate_validation`, ECE reduced by 53.9% on unseen jobs, 0 predictions flipped).
  - *Document Ingestion*: `src/inference/pdf_extractor.py` (pure-Python `pypdf`, in-memory, Unicode normalized, technical symbol preserving).
  - *Diagnostic Presentation*: `src/inference/explainability.py` (decoupled, non-destructive, non-causal `ExplainabilityResult`).
  - *Demonstration Application*: `app/streamlit_app.py` (Streamlit-based executive dashboard).
  - *Test Suite*: 67 automated regression, pipeline, and UI tests passing (`pytest -q`).
- **Forbidden Actions**:
  - No new model training, feature changes, or ensemble blending.
  - No LLM / LangChain / RAG / Vector DB wrappers.
  - No OCR dependencies or cloud container overhead.
- **Consequences**: The codebase is locked for evaluation and deployment readiness. Future work is restricted to real-world deployment containerization, domain taxonomy expansions, or user testing as documented in `README.md`.
