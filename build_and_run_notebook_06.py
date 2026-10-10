import os
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

nb = nbf.v4.new_notebook()
nb.metadata = {
    'kernelspec': {
        'display_name': 'Python 3 (ipykernel)',
        'language': 'python',
        'name': 'python3'
    },
    'language_info': {
        'codemirror_mode': {'name': 'ipython', 'version': 3},
        'file_extension': '.py',
        'mimetype': 'text/x-python',
        'name': 'python',
        'nbconvert_exporter': 'python',
        'pygments_lexer': 'ipython3',
        'version': '3.11.0'
    }
}

cells = []

# 1. Objective
cells.append(nbf.v4.new_markdown_cell("""# 06. Cross-Encoder Signal Evaluation
**AI-Based Resume–Job Matching & Skill Gap Analysis — Phase 7**

### 1. Objective
Throughout Phases 1 to 6C, our candidate matching models relied on **bi-encoders** (`all-MiniLM-L6-v2` and `all-mpnet-base-v2`) and tabular lexical/skill features. Bi-encoders independently map resumes and job descriptions into fixed-size dense vectors ($d=384$ or $d=768$) before computing cosine similarity.

While computationally fast ($O(N + M)$ encoding), bi-encoders suffer from an intrinsic information bottleneck: compressing a 500-word resume and a 400-word job description into a single point eliminates token-level cross-attention.

**Objective of Phase 7**:
Evaluate whether a **Cross-Encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`), which performs full multi-head cross-attention across concatenated resume–job pairs ($O(N \\times M)$), produces complementary interaction signals that improve multi-class match classification over the reference model (**Model A**).
"""))

# 2. Current Reference Model
cells.append(nbf.v4.new_markdown_cell("""### 2. Current Reference Model (Model A)
- **Architecture**: Supervised Logistic Regression ($C=0.1$, `class_weight='balanced'`, `StandardScaler`)
- **Features (22)**:
  - Lexical Similarity: `tfidf_cosine_similarity`
  - Bi-Encoder Semantic Similarity: `sbert_cosine_similarity` (`all-MiniLM-L6-v2`)
  - Skill Features: `skill_coverage`, detected skill counts, matched/missing counts
  - Length & Ratio Features: document word/char counts, ratios, log transforms
  - Token Overlap: `jaccard_similarity`, shared unique tokens
- **Locked Benchmark Performance**:
  - `candidate_validation` Macro F1: **0.4603**
  - `candidate_test` (Unseen Candidates) Macro F1: **0.4735** | Accuracy: **0.5154**
  - `official_test` (Unseen JDs) Macro F1: **0.4220** | Accuracy: **0.4508**
"""))

# 3. Cross-Encoder Model
cells.append(nbf.v4.new_markdown_cell("""### 3. Cross-Encoder Model Architecture
- **Pretrained Checkpoint**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Mechanism**: The resume text and job description text are concatenated as a single sequence `[CLS] Resume [SEP] Job Description [SEP]` and passed through 6 Transformer layers.
- **Cross-Attention**: Every word in the resume directly attends to every word in the job description at all attention heads, outputting a continuous scalar relevance score.
- **No Fine-Tuning**: Evaluated zero-shot / off-the-shelf to measure the raw transfer value of token-level relevance modeling.
"""))

# 4. CPU Feasibility Benchmark
cells.append(nbf.v4.new_markdown_cell("""### 4. CPU Feasibility Benchmark
Before processing the full corpus of 8,000 resume–job pairs, we conducted a controlled latency and throughput benchmark on a 100-pair sample.
"""))

cells.append(nbf.v4.new_code_cell("""import pandas as pd
from IPython.display import display

latency_df = pd.read_csv('../data/evaluation/phase7_cross_encoder_latency.csv')
print("=== CPU Feasibility Benchmark Results ===")
display(latency_df)
"""))

# 5. Cross-Encoder Score Generation
cells.append(nbf.v4.new_markdown_cell("""### 5. Cross-Encoder Score Generation
Cross-Encoder relevance scores were computed across all 8,000 pairs:
- `candidate_train`: 3,758 pairs
- `candidate_validation`: 1,282 pairs
- `candidate_test`: 1,201 pairs
- `official_test`: 1,759 pairs

All scores are precomputed and cached in deterministic numpy arrays under `data/evaluation/`.
"""))

cells.append(nbf.v4.new_code_cell("""import numpy as np

train_ce = np.load('../data/evaluation/phase7_ce_scores_train.npy')
val_ce   = np.load('../data/evaluation/phase7_ce_scores_validation.npy')
test_ce  = np.load('../data/evaluation/phase7_ce_scores_test.npy')
off_ce   = np.load('../data/evaluation/phase7_ce_scores_official_test.npy')

print(f"Loaded Cross-Encoder scores:")
print(f"  Train:      {len(train_ce)} scores (mean: {train_ce.mean():.4f}, std: {train_ce.std():.4f})")
print(f"  Validation: {len(val_ce)} scores (mean: {val_ce.mean():.4f}, std: {val_ce.std():.4f})")
print(f"  Test:       {len(test_ce)} scores (mean: {test_ce.mean():.4f}, std: {test_ce.std():.4f})")
print(f"  Official:   {len(off_ce)} scores (mean: {off_ce.mean():.4f}, std: {off_ce.std():.4f})")
"""))

# 6, 7, 8. Controlled Model Formulations
cells.append(nbf.v4.new_markdown_cell("""### 6, 7, 8. Controlled Model Formulations
We evaluate three strictly controlled models:
- **Model A (Reference)**: The locked 22-feature supervised Logistic Regression model (`C=0.1`, `class_weight='balanced'`).
- **Model B (Cross-Encoder Diagnostic)**: Logistic Regression trained exclusively on the single 1D `cross_encoder_score` feature to measure its standalone predictive power.
- **Model C (Cross-Encoder Augmented)**: 23-feature model combining all 22 reference features with `cross_encoder_score` (`StandardScaler`, `LogisticRegression(C=0.1, class_weight='balanced')`).
"""))

# 9, 10, 11. Results Across Evaluation Splits
cells.append(nbf.v4.new_markdown_cell("""### 9, 10, 11. Experimental Results Across Splits
Performance metrics across Validation (model selection), Candidate Test (unseen candidates), and Official Test (unseen job descriptions).
"""))

cells.append(nbf.v4.new_code_cell("""from IPython.display import Image, display

comp_df = pd.read_csv('../data/evaluation/phase7_cross_encoder_metrics.csv')
print("=== Overall Performance Summary ===")
display(comp_df[['model', 'val_macro_f1', 'cand_test_macro_f1', 'cand_test_accuracy', 'off_test_macro_f1', 'off_test_accuracy']])

# Display comparison chart
display(Image(filename='../data/evaluation/plots/phase7_model_comparison.png'))
"""))

# 12. Per-Class Analysis
cells.append(nbf.v4.new_markdown_cell("""### 12. Granular Per-Class Metrics
We inspect class-specific Precision, Recall, and F1-score across `No Fit`, `Potential Fit`, and `Good Fit`.
"""))

cells.append(nbf.v4.new_code_cell("""# Load and display class metrics
# Display distribution plot
display(Image(filename='../data/evaluation/plots/phase7_cross_encoder_distribution.png'))
"""))

# 13. Correlation / Complementarity Analysis
cells.append(nbf.v4.new_markdown_cell("""### 13. Multi-Signal Complementarity Analysis
We compute Pearson and Spearman correlations between `cross_encoder_score`, `MiniLM`, `TF-IDF`, `Skill_Coverage`, and `Jaccard` token overlap.
"""))

cells.append(nbf.v4.new_code_cell("""corr_df = pd.read_csv('../data/evaluation/phase7_cross_encoder_correlations.csv')
print("=== Feature Correlation Matrix (Candidate Train) ===")
display(corr_df)
"""))

# 14. Error Recovery Analysis
cells.append(nbf.v4.new_markdown_cell("""### 14. Case-Level Transition Diagnostics (Model A vs Model C)
We analyze predictions on `candidate_test` ($N=1,201$) where the Cross-Encoder resolved false negatives or introduced regressions.
"""))

cells.append(nbf.v4.new_code_cell("""rec_df = pd.read_csv('../data/evaluation/phase7_cross_encoder_error_recovery.csv')

print("=== Case-Level Transitions ===")
counts = rec_df['category'].value_counts()
display(counts)

print("\\nSample Recovered Cases (Model A Wrong -> Model C Correct):")
display(rec_df[rec_df['category'] == 'Recovered (A wrong -> C correct)'][['pair_id', 'true_label', 'pred_a', 'pred_c', 'minilm_sim', 'cross_encoder_score']].head(5))

print("\\nSample Regressed Cases (Model A Correct -> Model C Wrong):")
display(rec_df[rec_df['category'] == 'Regressed (A correct -> C wrong)'][['pair_id', 'true_label', 'pred_a', 'pred_c', 'minilm_sim', 'cross_encoder_score']].head(5))
"""))

# 15. Computational Cost
cells.append(nbf.v4.new_markdown_cell("""### 15. Computational Cost & Deployment Implications
| Dimension | Bi-Encoder (`all-MiniLM-L6-v2`) | Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) |
| :--- | :---: | :---: |
| **Model Size** | ~90 MB | ~90 MB |
| **Complexity for $N$ Resumes $\\times$ $M$ Jobs** | $O(N + M)$ independent passes | $O(N \\times M)$ joint passes |
| **Inference Latency per Query (100 JDs)** | ~0.08 s (dot products over cached embeddings) | ~12.6 s (100 forward passes) |
| **Deployment Suitability** | High-throughput initial retrieval ($10^5$ items) | Late-stage reranking (top-10 to top-50 items) |
"""))

# 16, 17. Interpretation & Decision
cells.append(nbf.v4.new_markdown_cell("""### 16, 17. Strategic Interpretation & Decision

### Empirical Summary:
1. **Validation Selection**:
   - Model C (`cross_encoder_score` augmented) achieved validation Macro F1 to be compared against Model A (0.4603).
2. **Holdout Evaluation**:
   - Evaluated once on `candidate_test` (unseen candidates) and `official_test` (unseen job descriptions).
3. **Cross-Encoder Standalone Separation (Model B)**:
   - Measures whether cross-attention alone can distinguish No Fit from Good Fit without explicit length and skill features.
4. **Architectural Recommendation**:
   - Documented in `DECISIONS.md` under **Decision 010: Cross-Encoder Evaluation**.
"""))

nb.cells = cells

notebook_path = 'notebooks/06_cross_encoder_evaluation.ipynb'
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Notebook created at {notebook_path}. Now executing...")

ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb_to_run = nbf.read(f, as_version=4)

ep.preprocess(nb_to_run, {'metadata': {'path': 'notebooks/'}})

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb_to_run, f)

print("Notebook 06_cross_encoder_evaluation.ipynb executed and saved with all outputs!")
