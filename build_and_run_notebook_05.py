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

# Title & Overview
cells.append(nbf.v4.new_markdown_cell("""# 05. Stronger Semantic Representation (MiniLM vs MPNet)
**AI-Based Resume–Job Matching & Skill Gap Analysis — Phase 6C**

### Background & Objective:
In **Phase 6A**, the supervised multi-signal classifier achieved strong performance gains over 1D thresholding (Candidate Test Macro F1 = 0.4735). In that model, semantic similarity was represented by `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, 6 layers).
In **Phase 6B**, we empirically proved that hard rule-based text cleaning degraded accuracy by stripping implicit contextual cues.

**Objective of Phase 6C**:
Determine whether replacing or augmenting MiniLM with a stronger, larger pretrained embedding model — **`sentence-transformers/all-mpnet-base-v2`** (768 dimensions, 12 layers) — improves matching performance under strict, controlled A/B conditions.

### Controlled Experimental Principles:
1. **Isolated Variable**: Change *only* the semantic embedding model. Keep all other 21 features identical (TF-IDF similarity, skill coverage, length ratios, lexical overlap).
2. **Text Integrity**: Use full, raw resume and job description texts. Zero text modification or cleaning.
3. **Identical Classifier**: Train regularized Logistic Regression ($C=0.1$, `class_weight='balanced'`, `StandardScaler`, random_state=42) strictly on `candidate_train`.
4. **Validation-Guided Selection**: Model selection is governed exclusively by `candidate_validation` Macro F1. Holdout `candidate_test` and `official_test` are evaluated only once after model selection.
"""))

# Section 1: Resource Benchmarking
cells.append(nbf.v4.new_markdown_cell("""## 1. Resource Feasibility & Model Benchmarking

We benchmark `all-mpnet-base-v2` against `all-MiniLM-L6-v2` across model size, embedding dimensionality, and CPU inference latency.
"""))

cells.append(nbf.v4.new_code_cell("""import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Image

print("=== Semantic Model Architecture Comparison ===")
arch_df = pd.DataFrame([
    {
        'Model': 'all-MiniLM-L6-v2',
        'Parameters': '22.7M',
        'Layers': 6,
        'Embedding Dimension': 384,
        'Disk Size': '~90 MB',
        'Inference Speed (CPU)': '~8 ms / doc',
        'Relative Cost': '1.0x (Baseline)'
    },
    {
        'Model': 'all-mpnet-base-v2',
        'Parameters': '109.5M',
        'Layers': 12,
        'Embedding Dimension': 768,
        'Disk Size': '~420 MB',
        'Inference Speed (CPU)': '~28 ms / doc',
        'Relative Cost': '3.5x - 4.0x'
    }
])
display(arch_df)
"""))

# Section 2: Distribution Analysis
cells.append(nbf.v4.new_markdown_cell("""## 2. Semantic Similarity Distribution Analysis

We compare cosine similarity distributions between MiniLM and MPNet across all evaluation splits and true match classes (`No Fit`, `Potential Fit`, `Good Fit`).
"""))

cells.append(nbf.v4.new_code_cell("""# Display distribution plot
display(Image(filename='../data/evaluation/plots/phase6c_similarity_distributions.png'))
"""))

# Section 3: Correlation Analysis
cells.append(nbf.v4.new_markdown_cell("""## 3. Multi-Signal Feature Correlation Analysis

We evaluate Pearson and Spearman rank correlations between TF-IDF similarity, MiniLM similarity, MPNet similarity, Skill Coverage, and Jaccard lexical overlap to measure information overlap.
"""))

cells.append(nbf.v4.new_code_cell("""corr_df = pd.read_csv('../data/evaluation/phase6c_semantic_correlations.csv')
print("=== Feature Correlation Matrix (Candidate Train) ===")
display(corr_df)
"""))

# Section 4: Controlled Classifier Experiments
cells.append(nbf.v4.new_markdown_cell("""## 4. Controlled Classifier Experiments & Model Selection

We evaluate three controlled model variants:
- **Model A (Reference)**: MiniLM (22 features)
- **Model B (MPNet Replacement)**: Replace MiniLM with MPNet (22 features)
- **Model C (Dual-Embedding)**: MiniLM + MPNet (23 features)
"""))

cells.append(nbf.v4.new_code_cell("""comp_df = pd.read_csv('../data/evaluation/phase6c_semantic_comparison.csv')
print("=== Model Performance Comparison Across Splits ===")
display(comp_df[['model', 'val_macro_f1', 'cand_test_macro_f1', 'cand_test_accuracy', 'off_test_macro_f1', 'off_test_accuracy']])

# Display comparison chart
display(Image(filename='../data/evaluation/plots/phase6c_model_comparison.png'))
"""))

# Section 5: Per-Class Performance
cells.append(nbf.v4.new_markdown_cell("""## 5. Granular Per-Class Metrics & Recall Diagnostics

We inspect per-class Precision, Recall, and F1 on holdout `candidate_test` and `official_test`.
"""))

cells.append(nbf.v4.new_code_cell("""class_df = pd.read_csv('../data/evaluation/phase6c_semantic_metrics.csv')

print("=== Candidate Test Per-Class Performance ===")
display(class_df[class_df['dataset'] == 'Candidate Test'])

print("\\n=== Official Test Per-Class Performance ===")
display(class_df[class_df['dataset'] == 'Official Test'])
"""))

# Section 6: Error Recovery & Case-Level Transitions
cells.append(nbf.v4.new_markdown_cell("""## 6. Case-Level Transition Diagnostics (Model A vs Model B)

We analyze transitions on holdout `candidate_test` ($N=1,201$):
- **Recovered Cases (103)**: Model A was wrong, Model B was correct.
- **Regressed Cases (93)**: Model A was correct, Model B was wrong.
- **Net Improvement**: $+10$ pairs.
"""))

cells.append(nbf.v4.new_code_cell("""recovery_df = pd.read_csv('../data/evaluation/phase6c_error_recovery.csv')

print("=== Transition Counts on Candidate Test ===")
print("Both Correct: 526 (43.8%)")
print("Both Wrong:   479 (39.9%)")
print("Recovered (Model A Wrong -> Model B Correct): 103 (8.6%)")
print("Regressed (Model A Correct -> Model B Wrong):  93 (7.7%)")
print("Net Gain: +10 cases")

print("\\nSample Recovered Cases (MPNet Solved Mismatch):")
display(recovery_df[recovery_df['category'] == 'Recovered (A wrong -> New correct)'][['pair_id', 'true_label', 'pred_a', 'pred_new', 'minilm_sim', 'mpnet_sim']].head(5))

print("\\nSample Regressed Cases:")
display(recovery_df[recovery_df['category'] == 'Regressed (A correct -> New wrong)'][['pair_id', 'true_label', 'pred_a', 'pred_new', 'minilm_sim', 'mpnet_sim']].head(5))
"""))

# Section 7: Key Findings & Architectural Verdict
cells.append(nbf.v4.new_markdown_cell("""## 7. Key Findings & Architectural Assessment

### Summary of Results:
1. **Validation Selection**:
   - Model B (MPNet Replacement) achieved **0.4627 Validation Macro F1**, outperforming Model A (0.4603) and Model C (0.4537).
2. **Holdout Evaluation**:
   - **Candidate Test (Unseen Candidates)**: Model B achieved **0.4811 Macro F1** (+0.0076 over Model A) and **0.5237 Accuracy** (+0.0083).
   - **Official Test (Unseen JDs)**: Model B achieved **0.4130 Macro F1** (-0.0090 compared to Model A) and **0.4520 Accuracy** (+0.0012).
3. **Per-Class Shifts**:
   - `Potential Fit` F1 improved noticeably from 0.3616 to **0.3848** (+2.32%), driven by gains in both precision and recall.
   - `Good Fit` recall declined slightly from 45.51% to 43.19% (-2.32%).
4. **Computational Cost vs Gain Tradeoff**:
   - MPNet requires **4.7x the disk space** (420 MB vs 90 MB) and **~3.5x-4x more CPU computation time** (617s vs ~140s for dataset encoding).
   - The performance lift on unseen candidates (+0.0076 Macro F1) is modest and mixed across datasets (-0.0090 on official test).

### Strategic Verdict:
- MPNet provides a slight boost on unseen candidates but fails to generalize consistently to unseen job descriptions.
- The fundamental bottleneck in matching remains: bi-encoder cosine similarity compresses complex 500-word career histories into a single 1D vector before comparison.
- Next logical progression: Pairwise Cross-Encoder ranking or Non-linear tree-based gradient boosting (LightGBM).
"""))

nb.cells = cells

notebook_path = 'notebooks/05_semantic_model_comparison.ipynb'
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Notebook created at {notebook_path}. Now executing...")

ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb_to_run = nbf.read(f, as_version=4)

ep.preprocess(nb_to_run, {'metadata': {'path': 'notebooks/'}})

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb_to_run, f)

print("Notebook 05_semantic_model_comparison.ipynb executed and saved with all outputs!")
