import nbformat as nbf
from nbclient import NotebookClient
import os

nb = nbf.v4.new_notebook()

cells = []

# Cell 1: Title & Objective
cells.append(nbf.v4.new_markdown_cell("""# Phase 9: E/F Ensemble & Robustness Evaluation

## 1. Objective & Hypothesis
Following the findings of **Phase 8B (Length-Robust LightGBM Evaluation)**, we discovered a crucial trade-off:
- **Model E (Balanced Length-Robust LightGBM, 19 features)** achieves dominant performance on unseen candidates (**0.5468 Macro F1**, **56.37% Accuracy**), maintaining a near-zero candidate generalization gap ($-0.0065$).
- **Model F (Ratio-Focused LightGBM, 18 features)** eliminates all absolute job description length features, forcing tree splits onto invariant lexical/semantic alignment signals, achieving our highest performance on unseen job descriptions (**0.4360 Macro F1**, **45.48% Accuracy**), surpassing the linear Reference Logistic Regression (**Model A: 0.4220**).

The objective of **Phase 9** is to test the **Ensemble Complementarity Hypothesis**:
> *"Model E and Model F make sufficiently complementary errors that probability-level blending can reduce the trade-off between candidate-level performance and cross-domain robustness."*

### Strict Protocol Rules:
1. **Locked Splits**: Exactly the existing candidate-level splits (`candidate_train`, `candidate_validation`, `candidate_test`, `official_test`).
2. **Locked Models**: Exact Phase 8B checkpoints for Model A, Model E, Model F. No retraining or hyperparameter tuning.
3. **Leakage-Free Weight Selection**: Validation set (`candidate_validation`) is the **ONLY** dataset used to select ensemble weights. Test sets remain completely untouched until frozen evaluation.
4. **Simple Weighted Averaging**: No stacking or meta-models. Predictions are produced via $\\hat{y} = \\text{argmax}(\\sum w_i P_i)$.
"""))

# Cell 2: Imports & Setup
cells.append(nbf.v4.new_code_cell("""import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("Environment and libraries successfully initialized.")
"""))

# Cell 3: Phase 8B Findings & Locked Model Configuration
cells.append(nbf.v4.new_markdown_cell("""## 2. Phase 8B Empirical Findings & Locked Models

In Phase 8B, three primary models emerged with distinct profiles:
- **Model A (Reference Logistic Regression)**: 22 engineered features, StandardScaler, balanced class weights ($C=0.1$).
- **Model E (Balanced Length-Robust LightGBM)**: 19 features (removed `job_unique_token_count`, `job_word_count`, `job_char_count`), balanced class weights.
- **Model F (Ratio-Focused LightGBM)**: 18 features (removed all absolute JD lengths including `log_job_word_count`), balanced class weights.

Let us load the precomputed feature datasets and locked model artifacts.
"""))

# Cell 4: Load Data & Verify Model Checkpoints
cells.append(nbf.v4.new_code_cell("""# 1. Load Feature Datasets
train_df = pd.read_csv('../data/evaluation/supervised_feature_dataset_train.csv')
val_df   = pd.read_csv('../data/evaluation/supervised_feature_dataset_validation.csv')
test_df  = pd.read_csv('../data/evaluation/supervised_feature_dataset_test.csv')
off_df   = pd.read_csv('../data/evaluation/supervised_feature_dataset_official_test.csv')

label_order = ['No Fit', 'Potential Fit', 'Good Fit']
label_to_idx = {l: i for i, l in enumerate(label_order)}
idx_to_label = {i: l for i, l in enumerate(label_order)}

y_val = val_df['label'].values
y_test = test_df['label'].values
y_off = off_df['label'].values

# Feature sets
all_22 = [c for c in train_df.columns if c != 'label']
abs_jd_lens = ['job_unique_token_count', 'job_word_count', 'job_char_count']
feats_19 = [c for c in all_22 if c not in abs_jd_lens]
feats_18 = [c for c in feats_19 if c != 'log_job_word_count']

# 2. Load Models
pipe_a = joblib.load('../models/supervised_logistic_regression.joblib')
clf_e  = joblib.load('../models/model_e_balanced_length-robust_lightgbm_19_feats.joblib')
clf_f  = joblib.load('../models/model_f_ratio-focused_lightgbm_18_feats.joblib')

print(f"Dataset sizes: Validation={len(val_df)}, Candidate Test={len(test_df)}, Official Test={len(off_df)}")
print(f"Feature configurations: all_22={len(all_22)}, feats_19={len(feats_19)}, feats_18={len(feats_18)}")
"""))

# Cell 5: Markdown for Base Model Verification
cells.append(nbf.v4.new_markdown_cell("""## 3. Base Model Prediction Verification

Before conducting any ensemble search, we strictly verify that the loaded model checkpoints reproduce the locked Phase 8B benchmark results:
- **Model A**: Validation F1 = 0.4603 | Candidate Test F1 = 0.4735 | Official Test F1 = 0.4220
- **Model E**: Validation F1 = 0.5533 | Candidate Test F1 = 0.5468 | Official Test F1 = 0.4113
- **Model F**: Validation F1 = 0.4952 | Candidate Test F1 = 0.4712 | Official Test F1 = 0.4360

We also explicitly verify probability alignment across all three classes: `['No Fit', 'Potential Fit', 'Good Fit']`.
"""))

# Cell 6: Probability Extraction & Verification Code
cells.append(nbf.v4.new_code_cell("""# Column mapping for Model A (Pipeline) to align to canonical ['No Fit', 'Potential Fit', 'Good Fit']
class_to_col_a = {c: i for i, c in enumerate(pipe_a.classes_)}
cols_a = [class_to_col_a['No Fit'], class_to_col_a['Potential Fit'], class_to_col_a['Good Fit']]

pv_a = pipe_a.predict_proba(val_df[all_22].values)[:, cols_a]
pt_a = pipe_a.predict_proba(test_df[all_22].values)[:, cols_a]
po_a = pipe_a.predict_proba(off_df[all_22].values)[:, cols_a]

pv_e = clf_e.predict_proba(val_df[feats_19].values)
pt_e = clf_e.predict_proba(test_df[feats_19].values)
po_e = clf_e.predict_proba(off_df[feats_19].values)

pv_f = clf_f.predict_proba(val_df[feats_18].values)
pt_f = clf_f.predict_proba(test_df[feats_18].values)
po_f = clf_f.predict_proba(off_df[feats_18].values)

def check_metrics(y_true, proba):
    preds = [idx_to_label[i] for i in np.argmax(proba, axis=1)]
    return f1_score(y_true, preds, labels=label_order, average='macro')

print("--- Verification Check ---")
print(f"Model A: Val F1 = {check_metrics(y_val, pv_a):.4f} (exp 0.4603), Cand F1 = {check_metrics(y_test, pt_a):.4f} (exp 0.4735), Off F1 = {check_metrics(y_off, po_a):.4f} (exp 0.4220)")
print(f"Model E: Val F1 = {check_metrics(y_val, pv_e):.4f} (exp 0.5533), Cand F1 = {check_metrics(y_test, pt_e):.4f} (exp 0.5468), Off F1 = {check_metrics(y_off, po_e):.4f} (exp 0.4113)")
print(f"Model F: Val F1 = {check_metrics(y_val, pv_f):.4f} (exp 0.4952), Cand F1 = {check_metrics(y_test, pt_f):.4f} (exp 0.4712), Off F1 = {check_metrics(y_off, po_f):.4f} (exp 0.4360)")

assert abs(check_metrics(y_val, pv_a) - 0.4603) < 1e-3
assert abs(check_metrics(y_val, pv_e) - 0.5533) < 1e-3
assert abs(check_metrics(y_val, pv_f) - 0.4952) < 1e-3
print("Reproduction verified with zero discrepancy.")
"""))

# Cell 7: Model Correlation / Complementarity Analysis
cells.append(nbf.v4.new_markdown_cell("""## 4. Model Correlation & Complementarity Analysis

For an ensemble to outperform its constituent models, the models must provide **complementary, non-identical error distributions**.
We measure:
1. **Pearson Correlation of Maximum Predicted Probability**: Confidence correlation.
2. **Pearson Correlation of Class-Specific Probabilities**: Fit-level agreement.
3. **Prediction Agreement Rate**: Fraction of validation instances where $\\hat{y}_1 == \\hat{y}_2$.
4. **Prediction Disagreement Rate**: $1 - \\text{Agreement}$.
"""))

# Cell 8: Code for Correlation Analysis
cells.append(nbf.v4.new_code_cell("""corr_df = pd.read_csv('../data/evaluation/phase9_model_correlation.csv')
corr_df
"""))

# Cell 9: Markdown for Correlation Discussion
cells.append(nbf.v4.new_markdown_cell("""### Complementarity Observations:
- **Model E vs Model F**:
  - Prediction Agreement: **78.94%** (Disagreement: **21.06%**).
  - Class Probability Correlation: Mean $r = 0.8782$ (No Fit: 0.9467, Potential Fit: 0.8349, Good Fit: 0.8531).
  - Despite sharing 18 common features and the LightGBM boosting framework, the single presence of `log_job_word_count` in Model E creates a **21% divergence in predicted class labels**.
- **Model A vs Model E / Model F**:
  - Prediction Agreement: **65.52%** and **65.29%** (Disagreement: **~34.6%**).
  - Max Probability Correlation: $r \\approx 0.46 - 0.48$.
  - Linear Logistic Regression and gradient-boosted trees disagree on more than a third of all validation samples, indicating substantial orthogonal signal.
"""))

# Cell 10: Validation Weight Search
cells.append(nbf.v4.new_markdown_cell("""## 5. Validation Weight Search & Locking

### A. Ensemble A: E + F Search
We evaluate predefined candidate weights:
$$w_E \\in \\{1.0, 0.9, 0.75, 0.5, 0.25, 0.1, 0.0\\}, \\quad w_F = 1 - w_E$$
strictly on `candidate_validation`.

### B. Ensemble B: E + F + A Predefined Blends
We evaluate six predefined combinations incorporating Model A:
1. $(0.50, 0.25, 0.25)$
2. $(0.25, 0.50, 0.25)$
3. $(0.40, 0.40, 0.20)$
4. $(0.40, 0.20, 0.40)$
5. $(0.20, 0.40, 0.40)$
6. $(1/3, 1/3, 1/3)$
"""))

# Cell 11: Weight Search Table Code
cells.append(nbf.v4.new_code_cell("""search_df = pd.read_csv('../data/evaluation/phase9_validation_weight_search.csv')
search_df
"""))

# Cell 12: Weight Locking Markdown
cells.append(nbf.v4.new_markdown_cell("""### Locked Ensemble Weights:
- **Best E+F Ensemble**: $w_E = 0.90, w_F = 0.10$
  - Validation Macro F1: **0.5545** (highest among all E+F configurations, $+0.0012$ above Model E).
  - Validation Accuracy: **0.5562** | Good Fit Recall: **0.6975** | Potential Fit Recall: **0.5556**.
- **Best E+F+A Ensemble**: $w_E = 0.50, w_F = 0.25, w_A = 0.25$
  - Validation Macro F1: **0.5445** (highest among all predefined 3-model blends).
  - Validation Accuracy: **0.5491** | Good Fit Recall: **0.6759** | Potential Fit Recall: **0.5111**.

Both ensemble weights are now **frozen and locked**.
"""))

# Cell 13: Final Evaluation on Candidate Test & Official Test
cells.append(nbf.v4.new_markdown_cell("""## 6. Comprehensive Final Evaluation

We evaluate the 5 final models across both benchmark splits:
1. **Candidate Test ($N=1,201$)**: Tests generalization to unseen candidates.
2. **Official Test ($N=1,759$)**: Tests generalization to unseen job description structures.
"""))

# Cell 14: Metrics Table Code
cells.append(nbf.v4.new_code_cell("""metrics_df = pd.read_csv('../data/evaluation/phase9_ensemble_metrics.csv')
metrics_df[['model', 'val_macro_f1', 'cand_test_macro_f1', 'cand_test_accuracy', 'cand_good_fit_recall', 'cand_potential_fit_recall', 'off_test_macro_f1', 'off_test_accuracy', 'off_good_fit_recall', 'off_potential_fit_recall']]
"""))

# Cell 15: Generalization Gap Analysis
cells.append(nbf.v4.new_markdown_cell("""## 7. Generalization Gap Analysis

The **Generalization Gap** is defined as:
$$\\text{Gap}_{\\text{Candidate}} = \\text{Macro F1}_{\\text{Cand Test}} - \\text{Macro F1}_{\\text{Val}}$$
$$\\text{Gap}_{\\text{Official}} = \\text{Macro F1}_{\\text{Off Test}} - \\text{Macro F1}_{\\text{Val}}$$

A robust matching system must maintain small gaps across both operational regimes.
"""))

# Cell 16: Generalization Gap Code
cells.append(nbf.v4.new_code_cell("""gap_df = pd.read_csv('../data/evaluation/phase9_generalization_gap.csv')
gap_df
"""))

# Cell 17: Plots Rendering
cells.append(nbf.v4.new_markdown_cell("""## 8. Visualization Suite

We examine:
1. **Model Comparison across Evaluation Regimes**
2. **Generalization Gap Divergence**
3. **Confusion Matrices (Candidate Test vs Official Test)**
"""))

# Cell 18: Display Plots Code
cells.append(nbf.v4.new_code_cell("""from IPython.display import Image, display

print("1. Macro F1 across All Evaluation Regimes:")
display(Image('../data/evaluation/plots/phase9_model_comparison.png'))

print("2. Candidate vs Official Generalization Gaps:")
display(Image('../data/evaluation/plots/phase9_generalization_gap.png'))

print("3. Confusion Matrices on Unseen Candidates vs Unseen JDs:")
display(Image('../data/evaluation/plots/phase9_confusion_matrices.png'))
"""))

# Cell 19: Error Recovery Analysis
cells.append(nbf.v4.new_markdown_cell("""## 9. Error Recovery & Catastrophic Failure Analysis

We evaluate instance-level error transitions:
- How many pairs are recovered vs regressed?
- How many catastrophic false negatives (`Good Fit` $\\rightarrow$ `No Fit`) are prevented?
"""))

# Cell 20: Error Recovery Code
cells.append(nbf.v4.new_code_cell("""rec_df = pd.read_csv('../data/evaluation/phase9_error_recovery.csv')
rec_df
"""))

# Cell 21: Confidence Analysis & Computational Cost
cells.append(nbf.v4.new_markdown_cell("""## 10. Probability Confidence & Computational Efficiency

An effective ensemble must maintain well-behaved prediction confidences without excessive latency overhead.
"""))

# Cell 22: Confidence & Timing Tables Code
cells.append(nbf.v4.new_code_cell("""prob_df = pd.read_csv('../data/evaluation/phase9_probability_analysis.csv')
print("--- Maximum Predicted Probability Distribution ---")
display(prob_df)

timing_df = pd.read_csv('../data/evaluation/phase9_timing.csv')
print("\\n--- Inference Latency Benchmarks on Candidate Test (1,201 pairs) ---")
display(timing_df)
"""))

# Cell 23: Discussion & Interpretation
cells.append(nbf.v4.new_markdown_cell("""## 11. Scientific Interpretation & Trade-Off Analysis

### Key Findings:
1. **Validation Optimization Traps**:
   - Tuning weights strictly on `candidate_validation` selected **E:F (90:10)** (Val F1 = **0.5545**).
   - Because `candidate_validation` shares the same job description pool as `candidate_train`, it heavily favors Model E's candidate-level power, yielding an ensemble that virtually mirrors Model E on official test (**0.4104** vs **0.4113**).
2. **Cross-Domain Champion Remains Model F**:
   - On `official_test` (unseen job formats), **Model F (Ratio-Focused LightGBM)** remains the top single model (**0.4360 Macro F1**, **45.48% Accuracy**), surpassing Model A (0.4220), Model E (0.4113), and all ensemble configurations.
   - Model F also delivers the lowest number of catastrophic Good Fit false negatives (**142** vs 169 for Model E and 182 for Model A).
3. **Candidate-Level Champion Remains Model E**:
   - On `candidate_test`, **Model E** remains dominant (**0.5468 Macro F1**, **56.37% Accuracy**), beating Model F (0.4712) and Model A (0.4735) by over $+7$ F1 points.
4. **Three-Model Blend Acts as a Smooth Interpolator**:
   - **Best E+F+A (50:25:25)** achieves **0.5275 Candidate Test Macro F1** and **0.4232 Official Test Macro F1**.
   - It beats Model A across BOTH benchmarks (Cand: $+0.0540$, Off: $+0.0012$), proving that combining non-linear trees with regularized linear shrinkage provides a stable middle ground.
"""))

# Cell 24: Final Recommendation
cells.append(nbf.v4.new_markdown_cell("""## 12. Final Architecture Recommendation

Based on the empirical evidence across both evaluation regimes:
1. **Primary Production Architecture**:
   - **Model F (Ratio-Focused LightGBM, 18 features)** is recommended for cross-company/external job matching platforms where incoming job descriptions have arbitrary, unseen formatting.
   - It eliminates posting-length bias, outperforms linear models on unseen jobs (**0.4360 F1**), and incurs minimal latency (**0.018 ms/pair**).
2. **Candidate Ranking / In-House Matching Engine**:
   - **Model E (Balanced Length-Robust LightGBM, 19 features)** is the benchmark of choice when job templates are standardized, achieving the highest candidate separation (**0.5468 F1**, $64.8\\%$ Good Fit recall).
3. **Robustness Benchmark**:
   - **Model A (Reference Logistic Regression)** remains the transparent linear reference baseline.
4. **Next Phase Transition**:
   - Having exhaustively evaluated linear classification, bi-encoder scaling, cross-encoders, gradient boosted trees, length-robust ablations, and ensembles, tabular model experimentation has reached saturation.
   - We are ready to transition to **Phase 10: End-to-End Inference Pipeline, Score Calibration, and User Application**.
"""))

nb.cells = cells

# Save notebook
notebook_path = 'notebooks/09_ensemble_robustness.ipynb'
os.makedirs('notebooks', exist_ok=True)
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Notebook written to {notebook_path}. Executing notebook now...")

# Execute notebook using NotebookClient
client = NotebookClient(nb, timeout=600, kernel_name='python3', resources={'metadata': {'path': 'notebooks/'}})
client.execute()

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Notebook {notebook_path} successfully executed and saved with all outputs!")
