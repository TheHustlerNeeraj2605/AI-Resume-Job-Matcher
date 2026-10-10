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
cells.append(nbf.v4.new_markdown_cell("""# 07. Non-Linear Tabular Classifier (LightGBM)
**AI-Based Resume–Job Matching & Skill Gap Analysis — Phase 8A**

### 1. Objective
Throughout Phases 6A through 7, our primary matching classifier has been a linear model (**Logistic Regression**, $C=0.1$, `class_weight='balanced'`) operating over 22 engineered features.

While linear models provide well-behaved decision boundaries and prevent runaway overfitting, they inherently suffer from a fundamental architectural limitation:
> **Hypothesis**: "Linear Logistic Regression cannot capture non-linear feature interactions among semantic similarity, lexical overlap, skill coverage, and document-length signals."

**Phase 8A Goal**:
Empirically test whether a non-linear gradient-boosted decision tree classifier (**LightGBM**) can learn feature interactions over the exact same 22 engineered features to improve multi-class matching performance over the locked Logistic Regression reference.
"""))

# 2. Locked Reference Model
cells.append(nbf.v4.new_markdown_cell("""### 2. Locked Reference Model (Phase 6A)
- **Classifier**: `StandardScaler` + `LogisticRegression(C=0.1, class_weight='balanced', max_iter=2000, random_state=42)`
- **Feature Set**: 22 engineered features (TF-IDF similarity, MiniLM SBERT similarity, skill counts/coverage, length counts/ratios, Jaccard overlap).
- **Locked Benchmark Performance**:
  - `candidate_validation` Macro F1: **0.4603**
  - `candidate_test` (Unseen Candidates) Macro F1: **0.4735** | Accuracy: **0.5154**
  - `official_test` (Unseen JDs) Macro F1: **0.4220** | Accuracy: **0.4508**
"""))

# 3. Dataset and Feature Audit
cells.append(nbf.v4.new_markdown_cell("""### 3. Dataset and Feature Audit
Before fitting any model, we verify dataset integrity, feature column alignment, absence of NaNs/infinite values, and the natural class distribution across splits.
"""))

cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np

train = pd.read_csv('../data/evaluation/supervised_feature_dataset_train.csv')
val   = pd.read_csv('../data/evaluation/supervised_feature_dataset_validation.csv')
test  = pd.read_csv('../data/evaluation/supervised_feature_dataset_test.csv')
off   = pd.read_csv('../data/evaluation/supervised_feature_dataset_official_test.csv')

f_cols = [c for c in train.columns if c != 'label']

audit_df = pd.DataFrame([
    {
        'Split': 'Candidate Train',
        'Pairs': len(train),
        'Features': len(f_cols),
        'NaN Count': train[f_cols].isna().sum().sum(),
        'Inf Count': np.isinf(train[f_cols].values).sum(),
        'No Fit %': f"{(train['label'] == 'No Fit').mean()*100:.1f}%",
        'Potential Fit %': f"{(train['label'] == 'Potential Fit').mean()*100:.1f}%",
        'Good Fit %': f"{(train['label'] == 'Good Fit').mean()*100:.1f}%"
    },
    {
        'Split': 'Candidate Validation',
        'Pairs': len(val),
        'Features': len(f_cols),
        'NaN Count': val[f_cols].isna().sum().sum(),
        'Inf Count': np.isinf(val[f_cols].values).sum(),
        'No Fit %': f"{(val['label'] == 'No Fit').mean()*100:.1f}%",
        'Potential Fit %': f"{(val['label'] == 'Potential Fit').mean()*100:.1f}%",
        'Good Fit %': f"{(val['label'] == 'Good Fit').mean()*100:.1f}%"
    },
    {
        'Split': 'Candidate Test',
        'Pairs': len(test),
        'Features': len(f_cols),
        'NaN Count': test[f_cols].isna().sum().sum(),
        'Inf Count': np.isinf(test[f_cols].values).sum(),
        'No Fit %': f"{(test['label'] == 'No Fit').mean()*100:.1f}%",
        'Potential Fit %': f"{(test['label'] == 'Potential Fit').mean()*100:.1f}%",
        'Good Fit %': f"{(test['label'] == 'Good Fit').mean()*100:.1f}%"
    },
    {
        'Split': 'Official Test',
        'Pairs': len(off),
        'Features': len(f_cols),
        'NaN Count': off[f_cols].isna().sum().sum(),
        'Inf Count': np.isinf(off[f_cols].values).sum(),
        'No Fit %': f"{(off['label'] == 'No Fit').mean()*100:.1f}%",
        'Potential Fit %': f"{(off['label'] == 'Potential Fit').mean()*100:.1f}%",
        'Good Fit %': f"{(off['label'] == 'Good Fit').mean()*100:.1f}%"
    }
])

display(audit_df)
"""))

# 4. Logistic Regression Reproduction
cells.append(nbf.v4.new_markdown_cell("""### 4. Logistic Regression Reproduction
We reproduce the locked reference model inside this notebook to guarantee a self-contained, reproducible comparison.
"""))

cells.append(nbf.v4.new_code_cell("""from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

label_order = ['No Fit', 'Potential Fit', 'Good Fit']
X_tr = train[f_cols].values
y_tr = train['label'].values

pipe_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(C=0.1, class_weight='balanced', max_iter=2000, random_state=42))
])
pipe_lr.fit(X_tr, y_tr)

for sname, df_s in [('Validation', val), ('Candidate Test', test), ('Official Test', off)]:
    y_true = df_s['label'].values
    y_pred = pipe_lr.predict(df_s[f_cols].values)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, labels=label_order, average='macro')
    print(f"{sname:<18} -> Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
"""))

# 5, 6. LightGBM Configuration & Training
cells.append(nbf.v4.new_markdown_cell("""### 5, 6. LightGBM Configuration & Training
We train LightGBM (`LGBMClassifier`) with the recommended controlled hyperparameter configuration:
- `objective='multiclass'`, `num_class=3`
- `n_estimators=300`, `learning_rate=0.03`
- `num_leaves=15`, `max_depth=5` (shallow trees to prevent overfitting)
- `subsample=0.8`, `colsample_bytree=0.8`
- `reg_alpha=0.1`, `reg_lambda=1.0` (L1/L2 regularization)
- Trained on the natural training distribution, with early stopping evaluated strictly on `candidate_validation`.
"""))

# 7, 8, 9. Experimental Performance Results
cells.append(nbf.v4.new_markdown_cell("""### 7, 8, 9. Experimental Results Across Splits
We compare Logistic Regression (Reference) against LightGBM (Natural Distribution) and a diagnostic Balanced Class Weight variant across Validation, Candidate Test, and Official Test.
"""))

cells.append(nbf.v4.new_code_cell("""from IPython.display import display, Image

metrics_df = pd.read_csv('../data/evaluation/phase8_lightgbm_metrics.csv')
print("=== Overall Performance Summary ===")
display(metrics_df[['model', 'val_macro_f1', 'cand_test_macro_f1', 'cand_test_accuracy', 'off_test_macro_f1', 'off_test_accuracy']])

# Display comparison chart
display(Image(filename='../data/evaluation/plots/phase8_lightgbm_model_comparison.png'))
"""))

# 10. Per-Class Analysis
cells.append(nbf.v4.new_markdown_cell("""### 10. Granular Per-Class Metrics Breakdown
We inspect precision, recall, and F1 across `No Fit`, `Potential Fit`, and `Good Fit`.
"""))

cells.append(nbf.v4.new_code_cell("""# Display per-class comparison table
from sklearn.metrics import classification_report

idx_to_label = {0: 'No Fit', 1: 'Potential Fit', 2: 'Good Fit'}
label_to_idx = {l: i for i, l in idx_to_label.items()}

import lightgbm as lgb
lgb_model = lgb.LGBMClassifier(
    objective='multiclass', num_class=3, n_estimators=300, learning_rate=0.03,
    num_leaves=15, max_depth=5, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=-1
)
lgb_model.fit(X_tr, [label_to_idx[l] for l in y_tr])

p_te_lgb = [idx_to_label[i] for i in lgb_model.predict(test[f_cols].values)]
p_te_lr  = pipe_lr.predict(test[f_cols].values)

print("=== Candidate Test: Logistic Regression ===")
print(classification_report(test['label'], p_te_lr, labels=label_order, digits=4))

print("=== Candidate Test: LightGBM (Natural) ===")
print(classification_report(test['label'], p_te_lgb, labels=label_order, digits=4))
"""))

# 11. Confusion Matrices
cells.append(nbf.v4.new_markdown_cell("""### 11. Confusion Matrix Visualizations
We display the confusion matrices for Logistic Regression and LightGBM across all three evaluation splits.
"""))

cells.append(nbf.v4.new_code_cell("""display(Image(filename='../data/evaluation/plots/phase8_lightgbm_confusion_matrix.png'))
"""))

# 12. Feature Importance
cells.append(nbf.v4.new_markdown_cell("""### 12. Feature Importance Analysis
We analyze LightGBM's gain-based and split-based feature importances and compare them with the linear coefficients from Logistic Regression.
"""))

cells.append(nbf.v4.new_code_cell("""feat_imp_df = pd.read_csv('../data/evaluation/phase8_lightgbm_feature_importance.csv')
print("=== Top 10 Features by LightGBM Gain Importance ===")
display(feat_imp_df.head(10))

display(Image(filename='../data/evaluation/plots/phase8_lightgbm_feature_importance.png'))
"""))

# 13. Non-Linear Interaction Analysis
cells.append(nbf.v4.new_markdown_cell("""### 13. Non-Linear Interaction Analysis
We evaluate four key feature pairings to see where decision boundaries benefit from non-linear thresholding:
1. SBERT Cosine Similarity × Jaccard Similarity
2. SBERT Cosine Similarity × Skill Coverage
3. TF-IDF Cosine Similarity × Skill Coverage
4. Skill Coverage × Resume-to-Job Word Ratio
"""))

cells.append(nbf.v4.new_code_cell("""display(Image(filename='../data/evaluation/plots/phase8_lightgbm_interactions.png'))
"""))

# 14. Error Recovery Analysis
cells.append(nbf.v4.new_markdown_cell("""### 14. Case-Level Transition Diagnostics (Logistic Regression vs LightGBM)
On `candidate_test` ($N=1,201$), LightGBM recovered **222 pairs** where Logistic Regression was wrong, while regressing on **133 pairs**, yielding a net accuracy gain of **+89 pairs (+7.41%)**.
"""))

cells.append(nbf.v4.new_code_cell("""rec_df = pd.read_csv('../data/evaluation/phase8_lightgbm_error_recovery.csv')

print("=== Error Recovery Transition Breakdown ===")
counts = rec_df['category'].value_counts()
display(counts)

print("\\nSample Recovered Cases (LR Wrong -> LightGBM Correct):")
display(rec_df[rec_df['category'] == 'Recovered (LR wrong -> LGBM correct)'][['pair_id', 'true_label', 'pred_lr', 'pred_lgb', 'sbert_sim', 'tfidf_sim', 'skill_cov']].head(5))

print("\\nSample Regressed Cases (LR Correct -> LightGBM Wrong):")
display(rec_df[rec_df['category'] == 'Regressed (LR correct -> LGBM wrong)'][['pair_id', 'true_label', 'pred_lr', 'pred_lgb', 'sbert_sim', 'tfidf_sim', 'skill_cov']].head(5))
"""))

# 15. Computational Cost
cells.append(nbf.v4.new_markdown_cell("""### 15. Computational Timing & Cost Benchmarks
We benchmark training time, prediction latency, and model size.
"""))

cells.append(nbf.v4.new_code_cell("""timing_df = pd.read_csv('../data/evaluation/phase8_lightgbm_timing.csv')
print("=== Computational Cost Comparison ===")
display(timing_df)
"""))

# 16, 17, 18. Interpretation & Decision
cells.append(nbf.v4.new_markdown_cell("""### 16, 17, 18. Strategic Interpretation & Recommendations

### Key Empirical Findings:
1. **Dramatic Lift on Unseen Candidates**:
   - **Validation Macro F1**: Rose from **0.4603** (LR) to **0.5662** (LightGBM) ($\mathbf{+0.1059}$).
   - **Candidate Test Macro F1**: Rose from **0.4735** (LR) to **0.5188** (LightGBM) ($\mathbf{+0.0453}$).
   - **Candidate Test Accuracy**: Jumped from **0.5154** to **0.5895** ($\mathbf{+7.41\\%}$ absolute lift!).
   - **Good Fit Precision**: Surged from **41.39%** to **62.20%** ($\mathbf{+20.81\\%}$).
   - **Good Fit Recall**: Increased from **45.51%** to **50.83%** ($\mathbf{+5.32\\%}$).
   - **Good Fit F1**: Increased from **0.4335** to **0.5594** ($\mathbf{+12.59\\%}$).

2. **Divergence on Unseen Job Descriptions (`official_test`)**:
   - Accuracy improved from **0.4508** to **0.4781** ($+2.73\\%$), and `No Fit` F1 rose from **0.5572** to **0.6309**.
   - However, Macro F1 dropped from **0.4220** to **0.3838** ($-0.0382$) because `Good Fit` recall declined from 38.65% to 15.50%.
   - **Root Cause**: LightGBM relies heavily on specific length feature cutoffs (`job_unique_token_count`, `job_word_count`). When applied to entirely new job descriptions with different formatting and lengths, tree cutoffs become overly conservative.

3. **Classification Rule Outcome**:
   - Matches **OUTCOME B**: LightGBM heavily improves validation and candidate test, but exhibits generalization sensitivity on official test.
   - Documented in `DECISIONS.md` under **Decision 011: Non-Linear Gradient-Boosted Classification**.
"""))

nb.cells = cells

notebook_path = 'notebooks/07_nonlinear_classifier_lightgbm.ipynb'
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Notebook created at {notebook_path}. Now executing...")

ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb_to_run = nbf.read(f, as_version=4)

ep.preprocess(nb_to_run, {'metadata': {'path': 'notebooks/'}})

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb_to_run, f)

print("Notebook 07_nonlinear_classifier_lightgbm.ipynb executed and saved with all outputs!")
