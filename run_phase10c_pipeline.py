import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.metrics import log_loss, accuracy_score, f1_score
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

os.makedirs('data/evaluation/plots', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("=" * 70)
print("STARTING PHASE 10C: PROBABILITY & CONFIDENCE CALIBRATION PIPELINE")
print("=" * 70)

# 1. Load Data & Models
clf_f = joblib.load('models/model_f_ratio-focused_lightgbm_18_feats.joblib')
val_feats = pd.read_csv('data/evaluation/supervised_feature_dataset_validation.csv')
test_feats = pd.read_csv('data/evaluation/supervised_feature_dataset_test.csv')
off_feats = pd.read_csv('data/evaluation/supervised_feature_dataset_official_test.csv')

feats_18 = [
    'tfidf_cosine_similarity', 'sbert_cosine_similarity', 'skill_coverage',
    'num_job_skills', 'num_resume_skills', 'num_matched_skills', 'num_missing_skills',
    'resume_word_count', 'resume_char_count', 'resume_to_job_word_ratio',
    'resume_to_job_char_ratio', 'log_resume_word_count', 'num_shared_unique_tokens',
    'jaccard_similarity', 'resume_unique_token_count', 'has_any_detected_job_skill',
    'has_any_detected_resume_skill', 'has_any_matched_skill'
]

label_order = ['No Fit', 'Potential Fit', 'Good Fit']
label_to_idx = {l: i for i, l in enumerate(label_order)}

y_val = np.array([label_to_idx[l] for l in val_feats['label']])
y_test = np.array([label_to_idx[l] for l in test_feats['label']])
y_off = np.array([label_to_idx[l] for l in off_feats['label']])

pv_raw = clf_f.predict_proba(val_feats[feats_18].values)
pt_raw = clf_f.predict_proba(test_feats[feats_18].values)
po_raw = clf_f.predict_proba(off_feats[feats_18].values)

eps = 1e-12
lv_raw = np.log(np.clip(pv_raw, eps, 1.0 - eps))
lt_raw = np.log(np.clip(pt_raw, eps, 1.0 - eps))
lo_raw = np.log(np.clip(po_raw, eps, 1.0 - eps))

# 2. Calibration Metrics Functions
def calc_brier_multi(y_true, prob_matrix):
    Y_onehot = np.eye(3)[y_true]
    return float(np.mean(np.sum((prob_matrix - Y_onehot)**2, axis=1)))

def calc_brier_per_class(y_true, prob_matrix):
    Y_onehot = np.eye(3)[y_true]
    return [round(float(x), 4) for x in np.mean((prob_matrix - Y_onehot)**2, axis=0)]

def calc_ece(y_true, prob_matrix, n_bins=10):
    preds = np.argmax(prob_matrix, axis=1)
    confs = np.max(prob_matrix, axis=1)
    corrects = (preds == y_true)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_data = []

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i+1]
        mask = (confs >= low) & (confs <= high) if i == n_bins - 1 else (confs >= low) & (confs < high)
        n_in_bin = int(np.sum(mask))
        if n_in_bin > 0:
            bin_acc = float(np.mean(corrects[mask]))
            bin_conf = float(np.mean(confs[mask]))
            gap = abs(bin_acc - bin_conf)
            ece += (n_in_bin / len(y_true)) * gap
            bin_data.append({
                'bin_idx': i, 'low': round(low, 2), 'high': round(high, 2),
                'count': n_in_bin, 'bin_acc': round(bin_acc, 4),
                'bin_conf': round(bin_conf, 4), 'gap': round(gap, 4)
            })
        else:
            bin_data.append({
                'bin_idx': i, 'low': round(low, 2), 'high': round(high, 2),
                'count': 0, 'bin_acc': 0.0, 'bin_conf': 0.0, 'gap': 0.0
            })

    return float(ece), bin_data, float(np.mean(confs))

# 3. Fit Calibration Methods STRICTLY on Validation Data
print("\n--- Fitting Calibration Methods on Candidate Validation Data ---")

# Method A: Temperature Scaling
def temp_nll(T):
    p = softmax(lv_raw / T[0], axis=1)
    return log_loss(y_val, p)

res_temp = minimize(temp_nll, x0=[1.0], bounds=[(0.05, 10.0)])
T_opt = float(res_temp.x[0])
print(f"Optimal Temperature Parameter T: {T_opt:.4f}")

pv_temp = softmax(lv_raw / T_opt, axis=1)
pt_temp = softmax(lt_raw / T_opt, axis=1)
po_temp = softmax(lo_raw / T_opt, axis=1)

# Method B: Platt / Vector Scaling (Multiclass Logistic Regression)
platt_clf = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=42)
platt_clf.fit(lv_raw, y_val)
pv_platt = platt_clf.predict_proba(lv_raw)
pt_platt = platt_clf.predict_proba(lt_raw)
po_platt = platt_clf.predict_proba(lo_raw)

# Method C: Isotonic Regression (One-vs-Rest)
iso_models = []
for c in range(3):
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    iso.fit(pv_raw[:, c], (y_val == c).astype(float))
    iso_models.append(iso)

def apply_isotonic(p_in):
    p_out = np.zeros_like(p_in)
    for c in range(3):
        p_out[:, c] = iso_models[c].predict(p_in[:, c])
    return p_out / p_out.sum(axis=1, keepdims=True)

pv_iso = apply_isotonic(pv_raw)
pt_iso = apply_isotonic(pt_raw)
po_iso = apply_isotonic(po_raw)

# 4. Validation Comparison Table
val_eval_rows = []
candidates = [
    ('Raw Model F', pv_raw),
    (f'Temperature Scaling (T={T_opt:.2f})', pv_temp),
    ('Platt / Vector Scaling', pv_platt),
    ('Isotonic Regression', pv_iso)
]

for name, p_mat in candidates:
    ll = float(log_loss(y_val, p_mat))
    brier = calc_brier_multi(y_val, p_mat)
    ece, _, m_conf = calc_ece(y_val, p_mat)
    preds = np.argmax(p_mat, axis=1)
    acc = float(accuracy_score(y_val, preds))
    f1 = float(f1_score(y_val, preds, average='macro'))
    n_changed = int(np.sum(preds != np.argmax(pv_raw, axis=1)))
    pct_changed = round((n_changed / len(y_val)) * 100, 2)

    val_eval_rows.append({
        'method': name,
        'val_log_loss': round(ll, 4),
        'val_brier_score': round(brier, 4),
        'val_ece': round(ece, 4),
        'val_mean_confidence': round(m_conf, 4),
        'val_accuracy': round(acc, 4),
        'val_macro_f1': round(f1, 4),
        'predictions_changed': n_changed,
        'pct_predictions_changed': pct_changed
    })

val_df_metrics = pd.DataFrame(val_eval_rows)
val_df_metrics.to_csv('data/evaluation/phase10c_validation_calibration.csv', index=False)
print("\nValidation Calibration Comparison:")
print(val_df_metrics.to_string(index=False))

# 5. Method Selection Rationale
# Temperature scaling maintains EXACT classification decisions (0 changes, Macro F1 preserved),
# while lowering Log Loss and Brier Score.
# Vector Scaling and Isotonic Regression distort the decision boundaries and degrade Macro F1 out of sample.
# Therefore, Temperature Scaling is selected as the production calibration method.

# 6. Final Evaluation Across All Three Benchmark Splits
comparison_rows = []
for split_name, y_true, p_raw, p_temp in [
    ('Candidate Validation', y_val, pv_raw, pv_temp),
    ('Candidate Test', y_test, pt_raw, pt_temp),
    ('Official Test', y_off, po_raw, po_temp)
]:
    for regime_name, p_mat in [('Model F (Raw)', p_raw), (f'Model F + Temperature Scaling (T={T_opt:.2f})', p_temp)]:
        ll = float(log_loss(y_true, p_mat))
        brier = calc_brier_multi(y_true, p_mat)
        per_class_brier = calc_brier_per_class(y_true, p_mat)
        ece, _, m_conf = calc_ece(y_true, p_mat)
        preds = np.argmax(p_mat, axis=1)
        acc = float(accuracy_score(y_true, preds))
        f1 = float(f1_score(y_true, preds, average='macro'))
        n_changed = int(np.sum(preds != np.argmax(p_raw, axis=1)))

        comparison_rows.append({
            'dataset': split_name,
            'model_regime': regime_name,
            'log_loss': round(ll, 4),
            'brier_score': round(brier, 4),
            'brier_no_fit': per_class_brier[0],
            'brier_pot_fit': per_class_brier[1],
            'brier_good_fit': per_class_brier[2],
            'ece': round(ece, 4),
            'mean_confidence': round(m_conf, 4),
            'accuracy': round(acc, 4),
            'macro_f1': round(f1, 4),
            'changed_predictions': n_changed
        })

comp_df = pd.DataFrame(comparison_rows)
comp_df.to_csv('data/evaluation/phase10c_calibration_comparison.csv', index=False)
print("\nFinal Multi-Dataset Calibration Comparison:")
print(comp_df[['dataset', 'model_regime', 'log_loss', 'brier_score', 'ece', 'mean_confidence', 'accuracy', 'macro_f1', 'changed_predictions']].to_string(index=False))

# 7. Reliability Diagram Data Persistence
_, val_bin_raw, _ = calc_ece(y_val, pv_raw)
_, val_bin_temp, _ = calc_ece(y_val, pv_temp)
_, off_bin_raw, _ = calc_ece(y_off, po_raw)
_, off_bin_temp, _ = calc_ece(y_off, po_temp)

rel_records = []
for b in val_bin_raw:
    rel_records.append({'dataset': 'Validation', 'regime': 'Raw', **b})
for b in val_bin_temp:
    rel_records.append({'dataset': 'Validation', 'regime': 'Calibrated (Temp)', **b})
for b in off_bin_raw:
    rel_records.append({'dataset': 'Official Test', 'regime': 'Raw', **b})
for b in off_bin_temp:
    rel_records.append({'dataset': 'Official Test', 'regime': 'Calibrated (Temp)', **b})

rel_df = pd.DataFrame(rel_records)
rel_df.to_csv('data/evaluation/phase10c_reliability_data.csv', index=False)

# 8. Visual Diagnostics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Plot 1: Reliability Diagrams
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, split_name, raw_bins, temp_bins, raw_ece, temp_ece in [
    (axes[0], 'Candidate Validation', val_bin_raw, val_bin_temp, val_df_metrics.loc[0, 'val_ece'], val_df_metrics.loc[1, 'val_ece']),
    (axes[1], 'Official Test (Unseen JDs)', off_bin_raw, off_bin_temp, comp_df.loc[4, 'ece'], comp_df.loc[5, 'ece'])
]:
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=1.5)

    r_x = [b['bin_conf'] for b in raw_bins if b['count'] > 0]
    r_y = [b['bin_acc'] for b in raw_bins if b['count'] > 0]
    ax.plot(r_x, r_y, 'o-', color='#E74C3C', label=f'Raw Model F (ECE = {raw_ece:.4f})', linewidth=2, markersize=6)

    t_x = [b['bin_conf'] for b in temp_bins if b['count'] > 0]
    t_y = [b['bin_acc'] for b in temp_bins if b['count'] > 0]
    ax.plot(t_x, t_y, 's-', color='#2ECC71', label=f'Calibrated T={T_opt:.2f} (ECE = {temp_ece:.4f})', linewidth=2, markersize=6)

    ax.set_title(f"Reliability Diagram: {split_name}", fontsize=12, fontweight='bold')
    ax.set_xlabel("Mean Predicted Confidence", fontsize=11)
    ax.set_ylabel("Empirical Accuracy", fontsize=11)
    ax.set_xlim(0.2, 1.0)
    ax.set_ylim(0.2, 1.0)
    ax.legend(frameon=True, fontsize=10)

plt.suptitle("Phase 10C: Calibration Reliability Curves (Raw vs Temperature Scaling)", fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('data/evaluation/plots/phase10c_reliability.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nSaved data/evaluation/plots/phase10c_reliability.png")

# Plot 2: Confidence Distributions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

conf_v_raw = np.max(pv_raw, axis=1)
conf_v_temp = np.max(pv_temp, axis=1)
conf_o_raw = np.max(po_raw, axis=1)
conf_o_temp = np.max(po_temp, axis=1)

axes[0].hist(conf_v_raw, bins=20, alpha=0.6, color='#E74C3C', label=f'Raw (Mean={conf_v_raw.mean():.3f})')
axes[0].hist(conf_v_temp, bins=20, alpha=0.6, color='#2ECC71', label=f'Calibrated (Mean={conf_v_temp.mean():.3f})')
axes[0].set_title("Validation Confidence Distribution", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Max Predicted Probability", fontsize=11)
axes[0].set_ylabel("Frequency", fontsize=11)
axes[0].legend(frameon=True)

axes[1].hist(conf_o_raw, bins=20, alpha=0.6, color='#E74C3C', label=f'Raw (Mean={conf_o_raw.mean():.3f})')
axes[1].hist(conf_o_temp, bins=20, alpha=0.6, color='#2ECC71', label=f'Calibrated (Mean={conf_o_temp.mean():.3f})')
axes[1].set_title("Official Test Confidence Distribution", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Max Predicted Probability", fontsize=11)
axes[1].set_ylabel("Frequency", fontsize=11)
axes[1].legend(frameon=True)

plt.suptitle("Phase 10C: Confidence Distributions (Softening Effect of Temperature Scaling)", fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('data/evaluation/plots/phase10c_confidence_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved data/evaluation/plots/phase10c_confidence_distribution.png")

# 9. Save Production Calibrator Artifact
calibrator_artifact = {
    'method': 'temperature_scaling',
    'temperature': T_opt,
    'fitted_on': 'candidate_validation.csv',
    'validation_samples': len(val_feats),
    'val_log_loss_raw': round(val_df_metrics.loc[0, 'val_log_loss'], 4),
    'val_log_loss_calibrated': round(val_df_metrics.loc[1, 'val_log_loss'], 4),
    'val_ece_raw': round(val_df_metrics.loc[0, 'val_ece'], 4),
    'val_ece_calibrated': round(val_df_metrics.loc[1, 'val_ece'], 4),
    'classes': label_order,
    'class_indices': [0, 1, 2]
}

joblib.dump(calibrator_artifact, 'models/model_f_calibrator.joblib')
print("\nSaved production calibration artifact to models/model_f_calibrator.joblib")
print("=" * 70)
print("PHASE 10C CALIBRATION PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 70)
