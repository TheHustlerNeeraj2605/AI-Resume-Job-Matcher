import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

os.makedirs('data/evaluation/plots', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("=" * 70)
print("STARTING PHASE 9: E/F ENSEMBLE & ROBUSTNESS EVALUATION PIPELINE")
print("=" * 70)

# 1. Load Data
train_df = pd.read_csv('data/evaluation/supervised_feature_dataset_train.csv')
val_df   = pd.read_csv('data/evaluation/supervised_feature_dataset_validation.csv')
test_df  = pd.read_csv('data/evaluation/supervised_feature_dataset_test.csv')
off_df   = pd.read_csv('data/evaluation/supervised_feature_dataset_official_test.csv')

label_order = ['No Fit', 'Potential Fit', 'Good Fit']
label_to_idx = {l: i for i, l in enumerate(label_order)}
idx_to_label = {i: l for i, l in enumerate(label_order)}

y_val = val_df['label'].values
y_test = test_df['label'].values
y_off = off_df['label'].values

all_22 = [c for c in train_df.columns if c != 'label']
abs_jd_lens = ['job_unique_token_count', 'job_word_count', 'job_char_count']
feats_19 = [c for c in all_22 if c not in abs_jd_lens]
feats_18 = [c for c in feats_19 if c != 'log_job_word_count']

# 2. Load Models
print("\nLoading Locked Base Models from Phase 8B...")
pipe_a = joblib.load('models/supervised_logistic_regression.joblib')
clf_e  = joblib.load('models/model_e_balanced_length-robust_lightgbm_19_feats.joblib')
clf_f  = joblib.load('models/model_f_ratio-focused_lightgbm_18_feats.joblib')

# Save standard named checkpoints if desired
joblib.dump(pipe_a, 'models/model_a_reference_lr.pkl')
joblib.dump(clf_e, 'models/model_e_balanced_len_rob_lgbm.pkl')
joblib.dump(clf_f, 'models/model_f_ratio_focused_lgbm.pkl')

# Column mapping for Model A to align to canonical ['No Fit', 'Potential Fit', 'Good Fit']
class_to_col_a = {c: i for i, c in enumerate(pipe_a.classes_)}
cols_a = [class_to_col_a['No Fit'], class_to_col_a['Potential Fit'], class_to_col_a['Good Fit']]

# Extract probabilities
pv_a = pipe_a.predict_proba(val_df[all_22].values)[:, cols_a]
pt_a = pipe_a.predict_proba(test_df[all_22].values)[:, cols_a]
po_a = pipe_a.predict_proba(off_df[all_22].values)[:, cols_a]

pv_e = clf_e.predict_proba(val_df[feats_19].values)
pt_e = clf_e.predict_proba(test_df[feats_19].values)
po_e = clf_e.predict_proba(off_df[feats_19].values)

pv_f = clf_f.predict_proba(val_df[feats_18].values)
pt_f = clf_f.predict_proba(test_df[feats_18].values)
po_f = clf_f.predict_proba(off_df[feats_18].values)

# 3. Base Model Reproduction Verification
def eval_preds(y_true, proba):
    preds = [idx_to_label[i] for i in np.argmax(proba, axis=1)]
    acc = accuracy_score(y_true, preds)
    pm = precision_score(y_true, preds, labels=label_order, average='macro', zero_division=0)
    rm = recall_score(y_true, preds, labels=label_order, average='macro', zero_division=0)
    f1 = f1_score(y_true, preds, labels=label_order, average='macro', zero_division=0)
    cr = classification_report(y_true, preds, labels=label_order, output_dict=True, digits=4)
    cm = confusion_matrix(y_true, preds, labels=label_order)
    return preds, acc, pm, rm, f1, cr, cm

preds_v_a, acc_v_a, pm_v_a, rm_v_a, f1_v_a, cr_v_a, _ = eval_preds(y_val, pv_a)
preds_t_a, acc_t_a, pm_t_a, rm_t_a, f1_t_a, cr_t_a, _ = eval_preds(y_test, pt_a)
preds_o_a, acc_o_a, pm_o_a, rm_o_a, f1_o_a, cr_o_a, _ = eval_preds(y_off, po_a)

preds_v_e, acc_v_e, pm_v_e, rm_v_e, f1_v_e, cr_v_e, _ = eval_preds(y_val, pv_e)
preds_t_e, acc_t_e, pm_t_e, rm_t_e, f1_t_e, cr_t_e, _ = eval_preds(y_test, pt_e)
preds_o_e, acc_o_e, pm_o_e, rm_o_e, f1_o_e, cr_o_e, _ = eval_preds(y_off, po_e)

preds_v_f, acc_v_f, pm_v_f, rm_v_f, f1_v_f, cr_v_f, _ = eval_preds(y_val, pv_f)
preds_t_f, acc_t_f, pm_t_f, rm_t_f, f1_t_f, cr_t_f, _ = eval_preds(y_test, pt_f)
preds_o_f, acc_o_f, pm_o_f, rm_o_f, f1_o_f, cr_o_f, _ = eval_preds(y_off, po_f)

print("\n--- Base Model Verification ---")
print(f"Model A: Val F1 = {f1_v_a:.4f} (exp 0.4603) | Cand F1 = {f1_t_a:.4f} (exp 0.4735) | Off F1 = {f1_o_a:.4f} (exp 0.4220)")
print(f"Model E: Val F1 = {f1_v_e:.4f} (exp 0.5533) | Cand F1 = {f1_t_e:.4f} (exp 0.5468) | Off F1 = {f1_o_e:.4f} (exp 0.4113)")
print(f"Model F: Val F1 = {f1_v_f:.4f} (exp 0.4952) | Cand F1 = {f1_t_f:.4f} (exp 0.4712) | Off F1 = {f1_o_f:.4f} (exp 0.4360)")

assert abs(f1_v_a - 0.4603) < 1e-3
assert abs(f1_t_a - 0.4735) < 1e-3
assert abs(f1_o_a - 0.4220) < 1e-3
assert abs(f1_v_e - 0.5533) < 1e-3
assert abs(f1_t_e - 0.5468) < 1e-3
assert abs(f1_o_e - 0.4113) < 1e-3
assert abs(f1_v_f - 0.4952) < 1e-3
assert abs(f1_t_f - 0.4712) < 1e-3
assert abs(f1_o_f - 0.4360) < 1e-3
print("Base model reproduction VERIFIED exactly!")

# 4. Model Correlation / Complementarity Analysis on Validation
print("\n--- Computing Model Correlation and Agreement on Validation ---")
corr_rows = []
pairs = [
    ('Model A vs Model E', pv_a, pv_e),
    ('Model A vs Model F', pv_a, pv_f),
    ('Model E vs Model F', pv_e, pv_f)
]

for name, p1, p2 in pairs:
    max_p1 = np.max(p1, axis=1)
    max_p2 = np.max(p2, axis=1)
    r_max, _ = pearsonr(max_p1, max_p2)

    r_c0, _ = pearsonr(p1[:, 0], p2[:, 0])
    r_c1, _ = pearsonr(p1[:, 1], p2[:, 1])
    r_c2, _ = pearsonr(p1[:, 2], p2[:, 2])
    mean_r_class = np.mean([r_c0, r_c1, r_c2])

    pred1 = np.argmax(p1, axis=1)
    pred2 = np.argmax(p2, axis=1)
    agreement = np.mean(pred1 == pred2)
    disagreement = 1.0 - agreement

    corr_rows.append({
        'pair': name,
        'max_prob_correlation': round(r_max, 4),
        'class_0_no_fit_correlation': round(r_c0, 4),
        'class_1_potential_fit_correlation': round(r_c1, 4),
        'class_2_good_fit_correlation': round(r_c2, 4),
        'mean_class_prob_correlation': round(mean_r_class, 4),
        'agreement_rate': round(agreement, 4),
        'disagreement_rate': round(disagreement, 4)
    })

corr_df = pd.DataFrame(corr_rows)
corr_df.to_csv('data/evaluation/phase9_model_correlation.csv', index=False)
print("Saved data/evaluation/phase9_model_correlation.csv")

# 5. Validation Weight Search: Ensemble A (E + F)
print("\n--- Validation Weight Search: Ensemble A (E + F) ---")
ef_weights = [
    (1.00, 0.00),
    (0.90, 0.10),
    (0.75, 0.25),
    (0.50, 0.50),
    (0.25, 0.75),
    (0.10, 0.90),
    (0.00, 1.00)
]

search_rows = []
for we, wf in ef_weights:
    p_ens = we * pv_e + wf * pv_f
    preds, acc, pm, rm, f1, cr, _ = eval_preds(y_val, p_ens)
    search_rows.append({
        'ensemble_type': 'E + F',
        'weight_e': round(we, 2),
        'weight_f': round(wf, 2),
        'weight_a': 0.00,
        'val_accuracy': round(acc, 4),
        'val_macro_precision': round(pm, 4),
        'val_macro_recall': round(rm, 4),
        'val_macro_f1': round(f1, 4),
        'no_fit_f1': round(cr['No Fit']['f1-score'], 4),
        'potential_fit_f1': round(cr['Potential Fit']['f1-score'], 4),
        'good_fit_f1': round(cr['Good Fit']['f1-score'], 4),
        'good_fit_recall': round(cr['Good Fit']['recall'], 4),
        'potential_fit_recall': round(cr['Potential Fit']['recall'], 4)
    })

# 6. Validation Evaluation: Ensemble B (E + F + A)
print("\n--- Validation Evaluation: Ensemble B (E + F + A) Predefined Blends ---")
efa_weights = [
    (0.50, 0.25, 0.25),
    (0.25, 0.50, 0.25),
    (0.40, 0.40, 0.20),
    (0.40, 0.20, 0.40),
    (0.20, 0.40, 0.40),
    (1/3, 1/3, 1/3)
]

for we, wf, wa in efa_weights:
    p_ens = we * pv_e + wf * pv_f + wa * pv_a
    preds, acc, pm, rm, f1, cr, _ = eval_preds(y_val, p_ens)
    search_rows.append({
        'ensemble_type': 'E + F + A',
        'weight_e': round(we, 4),
        'weight_f': round(wf, 4),
        'weight_a': round(wa, 4),
        'val_accuracy': round(acc, 4),
        'val_macro_precision': round(pm, 4),
        'val_macro_recall': round(rm, 4),
        'val_macro_f1': round(f1, 4),
        'no_fit_f1': round(cr['No Fit']['f1-score'], 4),
        'potential_fit_f1': round(cr['Potential Fit']['f1-score'], 4),
        'good_fit_f1': round(cr['Good Fit']['f1-score'], 4),
        'good_fit_recall': round(cr['Good Fit']['recall'], 4),
        'potential_fit_recall': round(cr['Potential Fit']['recall'], 4)
    })

search_df = pd.DataFrame(search_rows)
search_df.to_csv('data/evaluation/phase9_validation_weight_search.csv', index=False)
print("Saved data/evaluation/phase9_validation_weight_search.csv")

# Identify Best E+F and Best E+F+A strictly from Validation Search
ef_df = search_df[search_df['ensemble_type'] == 'E + F']
best_ef_idx = ef_df['val_macro_f1'].idxmax()
best_ef_row = ef_df.loc[best_ef_idx]
best_w_e, best_w_f = best_ef_row['weight_e'], best_ef_row['weight_f']

efa_df = search_df[search_df['ensemble_type'] == 'E + F + A']
best_efa_idx = efa_df['val_macro_f1'].idxmax()
best_efa_row = efa_df.loc[best_efa_idx]
best_w_e_efa, best_w_f_efa, best_w_a_efa = best_efa_row['weight_e'], best_efa_row['weight_f'], best_efa_row['weight_a']

print(f"\nLocked Best E+F: E={best_w_e:.2f}, F={best_w_f:.2f} (Val F1={best_ef_row['val_macro_f1']:.4f})")
print(f"Locked Best E+F+A: E={best_w_e_efa:.2f}, F={best_w_f_efa:.2f}, A={best_w_a_efa:.2f} (Val F1={best_efa_row['val_macro_f1']:.4f})")

# 7. Final Model Evaluation Dictionary
eval_models = {
    'Model A (Ref LR)': {
        'val_p': pv_a, 'test_p': pt_a, 'off_p': po_a
    },
    'Model E (Bal Len-Rob)': {
        'val_p': pv_e, 'test_p': pt_e, 'off_p': po_e
    },
    'Model F (Ratio-Focused)': {
        'val_p': pv_f, 'test_p': pt_f, 'off_p': po_f
    },
    f'Best E+F ({int(best_w_e*100)}:{int(best_w_f*100)})': {
        'val_p': best_w_e * pv_e + best_w_f * pv_f,
        'test_p': best_w_e * pt_e + best_w_f * pt_f,
        'off_p': best_w_e * po_e + best_w_f * po_f
    },
    f'Best E+F+A ({int(best_w_e_efa*100)}:{int(best_w_f_efa*100)}:{int(best_w_a_efa*100)})': {
        'val_p': best_w_e_efa * pv_e + best_w_f_efa * pv_f + best_w_a_efa * pv_a,
        'test_p': best_w_e_efa * pt_e + best_w_f_efa * pt_f + best_w_a_efa * pt_a,
        'off_p': best_w_e_efa * po_e + best_w_f_efa * po_f + best_w_a_efa * po_a
    }
}

# 8. Full Metrics Table & Generalization Gap Table
metrics_records = []
gap_records = []
cached_evals = {}

for name, p_dict in eval_models.items():
    pv, pt, po = p_dict['val_p'], p_dict['test_p'], p_dict['off_p']

    preds_v, av, pmv, rmv, f1v, crv, cmv = eval_preds(y_val, pv)
    preds_t, at, pmt, rmt, f1t, crt, cmt = eval_preds(y_test, pt)
    preds_o, ao, pmo, rmo, f1o, cro, cmo = eval_preds(y_off, po)

    cached_evals[name] = {
        'preds_v': preds_v, 'preds_t': preds_t, 'preds_o': preds_o,
        'cm_t': cmt, 'cm_o': cmo, 'cr_t': crt, 'cr_o': cro
    }

    metrics_records.append({
        'model': name,
        'val_accuracy': round(av, 4),
        'val_macro_precision': round(pmv, 4),
        'val_macro_recall': round(rmv, 4),
        'val_macro_f1': round(f1v, 4),
        'cand_test_accuracy': round(at, 4),
        'cand_test_macro_precision': round(pmt, 4),
        'cand_test_macro_recall': round(rmt, 4),
        'cand_test_macro_f1': round(f1t, 4),
        'cand_no_fit_f1': round(crt['No Fit']['f1-score'], 4),
        'cand_potential_fit_f1': round(crt['Potential Fit']['f1-score'], 4),
        'cand_good_fit_f1': round(crt['Good Fit']['f1-score'], 4),
        'cand_good_fit_recall': round(crt['Good Fit']['recall'], 4),
        'cand_potential_fit_recall': round(crt['Potential Fit']['recall'], 4),
        'off_test_accuracy': round(ao, 4),
        'off_test_macro_precision': round(pmo, 4),
        'off_test_macro_recall': round(rmo, 4),
        'off_test_macro_f1': round(f1o, 4),
        'off_no_fit_f1': round(cro['No Fit']['f1-score'], 4),
        'off_potential_fit_f1': round(cro['Potential Fit']['f1-score'], 4),
        'off_good_fit_f1': round(cro['Good Fit']['f1-score'], 4),
        'off_good_fit_recall': round(cro['Good Fit']['recall'], 4),
        'off_potential_fit_recall': round(cro['Potential Fit']['recall'], 4)
    })

    cand_gap = f1t - f1v
    off_gap  = f1o - f1v
    gap_records.append({
        'model': name,
        'val_macro_f1': round(f1v, 4),
        'cand_test_macro_f1': round(f1t, 4),
        'off_test_macro_f1': round(f1o, 4),
        'candidate_generalization_gap': round(cand_gap, 4),
        'official_generalization_gap': round(off_gap, 4)
    })

metrics_df = pd.DataFrame(metrics_records)
metrics_df.to_csv('data/evaluation/phase9_ensemble_metrics.csv', index=False)
print("Saved data/evaluation/phase9_ensemble_metrics.csv")

gap_df = pd.DataFrame(gap_records)
gap_df.to_csv('data/evaluation/phase9_generalization_gap.csv', index=False)
print("Saved data/evaluation/phase9_generalization_gap.csv")

# 9. Error Recovery Analysis
print("\n--- Error Recovery Analysis ---")
best_ef_name = f'Best E+F ({int(best_w_e*100)}:{int(best_w_f*100)})'
best_efa_name = f'Best E+F+A ({int(best_w_e_efa*100)}:{int(best_w_f_efa*100)}:{int(best_w_a_efa*100)})'

recovery_rows = []
for split_name, y_true, preds_key in [('Candidate Test', y_test, 'preds_t'), ('Official Test', y_off, 'preds_o')]:
    for ens_name in [best_ef_name, best_efa_name]:
        ens_p = np.array(cached_evals[ens_name][preds_key])
        for base_name in ['Model A (Ref LR)', 'Model E (Bal Len-Rob)', 'Model F (Ratio-Focused)']:
            base_p = np.array(cached_evals[base_name][preds_key])

            b_corr = (base_p == y_true)
            e_corr = (ens_p == y_true)
            both_c = int(np.sum(b_corr & e_corr))
            both_w = int(np.sum(~b_corr & ~e_corr))
            rec    = int(np.sum(~b_corr & e_corr))
            reg    = int(np.sum(b_corr & ~e_corr))
            net    = rec - reg

            # Catastrophic error counts for baseline vs ensemble
            base_gf_to_nf = int(np.sum((y_true == 'Good Fit') & (base_p == 'No Fit')))
            ens_gf_to_nf  = int(np.sum((y_true == 'Good Fit') & (ens_p == 'No Fit')))
            gf_nf_diff    = ens_gf_to_nf - base_gf_to_nf

            base_pf_to_nf = int(np.sum((y_true == 'Potential Fit') & (base_p == 'No Fit')))
            ens_pf_to_nf  = int(np.sum((y_true == 'Potential Fit') & (ens_p == 'No Fit')))
            pf_nf_diff    = ens_pf_to_nf - base_pf_to_nf

            recovery_rows.append({
                'dataset': split_name,
                'ensemble': ens_name,
                'baseline': base_name,
                'both_correct': both_c,
                'both_wrong': both_w,
                'recovered_pairs': rec,
                'regressed_pairs': reg,
                'net_gain': net,
                'baseline_good_fit_to_no_fit': base_gf_to_nf,
                'ensemble_good_fit_to_no_fit': ens_gf_to_nf,
                'delta_good_fit_to_no_fit': gf_nf_diff,
                'baseline_pot_fit_to_no_fit': base_pf_to_nf,
                'ensemble_pot_fit_to_no_fit': ens_pf_to_nf,
                'delta_pot_fit_to_no_fit': pf_nf_diff
            })

rec_df = pd.DataFrame(recovery_rows)
rec_df.to_csv('data/evaluation/phase9_error_recovery.csv', index=False)
print("Saved data/evaluation/phase9_error_recovery.csv")

# 10. Probability / Confidence Analysis
print("\n--- Probability / Confidence Analysis ---")
prob_rows = []
for split_name, p_key in [('Candidate Validation', 'val_p'), ('Candidate Test', 'test_p'), ('Official Test', 'off_p')]:
    for m_name, m_dict in eval_models.items():
        p_mat = m_dict[p_key]
        max_p = np.max(p_mat, axis=1)
        prob_rows.append({
            'dataset': split_name,
            'model': m_name,
            'mean_max_prob': round(float(np.mean(max_p)), 4),
            'median_max_prob': round(float(np.median(max_p)), 4),
            'min_max_prob': round(float(np.min(max_p)), 4),
            'max_max_prob': round(float(np.max(max_p)), 4),
            'std_max_prob': round(float(np.std(max_p)), 4)
        })

prob_df = pd.DataFrame(prob_rows)
prob_df.to_csv('data/evaluation/phase9_probability_analysis.csv', index=False)
print("Saved data/evaluation/phase9_probability_analysis.csv")

# 11. Computational Cost / Inference Timing Benchmarks on Candidate Test
print("\n--- Measuring Inference Timing on Candidate Test ---")
timing_rows = []
n_cand = len(test_df)
X_test_22 = test_df[all_22].values
X_test_19 = test_df[feats_19].values
X_test_18 = test_df[feats_18].values

# Warmup
_ = pipe_a.predict_proba(X_test_22)
_ = clf_e.predict_proba(X_test_19)
_ = clf_f.predict_proba(X_test_18)

# Model A
times = []
for _ in range(5):
    t0 = time.perf_counter()
    p_a = pipe_a.predict_proba(X_test_22)[:, cols_a]
    times.append(time.perf_counter() - t0)
t_a = np.mean(times)
timing_rows.append({
    'model': 'Model A (Ref LR)',
    'total_inference_time_s': round(t_a, 4),
    'time_per_pair_ms': round((t_a / n_cand) * 1000, 4)
})

# Model E
times = []
for _ in range(5):
    t0 = time.perf_counter()
    p_e = clf_e.predict_proba(X_test_19)
    times.append(time.perf_counter() - t0)
t_e = np.mean(times)
timing_rows.append({
    'model': 'Model E (Bal Len-Rob)',
    'total_inference_time_s': round(t_e, 4),
    'time_per_pair_ms': round((t_e / n_cand) * 1000, 4)
})

# Model F
times = []
for _ in range(5):
    t0 = time.perf_counter()
    p_f = clf_f.predict_proba(X_test_18)
    times.append(time.perf_counter() - t0)
t_f = np.mean(times)
timing_rows.append({
    'model': 'Model F (Ratio-Focused)',
    'total_inference_time_s': round(t_f, 4),
    'time_per_pair_ms': round((t_f / n_cand) * 1000, 4)
})

# Best E+F
times = []
for _ in range(5):
    t0 = time.perf_counter()
    p_e = clf_e.predict_proba(X_test_19)
    p_f = clf_f.predict_proba(X_test_18)
    p_ef = best_w_e * p_e + best_w_f * p_f
    _ = np.argmax(p_ef, axis=1)
    times.append(time.perf_counter() - t0)
t_ef = np.mean(times)
timing_rows.append({
    'model': best_ef_name,
    'total_inference_time_s': round(t_ef, 4),
    'time_per_pair_ms': round((t_ef / n_cand) * 1000, 4)
})

# Best E+F+A
times = []
for _ in range(5):
    t0 = time.perf_counter()
    p_a = pipe_a.predict_proba(X_test_22)[:, cols_a]
    p_e = clf_e.predict_proba(X_test_19)
    p_f = clf_f.predict_proba(X_test_18)
    p_efa = best_w_e_efa * p_e + best_w_f_efa * p_f + best_w_a_efa * p_a
    _ = np.argmax(p_efa, axis=1)
    times.append(time.perf_counter() - t0)
t_efa = np.mean(times)
timing_rows.append({
    'model': best_efa_name,
    'total_inference_time_s': round(t_efa, 4),
    'time_per_pair_ms': round((t_efa / n_cand) * 1000, 4)
})

timing_df = pd.DataFrame(timing_rows)
timing_df.to_csv('data/evaluation/phase9_timing.csv', index=False)
print("Saved data/evaluation/phase9_timing.csv")

# 12. Visualizations
print("\n--- Generating Plots ---")
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Plot 1: Confusion Matrices
fig, axes = plt.subplots(2, 5, figsize=(24, 9))
model_names_list = list(eval_models.keys())

for j, m_name in enumerate(model_names_list):
    cm_t = cached_evals[m_name]['cm_t']
    cm_o = cached_evals[m_name]['cm_o']

    sns.heatmap(cm_t, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0, j],
                xticklabels=['NF', 'PF', 'GF'], yticklabels=['NF', 'PF', 'GF'])
    axes[0, j].set_title(f"{m_name}\n(Cand Test)", fontsize=11, fontweight='bold')
    axes[0, j].set_ylabel('True' if j == 0 else '')
    axes[0, j].set_xlabel('Predicted')

    sns.heatmap(cm_o, annot=True, fmt='d', cmap='Greens', cbar=False, ax=axes[1, j],
                xticklabels=['NF', 'PF', 'GF'], yticklabels=['NF', 'PF', 'GF'])
    axes[1, j].set_title(f"{m_name}\n(Official Test)", fontsize=11, fontweight='bold')
    axes[1, j].set_ylabel('True' if j == 0 else '')
    axes[1, j].set_xlabel('Predicted')

plt.suptitle("Phase 9: Confusion Matrices Comparison (Candidate Test vs Official Test)", fontsize=15, fontweight='bold', y=0.99)
plt.tight_layout()
plt.savefig('data/evaluation/plots/phase9_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved data/evaluation/plots/phase9_confusion_matrices.png")

# Plot 2: Model Comparison Plot
fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.25
x_indices = np.arange(len(model_names_list))

val_f1s  = [metrics_df.loc[metrics_df['model'] == m, 'val_macro_f1'].values[0] for m in model_names_list]
cand_f1s = [metrics_df.loc[metrics_df['model'] == m, 'cand_test_macro_f1'].values[0] for m in model_names_list]
off_f1s  = [metrics_df.loc[metrics_df['model'] == m, 'off_test_macro_f1'].values[0] for m in model_names_list]

r1 = ax.bar(x_indices - bar_width, val_f1s, width=bar_width, label='Validation Macro F1', color='#4A90E2', alpha=0.9)
r2 = ax.bar(x_indices, cand_f1s, width=bar_width, label='Candidate Test Macro F1 (Unseen Candidates)', color='#50E3C2', alpha=0.9)
r3 = ax.bar(x_indices + bar_width, off_f1s, width=bar_width, label='Official Test Macro F1 (Unseen JDs)', color='#F5A623', alpha=0.9)

for rects in [r1, r2, r3]:
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f'{h:.4f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(x_indices)
ax.set_xticklabels(model_names_list, fontsize=10, fontweight='bold', rotation=12)
ax.set_ylabel('Macro F1 Score', fontsize=12, fontweight='bold')
ax.set_title('Phase 9 Model Performance Across Evaluation Regimes', fontsize=14, fontweight='bold')
ax.set_ylim(0.30, 0.65)
ax.axhline(0.4220, color='red', linestyle='--', linewidth=1, label='Model A Official Test Baseline (0.4220)')
ax.legend(frameon=True, fontsize=10, loc='upper right')
plt.tight_layout()
plt.savefig('data/evaluation/plots/phase9_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved data/evaluation/plots/phase9_model_comparison.png")

# Plot 3: Generalization Gap Plot
fig, ax = plt.subplots(figsize=(10, 6))
cand_gaps = [gap_df.loc[gap_df['model'] == m, 'candidate_generalization_gap'].values[0] for m in model_names_list]
off_gaps  = [gap_df.loc[gap_df['model'] == m, 'official_generalization_gap'].values[0] for m in model_names_list]

r_cand = ax.bar(x_indices - bar_width/2, cand_gaps, width=bar_width, label='Candidate Gen Gap (Cand Test - Val)', color='#2ECC71', alpha=0.9)
r_off  = ax.bar(x_indices + bar_width/2, off_gaps, width=bar_width, label='Official-JD Gen Gap (Off Test - Val)', color='#E74C3C', alpha=0.9)

for rects in [r_cand, r_off]:
    for rect in rects:
        h = rect.get_height()
        va = 'bottom' if h >= 0 else 'top'
        y_off = 3 if h >= 0 else -9
        ax.annotate(f'{h:+.4f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, y_off), textcoords="offset points", ha='center', va=va, fontsize=9, fontweight='bold')

ax.axhline(0, color='black', linestyle='-', linewidth=1)
ax.set_xticks(x_indices)
ax.set_xticklabels(model_names_list, fontsize=10, fontweight='bold', rotation=12)
ax.set_ylabel('Generalization Gap (Macro F1)', fontsize=12, fontweight='bold')
ax.set_title('Phase 9 Generalization Gaps: Candidate vs Cross-Domain Job Robustness', fontsize=14, fontweight='bold')
ax.set_ylim(-0.20, 0.05)
ax.legend(frameon=True, fontsize=10, loc='lower left')
plt.tight_layout()
plt.savefig('data/evaluation/plots/phase9_generalization_gap.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved data/evaluation/plots/phase9_generalization_gap.png")

print("\nAll Phase 9 pipeline steps completed successfully!")
