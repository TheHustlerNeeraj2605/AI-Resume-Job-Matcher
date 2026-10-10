import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

os.makedirs('data/evaluation/plots', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("=" * 70)
print("STARTING PHASE 8B: LENGTH-ROBUST LIGHTGBM EVALUATION PIPELINE")
print("=" * 70)

# 1. Load Data
train_df = pd.read_csv('data/evaluation/supervised_feature_dataset_train.csv')
val_df   = pd.read_csv('data/evaluation/supervised_feature_dataset_validation.csv')
test_df  = pd.read_csv('data/evaluation/supervised_feature_dataset_test.csv')
off_df   = pd.read_csv('data/evaluation/supervised_feature_dataset_official_test.csv')

train_raw = pd.read_csv('data/processed/candidate_train.csv')
val_raw   = pd.read_csv('data/processed/candidate_validation.csv')
test_raw  = pd.read_csv('data/processed/candidate_test.csv')
off_raw   = pd.read_csv('data/raw/test.csv')

label_order = ['No Fit', 'Potential Fit', 'Good Fit']
label_to_idx = {l: i for i, l in enumerate(label_order)}
idx_to_label = {i: l for i, l in enumerate(label_order)}

y_train = train_df['label'].values
y_train_idx = np.array([label_to_idx[l] for l in y_train])

y_val = val_df['label'].values
y_val_idx = np.array([label_to_idx[l] for l in y_val])

y_test = test_df['label'].values
y_test_idx = np.array([label_to_idx[l] for l in y_test])

y_off = off_df['label'].values
y_off_idx = np.array([label_to_idx[l] for l in y_off])

# Feature Sets
all_22 = [c for c in train_df.columns if c != 'label']
abs_jd_lens = ['job_unique_token_count', 'job_word_count', 'job_char_count']
feats_19 = [c for c in all_22 if c not in abs_jd_lens]
feats_18 = [c for c in feats_19 if c != 'log_job_word_count'] # pure ratio variant

print(f"Feature configurations: all_22={len(all_22)}, feats_19={len(feats_19)}, feats_18={len(feats_18)}")

# Helper for metric evaluation
def evaluate_predictions(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    pm  = precision_score(y_true, y_pred, labels=label_order, average='macro', zero_division=0)
    rm  = recall_score(y_true, y_pred, labels=label_order, average='macro', zero_division=0)
    f1  = f1_score(y_true, y_pred, labels=label_order, average='macro', zero_division=0)
    cr  = classification_report(y_true, y_pred, labels=label_order, output_dict=True, digits=4)
    cm  = confusion_matrix(y_true, y_pred, labels=label_order)
    return acc, pm, rm, f1, cr, cm

# LightGBM default hyperparams
lgb_base_params = dict(
    objective='multiclass', num_class=3, n_estimators=300, learning_rate=0.03,
    num_leaves=15, max_depth=5, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=-1
)

# 2. Train Models
models_dict = {}
timing_records = []

# Model A: Logistic Regression (Reference)
print("\nTraining Model A: Logistic Regression Reference...")
t0 = time.time()
pipe_a = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(C=0.1, class_weight='balanced', max_iter=2000, random_state=42))
])
pipe_a.fit(train_df[all_22].values, y_train)
t_train_a = time.time() - t0

t0 = time.time()
pv_a = pipe_a.predict(val_df[all_22].values)
t_pred_v_a = time.time() - t0
t0 = time.time()
pt_a = pipe_a.predict(test_df[all_22].values)
t_pred_t_a = time.time() - t0
t0 = time.time()
po_a = pipe_a.predict(off_df[all_22].values)
t_pred_o_a = time.time() - t0

models_dict['Model A: Logistic Regression (Reference)'] = {
    'preds_val': pv_a, 'preds_test': pt_a, 'preds_off': po_a,
    'model_obj': pipe_a, 'features': all_22
}
timing_records.append({
    'model': 'Model A: Logistic Regression (Reference)',
    'train_time_s': round(t_train_a, 4),
    'cand_test_pred_ms': round(t_pred_t_a * 1000, 2),
    'off_test_pred_ms': round(t_pred_o_a * 1000, 2),
    'model_size_kb': round(os.path.getsize('models/supervised_logistic_regression.joblib') / 1024, 2)
})

# Helper to train and record LightGBM variants
def train_lgbm_variant(name, feature_list, is_balanced):
    print(f"Training {name} ({len(feature_list)} features, balanced={is_balanced})...")
    kw = lgb_base_params.copy()
    if is_balanced:
        kw['class_weight'] = 'balanced'
    clf = lgb.LGBMClassifier(**kw)

    t0 = time.time()
    clf.fit(train_df[feature_list].values, y_train_idx)
    t_train = time.time() - t0

    t0 = time.time()
    pv = [idx_to_label[i] for i in clf.predict(val_df[feature_list].values)]
    t_pv = time.time() - t0

    t0 = time.time()
    pt = [idx_to_label[i] for i in clf.predict(test_df[feature_list].values)]
    t_pt = time.time() - t0

    t0 = time.time()
    po = [idx_to_label[i] for i in clf.predict(off_df[feature_list].values)]
    t_po = time.time() - t0

    save_path = f"models/{name.lower().replace(' ', '_').replace(':', '').replace('(', '').replace(')', '')}.joblib"
    joblib.dump(clf, save_path)

    models_dict[name] = {
        'preds_val': pv, 'preds_test': pt, 'preds_off': po,
        'model_obj': clf, 'features': feature_list
    }
    timing_records.append({
        'model': name,
        'train_time_s': round(t_train, 4),
        'cand_test_pred_ms': round(t_pt * 1000, 2),
        'off_test_pred_ms': round(t_po * 1000, 2),
        'model_size_kb': round(os.path.getsize(save_path) / 1024, 2)
    })

# Train Models B, C, D, E, F
train_lgbm_variant('Model B: Natural LightGBM (22 feats)', all_22, False)
train_lgbm_variant('Model C: Balanced LightGBM (22 feats)', all_22, True)
train_lgbm_variant('Model D: Length-Ablated LightGBM (19 feats)', feats_19, False)
train_lgbm_variant('Model E: Balanced Length-Robust LightGBM (19 feats)', feats_19, True)
train_lgbm_variant('Model F: Ratio-Focused LightGBM (18 feats)', feats_18, True)

# 3. Build Metrics Table & Generalization Gap Table
metrics_rows = []
gap_rows = []
per_class_rows = []

for m_name, m_info in models_dict.items():
    pv = m_info['preds_val']
    pt = m_info['preds_test']
    po = m_info['preds_off']

    av, pmv, rmv, f1v, crv, cmv = evaluate_predictions(y_val, pv)
    at, pmt, rmt, f1t, crt, cmt = evaluate_predictions(y_test, pt)
    ao, pmo, rmo, f1o, cro, cmo = evaluate_predictions(y_off, po)

    metrics_rows.append({
        'model': m_name,
        'val_accuracy': round(av, 4),
        'val_macro_precision': round(pmv, 4),
        'val_macro_recall': round(rmv, 4),
        'val_macro_f1': round(f1v, 4),
        'cand_test_accuracy': round(at, 4),
        'cand_test_macro_precision': round(pmt, 4),
        'cand_test_macro_recall': round(rmt, 4),
        'cand_test_macro_f1': round(f1t, 4),
        'off_test_accuracy': round(ao, 4),
        'off_test_macro_precision': round(pmo, 4),
        'off_test_macro_recall': round(rmo, 4),
        'off_test_macro_f1': round(f1o, 4)
    })

    cand_gap = f1t - f1v
    off_gap  = f1o - f1v
    gap_rows.append({
        'model': m_name,
        'val_macro_f1': round(f1v, 4),
        'cand_test_macro_f1': round(f1t, 4),
        'off_test_macro_f1': round(f1o, 4),
        'candidate_generalization_gap': round(cand_gap, 4),
        'official_generalization_gap': round(off_gap, 4)
    })

    for split_lbl, s_cr in [('Candidate Validation', crv), ('Candidate Test', crt), ('Official Test', cro)]:
        for lbl in label_order:
            per_class_rows.append({
                'dataset': split_lbl,
                'model': m_name,
                'label': lbl,
                'precision': round(s_cr[lbl]['precision'], 4),
                'recall': round(s_cr[lbl]['recall'], 4),
                'f1_score': round(s_cr[lbl]['f1-score'], 4),
                'support': s_cr[lbl]['support']
            })

metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv('data/evaluation/phase8b_length_robust_metrics.csv', index=False)
print("\nSaved data/evaluation/phase8b_length_robust_metrics.csv")

gap_df = pd.DataFrame(gap_rows)
gap_df.to_csv('data/evaluation/phase8b_generalization_gap.csv', index=False)
print("Saved data/evaluation/phase8b_generalization_gap.csv")

class_df = pd.DataFrame(per_class_rows)

timing_df = pd.DataFrame(timing_records)
timing_df.to_csv('data/evaluation/phase8b_timing.csv', index=False)
print("Saved data/evaluation/phase8b_timing.csv")

print("\n--- PERFORMANCE SUMMARY TABLE ---")
print(metrics_df[['model', 'val_macro_f1', 'cand_test_macro_f1', 'cand_test_accuracy', 'off_test_macro_f1', 'off_test_accuracy']].to_string(index=False))

print("\n--- GENERALIZATION GAP TABLE ---")
print(gap_df[['model', 'val_macro_f1', 'cand_test_macro_f1', 'candidate_generalization_gap', 'off_test_macro_f1', 'official_generalization_gap']].to_string(index=False))

# 4. Feature Importance Comparison
feat_imp_rows = []
for m_name in ['Model B: Natural LightGBM (22 feats)', 'Model C: Balanced LightGBM (22 feats)',
               'Model D: Length-Ablated LightGBM (19 feats)', 'Model E: Balanced Length-Robust LightGBM (19 feats)']:
    clf = models_dict[m_name]['model_obj']
    f_list = models_dict[m_name]['features']
    gain = clf.booster_.feature_importance(importance_type='gain')
    split = clf.booster_.feature_importance(importance_type='split')
    gain_pct = gain / gain.sum() * 100

    for f, g, s, gp in zip(f_list, gain, split, gain_pct):
        feat_imp_rows.append({
            'model': m_name,
            'feature': f,
            'gain_importance': round(g, 2),
            'split_importance': int(s),
            'gain_percentage': round(gp, 2)
        })

feat_imp_df = pd.DataFrame(feat_imp_rows)
feat_imp_df.to_csv('data/evaluation/phase8b_length_feature_importance.csv', index=False)
print("Saved data/evaluation/phase8b_length_feature_importance.csv")

# Gain of the 3 absolute length features:
print("\n--- ABSOLUTE LENGTH FEATURES GAIN PERCENTAGE ---")
for m_name in ['Model B: Natural LightGBM (22 feats)', 'Model C: Balanced LightGBM (22 feats)',
               'Model D: Length-Ablated LightGBM (19 feats)', 'Model E: Balanced Length-Robust LightGBM (19 feats)']:
    sub = feat_imp_df[feat_imp_df['model'] == m_name]
    len_gain = sub[sub['feature'].isin(abs_jd_lens)]['gain_percentage'].sum()
    print(f"{m_name:<45} -> Absolute Length Features Gain: {len_gain:.2f}%")

# 5. Error Recovery Analysis: Model B (Natural) vs Model E (Balanced Length-Robust)
print("\nAnalyzing Error Recovery: Model B vs Model E...")
test_eval = test_raw.copy()
test_eval['pred_b'] = models_dict['Model B: Natural LightGBM (22 feats)']['preds_test']
test_eval['pred_e'] = models_dict['Model E: Balanced Length-Robust LightGBM (19 feats)']['preds_test']
test_eval['corr_b'] = (test_eval['label'] == test_eval['pred_b'])
test_eval['corr_e'] = (test_eval['label'] == test_eval['pred_e'])

off_eval = off_raw.copy()
off_eval['pred_b'] = models_dict['Model B: Natural LightGBM (22 feats)']['preds_off']
off_eval['pred_e'] = models_dict['Model E: Balanced Length-Robust LightGBM (19 feats)']['preds_off']
off_eval['corr_b'] = (off_eval['label'] == off_eval['pred_b'])
off_eval['corr_e'] = (off_eval['label'] == off_eval['pred_e'])

rec_cand = test_eval[(~test_eval['corr_b']) & (test_eval['corr_e'])]
reg_cand = test_eval[(test_eval['corr_b']) & (~test_eval['corr_e'])]
both_corr_cand = test_eval[(test_eval['corr_b']) & (test_eval['corr_e'])]
both_wrng_cand = test_eval[(~test_eval['corr_b']) & (~test_eval['corr_e'])]

rec_off = off_eval[(~off_eval['corr_b']) & (off_eval['corr_e'])]
reg_off = off_eval[(off_eval['corr_b']) & (~off_eval['corr_e'])]

print(f"Candidate Test: Both Corr={len(both_corr_cand)}, Both Wrng={len(both_wrng_cand)}, Recov={len(rec_cand)}, Regres={len(reg_cand)} (Net {len(rec_cand)-len(reg_cand):+d})")
print(f"Official Test:  Recov={len(rec_off)}, Regres={len(reg_off)} (Net {len(rec_off)-len(reg_off):+d})")

# Specifically check Good Fit -> No Fit errors on official test
gf_to_nf_b = ((off_eval['label'] == 'Good Fit') & (off_eval['pred_b'] == 'No Fit')).sum()
gf_to_nf_e = ((off_eval['label'] == 'Good Fit') & (off_eval['pred_e'] == 'No Fit')).sum()
print(f"Official Test Good Fit misclassified as No Fit: Model B = {gf_to_nf_b} | Model E = {gf_to_nf_e} (Reduction: {gf_to_nf_b - gf_to_nf_e})")

recovery_rows = []
def sample_trans(df_eval, feat_source, split_name, cat_name, mask, n=7):
    sampled = df_eval[mask].head(n)
    for idx, r in sampled.iterrows():
        feat_r = feat_source.iloc[idx]
        recovery_rows.append({
            'split': split_name,
            'category': cat_name,
            'pair_id': idx,
            'true_label': r['label'],
            'pred_model_b': r['pred_b'],
            'pred_model_e': r['pred_e'],
            'resume_snippet': r['resume_text'][:150].replace('\n', ' ') + '...',
            'jd_snippet': r['job_description_text'][:150].replace('\n', ' ') + '...',
            'sbert_sim': round(feat_r['sbert_cosine_similarity'], 4),
            'tfidf_sim': round(feat_r['tfidf_cosine_similarity'], 4),
            'skill_cov': round(feat_r['skill_coverage'], 4),
            'resume_to_jd_ratio': round(feat_r['resume_to_job_word_ratio'], 3),
            'diagnostic_note': (
                f"Shifted from {r['pred_b']} to {r['pred_e']}. "
                f"SBERT={feat_r['sbert_cosine_similarity']:.3f}, TF-IDF={feat_r['tfidf_cosine_similarity']:.3f}, "
                f"SkillCov={feat_r['skill_coverage']:.2f}, WordRatio={feat_r['resume_to_job_word_ratio']:.2f}."
            )
        })

sample_trans(test_eval, test_df, 'Candidate Test', 'Recovered (B wrong -> E correct)', (~test_eval['corr_b']) & (test_eval['corr_e']))
sample_trans(test_eval, test_df, 'Candidate Test', 'Regressed (B correct -> E wrong)', (test_eval['corr_b']) & (~test_eval['corr_e']))
sample_trans(off_eval, off_df, 'Official Test', 'Recovered Official (B wrong -> E correct)', (~off_eval['corr_b']) & (off_eval['corr_e']))
sample_trans(off_eval, off_df, 'Official Test', 'Regressed Official (B correct -> E wrong)', (off_eval['corr_b']) & (~off_eval['corr_e']))

rec_df = pd.DataFrame(recovery_rows)
rec_df.to_csv('data/evaluation/phase8b_error_recovery.csv', index=False)
print("Saved data/evaluation/phase8b_error_recovery.csv")

# 6. Generate Plots
# Plot 1: Model Comparison Bar Chart Across Splits
plt.figure(figsize=(12, 6))
x = np.arange(len(metrics_df))
width = 0.25

plt.bar(x - width, metrics_df['val_macro_f1'], width, label='Validation Macro F1', color='#4C72B0')
plt.bar(x, metrics_df['cand_test_macro_f1'], width, label='Candidate Test Macro F1', color='#55A868')
plt.bar(x + width, metrics_df['off_test_macro_f1'], width, label='Official Test Macro F1', color='#C44E52')

plt.ylabel('Macro F1 Score', fontsize=12)
plt.title('Phase 8B: Model Macro F1 Comparison Across Evaluation Splits', fontsize=13, fontweight='bold')
plt.xticks(x, [m.replace(': ', '\n').replace(' (', '\n(') for m in metrics_df['model']], fontsize=9)
plt.ylim(0.30, 0.65)
plt.legend(fontsize=10)

for i in range(len(metrics_df)):
    plt.text(i - width, metrics_df['val_macro_f1'].iloc[i] + 0.005, f"{metrics_df['val_macro_f1'].iloc[i]:.3f}", ha='center', fontsize=8)
    plt.text(i, metrics_df['cand_test_macro_f1'].iloc[i] + 0.005, f"{metrics_df['cand_test_macro_f1'].iloc[i]:.3f}", ha='center', fontsize=8)
    plt.text(i + width, metrics_df['off_test_macro_f1'].iloc[i] + 0.005, f"{metrics_df['off_test_macro_f1'].iloc[i]:.3f}", ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('data/evaluation/plots/phase8b_model_comparison.png', dpi=300)
plt.close()
print("Saved data/evaluation/plots/phase8b_model_comparison.png")

# Plot 2: Generalization Gap Comparison
plt.figure(figsize=(10, 5))
x = np.arange(len(gap_df))
width = 0.35

plt.bar(x - width/2, gap_df['candidate_generalization_gap'], width, label='Candidate Gen Gap (Cand - Val)', color='#55A868')
plt.bar(x + width/2, gap_df['official_generalization_gap'], width, label='Official Gen Gap (Off - Val)', color='#C44E52')

plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.ylabel('Generalization Gap (Macro F1)', fontsize=11)
plt.title('Phase 8B: Generalization Gaps Across Architectures', fontsize=12, fontweight='bold')
plt.xticks(x, [m.replace(': ', '\n').replace(' (', '\n(') for m in gap_df['model']], fontsize=8)
plt.legend(fontsize=10)

for i in range(len(gap_df)):
    cg = gap_df['candidate_generalization_gap'].iloc[i]
    og = gap_df['official_generalization_gap'].iloc[i]
    plt.text(i - width/2, cg + (0.005 if cg >= 0 else -0.015), f"{cg:+.3f}", ha='center', fontsize=8)
    plt.text(i + width/2, og + (0.005 if og >= 0 else -0.015), f"{og:+.3f}", ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('data/evaluation/plots/phase8b_generalization_gap.png', dpi=300)
plt.close()
print("Saved data/evaluation/plots/phase8b_generalization_gap.png")

# Plot 3: Confusion Matrices (Models A, B, C, D, E for Candidate Test & Official Test)
cm_models = [
    ('Model A (Ref)', models_dict['Model A: Logistic Regression (Reference)']),
    ('Model B (Nat)', models_dict['Model B: Natural LightGBM (22 feats)']),
    ('Model C (Bal)', models_dict['Model C: Balanced LightGBM (22 feats)']),
    ('Model D (Len-Abl)', models_dict['Model D: Length-Ablated LightGBM (19 feats)']),
    ('Model E (Bal Len-Rob)', models_dict['Model E: Balanced Length-Robust LightGBM (19 feats)'])
]

fig, axes = plt.subplots(5, 2, figsize=(11, 18))
for row_idx, (m_lbl, m_data) in enumerate(cm_models):
    cm_t = confusion_matrix(y_test, m_data['preds_test'], labels=label_order)
    cm_o = confusion_matrix(y_off, m_data['preds_off'], labels=label_order)

    sns.heatmap(cm_t, annot=True, fmt='d', cmap='Blues', xticklabels=label_order, yticklabels=label_order, ax=axes[row_idx, 0])
    axes[row_idx, 0].set_title(f"Candidate Test: {m_lbl}", fontsize=10, fontweight='bold')
    axes[row_idx, 0].set_xlabel('Predicted')
    axes[row_idx, 0].set_ylabel('True')

    sns.heatmap(cm_o, annot=True, fmt='d', cmap='Oranges', xticklabels=label_order, yticklabels=label_order, ax=axes[row_idx, 1])
    axes[row_idx, 1].set_title(f"Official Test: {m_lbl}", fontsize=10, fontweight='bold')
    axes[row_idx, 1].set_xlabel('Predicted')
    axes[row_idx, 1].set_ylabel('True')

plt.tight_layout()
plt.savefig('data/evaluation/plots/phase8b_confusion_matrices.png', dpi=300)
plt.close()
print("Saved data/evaluation/plots/phase8b_confusion_matrices.png")

# Plot 4: Feature Importance Comparison (Top 10 across Models B, C, D, E)
top_feats = feat_imp_df[feat_imp_df['model'] == 'Model E: Balanced Length-Robust LightGBM (19 feats)'].sort_values('gain_percentage', ascending=False)['feature'].head(10).tolist()
comp_imp_df = feat_imp_df[feat_imp_df['feature'].isin(top_feats)]

plt.figure(figsize=(12, 6))
sns.barplot(data=comp_imp_df, y='feature', x='gain_percentage', hue='model', palette='tab10')
plt.title('Top 10 Feature Importance Shift: With vs. Without Absolute Length Features', fontsize=12, fontweight='bold')
plt.xlabel('Gain Importance (%)', fontsize=11)
plt.ylabel('Feature', fontsize=11)
plt.legend(title='Model Variant', fontsize=9)
plt.tight_layout()
plt.savefig('data/evaluation/plots/phase8b_feature_importance.png', dpi=300)
plt.close()
print("Saved data/evaluation/plots/phase8b_feature_importance.png")

print("\nPhase 8B execution complete!")
