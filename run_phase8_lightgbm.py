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
print("STARTING PHASE 8A: NON-LINEAR TABULAR CLASSIFIER (LIGHTGBM)")
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

feature_cols = [c for c in train_df.columns if c != 'label']
label_order = ['No Fit', 'Potential Fit', 'Good Fit']
label_to_idx = {l: i for i, l in enumerate(label_order)}
idx_to_label = {i: l for i, l in enumerate(label_order)}

X_train = train_df[feature_cols].values
y_train = train_df['label'].values
y_train_idx = np.array([label_to_idx[l] for l in y_train])

X_val = val_df[feature_cols].values
y_val = val_df['label'].values
y_val_idx = np.array([label_to_idx[l] for l in y_val])

X_test = test_df[feature_cols].values
y_test = test_df['label'].values
y_test_idx = np.array([label_to_idx[l] for l in y_test])

X_off = off_df[feature_cols].values
y_off = off_df['label'].values
y_off_idx = np.array([label_to_idx[l] for l in y_off])

# 2. Fit Model A: Logistic Regression (Reference)
print("\nFitting Model A (Logistic Regression Reference)...")
t0_lr_train = time.time()
pipe_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(C=0.1, class_weight='balanced', max_iter=2000, random_state=42))
])
pipe_lr.fit(X_train, y_train)
lr_train_time = time.time() - t0_lr_train

t0_lr_pred = time.time()
preds_val_lr  = pipe_lr.predict(X_val)
preds_test_lr = pipe_lr.predict(X_test)
preds_off_lr  = pipe_lr.predict(X_off)
lr_pred_time = time.time() - t0_lr_pred

# Save LR model
joblib.dump(pipe_lr, 'models/supervised_logistic_regression.joblib')

# 3. Fit Model B: LightGBM (Natural Distribution)
print("\nFitting Model B (LightGBM Natural Distribution)...")
lgb_model = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=3,
    n_estimators=300,
    learning_rate=0.03,
    num_leaves=15,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    verbosity=-1
)

t0_lgb_train = time.time()
lgb_model.fit(
    X_train, y_train_idx,
    eval_set=[(X_val, y_val_idx)],
    callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
)
lgb_train_time = time.time() - t0_lgb_train

t0_lgb_pred = time.time()
preds_val_lgb_idx  = lgb_model.predict(X_val)
preds_test_lgb_idx = lgb_model.predict(X_test)
preds_off_lgb_idx  = lgb_model.predict(X_off)
lgb_pred_time = time.time() - t0_lgb_pred

preds_val_lgb  = [idx_to_label[i] for i in preds_val_lgb_idx]
preds_test_lgb = [idx_to_label[i] for i in preds_test_lgb_idx]
preds_off_lgb  = [idx_to_label[i] for i in preds_off_lgb_idx]

# Save LightGBM model
joblib.dump(lgb_model, 'models/lightgbm_classifier.joblib')

# 4. Fit Model B (Balanced Class Weight Variant for Diagnostic Inspection)
print("Fitting LightGBM (Balanced Class Weight variant for diagnostic)...")
lgb_bal = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=3,
    n_estimators=300,
    learning_rate=0.03,
    num_leaves=15,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    class_weight='balanced',
    random_state=42,
    verbosity=-1
)
lgb_bal.fit(X_train, y_train_idx)
preds_val_bal  = [idx_to_label[i] for i in lgb_bal.predict(X_val)]
preds_test_bal = [idx_to_label[i] for i in lgb_bal.predict(X_test)]
preds_off_bal  = [idx_to_label[i] for i in lgb_bal.predict(X_off)]

# 5. Compute Metrics Across All Splits
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    pm = precision_score(y_true, y_pred, labels=label_order, average='macro', zero_division=0)
    rm = recall_score(y_true, y_pred, labels=label_order, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=label_order, average='macro', zero_division=0)
    cr = classification_report(y_true, y_pred, labels=label_order, output_dict=True, digits=4)
    cm = confusion_matrix(y_true, y_pred, labels=label_order)
    return acc, pm, rm, f1, cr, cm

models_eval = {
    'Logistic Regression (Reference)': (preds_val_lr, preds_test_lr, preds_off_lr),
    'LightGBM (Natural Distribution)': (preds_val_lgb, preds_test_lgb, preds_off_lgb),
    'LightGBM (Balanced Diagnostic)': (preds_val_bal, preds_test_bal, preds_off_bal)
}

metrics_rows = []
for m_name, (pv, pt, po) in models_eval.items():
    av, pmv, rmv, f1v, crv, cmv = compute_metrics(y_val, pv)
    at, pmt, rmt, f1t, crt, cmt = compute_metrics(y_test, pt)
    ao, pmo, rmo, f1o, cro, cmo = compute_metrics(y_off, po)

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

metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv('data/evaluation/phase8_lightgbm_metrics.csv', index=False)
print("Saved data/evaluation/phase8_lightgbm_metrics.csv")

# Per-Class Metrics Table
class_rows = []
for split_name, yt, preds_dict in [
    ('Candidate Validation', y_val, {m: p[0] for m, p in models_eval.items()}),
    ('Candidate Test', y_test, {m: p[1] for m, p in models_eval.items()}),
    ('Official Test', y_off, {m: p[2] for m, p in models_eval.items()})
]:
    for m_name, yp in preds_dict.items():
        _, _, _, _, cr, _ = compute_metrics(yt, yp)
        for lbl in label_order:
            class_rows.append({
                'dataset': split_name,
                'model': m_name,
                'label': lbl,
                'precision': round(cr[lbl]['precision'], 4),
                'recall': round(cr[lbl]['recall'], 4),
                'f1_score': round(cr[lbl]['f1-score'], 4),
                'support': cr[lbl]['support']
            })

class_df = pd.DataFrame(class_rows)
print("\n--- PERFORMANCE SUMMARY TABLE ---")
print(metrics_df[['model', 'val_macro_f1', 'cand_test_macro_f1', 'cand_test_accuracy', 'off_test_macro_f1', 'off_test_accuracy']].to_string(index=False))

# 6. Confusion Matrices Plot (6 subplots)
fig, axes = plt.subplots(3, 2, figsize=(12, 14))

cm_data = [
    ('Validation: Logistic Regression', confusion_matrix(y_val, preds_val_lr, labels=label_order), axes[0, 0]),
    ('Validation: LightGBM (Natural)',  confusion_matrix(y_val, preds_val_lgb, labels=label_order), axes[0, 1]),
    ('Candidate Test: Logistic Regression', confusion_matrix(y_test, preds_test_lr, labels=label_order), axes[1, 0]),
    ('Candidate Test: LightGBM (Natural)',  confusion_matrix(y_test, preds_test_lgb, labels=label_order), axes[1, 1]),
    ('Official Test: Logistic Regression', confusion_matrix(y_off, preds_off_lr, labels=label_order), axes[2, 0]),
    ('Official Test: LightGBM (Natural)',  confusion_matrix(y_off, preds_off_lgb, labels=label_order), axes[2, 1]),
]

for title, cm, ax in cm_data:
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_order, yticklabels=label_order, ax=ax)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

plt.tight_layout()
plt.savefig('data/evaluation/plots/phase8_lightgbm_confusion_matrix.png', dpi=300)
plt.close()
print("Saved data/evaluation/plots/phase8_lightgbm_confusion_matrix.png")

# 7. Model Comparison Bar Chart
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(3)
width = 0.25

rects1 = ax.bar(x - width, metrics_df['val_macro_f1'], width, label='Validation Macro F1', color='#4C72B0')
rects2 = ax.bar(x, metrics_df['cand_test_macro_f1'], width, label='Candidate Test Macro F1', color='#55A868')
rects3 = ax.bar(x + width, metrics_df['off_test_macro_f1'], width, label='Official Test Macro F1', color='#C44E52')

ax.set_ylabel('Macro F1 Score')
ax.set_title('Phase 8A: Model Performance Comparison Across Evaluation Splits')
ax.set_xticks(x)
ax.set_xticklabels(['Logistic Regression\n(Reference)', 'LightGBM\n(Natural)', 'LightGBM\n(Balanced Diag)'])
ax.set_ylim(0.30, 0.65)
ax.legend()

for rects in [rects1, rects2, rects3]:
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f'{h:.4f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('data/evaluation/plots/phase8_lightgbm_model_comparison.png', dpi=300)
plt.close()
print("Saved data/evaluation/plots/phase8_lightgbm_model_comparison.png")

# 8. Feature Importance (Gain & Split)
gain_imp  = lgb_model.booster_.feature_importance(importance_type='gain')
split_imp = lgb_model.booster_.feature_importance(importance_type='split')

lr_clf = pipe_lr.named_steps['clf']
# LR multiclass coefficients: shape (3, 22) -> average magnitude
lr_coef_mag = np.mean(np.abs(lr_clf.coef_), axis=0)

feat_imp_df = pd.DataFrame({
    'feature': feature_cols,
    'lgbm_gain_importance': gain_imp,
    'lgbm_split_importance': split_imp,
    'lgbm_gain_pct': (gain_imp / gain_imp.sum() * 100),
    'lr_mean_abs_coef': lr_coef_mag
}).sort_values('lgbm_gain_importance', ascending=False).reset_index(drop=True)

feat_imp_df.to_csv('data/evaluation/phase8_lightgbm_feature_importance.csv', index=False)
print("Saved data/evaluation/phase8_lightgbm_feature_importance.csv")

# Feature Importance Plot
plt.figure(figsize=(10, 8))
sns.barplot(data=feat_imp_df, y='feature', x='lgbm_gain_pct', palette='viridis')
plt.title('LightGBM Feature Importance (Gain % Contribution)', fontsize=12, fontweight='bold')
plt.xlabel('Gain Importance (%)')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('data/evaluation/plots/phase8_lightgbm_feature_importance.png', dpi=300)
plt.close()
print("Saved data/evaluation/plots/phase8_lightgbm_feature_importance.png")

# 9. Error Recovery Analysis on Candidate Test (Logistic Regression vs LightGBM Natural)
test_eval = test_raw.copy()
test_eval['pred_lr'] = preds_test_lr
test_eval['pred_lgb'] = preds_test_lgb
test_eval['correct_lr'] = (test_eval['label'] == test_eval['pred_lr'])
test_eval['correct_lgb'] = (test_eval['label'] == test_eval['pred_lgb'])

both_corr = test_eval[(test_eval['correct_lr']) & (test_eval['correct_lgb'])]
both_wrng = test_eval[(~test_eval['correct_lr']) & (~test_eval['correct_lgb'])]
recov     = test_eval[(~test_eval['correct_lr']) & (test_eval['correct_lgb'])]
regres    = test_eval[(test_eval['correct_lr']) & (~test_eval['correct_lgb'])]

print(f"\n--- CANDIDATE TEST TRANSITION MATRIX ---")
print(f"Both Correct: {len(both_corr)} ({len(both_corr)/len(test_eval)*100:.1f}%)")
print(f"Both Wrong:   {len(both_wrng)} ({len(both_wrng)/len(test_eval)*100:.1f}%)")
print(f"Recovered (LR Wrong -> LGBM Correct): {len(recov)} ({len(recov)/len(test_eval)*100:.1f}%)")
print(f"Regressed (LR Correct -> LGBM Wrong): {len(regres)} ({len(regres)/len(test_eval)*100:.1f}%)")
print(f"Net Change: {len(recov) - len(regres):+d} cases (Net Accuracy Delta: {(len(recov)-len(regres))/len(test_eval)*100:+.2f}%)")

def sample_cases(df_sub, cat_name, n=10):
    sampled = df_sub.head(n)
    rows = []
    for idx, r in sampled.iterrows():
        feat_row = test_df.iloc[idx]
        rows.append({
            'category': cat_name,
            'pair_id': idx,
            'true_label': r['label'],
            'pred_lr': r['pred_lr'],
            'pred_lgb': r['pred_lgb'],
            'resume_snippet': r['resume_text'][:160].replace('\n', ' ') + '...',
            'jd_snippet': r['job_description_text'][:160].replace('\n', ' ') + '...',
            'sbert_sim': round(feat_row['sbert_cosine_similarity'], 4),
            'tfidf_sim': round(feat_row['tfidf_cosine_similarity'], 4),
            'skill_cov': round(feat_row['skill_coverage'], 4),
            'jaccard': round(feat_row['jaccard_similarity'], 4),
            'job_words': int(feat_row['job_word_count']),
            'diagnostic_note': (
                f"Prediction changed from {r['pred_lr']} (LR) to {r['pred_lgb']} (LGBM). "
                f"SBERT={feat_row['sbert_cosine_similarity']:.3f}, TF-IDF={feat_row['tfidf_cosine_similarity']:.3f}, "
                f"SkillCov={feat_row['skill_coverage']:.2f}, Jaccard={feat_row['jaccard_similarity']:.3f}."
            )
        })
    return rows

recovery_rows = []
recovery_rows.extend(sample_cases(recov, 'Recovered (LR wrong -> LGBM correct)', 10))
recovery_rows.extend(sample_cases(regres, 'Regressed (LR correct -> LGBM wrong)', 10))
recovery_rows.extend(sample_cases(both_wrng, 'Both Wrong', 10))

rec_df = pd.DataFrame(recovery_rows)
rec_df.to_csv('data/evaluation/phase8_lightgbm_error_recovery.csv', index=False)
print("Saved data/evaluation/phase8_lightgbm_error_recovery.csv")

# 10. Computational Cost & Latency
lr_size_kb  = os.path.getsize('models/supervised_logistic_regression.joblib') / 1024
lgb_size_kb = os.path.getsize('models/lightgbm_classifier.joblib') / 1024

timing_df = pd.DataFrame([
    {
        'model': 'Logistic Regression (Reference)',
        'train_time_sec': round(lr_train_time, 4),
        'pred_time_candidate_test_ms': round(lr_pred_time * 1000, 2),
        'latency_per_pair_ms': round((lr_pred_time / (len(y_val) + len(y_test) + len(y_off))) * 1000, 4),
        'model_size_kb': round(lr_size_kb, 2)
    },
    {
        'model': 'LightGBM (Natural)',
        'train_time_sec': round(lgb_train_time, 4),
        'pred_time_candidate_test_ms': round(lgb_pred_time * 1000, 2),
        'latency_per_pair_ms': round((lgb_pred_time / (len(y_val) + len(y_test) + len(y_off))) * 1000, 4),
        'model_size_kb': round(lgb_size_kb, 2)
    }
])
timing_df.to_csv('data/evaluation/phase8_lightgbm_timing.csv', index=False)
print("Saved data/evaluation/phase8_lightgbm_timing.csv")
print("\n--- COMPUTATIONAL TIMING & COST ---")
print(timing_df.to_string(index=False))

print("\nPhase 8A execution complete!")
