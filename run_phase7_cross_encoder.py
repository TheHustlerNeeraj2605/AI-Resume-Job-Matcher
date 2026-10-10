import os
import time
import torch
import psutil
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import CrossEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

os.makedirs('data/evaluation/plots', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("=" * 70)
print("PHASE 7: CROSS-ENCODER SIGNAL EVALUATION PIPELINE")
print("=" * 70)

t_start = time.time()
torch.set_num_threads(8)

# 1. Load Data Splits and Existing Feature Datasets (Model A features)
print("Loading datasets and Phase 6A reference features...")
train_raw = pd.read_csv('data/processed/candidate_train.csv')
val_raw   = pd.read_csv('data/processed/candidate_validation.csv')
test_raw  = pd.read_csv('data/processed/candidate_test.csv')
off_raw   = pd.read_csv('data/raw/test.csv')

feat_train_a = pd.read_csv('data/evaluation/supervised_feature_dataset_train.csv')
feat_val_a   = pd.read_csv('data/evaluation/supervised_feature_dataset_validation.csv')
feat_test_a  = pd.read_csv('data/evaluation/supervised_feature_dataset_test.csv')
feat_off_a   = pd.read_csv('data/evaluation/supervised_feature_dataset_official_test.csv')

feature_cols_a = [c for c in feat_train_a.columns if c != 'label']
y_train = feat_train_a['label'].values
y_val   = feat_val_a['label'].values
y_test  = feat_test_a['label'].values
y_off   = feat_off_a['label'].values

print(f"Candidate Train:      {len(train_raw)} pairs")
print(f"Candidate Validation: {len(val_raw)} pairs")
print(f"Candidate Test:       {len(test_raw)} pairs")
print(f"Official Test:        {len(off_raw)} pairs")

# 2. Cross-Encoder Model Loading
print("\nLoading cross-encoder/ms-marco-MiniLM-L-6-v2 ...")
t0_model = time.time()
ce_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
t_load = time.time() - t0_model
print(f"Model loaded in {t_load:.2f}s.")

# 3. Compute or Load Cross-Encoder Scores with Checkpointing
def get_or_compute_ce_scores(df, cache_path, split_name):
    if os.path.exists(cache_path):
        print(f"Loading cached Cross-Encoder scores for {split_name} from {cache_path}...")
        return np.load(cache_path)

    print(f"Computing Cross-Encoder scores for {split_name} ({len(df)} pairs)...")
    pairs = list(zip(df['resume_text'], df['job_description_text']))
    t0 = time.time()
    scores = ce_model.predict(pairs, batch_size=64, show_progress_bar=True)
    dt = time.time() - t0
    print(f"Finished {split_name} in {dt:.2f}s ({dt/len(df)*1000:.1f}ms/pair).")
    np.save(cache_path, scores)
    return scores

train_ce_scores = get_or_compute_ce_scores(train_raw, 'data/evaluation/phase7_ce_scores_train.npy', 'candidate_train')
val_ce_scores   = get_or_compute_ce_scores(val_raw,   'data/evaluation/phase7_ce_scores_validation.npy', 'candidate_val')
test_ce_scores  = get_or_compute_ce_scores(test_raw,  'data/evaluation/phase7_ce_scores_test.npy', 'candidate_test')
off_ce_scores   = get_or_compute_ce_scores(off_raw,   'data/evaluation/phase7_ce_scores_official_test.npy', 'official_test')

print("All Cross-Encoder scores ready!")

# 4. Feature Matrices Construction
# Model A: Reference (22 features)
X_train_a = feat_train_a[feature_cols_a]
X_val_a   = feat_val_a[feature_cols_a]
X_test_a  = feat_test_a[feature_cols_a]
X_off_a   = feat_off_a[feature_cols_a]

# Model B: Cross-Encoder Only (1 feature)
X_train_b = pd.DataFrame({'cross_encoder_score': train_ce_scores})
X_val_b   = pd.DataFrame({'cross_encoder_score': val_ce_scores})
X_test_b  = pd.DataFrame({'cross_encoder_score': test_ce_scores})
X_off_b   = pd.DataFrame({'cross_encoder_score': off_ce_scores})

# Model C: 22 features + cross_encoder_score (23 features)
X_train_c = feat_train_a[feature_cols_a].copy()
X_train_c['cross_encoder_score'] = train_ce_scores

X_val_c = feat_val_a[feature_cols_a].copy()
X_val_c['cross_encoder_score'] = val_ce_scores

X_test_c = feat_test_a[feature_cols_a].copy()
X_test_c['cross_encoder_score'] = test_ce_scores

X_off_c = feat_off_a[feature_cols_a].copy()
X_off_c['cross_encoder_score'] = off_ce_scores

# 5. Complementarity Analysis (Correlations)
print("\n" + "=" * 70)
print("FEATURE CORRELATION & COMPLEMENTARITY ANALYSIS (CANDIDATE TRAIN)")
print("=" * 70)

corr_data = pd.DataFrame({
    'Cross_Encoder': train_ce_scores,
    'MiniLM': feat_train_a['sbert_cosine_similarity'],
    'TF-IDF': feat_train_a['tfidf_cosine_similarity'],
    'Skill_Coverage': feat_train_a['skill_coverage'],
    'Jaccard': feat_train_a['jaccard_similarity']
})

p_corr = corr_data.corr(method='pearson')
s_corr = corr_data.corr(method='spearman')

print("\n--- Pearson Correlation Matrix ---")
print(p_corr.round(4))
print("\n--- Spearman Rank Correlation Matrix ---")
print(s_corr.round(4))

corr_export = []
for c1 in corr_data.columns:
    for c2 in corr_data.columns:
        if c1 < c2:
            corr_export.append({
                'feature_1': c1,
                'feature_2': c2,
                'pearson_correlation': round(p_corr.loc[c1, c2], 4),
                'spearman_correlation': round(s_corr.loc[c1, c2], 4)
            })
corr_df = pd.DataFrame(corr_export)
corr_df.to_csv('data/evaluation/phase7_cross_encoder_correlations.csv', index=False)
print("Saved data/evaluation/phase7_cross_encoder_correlations.csv")

# Plot Cross-Encoder Distribution by class
fig, ax = plt.subplots(figsize=(8, 5))
for lbl, color in [('No Fit', 'red'), ('Potential Fit', 'orange'), ('Good Fit', 'green')]:
    mask = (y_test == lbl)
    sns.kdeplot(test_ce_scores[mask], ax=ax, label=lbl, color=color, fill=True, alpha=0.25)
ax.set_title('Cross-Encoder Score Distribution by True Class\n(Candidate Test, N=1,201)')
ax.set_xlabel('Cross-Encoder Score (ms-marco-MiniLM-L-6-v2)')
ax.set_ylabel('Density')
ax.legend()
plt.tight_layout()
plt.savefig('data/evaluation/plots/phase7_cross_encoder_distribution.png', dpi=300)
plt.close()
print("Saved data/evaluation/plots/phase7_cross_encoder_distribution.png")

# 6. Controlled Classifier Training & Evaluation
label_order = ['No Fit', 'Potential Fit', 'Good Fit']

def train_and_eval(X_tr, X_va, X_te, X_of, m_name):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=0.1, class_weight='balanced', max_iter=2000, random_state=42))
    ])
    pipe.fit(X_tr, y_train)

    p_va = pipe.predict(X_va)
    p_te = pipe.predict(X_te)
    p_of = pipe.predict(X_of)

    def get_metrics(y_t, y_p):
        acc = accuracy_score(y_t, y_p)
        p_m = precision_score(y_t, y_p, labels=label_order, average='macro', zero_division=0)
        r_m = recall_score(y_t, y_p, labels=label_order, average='macro', zero_division=0)
        f_m = f1_score(y_t, y_p, labels=label_order, average='macro', zero_division=0)
        cr  = classification_report(y_t, y_p, labels=label_order, output_dict=True, digits=4)
        cm  = confusion_matrix(y_t, y_p, labels=label_order)
        return acc, p_m, r_m, f_m, cr, cm

    return pipe, (p_va, p_te, p_of), (get_metrics(y_val, p_va), get_metrics(y_test, p_te), get_metrics(y_off, p_of))

print("\n" + "=" * 70)
print("TRAINING & EVALUATING CONTROLLED MODELS")
print("=" * 70)

pipe_a, preds_a, m_a = train_and_eval(X_train_a, X_val_a, X_test_a, X_off_a, "Model A (Reference 22-feat)")
pipe_b, preds_b, m_b = train_and_eval(X_train_b, X_val_b, X_test_b, X_off_b, "Model B (Cross-Encoder Only)")
pipe_c, preds_c, m_c = train_and_eval(X_train_c, X_val_c, X_test_c, X_off_c, "Model C (Cross-Encoder Augmented 23-feat)")

# Save Model C pipeline
joblib.dump(pipe_c, 'models/supervised_logistic_regression_cross_encoder.joblib')

# Build metrics summary table
comparison_rows = []
for m_name, (v_m, t_m, o_m) in [
    ('Model A (Reference 22-feat)', m_a),
    ('Model B (Cross-Encoder Only)', m_b),
    ('Model C (Cross-Encoder Augmented 23-feat)', m_c)
]:
    comparison_rows.append({
        'model': m_name,
        'val_accuracy': round(v_m[0], 4),
        'val_macro_precision': round(v_m[1], 4),
        'val_macro_recall': round(v_m[2], 4),
        'val_macro_f1': round(v_m[3], 4),
        'cand_test_accuracy': round(t_m[0], 4),
        'cand_test_macro_precision': round(t_m[1], 4),
        'cand_test_macro_recall': round(t_m[2], 4),
        'cand_test_macro_f1': round(t_m[3], 4),
        'off_test_accuracy': round(o_m[0], 4),
        'off_test_macro_precision': round(o_m[1], 4),
        'off_test_macro_recall': round(o_m[2], 4),
        'off_test_macro_f1': round(o_m[3], 4)
    })

comp_df = pd.DataFrame(comparison_rows)
comp_df.to_csv('data/evaluation/phase7_cross_encoder_metrics.csv', index=False)
print("Saved data/evaluation/phase7_cross_encoder_metrics.csv")

print("\n--- PERFORMANCE SUMMARY TABLE ---")
print(comp_df[['model', 'val_macro_f1', 'cand_test_macro_f1', 'cand_test_accuracy', 'off_test_macro_f1', 'off_test_accuracy']].to_string(index=False))

# Per-Class Metrics Table
class_rows = []
for split_name, s_idx in [('Candidate Test', 1), ('Official Test', 2)]:
    for m_name, m_tuple in [
        ('Model A (Reference 22-feat)', m_a),
        ('Model B (Cross-Encoder Only)', m_b),
        ('Model C (Cross-Encoder Augmented 23-feat)', m_c)
    ]:
        cr = m_tuple[s_idx][4]
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
print("\n--- CANDIDATE TEST PER-CLASS BREAKDOWN ---")
print(class_df[class_df['dataset'] == 'Candidate Test'].to_string(index=False))

# Plot Comparison Bar Chart
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(3)
width = 0.25

rects1 = ax.bar(x - width, comp_df['val_macro_f1'], width, label='Validation Macro F1', color='#4C72B0')
rects2 = ax.bar(x, comp_df['cand_test_macro_f1'], width, label='Candidate Test Macro F1', color='#55A868')
rects3 = ax.bar(x + width, comp_df['off_test_macro_f1'], width, label='Official Test Macro F1', color='#C44E52')

ax.set_ylabel('Macro F1 Score')
ax.set_title('Phase 7: Cross-Encoder Controlled Evaluation Across Splits')
ax.set_xticks(x)
ax.set_xticklabels(['Model A\n(Reference)', 'Model B\n(CE Only)', 'Model C\n(CE Augmented)'])
ax.set_ylim(0.30, 0.52)
ax.legend()

for rects in [rects1, rects2, rects3]:
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f'{h:.4f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('data/evaluation/plots/phase7_model_comparison.png', dpi=300)
plt.close()
print("Saved data/evaluation/plots/phase7_model_comparison.png")

# 7. Error Recovery Analysis on Candidate Test (Model A vs Model C)
test_eval_df = test_raw.copy()
test_eval_df['pred_a'] = preds_a[1]
test_eval_df['pred_c'] = preds_c[1]
test_eval_df['correct_a'] = (test_eval_df['label'] == test_eval_df['pred_a'])
test_eval_df['correct_c'] = (test_eval_df['label'] == test_eval_df['pred_c'])

both_correct = test_eval_df[(test_eval_df['correct_a']) & (test_eval_df['correct_c'])]
both_wrong   = test_eval_df[(~test_eval_df['correct_a']) & (~test_eval_df['correct_c'])]
recovered    = test_eval_df[(~test_eval_df['correct_a']) & (test_eval_df['correct_c'])]
regressed    = test_eval_df[(test_eval_df['correct_a']) & (~test_eval_df['correct_c'])]

print("\n--- CANDIDATE TEST TRANSITION MATRIX (MODEL A -> MODEL C) ---")
print(f"Both Correct: {len(both_correct)} ({len(both_correct)/len(test_raw)*100:.1f}%)")
print(f"Both Wrong:   {len(both_wrong)} ({len(both_wrong)/len(test_raw)*100:.1f}%)")
print(f"Recovered (Model A Wrong -> Model C Correct): {len(recovered)} ({len(recovered)/len(test_raw)*100:.1f}%)")
print(f"Regressed (Model A Correct -> Model C Wrong): {len(regressed)} ({len(regressed)/len(test_raw)*100:.1f}%)")
print(f"Net Change: {len(recovered) - len(regressed):+d} cases")

def sample_transitions(df_sub, cat_name, n=10):
    sampled = df_sub.head(n)
    rows = []
    for idx, r in sampled.iterrows():
        rows.append({
            'category': cat_name,
            'pair_id': idx,
            'true_label': r['label'],
            'pred_a': r['pred_a'],
            'pred_c': r['pred_c'],
            'resume_snippet': r['resume_text'][:160].replace('\n', ' ') + '...',
            'jd_snippet': r['job_description_text'][:160].replace('\n', ' ') + '...',
            'minilm_sim': round(feat_test_a['sbert_cosine_similarity'].iloc[idx], 4),
            'cross_encoder_score': round(test_ce_scores[idx], 4),
            'diagnostic_explanation': (
                f"Shifted from {r['pred_a']} to {r['pred_c']}. "
                f"MiniLM similarity: {feat_test_a['sbert_cosine_similarity'].iloc[idx]:.4f}, "
                f"Cross-Encoder score: {test_ce_scores[idx]:.4f}."
            )
        })
    return rows

recovery_rows = []
recovery_rows.extend(sample_transitions(recovered, 'Recovered (A wrong -> C correct)', 10))
recovery_rows.extend(sample_transitions(regressed, 'Regressed (A correct -> C wrong)', 10))
recovery_rows.extend(sample_transitions(both_wrong, 'Both Wrong', 10))

recovery_df = pd.DataFrame(recovery_rows)
recovery_df.to_csv('data/evaluation/phase7_cross_encoder_error_recovery.csv', index=False)
print("Saved data/evaluation/phase7_cross_encoder_error_recovery.csv")

print(f"\nPhase 7 pipeline finished in {time.time() - t_start:.2f}s.")
