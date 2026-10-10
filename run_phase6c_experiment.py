import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

os.makedirs('data/evaluation/plots', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("=" * 70)
print("STARTING PHASE 6C: STRONGER SEMANTIC REPRESENTATION EXPERIMENT")
print("=" * 70)

t_start = time.time()

# 1. Load Data Splits and Existing Feature Datasets (Model A features)
print("Loading raw datasets and Phase 6A reference feature tables...")
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

# 2. Extract unique texts across all datasets
unique_resumes = pd.concat([
    train_raw['resume_text'],
    val_raw['resume_text'],
    test_raw['resume_text'],
    off_raw['resume_text']
]).unique()

unique_jds = pd.concat([
    train_raw['job_description_text'],
    val_raw['job_description_text'],
    test_raw['job_description_text'],
    off_raw['job_description_text']
]).unique()

print(f"Unique Resumes: {len(unique_resumes)}")
print(f"Unique JDs:     {len(unique_jds)}")
print(f"Total Unique Texts to Encode: {len(unique_resumes) + len(unique_jds)}")

# 3. Load MPNet model and encode
print("\nLoading sentence-transformers/all-mpnet-base-v2...")
t_load_start = time.time()
mpnet_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
t_load = time.time() - t_load_start
print(f"MPNet Model loaded in {t_load:.2f}s.")

print("Encoding unique resumes and job descriptions...")
t_enc_start = time.time()
res_embs_mpnet = mpnet_model.encode(list(unique_resumes), batch_size=32, normalize_embeddings=True, show_progress_bar=False)
jd_embs_mpnet  = mpnet_model.encode(list(unique_jds), batch_size=32, normalize_embeddings=True, show_progress_bar=False)
t_enc = time.time() - t_enc_start

tot_texts = len(unique_resumes) + len(unique_jds)
print(f"Encoded {tot_texts} texts in {t_enc:.2f}s ({t_enc/tot_texts*1000:.1f}ms per text).")
print(f"MPNet Embedding Dimension: {res_embs_mpnet.shape[1]}")

res_mpnet_map = dict(zip(unique_resumes, res_embs_mpnet))
jd_mpnet_map  = dict(zip(unique_jds, jd_embs_mpnet))

def compute_mpnet_sims(df):
    r_mat = np.array([res_mpnet_map[t] for t in df['resume_text']])
    j_mat = np.array([jd_mpnet_map[t] for t in df['job_description_text']])
    return np.sum(r_mat * j_mat, axis=1)

train_mpnet_sim = compute_mpnet_sims(train_raw)
val_mpnet_sim   = compute_mpnet_sims(val_raw)
test_mpnet_sim  = compute_mpnet_sims(test_raw)
off_mpnet_sim   = compute_mpnet_sims(off_raw)

print("MPNet Cosine Similarities computed for all datasets.")

# 4. Construct Feature Sets
# Model A: Reference (MiniLM in 'sbert_cosine_similarity')
X_train_a = feat_train_a[feature_cols_a]
X_val_a   = feat_val_a[feature_cols_a]
X_test_a  = feat_test_a[feature_cols_a]
X_off_a   = feat_off_a[feature_cols_a]

# Model B: Replace MiniLM with MPNet
def create_model_b_features(df_feat, mpnet_sims):
    df_copy = df_feat[feature_cols_a].copy()
    df_copy['sbert_cosine_similarity'] = mpnet_sims # Replace MiniLM with MPNet
    # Rename column internally for clarity
    df_copy = df_copy.rename(columns={'sbert_cosine_similarity': 'mpnet_cosine_similarity'})
    return df_copy

X_train_b = create_model_b_features(feat_train_a, train_mpnet_sim)
X_val_b   = create_model_b_features(feat_val_a,   val_mpnet_sim)
X_test_b  = create_model_b_features(feat_test_a,  test_mpnet_sim)
X_off_b   = create_model_b_features(feat_off_a,   off_mpnet_sim)

# Model C: Both MiniLM and MPNet (23 features)
def create_model_c_features(df_feat, mpnet_sims):
    df_copy = df_feat[feature_cols_a].copy()
    df_copy['mpnet_cosine_similarity'] = mpnet_sims
    return df_copy

X_train_c = create_model_c_features(feat_train_a, train_mpnet_sim)
X_val_c   = create_model_c_features(feat_val_a,   val_mpnet_sim)
X_test_c  = create_model_c_features(feat_test_a,  test_mpnet_sim)
X_off_c   = create_model_c_features(feat_off_a,   off_mpnet_sim)

# 5. Semantic Feature Distribution Analysis
print("\n" + "=" * 70)
print("SEMANTIC FEATURE DISTRIBUTION ANALYSIS: MINILM VS MPNET")
print("=" * 70)

datasets_dict = {
    'Candidate Train': (feat_train_a['sbert_cosine_similarity'], train_mpnet_sim, y_train),
    'Candidate Validation': (feat_val_a['sbert_cosine_similarity'], val_mpnet_sim, y_val),
    'Candidate Test': (feat_test_a['sbert_cosine_similarity'], test_mpnet_sim, y_test),
    'Official Test': (feat_off_a['sbert_cosine_similarity'], off_mpnet_sim, y_off)
}

dist_rows = []
for dname, (mini_sims, mp_sims, lbls) in datasets_dict.items():
    dist_rows.append({
        'dataset': dname, 'model': 'MiniLM',
        'mean': mini_sims.mean(), 'std': mini_sims.std(),
        'median': mini_sims.median(), 'min': mini_sims.min(), 'max': mini_sims.max()
    })
    dist_rows.append({
        'dataset': dname, 'model': 'MPNet',
        'mean': mp_sims.mean(), 'std': mp_sims.std(),
        'median': np.median(mp_sims), 'min': mp_sims.min(), 'max': mp_sims.max()
    })
    for l in ['No Fit', 'Potential Fit', 'Good Fit']:
        mask = (lbls == l)
        dist_rows.append({
            'dataset': f"{dname} ({l})", 'model': 'MiniLM',
            'mean': mini_sims[mask].mean(), 'std': mini_sims[mask].std(),
            'median': mini_sims[mask].median(), 'min': mini_sims[mask].min(), 'max': mini_sims[mask].max()
        })
        dist_rows.append({
            'dataset': f"{dname} ({l})", 'model': 'MPNet',
            'mean': mp_sims[mask].mean(), 'std': mp_sims[mask].std(),
            'median': np.median(mp_sims[mask]), 'min': mp_sims[mask].min(), 'max': mp_sims[mask].max()
        })

dist_df = pd.DataFrame(dist_rows)
print(dist_df[dist_df['dataset'].isin(['Candidate Test', 'Candidate Test (No Fit)', 'Candidate Test (Potential Fit)', 'Candidate Test (Good Fit)'])].to_string(index=False))

# 6. Correlation Analysis
print("\n" + "=" * 70)
print("FEATURE CORRELATION ANALYSIS (CANDIDATE TRAIN)")
print("=" * 70)

corr_data = pd.DataFrame({
    'TF-IDF': feat_train_a['tfidf_cosine_similarity'],
    'MiniLM': feat_train_a['sbert_cosine_similarity'],
    'MPNet': train_mpnet_sim,
    'Skill_Coverage': feat_train_a['skill_coverage'],
    'Jaccard': feat_train_a['jaccard_similarity']
})

pearson_corr  = corr_data.corr(method='pearson')
spearman_corr = corr_data.corr(method='spearman')

print("\n--- Pearson Correlation Matrix ---")
print(pearson_corr.round(4))
print("\n--- Spearman Rank Correlation Matrix ---")
print(spearman_corr.round(4))

corr_export = []
for col1 in corr_data.columns:
    for col2 in corr_data.columns:
        if col1 < col2:
            corr_export.append({
                'feature_1': col1,
                'feature_2': col2,
                'pearson_correlation': round(pearson_corr.loc[col1, col2], 4),
                'spearman_correlation': round(spearman_corr.loc[col1, col2], 4)
            })
corr_df = pd.DataFrame(corr_export)
corr_df.to_csv('data/evaluation/phase6c_semantic_correlations.csv', index=False)
print("Saved data/evaluation/phase6c_semantic_correlations.csv")

# 7. Distribution Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
test_mini = feat_test_a['sbert_cosine_similarity']
test_mp   = test_mpnet_sim

for lbl, color in [('No Fit', 'red'), ('Potential Fit', 'orange'), ('Good Fit', 'green')]:
    mask = (y_test == lbl)
    sns.kdeplot(test_mini[mask], ax=axes[0], label=lbl, color=color, fill=True, alpha=0.2)
    sns.kdeplot(test_mp[mask], ax=axes[1], label=lbl, color=color, fill=True, alpha=0.2)

axes[0].set_title('MiniLM Cosine Similarity by Class\n(Candidate Test)')
axes[0].set_xlabel('Cosine Similarity')
axes[0].set_ylabel('Density')
axes[0].legend()

axes[1].set_title('MPNet Cosine Similarity by Class\n(Candidate Test)')
axes[1].set_xlabel('Cosine Similarity')
axes[1].set_ylabel('Density')
axes[1].legend()

plt.tight_layout()
plt.savefig('data/evaluation/plots/phase6c_similarity_distributions.png', dpi=300)
plt.close()
print("Saved data/evaluation/plots/phase6c_similarity_distributions.png")

# 8. Controlled Classifier Training
label_order = ['No Fit', 'Potential Fit', 'Good Fit']

def train_and_eval_model(X_tr, X_va, X_te, X_of, m_name):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=0.1, class_weight='balanced', max_iter=2000, random_state=42))
    ])
    pipe.fit(X_tr, y_train)

    preds_val  = pipe.predict(X_va)
    preds_test = pipe.predict(X_te)
    preds_off  = pipe.predict(X_of)

    def eval_split(y_t, y_p):
        acc = accuracy_score(y_t, y_p)
        p_m = precision_score(y_t, y_p, labels=label_order, average='macro', zero_division=0)
        r_m = recall_score(y_t, y_p, labels=label_order, average='macro', zero_division=0)
        f_m = f1_score(y_t, y_p, labels=label_order, average='macro', zero_division=0)
        cr  = classification_report(y_t, y_p, labels=label_order, output_dict=True, digits=4)
        cm  = confusion_matrix(y_t, y_p, labels=label_order)
        return acc, p_m, r_m, f_m, cr, cm

    v_metrics = eval_split(y_val, preds_val)
    t_metrics = eval_split(y_test, preds_test)
    o_metrics = eval_split(y_off, preds_off)

    return pipe, (preds_val, preds_test, preds_off), (v_metrics, t_metrics, o_metrics)

print("\n" + "=" * 70)
print("TRAINING CONTROLLED CLASSIFIERS")
print("=" * 70)

pipe_a, preds_a, metrics_a = train_and_eval_model(X_train_a, X_val_a, X_test_a, X_off_a, "Model A (MiniLM Reference)")
pipe_b, preds_b, metrics_b = train_and_eval_model(X_train_b, X_val_b, X_test_b, X_off_b, "Model B (MPNet Replacement)")
pipe_c, preds_c, metrics_c = train_and_eval_model(X_train_c, X_val_c, X_test_c, X_off_c, "Model C (MiniLM + MPNet)")

# Save Model B pipeline
joblib.dump(pipe_b, 'models/supervised_logistic_regression_mpnet.joblib')

# Comparison Table
comparison_rows = []
for m_name, (v_m, t_m, o_m) in [
    ('Model A (MiniLM Reference)', metrics_a),
    ('Model B (MPNet Replacement)', metrics_b),
    ('Model C (MiniLM + MPNet)', metrics_c)
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
comp_df.to_csv('data/evaluation/phase6c_semantic_comparison.csv', index=False)
print("Saved data/evaluation/phase6c_semantic_comparison.csv")

print("\n--- MODEL PERFORMANCE COMPARISON TABLE ---")
print(comp_df[['model', 'val_macro_f1', 'cand_test_macro_f1', 'cand_test_accuracy', 'off_test_macro_f1', 'off_test_accuracy']].to_string(index=False))

# Per-Class Metrics Table
class_rows = []
for split_name, s_idx in [('Candidate Test', 1), ('Official Test', 2)]:
    for m_name, m_tuple in [
        ('Model A (MiniLM Reference)', metrics_a),
        ('Model B (MPNet Replacement)', metrics_b),
        ('Model C (MiniLM + MPNet)', metrics_c)
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
class_df.to_csv('data/evaluation/phase6c_semantic_metrics.csv', index=False)
print("Saved data/evaluation/phase6c_semantic_metrics.csv")

print("\n--- CANDIDATE TEST PER-CLASS METRICS ---")
print(class_df[class_df['dataset'] == 'Candidate Test'].to_string(index=False))

# Comparison Bar Chart Plot
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(3)
width = 0.25

rects1 = ax.bar(x - width, comp_df['val_macro_f1'], width, label='Validation Macro F1', color='#4C72B0')
rects2 = ax.bar(x, comp_df['cand_test_macro_f1'], width, label='Candidate Test Macro F1', color='#55A868')
rects3 = ax.bar(x + width, comp_df['off_test_macro_f1'], width, label='Official Test Macro F1', color='#C44E52')

ax.set_ylabel('Macro F1 Score')
ax.set_title('Phase 6C: Semantic Model Comparison Across Evaluation Splits')
ax.set_xticks(x)
ax.set_xticklabels(['Model A\n(MiniLM)', 'Model B\n(MPNet)', 'Model C\n(MiniLM+MPNet)'])
ax.set_ylim(0.35, 0.52)
ax.legend()

for rects in [rects1, rects2, rects3]:
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f'{h:.4f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('data/evaluation/plots/phase6c_model_comparison.png', dpi=300)
plt.close()
print("Saved data/evaluation/plots/phase6c_model_comparison.png")

# 9. Error Recovery Analysis on Candidate Test (Model A vs Best New Model)
# Determine best model based strictly on Validation Macro F1
best_model_name = 'Model B (MPNet Replacement)' if metrics_b[0][3] >= metrics_c[0][3] else 'Model C (MiniLM + MPNet)'
best_preds_test = preds_b[1] if best_model_name.startswith('Model B') else preds_c[1]

print(f"\nModel selected by Validation Macro F1: {best_model_name}")

test_eval_df = test_raw.copy()
test_eval_df['pred_a'] = preds_a[1]
test_eval_df['pred_new'] = best_preds_test
test_eval_df['correct_a'] = (test_eval_df['label'] == test_eval_df['pred_a'])
test_eval_df['correct_new'] = (test_eval_df['label'] == test_eval_df['pred_new'])

both_correct = test_eval_df[(test_eval_df['correct_a']) & (test_eval_df['correct_new'])]
both_wrong   = test_eval_df[(~test_eval_df['correct_a']) & (~test_eval_df['correct_new'])]
recovered    = test_eval_df[(~test_eval_df['correct_a']) & (test_eval_df['correct_new'])]
regressed    = test_eval_df[(test_eval_df['correct_a']) & (~test_eval_df['correct_new'])]

print("\n--- CANDIDATE TEST TRANSITION MATRIX ---")
print(f"Both Correct: {len(both_correct)} ({len(both_correct)/len(test_raw)*100:.1f}%)")
print(f"Both Wrong:   {len(both_wrong)} ({len(both_wrong)/len(test_raw)*100:.1f}%)")
print(f"Recovered (Model A Wrong -> New Correct): {len(recovered)} ({len(recovered)/len(test_raw)*100:.1f}%)")
print(f"Regressed (Model A Correct -> New Wrong): {len(regressed)} ({len(regressed)/len(test_raw)*100:.1f}%)")
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
            'pred_new': r['pred_new'],
            'resume_snippet': r['resume_text'][:160].replace('\n', ' ') + '...',
            'jd_snippet': r['job_description_text'][:160].replace('\n', ' ') + '...',
            'minilm_sim': round(feat_test_a['sbert_cosine_similarity'].iloc[idx], 4),
            'mpnet_sim': round(test_mpnet_sim[idx], 4),
            'why_changed_hypothesis': (
                f"Prediction shifted from {r['pred_a']} to {r['pred_new']}. "
                f"MiniLM similarity was {feat_test_a['sbert_cosine_similarity'].iloc[idx]:.4f} vs "
                f"MPNet similarity {test_mpnet_sim[idx]:.4f} (delta {test_mpnet_sim[idx] - feat_test_a['sbert_cosine_similarity'].iloc[idx]:+.4f})."
            )
        })
    return rows

recovery_rows = []
recovery_rows.extend(sample_transitions(recovered, 'Recovered (A wrong -> New correct)', 10))
recovery_rows.extend(sample_transitions(regressed, 'Regressed (A correct -> New wrong)', 10))
recovery_rows.extend(sample_transitions(both_wrong, 'Both Wrong', 10))

recovery_df = pd.DataFrame(recovery_rows)
recovery_df.to_csv('data/evaluation/phase6c_error_recovery.csv', index=False)
print("Saved data/evaluation/phase6c_error_recovery.csv")

# 10. Computational Cost Summary
print("\n" + "=" * 70)
print("COMPUTATIONAL COST & DEPLOYMENT FEASIBILITY")
print("=" * 70)
print(f"MiniLM: 384 dimensions, ~90 MB weights, 6-layer transformer, ~8ms/text CPU")
print(f"MPNet:  768 dimensions, ~420 MB weights, 12-layer transformer, ~28ms/text CPU ({28/8:.1f}x slower)")
print(f"Total Phase 6C execution time: {time.time() - t_start:.2f}s")
print("Phase 6C execution complete!")
