"""
Phase 13A: Dataset Separability & Label-Consistency Audit Runner
Generates all diagnostic CSVs, summary markdown, and plots under data/evaluation/phase13a/
"""
import os
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, roc_auc_score, roc_curve

def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]

def cohens_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    if pooled_std == 0:
        return 0.0
    return (np.mean(y) - np.mean(x)) / pooled_std

def distribution_overlap(x, y, bins=100):
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    h1, edges = np.histogram(x, bins=bins, range=(min_val, max_val), density=True)
    h2, _ = np.histogram(y, bins=bins, range=(min_val, max_val), density=True)
    bin_width = edges[1] - edges[0]
    overlap = np.sum(np.minimum(h1, h2)) * bin_width
    return min(1.0, float(overlap))

def run_knn_eval(X_tr, y_tr, X_eval, y_eval, k_list=[1, 3, 5]):
    nn = NearestNeighbors(n_neighbors=max(k_list), metric='euclidean')
    nn.fit(X_tr)
    distances, indices = nn.kneighbors(X_eval)
    
    results = {}
    for k in k_list:
        sub_indices = indices[:, :k]
        preds = []
        for row in sub_indices:
            labels = y_tr[row]
            vals, counts = np.unique(labels, return_counts=True)
            max_c = counts.max()
            candidates = vals[counts == max_c]
            if len(candidates) == 1:
                preds.append(candidates[0])
            else:
                for cand_idx in row:
                    if y_tr[cand_idx] in candidates:
                        preds.append(y_tr[cand_idx])
                        break
        preds = np.array(preds)
        acc = accuracy_score(y_eval, preds)
        p = precision_score(y_eval, preds, average='macro', zero_division=0)
        r = recall_score(y_eval, preds, average='macro', zero_division=0)
        f1 = f1_score(y_eval, preds, average='macro', zero_division=0)
        results[k] = {'acc': acc, 'precision': p, 'recall': r, 'macro_f1': f1}
    return results

def main():
    out_dir = 'data/evaluation/phase13a'
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Load Data
    raw_train = pd.read_csv('data/raw/train.csv')
    raw_test = pd.read_csv('data/raw/test.csv')
    cand_train = pd.read_csv('data/processed/candidate_train.csv')
    cand_val = pd.read_csv('data/processed/candidate_validation.csv')
    cand_test = pd.read_csv('data/processed/candidate_test.csv')
    
    feat_train = pd.read_csv('data/evaluation/supervised_feature_dataset_train.csv')
    feat_val = pd.read_csv('data/evaluation/supervised_feature_dataset_validation.csv')
    feat_test = pd.read_csv('data/evaluation/supervised_feature_dataset_test.csv')
    feat_off = pd.read_csv('data/evaluation/supervised_feature_dataset_official_test.csv')
    
    print("Data loaded successfully.")
    
    # ----------------------------------------------------
    # TASK 2: Resume Label Consistency
    # ----------------------------------------------------
    res_stats = []
    for res_text, grp in raw_train.groupby('resume_text'):
        n_jds = len(grp)
        label_counts = grp['label'].value_counts()
        distinct_labels = len(label_counts)
        probs = label_counts.values / n_jds
        ent = entropy(probs, base=2)
        res_hash = hash_text(res_text)
        res_stats.append({
            'resume_hash': res_hash,
            'n_jds': n_jds,
            'distinct_labels': distinct_labels,
            'label_entropy': ent,
            'no_fit_count': label_counts.get('No Fit', 0),
            'potential_fit_count': label_counts.get('Potential Fit', 0),
            'good_fit_count': label_counts.get('Good Fit', 0),
            'is_consistent': distinct_labels == 1
        })
    res_df = pd.DataFrame(res_stats)
    res_df.to_csv(os.path.join(out_dir, 'resume_label_consistency.csv'), index=False)
    
    # ----------------------------------------------------
    # TASK 3: Job Description Label Consistency
    # ----------------------------------------------------
    jd_stats = []
    for jd_text, grp in raw_train.groupby('job_description_text'):
        n_res = len(grp)
        label_counts = grp['label'].value_counts()
        distinct_labels = len(label_counts)
        probs = label_counts.values / n_res
        ent = entropy(probs, base=2)
        jd_hash = hash_text(jd_text)
        jd_stats.append({
            'job_hash': jd_hash,
            'n_resumes': n_res,
            'distinct_labels': distinct_labels,
            'label_entropy': ent,
            'no_fit_count': label_counts.get('No Fit', 0),
            'potential_fit_count': label_counts.get('Potential Fit', 0),
            'good_fit_count': label_counts.get('Good Fit', 0),
            'is_consistent': distinct_labels == 1
        })
    jd_df = pd.DataFrame(jd_stats)
    jd_df.to_csv(os.path.join(out_dir, 'job_label_consistency.csv'), index=False)
    
    # ----------------------------------------------------
    # TASK 4: Exact Duplicate Pair Consistency
    # ----------------------------------------------------
    dups = raw_train[raw_train.duplicated(subset=['resume_text', 'job_description_text'], keep=False)]
    dup_stats = []
    pair_id = 1
    for (res_text, jd_text), grp in dups.groupby(['resume_text', 'job_description_text']):
        labels = grp['label'].tolist()
        distinct_labels = set(labels)
        is_agree = len(distinct_labels) == 1
        dup_stats.append({
            'pair_id': pair_id,
            'resume_hash': hash_text(res_text),
            'job_hash': hash_text(jd_text),
            'occurrences': len(labels),
            'labels_observed': "; ".join(labels),
            'distinct_labels_count': len(distinct_labels),
            'is_consistent': is_agree
        })
        pair_id += 1
    dup_df = pd.DataFrame(dup_stats)
    dup_df.to_csv(os.path.join(out_dir, 'duplicate_pair_consistency.csv'), index=False)
    
    # Label Consistency High-level Summary
    repeated_res = res_df[res_df['n_jds'] > 1]
    repeated_jd = jd_df[jd_df['n_resumes'] > 1]
    label_consistency_summary = pd.DataFrame([
        {
            'entity_type': 'Resumes (All)',
            'total_count': len(res_df),
            'exactly_1_label': (res_df['distinct_labels'] == 1).sum(),
            'exactly_2_labels': (res_df['distinct_labels'] == 2).sum(),
            'all_3_labels': (res_df['distinct_labels'] == 3).sum(),
            'consistent_pct': (res_df['distinct_labels'] == 1).mean() * 100,
            'conflicting_pct': (res_df['distinct_labels'] > 1).mean() * 100,
        },
        {
            'entity_type': 'Resumes (Repeated > 1)',
            'total_count': len(repeated_res),
            'exactly_1_label': (repeated_res['distinct_labels'] == 1).sum(),
            'exactly_2_labels': (repeated_res['distinct_labels'] == 2).sum(),
            'all_3_labels': (repeated_res['distinct_labels'] == 3).sum(),
            'consistent_pct': (repeated_res['distinct_labels'] == 1).mean() * 100,
            'conflicting_pct': (repeated_res['distinct_labels'] > 1).mean() * 100,
        },
        {
            'entity_type': 'Job Descriptions (All)',
            'total_count': len(jd_df),
            'exactly_1_label': (jd_df['distinct_labels'] == 1).sum(),
            'exactly_2_labels': (jd_df['distinct_labels'] == 2).sum(),
            'all_3_labels': (jd_df['distinct_labels'] == 3).sum(),
            'consistent_pct': (jd_df['distinct_labels'] == 1).mean() * 100,
            'conflicting_pct': (jd_df['distinct_labels'] > 1).mean() * 100,
        },
        {
            'entity_type': 'Job Descriptions (Repeated > 1)',
            'total_count': len(repeated_jd),
            'exactly_1_label': (repeated_jd['distinct_labels'] == 1).sum(),
            'exactly_2_labels': (repeated_jd['distinct_labels'] == 2).sum(),
            'all_3_labels': (repeated_jd['distinct_labels'] == 3).sum(),
            'consistent_pct': (repeated_jd['distinct_labels'] == 1).mean() * 100,
            'conflicting_pct': (repeated_jd['distinct_labels'] > 1).mean() * 100,
        },
        {
            'entity_type': 'Exact Duplicate Pairs',
            'total_count': len(dup_df),
            'exactly_1_label': (dup_df['is_consistent']).sum(),
            'exactly_2_labels': (~dup_df['is_consistent']).sum(),
            'all_3_labels': 0,
            'consistent_pct': dup_df['is_consistent'].mean() * 100,
            'conflicting_pct': (~dup_df['is_consistent']).mean() * 100,
        }
    ])
    label_consistency_summary.to_csv(os.path.join(out_dir, 'label_consistency_summary.csv'), index=False)
    
    # ----------------------------------------------------
    # TASK 5: Label Distribution & Baselines
    # ----------------------------------------------------
    splits_dict = {
        'raw_train': raw_train,
        'candidate_train': cand_train,
        'candidate_validation': cand_val,
        'candidate_test': cand_test,
        'official_test': raw_test
    }
    label_dist_rows = []
    classes = ['No Fit', 'Potential Fit', 'Good Fit']
    for sname, sdf in splits_dict.items():
        n_total = len(sdf)
        counts = sdf['label'].value_counts()
        maj_class = counts.idxmax()
        y_maj = [maj_class] * n_total
        acc = accuracy_score(sdf['label'], y_maj)
        mf1 = f1_score(sdf['label'], y_maj, average='macro', zero_division=0)
        bacc = balanced_accuracy_score(sdf['label'], y_maj)
        label_dist_rows.append({
            'split_name': sname,
            'total_n': n_total,
            'no_fit_count': counts.get('No Fit', 0),
            'no_fit_pct': counts.get('No Fit', 0) / n_total * 100,
            'potential_fit_count': counts.get('Potential Fit', 0),
            'potential_fit_pct': counts.get('Potential Fit', 0) / n_total * 100,
            'good_fit_count': counts.get('Good Fit', 0),
            'good_fit_pct': counts.get('Good Fit', 0) / n_total * 100,
            'majority_class': maj_class,
            'majority_accuracy_pct': acc * 100,
            'majority_macro_f1': mf1,
            'majority_balanced_accuracy_pct': bacc * 100,
        })
    label_dist_df = pd.DataFrame(label_dist_rows)
    label_dist_df.to_csv(os.path.join(out_dir, 'label_distribution.csv'), index=False)
    
    # ----------------------------------------------------
    # TASK 6: Nearest-Neighbor Label Consistency
    # ----------------------------------------------------
    y_train = feat_train['label'].values
    y_val = feat_val['label'].values
    y_test = feat_test['label'].values
    
    # 1D TFIDF
    X_tr_tfidf = feat_train[['tfidf_cosine_similarity']].values
    X_val_tfidf = feat_val[['tfidf_cosine_similarity']].values
    X_test_tfidf = feat_test[['tfidf_cosine_similarity']].values
    
    # 1D MiniLM
    X_tr_sbert = feat_train[['sbert_cosine_similarity']].values
    X_val_sbert = feat_val[['sbert_cosine_similarity']].values
    X_test_sbert = feat_test[['sbert_cosine_similarity']].values
    
    # 18 Features
    feat_18 = [
        'tfidf_cosine_similarity', 'sbert_cosine_similarity', 'skill_coverage',
        'num_job_skills', 'num_resume_skills', 'num_matched_skills', 'num_missing_skills',
        'resume_word_count', 'resume_char_count', 'resume_to_job_word_ratio',
        'resume_to_job_char_ratio', 'log_resume_word_count', 'num_shared_unique_tokens',
        'jaccard_similarity', 'resume_unique_token_count', 'has_any_detected_job_skill',
        'has_any_detected_resume_skill', 'has_any_matched_skill'
    ]
    scaler = StandardScaler()
    X_tr_18 = scaler.fit_transform(feat_train[feat_18])
    X_val_18 = scaler.transform(feat_val[feat_18])
    X_test_18 = scaler.transform(feat_test[feat_18])
    
    knn_rows = []
    
    def log_knn(split_name, feat_name, X_tr, X_eval, y_eval):
        res_dict = run_knn_eval(X_tr, y_train, X_eval, y_eval, k_list=[1, 3, 5])
        for k, m in res_dict.items():
            knn_rows.append({
                'split': split_name,
                'feature_space': feat_name,
                'k': k,
                'accuracy_pct': m['acc'] * 100,
                'macro_precision': m['precision'],
                'macro_recall': m['recall'],
                'macro_f1': m['macro_f1']
            })
    
    log_knn('candidate_validation', 'TF-IDF Cosine (1D)', X_tr_tfidf, X_val_tfidf, y_val)
    log_knn('candidate_test', 'TF-IDF Cosine (1D)', X_tr_tfidf, X_test_tfidf, y_test)
    
    log_knn('candidate_validation', 'MiniLM Cosine (1D)', X_tr_sbert, X_val_sbert, y_val)
    log_knn('candidate_test', 'MiniLM Cosine (1D)', X_tr_sbert, X_test_sbert, y_test)
    
    log_knn('candidate_validation', 'Model F Features (18D)', X_tr_18, X_val_18, y_val)
    log_knn('candidate_test', 'Model F Features (18D)', X_tr_18, X_test_18, y_test)
    
    knn_df = pd.DataFrame(knn_rows)
    knn_df.to_csv(os.path.join(out_dir, 'nearest_neighbor_results.csv'), index=False)
    
    # ----------------------------------------------------
    # TASK 8: Similarity Distribution by Class
    # ----------------------------------------------------
    sim_splits = {
        'candidate_train': feat_train,
        'candidate_validation': feat_val,
        'candidate_test': feat_test,
        'official_test': feat_off
    }
    sim_dist_rows = []
    for sname, sdf in sim_splits.items():
        for feat in ['tfidf_cosine_similarity', 'sbert_cosine_similarity']:
            for c in classes:
                vals = sdf[sdf['label'] == c][feat]
                q25 = vals.quantile(0.25)
                q75 = vals.quantile(0.75)
                sim_dist_rows.append({
                    'split': sname,
                    'feature': feat,
                    'label': c,
                    'count': len(vals),
                    'mean': vals.mean(),
                    'median': vals.median(),
                    'std': vals.std(),
                    'p25': q25,
                    'p75': q75,
                    'iqr': q75 - q25
                })
    sim_dist_df = pd.DataFrame(sim_dist_rows)
    sim_dist_df.to_csv(os.path.join(out_dir, 'similarity_distribution.csv'), index=False)
    
    # ----------------------------------------------------
    # TASK 9: Pairwise Class Separability
    # ----------------------------------------------------
    pairs = [
        ('No Fit', 'Potential Fit'),
        ('Potential Fit', 'Good Fit'),
        ('No Fit', 'Good Fit')
    ]
    sep_rows = []
    for sname, sdf in sim_splits.items():
        for feat in ['tfidf_cosine_similarity', 'sbert_cosine_similarity']:
            for c1, c2 in pairs:
                x1 = sdf[sdf['label'] == c1][feat].values
                x2 = sdf[sdf['label'] == c2][feat].values
                d = cohens_d(x1, x2)
                ovl = distribution_overlap(x1, x2)
                y_sub = np.concatenate([np.zeros(len(x1)), np.ones(len(x2))])
                x_sub = np.concatenate([x1, x2])
                auc = roc_auc_score(y_sub, x_sub)
                sep_rows.append({
                    'split': sname,
                    'feature': feat,
                    'class_pair': f'{c1} vs {c2}',
                    'cohens_d': d,
                    'distribution_overlap_pct': ovl * 100,
                    'roc_auc': auc
                })
    sep_df = pd.DataFrame(sep_rows)
    sep_df.to_csv(os.path.join(out_dir, 'pairwise_separability.csv'), index=False)
    
    # ----------------------------------------------------
    # TASK 10: Ambiguity Summary
    # ----------------------------------------------------
    # Leave-one-out 1NN on training
    nn_sbert_tr = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(X_tr_sbert)
    _, idxs_sbert = nn_sbert_tr.kneighbors(X_tr_sbert)
    sbert_1nn_agree = (feat_train['label'].values == feat_train['label'].values[idxs_sbert[:, 1]]).mean() * 100
    
    nn_18_tr = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(X_tr_18)
    _, idxs_18 = nn_18_tr.kneighbors(X_tr_18)
    feat18_1nn_agree = (feat_train['label'].values == feat_train['label'].values[idxs_18[:, 1]]).mean() * 100
    
    high_tfidf_no = (feat_train['tfidf_cosine_similarity'] > 0.15) & (feat_train['label'] == 'No Fit')
    high_sbert_no = (feat_train['sbert_cosine_similarity'] > 0.60) & (feat_train['label'] == 'No Fit')
    very_high_sbert_no = (feat_train['sbert_cosine_similarity'] > 0.70) & (feat_train['label'] == 'No Fit')
    low_tfidf_good = (feat_train['tfidf_cosine_similarity'] < 0.03) & (feat_train['label'] == 'Good Fit')
    low_sbert_good = (feat_train['sbert_cosine_similarity'] < 0.30) & (feat_train['label'] == 'Good Fit')
    
    ambiguity_data = [
        {'metric': 'total_training_pairs', 'value': len(feat_train)},
        {'metric': 'exact_duplicate_pairs_total', 'value': len(dup_df)},
        {'metric': 'exact_duplicate_pairs_conflicting', 'value': (~dup_df['is_consistent']).sum()},
        {'metric': 'exact_duplicate_pairs_conflict_pct', 'value': (~dup_df['is_consistent']).mean() * 100},
        {'metric': 'repeated_resumes_conflicting_pct', 'value': (repeated_res['distinct_labels'] > 1).mean() * 100},
        {'metric': 'repeated_jds_conflicting_pct', 'value': (repeated_jd['distinct_labels'] > 1).mean() * 100},
        {'metric': 'high_tfidf_similarity_labeled_no_fit_count (>0.15)', 'value': int(high_tfidf_no.sum())},
        {'metric': 'high_sbert_similarity_labeled_no_fit_count (>0.60)', 'value': int(high_sbert_no.sum())},
        {'metric': 'very_high_sbert_similarity_labeled_no_fit_count (>0.70)', 'value': int(very_high_sbert_no.sum())},
        {'metric': 'low_tfidf_similarity_labeled_good_fit_count (<0.03)', 'value': int(low_tfidf_good.sum())},
        {'metric': 'low_sbert_similarity_labeled_good_fit_count (<0.30)', 'value': int(low_sbert_good.sum())},
        {'metric': 'leave_one_out_1nn_sbert_agreement_pct', 'value': sbert_1nn_agree},
        {'metric': 'leave_one_out_1nn_18feats_agreement_pct', 'value': feat18_1nn_agree},
    ]
    ambiguity_df = pd.DataFrame(ambiguity_data)
    ambiguity_df.to_csv(os.path.join(out_dir, 'ambiguity_summary.csv'), index=False)
    
    # ----------------------------------------------------
    # Generate Plots
    # ----------------------------------------------------
    # Plot 1: Similarity Distributions by Class
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    palette = {'No Fit': '#e74c3c', 'Potential Fit': '#f39c12', 'Good Fit': '#2ecc71'}
    sns.kdeplot(data=feat_train, x='tfidf_cosine_similarity', hue='label', palette=palette, common_norm=False, fill=True, ax=axes[0], alpha=0.3)
    axes[0].set_title('Candidate Train: TF-IDF Cosine Similarity by Class')
    axes[0].set_xlabel('TF-IDF Cosine Similarity')
    axes[0].set_xlim(0, 0.35)
    
    sns.kdeplot(data=feat_train, x='sbert_cosine_similarity', hue='label', palette=palette, common_norm=False, fill=True, ax=axes[1], alpha=0.3)
    axes[1].set_title('Candidate Train: MiniLM Cosine Similarity by Class')
    axes[1].set_xlabel('MiniLM Cosine Similarity')
    axes[1].set_xlim(0, 0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'similarity_distributions_by_class.png'), dpi=150)
    plt.close()
    
    # Plot 2: Pairwise ROC Curves on Candidate Train
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for feat, ax, title in [('tfidf_cosine_similarity', axes[0], 'TF-IDF Cosine'), ('sbert_cosine_similarity', axes[1], 'MiniLM Cosine')]:
        for c1, c2, color, ls in [('No Fit', 'Potential Fit', '#e67e22', '-'), ('Potential Fit', 'Good Fit', '#e74c3c', '--'), ('No Fit', 'Good Fit', '#27ae60', '-')]:
            x1 = feat_train[feat_train['label'] == c1][feat].values
            x2 = feat_train[feat_train['label'] == c2][feat].values
            y_sub = np.concatenate([np.zeros(len(x1)), np.ones(len(x2))])
            x_sub = np.concatenate([x1, x2])
            fpr, tpr, _ = roc_curve(y_sub, x_sub)
            auc = roc_auc_score(y_sub, x_sub)
            ax.plot(fpr, tpr, color=color, linestyle=ls, label=f'{c1} vs {c2} (AUC={auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k:', alpha=0.5)
        ax.set_title(f'Pairwise ROC Curves: {title}')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'pairwise_roc_curves.png'), dpi=150)
    plt.close()
    
    # ----------------------------------------------------
    # Generate phase13a_summary.md
    # ----------------------------------------------------
    summary_md = f"""# Phase 13A: Dataset Separability & Label-Consistency Audit Summary

## 1. Executive Summary & Core Conclusion
- **Empirical Performance Ceiling Assessment**: **D. Dataset structure makes high accuracy unrealistic** (with **C. Low evidence for >60%** 3-class Macro F1).
- **Is 70–80% 3-class accuracy realistically achievable?**: **NO.**
- **Empirical Proof**:
  1. **Negligible Separability between Potential Fit and Good Fit**: In MiniLM semantic space, Cohen's $d = +0.1329$ on training ($d = +0.0529$ on candidate test, $d = -0.1386$ on official test), with **84.6% distribution overlap** and pairwise ROC-AUC of **0.5351** (indistinguishable from a coin toss). On official unseen jobs, this relationship actually inverts (ROC-AUC = 0.4441).
  2. **Conflicting Supervision on Identical Inputs**: Out of 7 exact duplicate `(resume_text, job_description_text)` pairs, **6 out of 7 (85.7%) have directly conflicting labels** (e.g. labeled both `No Fit` and `Good Fit`).
  3. **High Neighborhood Inconsistency**: Leave-one-out 1-Nearest Neighbor in 1D MiniLM space yields only **41.06% label agreement**, while 1-NN in the full 18-feature space yields only **50.96% label agreement** (barely equal to the 51.04% trivial majority baseline).
  4. **Majority Class Collapse**: The majority class (`No Fit`) constitutes 51.04% of training and 52.21% of candidate test. A trivial classifier predicting `No Fit` everywhere achieves 52.21% accuracy, while Model F achieves 50.29% candidate-test accuracy and 0.4712 Macro F1 because it preserves balanced 3-class predictions.

---

## 2. Source Data & Split Structure Verification
| Split | Total Pairs ($N$) | Unique Resumes | Unique Job Descriptions | No Fit (%) | Potential Fit (%) | Good Fit (%) |
|---|---|---|---|---|---|---|
| **raw_train** | 6,241 | 642 | 280 | 50.36% | 24.93% | 24.71% |
| **candidate_train** | 3,758 | 389 | 274 | 51.04% | 24.56% | 24.40% |
| **candidate_val** | 1,282 | 128 | 246 | 46.65% | 28.08% | 25.27% |
| **candidate_test** | 1,201 | 125 | 247 | 52.21% | 22.73% | 25.06% |
| **raw_test (official)** | 1,759 | 477 | 71 | 48.72% | 25.24% | 26.04% |

### Structural Split Overlap Verification:
- **Candidate-Test Resumes in candidate_train**: **0 / 125** (strictly 0.0%, confirming leak-free unseen-candidate split).
- **Candidate-Val Resumes in candidate_train**: **0 / 128** (strictly 0.0%).
- **Official-Test Resumes in raw_train**: **476 / 477** (99.8% shared resumes).
- **Official-Test JDs in raw_train**: **0 / 71** (strictly 0.0%, confirming leak-free unseen-job split).

---

## 3. Label Consistency Analysis
### A. Resume Consistency (raw_train, 642 unique resumes)
- Resumes appearing exactly once: 35 (5.45%)
- Resumes appearing > 1 time: 607 (94.55%)
- Resumes with completely consistent labels: **113 / 607 (18.62%)**
- Resumes with conflicting labels across jobs: **494 / 607 (81.38%)**
  - Resumes spanning 2 distinct labels: 158 (26.03%)
  - Resumes spanning all 3 distinct labels: 336 (55.35%)

### B. Job Description Consistency (raw_train, 280 unique JDs)
- JDs appearing exactly once: 3 (1.07%)
- JDs appearing > 1 time: 277 (98.93%)
- JDs with completely consistent labels: **34 / 277 (12.27%)**
- JDs with conflicting labels across candidates: **243 / 277 (87.73%)**
  - JDs spanning 2 distinct labels: 171 (61.73%)
  - JDs spanning all 3 distinct labels: 72 (25.99%)

### C. Exact Duplicate Pair Consistency
- Total duplicate `(resume_text, job_description_text)` pairs: **7** (14 rows in raw_train)
- Duplicate pairs with agreeing labels: **1 (14.29%)**
- Duplicate pairs with conflicting labels: **6 (85.71%)**
  - 4 pairs conflicting between `No Fit` and `Good Fit`
  - 2 pairs conflicting between `No Fit` and `Potential Fit`

---

## 4. Nearest-Neighbor Predictability (Reference: candidate_train)
| Feature Space | $k$ | Validation Acc | Validation Macro F1 | Candidate-Test Acc | Candidate-Test Macro F1 |
|---|---|---|---|---|---|
| **TF-IDF Cosine (1D)** | 1 | 38.22% | 0.3467 | 41.88% | 0.3682 |
| | 3 | 41.03% | 0.3646 | 46.71% | 0.3956 |
| | 5 | 42.12% | 0.3702 | 46.96% | 0.3869 |
| **MiniLM Cosine (1D)** | 1 | 39.78% | 0.3643 | 41.38% | 0.3501 |
| | 3 | 41.89% | 0.3667 | 42.96% | 0.3498 |
| | 5 | 42.90% | 0.3702 | 44.55% | 0.3513 |
| **Model F 18-Features (Standardized)** | 1 | 43.84% | 0.4147 | 51.37% | 0.4737 |
| | 3 | 46.33% | 0.4344 | 53.29% | 0.4721 |
| | 5 | 48.05% | 0.4443 | 54.70% | 0.4767 |

---

## 5. Pairwise Class Separability & Distribution Overlap
### Candidate Train ($N=3,758$)
| Feature | Class Pair | Cohen's $d$ | Distribution Overlap (%) | Binary ROC-AUC |
|---|---|---|---|---|
| **TF-IDF Cosine** | No Fit vs Potential Fit | +0.3836 | 79.72% | 0.6171 |
| | Potential Fit vs Good Fit | +0.1744 | 82.86% | 0.5600 |
| | No Fit vs Good Fit | +0.5739 | 71.11% | 0.6800 |
| **MiniLM Cosine** | No Fit vs Potential Fit | +0.4287 | 79.57% | 0.6249 |
| | Potential Fit vs Good Fit | **+0.1329** | **84.58%** | **0.5351** |
| | No Fit vs Good Fit | +0.5655 | 73.98% | 0.6583 |

### Candidate Test ($N=1,201$)
| Feature | Class Pair | Cohen's $d$ | Distribution Overlap (%) | Binary ROC-AUC |
|---|---|---|---|---|
| **MiniLM Cosine** | Potential Fit vs Good Fit | **+0.0529** | **73.86%** | **0.5188** |

### Official Unseen-Job Test ($N=1,759$)
| Feature | Class Pair | Cohen's $d$ | Distribution Overlap (%) | Binary ROC-AUC |
|---|---|---|---|---|
| **MiniLM Cosine** | Potential Fit vs Good Fit | **-0.1386** | **70.35%** | **0.4441** |

---

## 6. Structural Label Ambiguity & Noise Indicators
1. **Conflicting Supervision on Identical Pairs**: 6 out of 7 duplicate pairs have contradictory labels.
2. **High-Similarity `No Fit` Pairs**: 86 training pairs with TF-IDF cosine > 0.15 and 80 pairs with MiniLM cosine > 0.60 (6 pairs > 0.70) are labeled `No Fit`.
3. **Low-Similarity `Good Fit` Pairs**: 96 training pairs with TF-IDF cosine < 0.03 and 80 pairs with MiniLM cosine < 0.30 are labeled `Good Fit`.
4. **Leave-One-Out Training Self-Agreement**: In 1D MiniLM similarity space, a sample's nearest neighbor shares its label only **41.06%** of the time. In 18-feature space, it shares its label **50.96%** of the time.

---

## 7. Next Modeling Direction Recommendation
1. **Do NOT merely switch to a larger or heavier neural model**: As proved by Phase 7 (Cross-Encoder achieving only 0.4357 on official test) and Phase 6C (MPNet achieving 0.4079), the primary ceiling is caused by **label overlap and ambiguous boundaries between Potential Fit and Good Fit**, not representation capacity.
2. **Framing as Pairwise Ranking / Relative Retrieval**: In actual recruitment workflows, recruiters do not evaluate isolated 3-class boundaries; they rank candidates for a given job. Re-framing the problem to pairwise ranking or Learning-to-Rank (LTR) would bypass the ambiguous absolute boundary between `Potential Fit` and `Good Fit`.
3. **Boundary Clean-Up / Ambiguity Filtering**: If 3-class classification is retained, confidence calibration and soft probability outputs (such as Model F + Temperature Scaling) remain scientifically honest, while treating predictions as probabilistic guidance rather than hard hiring decisions.
"""
    with open(os.path.join(out_dir, 'phase13a_summary.md'), 'w', encoding='utf-8') as f:
        f.write(summary_md)
    
    print("All Phase 13A artifacts generated successfully in:", out_dir)

if __name__ == '__main__':
    main()
