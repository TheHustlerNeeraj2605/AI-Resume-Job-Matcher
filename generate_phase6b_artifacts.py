import os
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

os.makedirs('data/evaluation/plots', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("Starting Phase 6B Artifact Generation...")

# 1. Load Data
train_df = pd.read_csv('data/processed/candidate_train.csv')
val_df   = pd.read_csv('data/processed/candidate_validation.csv')
test_df  = pd.read_csv('data/processed/candidate_test.csv')
off_df   = pd.read_csv('data/raw/test.csv')

# Section Headings & Boilerplate rules
HEADINGS = [
    (r'(?:job|position|role)\s+(?:summary|overview|description|details|scope)|about\s+the\s+role', 'CORE_SUMMARY'),
    (r'(?:essential\s+)?(?:key\s+|primary\s+)?responsibilities(?:\s+and\s+duties)?|(?:essential\s+)?duties(?:\s+and\s+responsibilities)?|what\s+you(?:(?:\'ll)|\s+will)\s+do|in\s+this\s+role|role\s+(?:and\s+)?responsibilities|day\s+to\s+day', 'CORE_RESPONSIBILITIES'),
    (r'(?:minimum|basic|required|preferred)?\s*qualifications|(?:minimum|basic|required|preferred)?\s*requirements|(?:required|preferred|technical|key)?\s*skills(?:\s+and\s+qualifications)?|(?:required\s+)?experience(?:\s+required|\s+and\s+education)?|education\s+(?:&|and)\s+experience|what\s+you\s+bring|you\s+will\s+be\s+a\s+great\s+fit\s+if|great\s+fit\s+if', 'CORE_REQUIREMENTS'),
    (r'equal\s+(?:employment\s+)?opportunity(?:\s+employer|\s+employment\s+statement)?|eeo\s+statement|affirmative\s+action|reasonable\s+accommodations?', 'BP_EEO'),
    (r'about\s+(?:us|the\s+company|our\s+company|capgemini|the\s+team)|who\s+we\s+are|life\s+at\s+\w+|why\s+(?:work\s+with\s+us|join\s+us)|the\s+company', 'BP_COMPANY'),
    (r'(?:our\s+)?benefits(?:\s+include|\s+package)?|what\s+we\s+offer|perks\s+(?:and\s+benefits)?|total\s+rewards|compensation\s+(?:and\s+benefits|range)?|salary\s+range', 'BP_BENEFITS'),
    (r'how\s+to\s+apply|application\s+(?:instructions|process)|notice\s+to\s+(?:third\s+party|agencies|recruiters)|disclaimer', 'BP_APPLY')
]

combined_heading_regex = re.compile(
    r'(?:^|\n|\.\s+|:\s+)\b(' + '|'.join([p for p, _ in HEADINGS]) + r')\b\s*(?:[:\n\-]|\s{2,})',
    re.IGNORECASE
)

INLINE_BOILERPLATE = [
    r'all\s+qualified\s+applicants\s+will\s+receive\s+consideration\s+for\s+employment\s+without\s+regard\s+to[^\.\n]*(?:race|color|religion|sex|gender|sexual\s+orientation|national\s+origin|disability|veteran)[^\.\n]*[\.\n]?',
    r'we\s+are\s+an\s+equal\s+opportunity\s+employer[^\.\n]*[\.\n]?',
    r'equal\s+opportunity\s+employer(?:\s+minorities\/females\/veterans\/disability)?[\.\n]?',
    r'we\s+do\s+not\s+discriminate\s+on\s+the\s+basis\s+of[^\.\n]*[\.\n]?',
    r'to\s+request\s+(?:an?|reasonable)\s+accommodation[^\.\n]*[\.\n]?',
    r'if\s+you\s+need\s+(?:a\s+)?reasonable\s+accommodation[^\.\n]*[\.\n]?',
    r'click\s+the\s+following\s+link\s+for\s+more\s+information\s+on\s+your\s+rights\s+as\s+an\s+applicant[^\.\n]*',
    r'net2source\s+inc\.\s+is\s+an\s+award-winning\s+total\s+workforce\s+solutions\s+company[^\.\n]*coach\s+to\s+our\s+consultants\.?'
]
compiled_inline_bp = [re.compile(p, re.IGNORECASE) for p in INLINE_BOILERPLATE]

def analyze_and_clean_jd(text):
    if not isinstance(text, str) or not text.strip():
        return "", [], False

    original = text.strip()
    working = original
    detected_patterns = []

    # 1. Inline boilerplate
    for i, bp in enumerate(compiled_inline_bp):
        if bp.search(working):
            detected_patterns.append(f"inline_pattern_{i}")
            working = bp.sub(" ", working)

    # 2. Section segmentation
    matches = list(combined_heading_regex.finditer(working))
    sections_detected = []
    if len(matches) >= 1:
        sections = []
        if matches[0].start() > 0:
            pre_text = working[:matches[0].start()].strip()
            sections.append(('INTRO', pre_text))

        for i in range(len(matches)):
            curr_match = matches[i]
            heading_text = curr_match.group(1).lower()
            start_pos = curr_match.end()
            end_pos = matches[i+1].start() if i + 1 < len(matches) else len(working)
            section_content = working[start_pos:end_pos].strip()

            # Identify heading category
            cat = 'UNKNOWN'
            for p, c in HEADINGS:
                if re.match(p, heading_text, re.IGNORECASE):
                    cat = c
                    break
            sections_detected.append(cat)

            if cat.startswith('CORE_'):
                sections.append(('CORE', heading_text + ": " + section_content))
            elif cat.startswith('BP_'):
                sections.append(('BOILERPLATE', section_content))
            else:
                sections.append(('OTHER', heading_text + ": " + section_content))

        core_chunks = [content for stype, content in sections if stype in ('CORE', 'OTHER')]
        if core_chunks:
            extracted_core = "\n\n".join(core_chunks).strip()
            if len(extracted_core.split()) >= 40 and len(extracted_core) >= 200:
                working = extracted_core

    cleaned = re.sub(r'[ \t]+', ' ', working)
    cleaned = re.sub(r'\n\s*\n+', '\n\n', cleaned).strip()

    fallback_triggered = False
    if len(cleaned) < 120 or len(cleaned) < 0.20 * len(original):
        cleaned = original
        fallback_triggered = True

    return cleaned, list(set(sections_detected + detected_patterns)), fallback_triggered

all_unique_jds = pd.concat([
    train_df['job_description_text'],
    val_df['job_description_text'],
    test_df['job_description_text'],
    off_df['job_description_text']
]).unique()

cleaning_results = {}
for jd in all_unique_jds:
    c_text, sects, fb = analyze_and_clean_jd(jd)
    cleaning_results[jd] = {
        'cleaned_text': c_text,
        'sections_detected': sects,
        'fallback_triggered': fb,
        'orig_words': len(jd.split()),
        'cleaned_words': len(c_text.split()),
        'orig_chars': len(jd),
        'cleaned_chars': len(c_text),
        'pct_retained': (len(c_text.split()) / len(jd.split()) * 100) if len(jd.split()) > 0 else 100.0,
        'is_changed': jd.strip() != c_text.strip()
    }

# Build phase6b_cleaning_statistics.csv
tot_jds = len(all_unique_jds)
changed_count = sum(1 for r in cleaning_results.values() if r['is_changed'])
fb_count = sum(1 for r in cleaning_results.values() if r['fallback_triggered'])
orig_w = [r['orig_words'] for r in cleaning_results.values()]
clean_w = [r['cleaned_words'] for r in cleaning_results.values()]
orig_c = [r['orig_chars'] for r in cleaning_results.values()]
clean_c = [r['cleaned_chars'] for r in cleaning_results.values()]
pct_ret = [r['pct_retained'] for r in cleaning_results.values()]

pattern_counts = {}
for r in cleaning_results.values():
    for p in r['sections_detected']:
        pattern_counts[p] = pattern_counts.get(p, 0) + 1

stats_rows = [
    {'metric': 'total_unique_jds', 'value': tot_jds},
    {'metric': 'modified_jds_count', 'value': changed_count},
    {'metric': 'modified_jds_percentage', 'value': f"{changed_count / tot_jds * 100:.2f}%"},
    {'metric': 'fallback_triggered_count', 'value': fb_count},
    {'metric': 'orig_word_count_mean', 'value': f"{np.mean(orig_w):.2f}"},
    {'metric': 'orig_word_count_median', 'value': f"{np.median(orig_w):.2f}"},
    {'metric': 'orig_word_count_min', 'value': np.min(orig_w)},
    {'metric': 'orig_word_count_max', 'value': np.max(orig_w)},
    {'metric': 'orig_word_count_std', 'value': f"{np.std(orig_w):.2f}"},
    {'metric': 'cleaned_word_count_mean', 'value': f"{np.mean(clean_w):.2f}"},
    {'metric': 'cleaned_word_count_median', 'value': f"{np.median(clean_w):.2f}"},
    {'metric': 'cleaned_word_count_min', 'value': np.min(clean_w)},
    {'metric': 'cleaned_word_count_max', 'value': np.max(clean_w)},
    {'metric': 'cleaned_word_count_std', 'value': f"{np.std(clean_w):.2f}"},
    {'metric': 'orig_char_count_mean', 'value': f"{np.mean(orig_c):.2f}"},
    {'metric': 'orig_char_count_median', 'value': f"{np.median(orig_c):.2f}"},
    {'metric': 'cleaned_char_count_mean', 'value': f"{np.mean(clean_c):.2f}"},
    {'metric': 'cleaned_char_count_median', 'value': f"{np.median(clean_c):.2f}"},
    {'metric': 'pct_retained_mean', 'value': f"{np.mean(pct_ret):.2f}%"},
    {'metric': 'pct_retained_median', 'value': f"{np.median(pct_ret):.2f}%"},
    {'metric': 'pct_retained_min', 'value': f"{np.min(pct_ret):.2f}%"},
    {'metric': 'pct_retained_max', 'value': f"{np.max(pct_ret):.2f}%"}
]
for pat, cnt in sorted(pattern_counts.items(), key=lambda x: -x[1]):
    stats_rows.append({'metric': f"detected_pattern_{pat}", 'value': cnt})

stats_df = pd.DataFrame(stats_rows)
stats_df.to_csv('data/evaluation/phase6b_cleaning_statistics.csv', index=False)
print("Saved data/evaluation/phase6b_cleaning_statistics.csv")

# Build phase6b_example_pairs.csv (10 representative examples)
example_rows = []
sorted_by_removed = sorted(cleaning_results.items(), key=lambda x: (x[1]['orig_chars'] - x[1]['cleaned_chars']), reverse=True)
selected_examples = sorted_by_removed[:7] + [x for x in sorted_by_removed if not x[1]['is_changed']][:3]

for jd, info in selected_examples:
    example_rows.append({
        'original_text_snippet': jd[:200].replace('\n', ' ') + '...',
        'cleaned_text_snippet': info['cleaned_text'][:200].replace('\n', ' ') + '...',
        'original_chars': info['orig_chars'],
        'cleaned_chars': info['cleaned_chars'],
        'chars_removed': info['orig_chars'] - info['cleaned_chars'],
        'words_removed': info['orig_words'] - info['cleaned_words'],
        'pct_retained': f"{info['pct_retained']:.2f}%",
        'sections_detected': "; ".join(info['sections_detected']),
        'fallback_triggered': info['fallback_triggered']
    })
example_df = pd.DataFrame(example_rows)
example_df.to_csv('data/evaluation/phase6b_example_pairs.csv', index=False)
print("Saved data/evaluation/phase6b_example_pairs.csv")

# Plot length before vs after
plt.figure(figsize=(10, 5))
sns.histplot(orig_w, color='skyblue', label='Original JD Words', kde=True, bins=30, alpha=0.5)
sns.histplot(clean_w, color='orange', label='Cleaned JD Words', kde=True, bins=30, alpha=0.5)
plt.title('Job Description Word Count Distribution: Before vs After Cleaning (Phase 6B)')
plt.xlabel('Word Count')
plt.ylabel('Density / Frequency')
plt.legend()
plt.tight_layout()
plt.savefig('data/evaluation/plots/phase6b_length_before_after.png', dpi=300)
plt.close()
print("Saved data/evaluation/plots/phase6b_length_before_after.png")

# Now recompute or build features for Model B
jd_clean_map = {jd: cleaning_results[jd]['cleaned_text'] for jd in all_unique_jds}
train_clean_jds = train_df['job_description_text'].map(jd_clean_map)
val_clean_jds   = val_df['job_description_text'].map(jd_clean_map)
test_clean_jds  = test_df['job_description_text'].map(jd_clean_map)
off_clean_jds   = off_df['job_description_text'].map(jd_clean_map)

# Load SBERT model & embeddings
print("Encoding with SBERT...")
sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
unique_resumes = pd.concat([train_df['resume_text'], val_df['resume_text'], test_df['resume_text'], off_df['resume_text']]).unique()
res_embs_arr = sbert_model.encode(list(unique_resumes), batch_size=64, normalize_embeddings=True, show_progress_bar=False)
res_emb_map = dict(zip(unique_resumes, res_embs_arr))

unique_cleaned_jds = list(set(jd_clean_map.values()))
jd_embs_arr = sbert_model.encode(unique_cleaned_jds, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
clean_jd_emb_map = dict(zip(unique_cleaned_jds, jd_embs_arr))

def get_sbert_sims(res_s, jd_s):
    r_mat = np.array([res_emb_map[t] for t in res_s])
    j_mat = np.array([clean_jd_emb_map[t] for t in jd_s])
    return np.sum(r_mat * j_mat, axis=1)

train_sbert_b = get_sbert_sims(train_df['resume_text'], train_clean_jds)
val_sbert_b   = get_sbert_sims(val_df['resume_text'], val_clean_jds)
test_sbert_b  = get_sbert_sims(test_df['resume_text'], test_clean_jds)
off_sbert_b   = get_sbert_sims(off_df['resume_text'], off_clean_jds)

# TF-IDF strictly on candidate_train
tfidf_vec = TfidfVectorizer(stop_words='english', lowercase=True, norm='l2')
train_corpus_b = pd.concat([train_df['resume_text'], train_clean_jds], ignore_index=True)
tfidf_vec.fit(train_corpus_b)

def get_tfidf_sims(res_s, jd_s):
    r_mat = tfidf_vec.transform(res_s)
    j_mat = tfidf_vec.transform(jd_s)
    return np.asarray(r_mat.multiply(j_mat).sum(axis=1)).ravel()

train_tfidf_b = get_tfidf_sims(train_df['resume_text'], train_clean_jds)
val_tfidf_b   = get_tfidf_sims(val_df['resume_text'], val_clean_jds)
test_tfidf_b  = get_tfidf_sims(test_df['resume_text'], test_clean_jds)
off_tfidf_b   = get_tfidf_sims(off_df['resume_text'], off_clean_jds)

# Skills & Lexical
SKILL_TAXONOMY = {
    "python": ["python", "python3", "py"], "c++": ["c++", "cpp"], "java": ["java"],
    "javascript": ["javascript", "js", "ecmascript"], "typescript": ["typescript", "ts"],
    "r": ["r programming", "r language", "r-project", "rstudio"], "scala": ["scala"],
    "sql": ["sql", "structured query language"], "nosql": ["nosql"], "html": ["html", "html5"],
    "css": ["css", "css3"], "bash": ["bash", "shell scripting", "shell script"],
    "pytorch": ["pytorch", "torch"], "tensorflow": ["tensorflow", "tf"], "keras": ["keras"],
    "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"], "pandas": ["pandas"],
    "numpy": ["numpy"], "scipy": ["scipy"], "matplotlib": ["matplotlib"], "seaborn": ["seaborn"],
    "spark": ["spark", "pyspark", "apache spark"], "hadoop": ["hadoop", "apache hadoop"],
    "kafka": ["kafka", "apache kafka"], "airflow": ["airflow", "apache airflow"],
    "tableau": ["tableau"], "power bi": ["power bi", "powerbi"], "excel": ["excel", "ms excel", "microsoft excel"],
    "aws": ["aws", "amazon web services", "amazon ec2", "amazon s3"],
    "azure": ["azure", "microsoft azure"], "gcp": ["gcp", "google cloud", "google cloud platform"],
    "docker": ["docker"], "kubernetes": ["kubernetes", "k8s"], "git": ["git", "github", "gitlab"],
    "ci/cd": ["ci/cd", "cicd", "continuous integration", "continuous deployment"],
    "linux": ["linux", "unix", "ubuntu"],
    "rest api": ["rest api", "restful", "rest apis", "rest web services", "fastapi", "flask"],
    "graphql": ["graphql"], "mongodb": ["mongodb", "mongo"], "postgresql": ["postgresql", "postgres"],
    "mysql": ["mysql"], "redis": ["redis"], "nlp": ["nlp", "natural language processing"],
    "llm": ["llm", "llms", "large language model", "large language models"],
    "rag": ["rag", "retrieval augmented generation"], "computer vision": ["computer vision", "opencv"],
    "deep learning": ["deep learning", "neural networks", "neural network"],
    "machine learning": ["machine learning", "supervised learning", "unsupervised learning"],
    "bert": ["bert", "sentence-bert", "sbert"], "transformers": ["transformers", "huggingface", "hugging face"]
}
compiled_patterns = []
for canonical, aliases in SKILL_TAXONOMY.items():
    for alias in aliases:
        if alias in ["c++"]: pat = r'(?<![a-zA-Z0-9_])c\+\+(?![a-zA-Z0-9_\+])'
        elif alias in ["r"]: pat = r'(?<![a-zA-Z0-9_])r(?![a-zA-Z0-9_])'
        elif alias in ["c"]: pat = r'(?<![a-zA-Z0-9_])c(?![a-zA-Z0-9_\+])'
        else: pat = r'(?<![a-zA-Z0-9_])' + re.escape(alias) + r'(?![a-zA-Z0-9_])'
        compiled_patterns.append((re.compile(pat, re.IGNORECASE), canonical))

def extract_skills_set(text):
    if not isinstance(text, str) or not text.strip(): return set()
    cleaned = re.sub(r'\s+', ' ', text)
    found = set()
    for reg, can in compiled_patterns:
        if reg.search(cleaned): found.add(can)
    return found

token_pattern = re.compile(r'\b[a-zA-Z0-9_]+\b')
def tokenize_set(text):
    if not isinstance(text, str) or not text.strip(): return set()
    return set(token_pattern.findall(text.lower()))

res_skills_map = {t: extract_skills_set(t) for t in unique_resumes}
clean_jd_skills_map = {t: extract_skills_set(t) for t in unique_cleaned_jds}
res_tokens_map = {t: tokenize_set(t) for t in unique_resumes}
clean_jd_tokens_map = {t: tokenize_set(t) for t in unique_cleaned_jds}
res_words_map = {t: len(str(t).split()) for t in unique_resumes}
clean_jd_words_map = {t: len(str(t).split()) for t in unique_cleaned_jds}
res_chars_map = {t: len(str(t)) for t in unique_resumes}
clean_jd_chars_map = {t: len(str(t)) for t in unique_cleaned_jds}

def build_features(df, clean_jds, tfidf_sims, sbert_sims):
    n_jsk, n_rsk, n_msk, n_mis, cov_l = [], [], [], [], []
    rw_l, jw_l, rc_l, jc_l = [], [], [], []
    rw_rat, rc_rat, log_rw, log_jw = [], [], [], []
    sh_tok, jacc_l, r_utok, j_utok = [], [], [], []
    has_j_sk, has_r_sk, has_m_sk = [], [], []

    for i in range(len(df)):
        rt = df['resume_text'].iloc[i]
        jt = clean_jds.iloc[i]

        r_sk = res_skills_map[rt]
        j_sk = clean_jd_skills_map[jt]
        m_sk = r_sk.intersection(j_sk)
        miss_sk = j_sk.difference(r_sk)
        cov = len(m_sk) / len(j_sk) if len(j_sk) > 0 else 0.0

        rw = res_words_map[rt]
        jw = clean_jd_words_map[jt]
        rc = res_chars_map[rt]
        jc = clean_jd_chars_map[jt]

        r_tok = res_tokens_map[rt]
        j_tok = clean_jd_tokens_map[jt]
        shared = len(r_tok.intersection(j_tok))
        union  = len(r_tok.union(j_tok))
        jacc   = shared / union if union > 0 else 0.0

        n_jsk.append(len(j_sk))
        n_rsk.append(len(r_sk))
        n_msk.append(len(m_sk))
        n_mis.append(len(miss_sk))
        cov_l.append(cov)

        rw_l.append(rw)
        jw_l.append(jw)
        rc_l.append(rc)
        jc_l.append(jc)

        rw_rat.append(rw / (jw + 1e-5))
        rc_rat.append(rc / (jc + 1e-5))
        log_rw.append(np.log1p(rw))
        log_jw.append(np.log1p(jw))

        sh_tok.append(shared)
        jacc_l.append(jacc)
        r_utok.append(len(r_tok))
        j_utok.append(len(j_tok))

        has_j_sk.append(1 if len(j_sk) > 0 else 0)
        has_r_sk.append(1 if len(r_sk) > 0 else 0)
        has_m_sk.append(1 if len(m_sk) > 0 else 0)

    return pd.DataFrame({
        'tfidf_cosine_similarity': tfidf_sims,
        'sbert_cosine_similarity': sbert_sims,
        'skill_coverage': cov_l,
        'num_job_skills': n_jsk,
        'num_resume_skills': n_rsk,
        'num_matched_skills': n_msk,
        'num_missing_skills': n_mis,
        'resume_word_count': rw_l,
        'job_word_count': jw_l,
        'resume_char_count': rc_l,
        'job_char_count': jc_l,
        'resume_to_job_word_ratio': rw_rat,
        'resume_to_job_char_ratio': rc_rat,
        'log_resume_word_count': log_rw,
        'log_job_word_count': log_jw,
        'num_shared_unique_tokens': sh_tok,
        'jaccard_similarity': jacc_l,
        'resume_unique_token_count': r_utok,
        'job_unique_token_count': j_utok,
        'has_any_detected_job_skill': has_j_sk,
        'has_any_detected_resume_skill': has_r_sk,
        'has_any_matched_skill': has_m_sk
    })

X_train_b = build_features(train_df, train_clean_jds, train_tfidf_b, train_sbert_b)
X_val_b   = build_features(val_df,   val_clean_jds,   val_tfidf_b,   val_sbert_b)
X_test_b  = build_features(test_df,  test_clean_jds,  test_tfidf_b,  test_sbert_b)
X_off_b   = build_features(off_df,   off_clean_jds,   off_tfidf_b,   off_sbert_b)

y_train = train_df['label'].values
y_val   = val_df['label'].values
y_test  = test_df['label'].values
y_off   = off_df['label'].values

pipe_b = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(C=0.1, class_weight='balanced', max_iter=2000, random_state=42))
])
pipe_b.fit(X_train_b, y_train)

# Save Model B pipeline
joblib.dump(pipe_b, 'models/supervised_logistic_regression_phase6b.joblib')
print("Saved models/supervised_logistic_regression_phase6b.joblib")

# Predictions
val_preds_b  = pipe_b.predict(X_val_b)
test_preds_b = pipe_b.predict(X_test_b)
off_preds_b  = pipe_b.predict(X_off_b)

# Model A
pipe_a = joblib.load('models/supervised_logistic_regression.joblib')
feat_val_a  = pd.read_csv('data/evaluation/supervised_feature_dataset_validation.csv')
feat_test_a = pd.read_csv('data/evaluation/supervised_feature_dataset_test.csv')
feat_off_a  = pd.read_csv('data/evaluation/supervised_feature_dataset_official_test.csv')
f_cols = [c for c in feat_val_a.columns if c != 'label']

val_preds_a  = pipe_a.predict(feat_val_a[f_cols])
test_preds_a = pipe_a.predict(feat_test_a[f_cols])
off_preds_a  = pipe_a.predict(feat_off_a[f_cols])

label_order = ['No Fit', 'Potential Fit', 'Good Fit']

def compute_all_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p_m = precision_score(y_true, y_pred, labels=label_order, average='macro', zero_division=0)
    r_m = recall_score(y_true, y_pred, labels=label_order, average='macro', zero_division=0)
    f_m = f1_score(y_true, y_pred, labels=label_order, average='macro', zero_division=0)
    cr  = classification_report(y_true, y_pred, labels=label_order, output_dict=True, digits=4)
    return acc, p_m, r_m, f_m, cr

# Model Comparison Table
comp_rows = []
for split_name, y_t, y_pa, y_pb in [
    ('Validation', y_val, val_preds_a, val_preds_b),
    ('Candidate Test', y_test, test_preds_a, test_preds_b),
    ('Official Test', y_off, off_preds_a, off_preds_b)
]:
    acc_a, p_a, r_a, f_a, cr_a = compute_all_metrics(y_t, y_pa)
    acc_b, p_b, r_b, f_b, cr_b = compute_all_metrics(y_t, y_pb)

    comp_rows.append({
        'dataset': split_name,
        'model': 'Model A (Raw JDs)',
        'accuracy': round(acc_a, 4),
        'macro_precision': round(p_a, 4),
        'macro_recall': round(r_a, 4),
        'macro_f1': round(f_a, 4)
    })
    comp_rows.append({
        'dataset': split_name,
        'model': 'Model B (Cleaned JDs)',
        'accuracy': round(acc_b, 4),
        'macro_precision': round(p_b, 4),
        'macro_recall': round(r_b, 4),
        'macro_f1': round(f_b, 4)
    })
    comp_rows.append({
        'dataset': split_name,
        'model': 'Delta (Model B - Model A)',
        'accuracy': round(acc_b - acc_a, 4),
        'macro_precision': round(p_b - p_a, 4),
        'macro_recall': round(r_b - r_a, 4),
        'macro_f1': round(f_b - f_a, 4)
    })

comp_df = pd.DataFrame(comp_rows)
comp_df.to_csv('data/evaluation/phase6b_model_comparison.csv', index=False)
print("Saved data/evaluation/phase6b_model_comparison.csv")

# Per-Class Metrics Table
class_rows = []
for split_name, y_t, y_pa, y_pb in [
    ('Candidate Test', y_test, test_preds_a, test_preds_b),
    ('Official Test', y_off, off_preds_a, off_preds_b)
]:
    _, _, _, _, cr_a = compute_all_metrics(y_t, y_pa)
    _, _, _, _, cr_b = compute_all_metrics(y_t, y_pb)
    for lbl in label_order:
        class_rows.append({
            'dataset': split_name,
            'model': 'Model A (Raw JDs)',
            'label': lbl,
            'precision': round(cr_a[lbl]['precision'], 4),
            'recall': round(cr_a[lbl]['recall'], 4),
            'f1_score': round(cr_a[lbl]['f1-score'], 4),
            'support': cr_a[lbl]['support']
        })
        class_rows.append({
            'dataset': split_name,
            'model': 'Model B (Cleaned JDs)',
            'label': lbl,
            'precision': round(cr_b[lbl]['precision'], 4),
            'recall': round(cr_b[lbl]['recall'], 4),
            'f1_score': round(cr_b[lbl]['f1-score'], 4),
            'support': cr_b[lbl]['support']
        })
        class_rows.append({
            'dataset': split_name,
            'model': 'Delta (B - A)',
            'label': lbl,
            'precision': round(cr_b[lbl]['precision'] - cr_a[lbl]['precision'], 4),
            'recall': round(cr_b[lbl]['recall'] - cr_a[lbl]['recall'], 4),
            'f1_score': round(cr_b[lbl]['f1-score'] - cr_a[lbl]['f1-score'], 4),
            'support': cr_b[lbl]['support']
        })
class_df = pd.DataFrame(class_rows)
class_df.to_csv('data/evaluation/phase6b_class_metrics.csv', index=False)
print("Saved data/evaluation/phase6b_class_metrics.csv")

# Confusion Matrices Plots
cm_a = confusion_matrix(y_test, test_preds_a, labels=label_order)
cm_b = confusion_matrix(y_test, test_preds_b, labels=label_order)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm_a, annot=True, fmt='d', cmap='Blues', xticklabels=label_order, yticklabels=label_order, ax=ax)
ax.set_title('Model A (Raw JDs) Confusion Matrix\nCandidate Test')
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
plt.tight_layout()
plt.savefig('data/evaluation/plots/phase6b_confusion_original.png', dpi=300)
plt.close()

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm_b, annot=True, fmt='d', cmap='Greens', xticklabels=label_order, yticklabels=label_order, ax=ax)
ax.set_title('Model B (Cleaned JDs) Confusion Matrix\nCandidate Test')
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
plt.tight_layout()
plt.savefig('data/evaluation/plots/phase6b_confusion_cleaned.png', dpi=300)
plt.close()
print("Saved confusion matrix plots.")

# Build phase6b_error_recovery.csv
# Inspect Candidate Test predictions
test_eval_df = test_df.copy()
test_eval_df['pred_a'] = test_preds_a
test_eval_df['pred_b'] = test_preds_b
test_eval_df['correct_a'] = (test_eval_df['label'] == test_eval_df['pred_a'])
test_eval_df['correct_b'] = (test_eval_df['label'] == test_eval_df['pred_b'])

recovered = test_eval_df[(~test_eval_df['correct_a']) & (test_eval_df['correct_b'])].copy()
regressed = test_eval_df[(test_eval_df['correct_a']) & (~test_eval_df['correct_b'])].copy()
both_wrong = test_eval_df[(~test_eval_df['correct_a']) & (~test_eval_df['correct_b'])].copy()

print(f"Candidate Test Transitions: Recovered = {len(recovered)}, Regressed = {len(regressed)}, Both Wrong = {len(both_wrong)}, Both Correct = {len(test_eval_df[(test_eval_df['correct_a']) & (test_eval_df['correct_b'])])}")

def sample_cases(df_sub, cat_name, n=10):
    sampled = df_sub.head(n)
    rows = []
    for idx, r in sampled.iterrows():
        orig_jd = r['job_description_text']
        cleaned_jd = cleaning_results[orig_jd]['cleaned_text']
        rows.append({
            'category': cat_name,
            'pair_id': idx,
            'true_label': r['label'],
            'pred_a': r['pred_a'],
            'pred_b': r['pred_b'],
            'resume_snippet': r['resume_text'][:150].replace('\n', ' ') + '...',
            'original_jd_snippet': orig_jd[:150].replace('\n', ' ') + '...',
            'cleaned_jd_snippet': cleaned_jd[:150].replace('\n', ' ') + '...',
            'orig_jd_words': len(orig_jd.split()),
            'cleaned_jd_words': len(cleaned_jd.split()),
            'why_changed_hypothesis': (
                f"Model B switched from {r['pred_a']} to {r['pred_b']}. "
                f"Boilerplate removal altered word count ({len(orig_jd.split())} -> {len(cleaned_jd.split())}) "
                f"and shifted similarity scores."
            )
        })
    return rows

recovery_rows = []
recovery_rows.extend(sample_cases(recovered, 'Recovered (A wrong -> B correct)', 10))
recovery_rows.extend(sample_cases(regressed, 'Regressed (A correct -> B wrong)', 10))
recovery_rows.extend(sample_cases(both_wrong, 'Both Wrong', 10))

recovery_df = pd.DataFrame(recovery_rows)
recovery_df.to_csv('data/evaluation/phase6b_error_recovery.csv', index=False)
print("Saved data/evaluation/phase6b_error_recovery.csv")

print("All Phase 6B artifacts generated successfully!")
