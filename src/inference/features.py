import os
import re
import joblib
import numpy as np
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer

from .skills import (
    extract_skills_phase3,
    extract_skills_phase6
)

MODEL_F_FEATURES: List[str] = [
    'tfidf_cosine_similarity',
    'sbert_cosine_similarity',
    'skill_coverage',
    'num_job_skills',
    'num_resume_skills',
    'num_matched_skills',
    'num_missing_skills',
    'resume_word_count',
    'resume_char_count',
    'resume_to_job_word_ratio',
    'resume_to_job_char_ratio',
    'log_resume_word_count',
    'num_shared_unique_tokens',
    'jaccard_similarity',
    'resume_unique_token_count',
    'has_any_detected_job_skill',
    'has_any_detected_resume_skill',
    'has_any_matched_skill',
]

TOKEN_PATTERN = re.compile(r'\b[a-zA-Z0-9_]+\b')


class FeatureExtractor:
    """
    Production feature extractor reproducing training-time semantics for Model F.
    Extracts the exact 18 ratio-focused features in deterministic order.
    """

    def __init__(
        self,
        vectorizer_path: Optional[str] = None,
        sbert_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        sbert_model: Optional[SentenceTransformer] = None
    ):
        if vectorizer_path is None:
            # Default to repository models directory
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            vectorizer_path = os.path.join(base_dir, 'models', 'tfidf_vectorizer.joblib')

        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(
                f"TF-IDF vectorizer artifact not found at {vectorizer_path}. "
                "Ensure models/tfidf_vectorizer.joblib is present."
            )

        self.vectorizer_path = vectorizer_path
        self.vectorizer = joblib.load(vectorizer_path)

        if sbert_model is not None:
            self.sbert_model = sbert_model
        else:
            self.sbert_model = SentenceTransformer(sbert_model_name)

    def extract_features(self, resume_text: str, job_description_text: str) -> Dict[str, float]:
        """
        Extract the exact 18 Model F features for a single (resume, job description) pair.
        """
        # 1. TF-IDF Cosine Similarity
        r_vec = self.vectorizer.transform([resume_text])
        j_vec = self.vectorizer.transform([job_description_text])
        tfidf_sim = float(r_vec.multiply(j_vec).sum())

        # 2. Sentence-BERT Cosine Similarity (MiniLM)
        # normalize_embeddings=True ensures inner product equals cosine similarity
        r_emb = self.sbert_model.encode([resume_text], normalize_embeddings=True)
        j_emb = self.sbert_model.encode([job_description_text], normalize_embeddings=True)
        sbert_sim = float(np.sum(r_emb * j_emb))

        # 3. Phase 3 Skill Coverage
        r_sk3 = set(extract_skills_phase3(resume_text))
        j_sk3 = set(extract_skills_phase3(job_description_text))
        m_sk3 = r_sk3.intersection(j_sk3)
        skill_coverage = float(len(m_sk3) / len(j_sk3)) if len(j_sk3) > 0 else 0.0

        # 4. Phase 6A Skill Counts
        r_sk6 = extract_skills_phase6(resume_text)
        j_sk6 = extract_skills_phase6(job_description_text)
        m_sk6 = r_sk6.intersection(j_sk6)
        miss_sk6 = j_sk6.difference(r_sk6)

        num_job_skills = float(len(j_sk6))
        num_resume_skills = float(len(r_sk6))
        num_matched_skills = float(len(m_sk6))
        num_missing_skills = float(len(miss_sk6))

        # 5. Length & Ratio Features
        rw = len(str(resume_text).split())
        jw = len(str(job_description_text).split())
        rc = len(str(resume_text))
        jc = len(str(job_description_text))

        resume_word_count = float(rw)
        resume_char_count = float(rc)
        resume_to_job_word_ratio = float(rw / (jw + 1e-5))
        resume_to_job_char_ratio = float(rc / (jc + 1e-5))
        log_resume_word_count = float(np.log1p(rw))

        # 6. Lexical Overlap & Token Features
        r_tokens = set(TOKEN_PATTERN.findall(resume_text.lower()))
        j_tokens = set(TOKEN_PATTERN.findall(job_description_text.lower()))
        shared_tokens = len(r_tokens.intersection(j_tokens))
        union_tokens = len(r_tokens.union(j_tokens))
        jaccard_similarity = float(shared_tokens / union_tokens) if union_tokens > 0 else 0.0

        num_shared_unique_tokens = float(shared_tokens)
        resume_unique_token_count = float(len(r_tokens))

        # 7. Binary Indicator Features
        has_any_detected_job_skill = 1.0 if len(j_sk6) > 0 else 0.0
        has_any_detected_resume_skill = 1.0 if len(r_sk6) > 0 else 0.0
        has_any_matched_skill = 1.0 if len(m_sk6) > 0 else 0.0

        return {
            'tfidf_cosine_similarity': tfidf_sim,
            'sbert_cosine_similarity': sbert_sim,
            'skill_coverage': skill_coverage,
            'num_job_skills': num_job_skills,
            'num_resume_skills': num_resume_skills,
            'num_matched_skills': num_matched_skills,
            'num_missing_skills': num_missing_skills,
            'resume_word_count': resume_word_count,
            'resume_char_count': resume_char_count,
            'resume_to_job_word_ratio': resume_to_job_word_ratio,
            'resume_to_job_char_ratio': resume_to_job_char_ratio,
            'log_resume_word_count': log_resume_word_count,
            'num_shared_unique_tokens': num_shared_unique_tokens,
            'jaccard_similarity': jaccard_similarity,
            'resume_unique_token_count': resume_unique_token_count,
            'has_any_detected_job_skill': has_any_detected_job_skill,
            'has_any_detected_resume_skill': has_any_detected_resume_skill,
            'has_any_matched_skill': has_any_matched_skill,
        }

    def to_feature_vector(self, feature_dict: Dict[str, float]) -> np.ndarray:
        """
        Convert feature dictionary to a 2D numpy array with shape (1, 18) in exact Model F order.
        """
        vector = [feature_dict[feat] for feat in MODEL_F_FEATURES]
        return np.array(vector, dtype=float).reshape(1, -1)
