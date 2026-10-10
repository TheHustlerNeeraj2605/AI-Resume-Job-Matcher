import os
import joblib
import numpy as np
from typing import Optional, List, Dict, Any
from scipy.special import softmax
from sentence_transformers import SentenceTransformer

from .schemas import InferenceResult, ExplainabilityResult
from .features import FeatureExtractor, MODEL_F_FEATURES
from .skills import analyze_skill_gap
from .explainability import build_explanation


class ResumeJobMatcher:
    """
    Production inference engine for AI-based Resume-Job Matching and Skill Gap Analysis.
    Wraps Model F (Ratio-Focused LightGBM, 18 features).
    """

    LABEL_ORDER: List[str] = ['No Fit', 'Potential Fit', 'Good Fit']

    def __init__(
        self,
        model_path: Optional[str] = None,
        vectorizer_path: Optional[str] = None,
        sbert_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        sbert_model: Optional[SentenceTransformer] = None,
        calibrator_path: Optional[str] = None,
        apply_calibration: bool = True,
    ):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

        if model_path is None:
            # Prefer model_f_ratio-focused_lightgbm_18_feats.joblib or model_f_ratio_focused_lgbm.pkl
            p1 = os.path.join(base_dir, 'models', 'model_f_ratio-focused_lightgbm_18_feats.joblib')
            p2 = os.path.join(base_dir, 'models', 'model_f_ratio_focused_lgbm.pkl')
            model_path = p1 if os.path.exists(p1) else p2

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model F checkpoint not found at {model_path}. "
                "Ensure models/model_f_ratio-focused_lightgbm_18_feats.joblib is present."
            )

        self.model_path = model_path
        self.model = joblib.load(model_path)

        # Optional post-hoc calibrator
        self.apply_calibration = apply_calibration
        if calibrator_path is None:
            default_cal_path = os.path.join(base_dir, 'models', 'model_f_calibrator.joblib')
            calibrator_path = default_cal_path if os.path.exists(default_cal_path) else None

        self.calibrator = joblib.load(calibrator_path) if calibrator_path and os.path.exists(calibrator_path) else None

        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(
            vectorizer_path=vectorizer_path,
            sbert_model_name=sbert_model_name,
            sbert_model=sbert_model
        )

    def _validate_inputs(self, resume_text: str, job_description_text: str) -> None:
        """Validate input texts for non-emptiness, types, and minimum length requirements."""
        if resume_text is None or not isinstance(resume_text, str):
            raise ValueError("Resume text must be a non-null string.")

        if job_description_text is None or not isinstance(job_description_text, str):
            raise ValueError("Job description text must be a non-null string.")

        clean_resume = resume_text.strip()
        clean_jd = job_description_text.strip()

        if not clean_resume:
            raise ValueError("Resume text cannot be empty or whitespace only.")

        if not clean_jd:
            raise ValueError("Job description text cannot be empty or whitespace only.")

        if len(clean_resume) < 10 or len(clean_resume.split()) < 3:
            raise ValueError("Resume text is too short to produce meaningful matching (minimum 3 words required).")

        if len(clean_jd) < 10 or len(clean_jd.split()) < 3:
            raise ValueError("Job description text is too short to produce meaningful matching (minimum 3 words required).")

    def predict(self, resume_text: str, job_description_text: str) -> InferenceResult:
        """
        Execute the end-to-end inference pipeline for a (resume, job description) pair.

        Parameters:
            resume_text: Raw or preprocessed candidate resume text.
            job_description_text: Raw or preprocessed job description text.

        Returns:
            InferenceResult object with predicted class, probabilities, confidence,
            feature values, matched skills, potential skill gaps, and explanation metadata.
        """
        # 1. Validate inputs
        self._validate_inputs(resume_text, job_description_text)

        # 2. Extract 18 Model F features
        features = self.feature_extractor.extract_features(resume_text, job_description_text)
        feature_vector = self.feature_extractor.to_feature_vector(features)

        # 3. Model F Raw Probability Prediction
        raw_probs = self.model.predict_proba(feature_vector)[0]
        raw_class_probabilities: Dict[str, float] = {
            self.LABEL_ORDER[i]: float(raw_probs[i])
            for i in range(len(self.LABEL_ORDER))
        }

        # 4. Calibration (Temperature Scaling)
        if self.calibrator is not None and self.apply_calibration:
            method = self.calibrator.get("method", "temperature_scaling")
            if method == "temperature_scaling":
                T = float(self.calibrator.get("temperature", 1.0))
                logits = np.log(np.clip(raw_probs, 1e-12, 1.0 - 1e-12))
                cal_probs = softmax(logits / T)
            else:
                cal_probs = raw_probs
                T = 1.0

            class_probabilities: Dict[str, float] = {
                self.LABEL_ORDER[i]: float(cal_probs[i])
                for i in range(len(self.LABEL_ORDER))
            }
            confidence = float(np.max(cal_probs))
            predicted_idx = int(np.argmax(cal_probs))
            confidence_type = f"Calibrated Model Confidence (Temperature Scaling T={T:.2f})"
            calibration_applied = True
        else:
            class_probabilities = raw_class_probabilities.copy()
            confidence = float(np.max(raw_probs))
            predicted_idx = int(np.argmax(raw_probs))
            confidence_type = "Model Confidence (Maximum Predicted Class Probability)"
            calibration_applied = False

        predicted_label = self.LABEL_ORDER[predicted_idx]

        # 5. Skill Gap Analysis
        skill_analysis = analyze_skill_gap(resume_text, job_description_text)

        # 6. Metadata for Streamlit / Explainability
        metadata: Dict[str, Any] = {
            "model_name": "Model F (Ratio-Focused LightGBM)",
            "model_architecture": "Gradient Boosted Decision Trees (18 features)",
            "model_version": "Phase 10C Production",
            "feature_names": MODEL_F_FEATURES,
            "num_features": len(MODEL_F_FEATURES),
            "resume_detected_skills_count": len(skill_analysis["resume_skills"]),
            "job_required_skills_count": len(skill_analysis["job_skills"]),
            "matched_skills_count": len(skill_analysis["matched_skills"]),
            "potential_skill_gaps_count": len(skill_analysis["potential_skill_gaps"]),
            "skill_coverage_pct": round(skill_analysis["skill_coverage"] * 100, 1),
            "confidence_type": confidence_type,
            "calibration_applied": calibration_applied,
            "calibration_temperature": self.calibrator.get("temperature", None) if self.calibrator else None,
        }

        return InferenceResult(
            predicted_label=predicted_label,
            class_probabilities=class_probabilities,
            confidence=confidence,
            features=features,
            matched_skills=skill_analysis["matched_skills"],
            potential_skill_gaps=skill_analysis["potential_skill_gaps"],
            raw_class_probabilities=raw_class_probabilities,
            metadata=metadata
        )

    def predict_from_pdf(self, pdf_source: Any, job_description_text: str) -> InferenceResult:
        """
        Convenience method to extract clean text from a PDF resume and execute matching.

        Parameters:
            pdf_source: File path (str/Path), bytes, or binary stream to a PDF resume.
            job_description_text: Raw or preprocessed job description text.

        Returns:
            InferenceResult object.
        """
        from .pdf_extractor import extract_resume_text
        resume_text = extract_resume_text(pdf_source)
        return self.predict(resume_text, job_description_text)

    def explain(self, result: InferenceResult) -> ExplainabilityResult:
        """
        Transform an InferenceResult into a presentation-ready ExplainabilityResult.

        Parameters:
            result: InferenceResult instance.

        Returns:
            ExplainabilityResult instance.
        """
        return build_explanation(result)

    def predict_and_explain(self, resume_text: str, job_description_text: str) -> ExplainabilityResult:
        """
        End-to-end convenience method: executes inference and builds presentation explainability.

        Parameters:
            resume_text: Raw or preprocessed candidate resume text.
            job_description_text: Raw or preprocessed job description text.

        Returns:
            ExplainabilityResult instance.
        """
        result = self.predict(resume_text, job_description_text)
        return self.explain(result)
