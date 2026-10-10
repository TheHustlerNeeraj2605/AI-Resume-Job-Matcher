from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass
class InferenceResult:
    """Structured result returned by the Resume-Job matching inference pipeline."""
    predicted_label: str
    class_probabilities: Dict[str, float]
    confidence: float
    features: Dict[str, float]
    matched_skills: List[str]
    potential_skill_gaps: List[str]
    raw_class_probabilities: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a serializable dictionary for downstream consumers/APIs."""
        data = {
            "predicted_label": self.predicted_label,
            "class_probabilities": self.class_probabilities,
            "confidence": self.confidence,
            "features": self.features,
            "matched_skills": self.matched_skills,
            "potential_skill_gaps": self.potential_skill_gaps,
            "metadata": self.metadata,
        }
        if self.raw_class_probabilities is not None:
            data["raw_class_probabilities"] = self.raw_class_probabilities
        return data


@dataclass
class ExplainabilityResult:
    """Structured, UI-ready diagnostic presentation result for Streamlit and dashboard consumers."""
    predicted_label: str
    confidence: float
    calibrated_probabilities: Dict[str, float]
    raw_probabilities: Dict[str, float]
    matched_skills: List[str]
    potential_skill_gaps: List[str]
    skill_coverage_percentage: Optional[float]
    job_required_skill_count: int
    resume_detected_skill_count: int
    matched_skill_count: int
    missing_skill_count: int
    matched_skill_summary: Dict[str, Any]
    potential_gap_summary: Dict[str, Any]
    signal_summary: Dict[str, float]
    signal_flags: Dict[str, bool]
    explanation_summary: str
    skill_message: str
    confidence_label: str = "Model Confidence"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert explainability result to a serializable dictionary for UI/Streamlit rendering."""
        return {
            "predicted_label": self.predicted_label,
            "confidence": self.confidence,
            "confidence_label": self.confidence_label,
            "calibrated_probabilities": self.calibrated_probabilities,
            "raw_probabilities": self.raw_probabilities,
            "matched_skills": self.matched_skills,
            "potential_skill_gaps": self.potential_skill_gaps,
            "skill_coverage_percentage": self.skill_coverage_percentage,
            "job_required_skill_count": self.job_required_skill_count,
            "resume_detected_skill_count": self.resume_detected_skill_count,
            "matched_skill_count": self.matched_skill_count,
            "missing_skill_count": self.missing_skill_count,
            "matched_skill_summary": self.matched_skill_summary,
            "potential_gap_summary": self.potential_gap_summary,
            "signal_summary": self.signal_summary,
            "signal_flags": self.signal_flags,
            "explanation_summary": self.explanation_summary,
            "skill_message": self.skill_message,
            "metadata": self.metadata,
        }
