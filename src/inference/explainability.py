"""
Explainability & Presentation Layer for AI-Based Resume-Job Matching.

Transforms raw InferenceResult objects into structured, UI-ready diagnostic
summaries for the Streamlit dashboard or downstream consumers.

Design Guarantees:
1. Purely non-destructive: Never alters predicted labels or probabilities.
2. Honest terminology: Uses 'Model Confidence', 'Matched Skills', 'Potential Skill Gaps'.
3. Descriptive, not causal: Signals summarize alignment without claiming why a candidate was accepted/rejected.
4. Distinguishes 'No detected skills' from '0% skill coverage'.
"""

from typing import Dict, List, Any, Optional
from .schemas import InferenceResult, ExplainabilityResult


# Standard descriptive summaries by predicted label (non-causal)
SUMMARY_TEMPLATES: Dict[str, str] = {
    "Good Fit": (
        "Strong overall alignment based on the model's learned combination "
        "of semantic similarity, lexical overlap, and skill signals."
    ),
    "Potential Fit": (
        "Moderate alignment. The model identifies some relevant overlap, "
        "but there are also meaningful gaps or weaker alignment signals."
    ),
    "No Fit": (
        "Limited overall alignment between the resume and the job description "
        "based on the model's learned signals."
    )
}

# Presentation heuristic thresholds for UI badges (strictly visual heuristics, not model decision rules)
SEMANTIC_OVERLAP_HEURISTIC_THRESHOLD = 0.65
LEXICAL_OVERLAP_HEURISTIC_THRESHOLD = 0.05


def build_explanation(result: InferenceResult) -> ExplainabilityResult:
    """
    Transform an InferenceResult into a structured, presentation-ready ExplainabilityResult.

    Parameters:
        result: An InferenceResult produced by ResumeJobMatcher.predict().

    Returns:
        ExplainabilityResult with structured diagnostics, preserved probabilities,
        explicit skill metrics, supporting signals, and non-causal summaries.
    """
    if not isinstance(result, InferenceResult):
        raise TypeError(f"Expected InferenceResult instance, got {type(result).__name__}")

    # 1. Overall Result & Probabilities (guaranteed non-destructive)
    predicted_label = result.predicted_label
    confidence = float(result.confidence)
    calibrated_probabilities = dict(result.class_probabilities)
    raw_probabilities = (
        dict(result.raw_class_probabilities)
        if result.raw_class_probabilities is not None
        else dict(result.class_probabilities)
    )

    # 2. Skill Metrics & Counts
    matched_skills = list(result.matched_skills)
    potential_skill_gaps = list(result.potential_skill_gaps)
    matched_skill_count = len(matched_skills)
    missing_skill_count = len(potential_skill_gaps)

    # Read from metadata if present, fallback to skill lists
    job_required_skill_count = int(
        result.metadata.get("job_required_skills_count", matched_skill_count + missing_skill_count)
    )
    resume_detected_skill_count = int(
        result.metadata.get(
            "resume_detected_skills_count",
            int(result.features.get("num_resume_skills", matched_skill_count))
        )
    )

    # Skill Coverage Percentage:
    # Explicitly distinguish between "no skills detected in JD" (None) and "0% coverage" (0.0)
    if job_required_skill_count > 0:
        skill_coverage_percentage: Optional[float] = round(
            (matched_skill_count / job_required_skill_count) * 100.0, 1
        )
    else:
        skill_coverage_percentage = None

    # 3. Descriptive Skill Message & Summaries
    if job_required_skill_count == 0 and resume_detected_skill_count == 0:
        skill_message = "No technical skills were detected by the current taxonomy in either the resume or the job description."
    elif job_required_skill_count == 0:
        skill_message = "No technical skills were detected by the current taxonomy in the job description."
    elif resume_detected_skill_count == 0:
        skill_message = "No technical skills were detected by the current taxonomy in the resume."
    else:
        skill_message = (
            f"Detected {matched_skill_count} matched skill(s) and {missing_skill_count} "
            f"potential skill gap(s) out of {job_required_skill_count} job requirement(s)."
        )

    matched_skill_summary: Dict[str, Any] = {
        "count": matched_skill_count,
        "skills": matched_skills,
        "description": (
            f"{matched_skill_count} skill(s) detected in both resume and job description."
            if matched_skill_count > 0
            else "No overlapping skills detected between resume and job description."
        )
    }

    potential_gap_summary: Dict[str, Any] = {
        "count": missing_skill_count,
        "skills": potential_skill_gaps,
        "description": (
            f"{missing_skill_count} skill(s) mentioned in job description but not detected in resume."
            if missing_skill_count > 0
            else "No skill gaps identified based on the detected taxonomy."
        )
    }

    # 4. Human-Readable Explanation Summary (Descriptive, not causal)
    explanation_summary = SUMMARY_TEMPLATES.get(
        predicted_label,
        f"Classification result: {predicted_label}."
    )

    # 5. Supporting Model Signals (from result.features)
    signal_keys = [
        "tfidf_cosine_similarity",
        "sbert_cosine_similarity",
        "skill_coverage",
        "jaccard_similarity",
        "resume_to_job_word_ratio",
        "resume_to_job_char_ratio",
        "num_job_skills",
        "num_resume_skills",
        "num_matched_skills",
        "num_missing_skills",
    ]
    signal_summary: Dict[str, float] = {}
    for k in signal_keys:
        if k in result.features:
            signal_summary[k] = round(float(result.features[k]), 4)

    # 6. Optional UI Signal Flags (Deterministic presentation heuristics)
    sbert_sim = float(result.features.get("sbert_cosine_similarity", 0.0))
    jaccard_sim = float(result.features.get("jaccard_similarity", 0.0))
    tfidf_sim = float(result.features.get("tfidf_cosine_similarity", 0.0))

    signal_flags: Dict[str, bool] = {
        "has_strong_semantic_overlap": sbert_sim >= SEMANTIC_OVERLAP_HEURISTIC_THRESHOLD,
        "has_lexical_overlap": (
            jaccard_sim > LEXICAL_OVERLAP_HEURISTIC_THRESHOLD
            or tfidf_sim > LEXICAL_OVERLAP_HEURISTIC_THRESHOLD
        ),
        "has_detected_skill_match": matched_skill_count > 0,
        "has_potential_skill_gaps": missing_skill_count > 0,
    }

    # 7. Metadata
    metadata: Dict[str, Any] = {
        **result.metadata,
        "explainability_layer": "Phase 10D Explainability",
        "confidence_label": "Model Confidence",
        "presentation_guidance": (
            "Model confidence reflects internal classification certainty among "
            "[No Fit, Potential Fit, Good Fit]. It is not a hiring probability or interview chance."
        ),
        "skill_extraction_limitation": (
            "Skill matching is based on deterministic taxonomy matching and does not establish "
            "actual candidate proficiency or exhaustive job requirements."
        ),
    }

    return ExplainabilityResult(
        predicted_label=predicted_label,
        confidence=confidence,
        confidence_label="Model Confidence",
        calibrated_probabilities=calibrated_probabilities,
        raw_probabilities=raw_probabilities,
        matched_skills=matched_skills,
        potential_skill_gaps=potential_skill_gaps,
        skill_coverage_percentage=skill_coverage_percentage,
        job_required_skill_count=job_required_skill_count,
        resume_detected_skill_count=resume_detected_skill_count,
        matched_skill_count=matched_skill_count,
        missing_skill_count=missing_skill_count,
        matched_skill_summary=matched_skill_summary,
        potential_gap_summary=potential_gap_summary,
        signal_summary=signal_summary,
        signal_flags=signal_flags,
        explanation_summary=explanation_summary,
        skill_message=skill_message,
        metadata=metadata
    )
