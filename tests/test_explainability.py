"""
Tests for Phase 10D Explainability & Skill-Gap Presentation Layer.

Ensures that:
1. Explainability layer transforms InferenceResult into structured, presentation-ready diagnostics.
2. Model F predictions, class probabilities, and confidence values are 100% invariant (never altered).
3. Calibrated and raw probabilities are both explicitly preserved.
4. Skill gap metrics are consistent with deterministic extraction.
5. When zero skills are detected in the job description, skill_coverage_percentage is None (not 0.0%).
6. Descriptive summaries and signal flags adhere strictly to non-causal presentation guidelines.
"""

import pytest
from src.inference import (
    ResumeJobMatcher,
    InferenceResult,
    ExplainabilityResult,
    build_explanation
)


@pytest.fixture(scope="module")
def matcher():
    """Module-level matcher instance to share across test runs."""
    return ResumeJobMatcher()


# Sample realistic test pairs
GOOD_FIT_RESUME = (
    "Senior Python Engineer with 7 years of deep experience in Machine Learning, Scikit-Learn, "
    "PyTorch, Pandas, NumPy, SQL databases, Docker, and AWS cloud architecture."
)
GOOD_FIT_JD = (
    "Seeking Senior Machine Learning Engineer with Python, PyTorch, Scikit-Learn, Pandas, "
    "Docker, and AWS experience to train production models."
)

POTENTIAL_FIT_RESUME = (
    "Junior Web Developer with 2 years of experience in JavaScript, HTML, CSS, React, and Python basics."
)
POTENTIAL_FIT_JD = (
    "Looking for a Full-Stack Python and React Developer with experience in AWS, Docker, and SQL."
)

NO_FIT_RESUME = (
    "Registered Nurse and Clinical Healthcare Specialist with 10 years of patient triage, "
    "emergency medicine, healthcare administration, and CPR certification."
)
NO_FIT_JD = (
    "Seeking a Principal Cloud Architect proficient in Kubernetes, Terraform, Golang, "
    "microservices design, and GCP infrastructure."
)

NO_SKILLS_RESUME = (
    "Responsible leader with broad experience directing cross-functional business initiatives, "
    "coordinating external communications, and overseeing organizational workflows."
)
NO_SKILLS_JD = (
    "Seeking an experienced professional for corporate program coordination, administrative oversight, "
    "and strategic partner communication."
)


def test_good_fit_result_generates_explanation(matcher):
    """1. Good Fit result generates valid ExplainabilityResult with non-causal summary."""
    res = matcher.predict(GOOD_FIT_RESUME, GOOD_FIT_JD)
    exp = build_explanation(res)

    assert isinstance(exp, ExplainabilityResult)
    assert exp.predicted_label == res.predicted_label
    assert "Strong overall alignment" in exp.explanation_summary
    assert exp.confidence_label == "Model Confidence"


def test_potential_fit_result_generates_explanation(matcher):
    """2. Potential Fit result generates valid ExplainabilityResult."""
    res = matcher.predict(POTENTIAL_FIT_RESUME, POTENTIAL_FIT_JD)
    exp = build_explanation(res)

    assert isinstance(exp, ExplainabilityResult)
    assert exp.predicted_label == res.predicted_label
    if exp.predicted_label == "Potential Fit":
        assert "Moderate alignment" in exp.explanation_summary


def test_no_fit_result_generates_explanation(matcher):
    """3. No Fit result generates valid ExplainabilityResult with non-causal summary."""
    res = matcher.predict(NO_FIT_RESUME, NO_FIT_JD)
    exp = build_explanation(res)

    assert isinstance(exp, ExplainabilityResult)
    assert exp.predicted_label == res.predicted_label
    assert exp.predicted_label == "No Fit"
    assert "Limited overall alignment" in exp.explanation_summary


def test_all_three_calibrated_probabilities_are_preserved(matcher):
    """4. All three calibrated probabilities are preserved with identical keys."""
    res = matcher.predict(GOOD_FIT_RESUME, GOOD_FIT_JD)
    exp = build_explanation(res)

    assert set(exp.calibrated_probabilities.keys()) == {"No Fit", "Potential Fit", "Good Fit"}
    for label in ["No Fit", "Potential Fit", "Good Fit"]:
        assert exp.calibrated_probabilities[label] == res.class_probabilities[label]


def test_raw_probabilities_are_preserved(matcher):
    """5. Raw probabilities are preserved and explicitly exposed."""
    res = matcher.predict(GOOD_FIT_RESUME, GOOD_FIT_JD)
    exp = build_explanation(res)

    assert set(exp.raw_probabilities.keys()) == {"No Fit", "Potential Fit", "Good Fit"}
    if res.raw_class_probabilities is not None:
        for label in ["No Fit", "Potential Fit", "Good Fit"]:
            assert exp.raw_probabilities[label] == res.raw_class_probabilities[label]


def test_probability_values_and_confidence_strictly_unchanged(matcher):
    """6. Probability values and confidence remain strictly identical."""
    res = matcher.predict(GOOD_FIT_RESUME, GOOD_FIT_JD)
    exp = build_explanation(res)

    assert exp.confidence == res.confidence
    assert exp.confidence == max(exp.calibrated_probabilities.values())
    for k, v in res.class_probabilities.items():
        assert exp.calibrated_probabilities[k] == v


def test_predicted_label_remains_unchanged(matcher):
    """7. Predicted label remains identical between InferenceResult and ExplainabilityResult."""
    for resume, jd in [(GOOD_FIT_RESUME, GOOD_FIT_JD), (NO_FIT_RESUME, NO_FIT_JD)]:
        res = matcher.predict(resume, jd)
        exp = build_explanation(res)
        assert exp.predicted_label == res.predicted_label


def test_matched_skills_are_preserved(matcher):
    """8. Matched skills list is identical."""
    res = matcher.predict(GOOD_FIT_RESUME, GOOD_FIT_JD)
    exp = build_explanation(res)

    assert exp.matched_skills == res.matched_skills
    assert exp.matched_skill_count == len(res.matched_skills)
    assert exp.matched_skill_summary["skills"] == res.matched_skills
    assert exp.matched_skill_summary["count"] == len(res.matched_skills)


def test_potential_skill_gaps_are_preserved(matcher):
    """9. Potential skill gaps list is identical."""
    res = matcher.predict(POTENTIAL_FIT_RESUME, POTENTIAL_FIT_JD)
    exp = build_explanation(res)

    assert exp.potential_skill_gaps == res.potential_skill_gaps
    assert exp.missing_skill_count == len(res.potential_skill_gaps)
    assert exp.potential_gap_summary["skills"] == res.potential_skill_gaps
    assert exp.potential_gap_summary["count"] == len(res.potential_skill_gaps)


def test_zero_detected_job_skills_returns_none_coverage_not_zero_percent(matcher):
    """10. Zero detected job skills returns None coverage rather than false 0%."""
    res = matcher.predict(NO_SKILLS_RESUME, NO_SKILLS_JD)
    exp = build_explanation(res)

    assert exp.job_required_skill_count == 0
    # Crucial distinction: None rather than 0.0%
    assert exp.skill_coverage_percentage is None
    assert "No technical skills were detected by the current taxonomy" in exp.skill_message


def test_skills_required_with_zero_match_returns_zero_percent(matcher):
    """11. When job requires skills but candidate matches zero, returns 0.0% coverage."""
    # Nurse resume against Python/PyTorch JD
    res = matcher.predict(NO_FIT_RESUME, GOOD_FIT_JD)
    exp = build_explanation(res)

    assert exp.job_required_skill_count > 0
    assert exp.matched_skill_count == 0
    assert exp.skill_coverage_percentage == 0.0


def test_skill_counts_are_consistent(matcher):
    """12. Skill counts are mathematically consistent."""
    res = matcher.predict(GOOD_FIT_RESUME, GOOD_FIT_JD)
    exp = build_explanation(res)

    assert exp.matched_skill_count == len(exp.matched_skills)
    assert exp.missing_skill_count == len(exp.potential_skill_gaps)
    # In Phase 3 taxonomy, job_skills = matched + missing
    assert exp.job_required_skill_count == exp.matched_skill_count + exp.missing_skill_count


def test_supporting_signals_extracted_cleanly(matcher):
    """13. Supporting signals contain required model features."""
    res = matcher.predict(GOOD_FIT_RESUME, GOOD_FIT_JD)
    exp = build_explanation(res)

    signals = exp.signal_summary
    assert "tfidf_cosine_similarity" in signals
    assert "sbert_cosine_similarity" in signals
    assert "skill_coverage" in signals
    assert "jaccard_similarity" in signals
    assert "resume_to_job_word_ratio" in signals
    assert "resume_to_job_char_ratio" in signals
    assert "num_job_skills" in signals
    assert "num_resume_skills" in signals
    assert "num_matched_skills" in signals
    assert "num_missing_skills" in signals


def test_signal_flags_are_boolean_and_deterministic(matcher):
    """14. Signal flags are deterministic booleans."""
    res = matcher.predict(GOOD_FIT_RESUME, GOOD_FIT_JD)
    exp = build_explanation(res)

    flags = exp.signal_flags
    assert isinstance(flags["has_strong_semantic_overlap"], bool)
    assert isinstance(flags["has_lexical_overlap"], bool)
    assert isinstance(flags["has_detected_skill_match"], bool)
    assert isinstance(flags["has_potential_skill_gaps"], bool)


def test_output_schema_is_deterministic_and_serializable(matcher):
    """15. Output schema can be converted to dictionary for Streamlit."""
    res = matcher.predict(GOOD_FIT_RESUME, GOOD_FIT_JD)
    exp = build_explanation(res)
    d = exp.to_dict()

    assert isinstance(d, dict)
    assert "predicted_label" in d
    assert "confidence" in d
    assert "calibrated_probabilities" in d
    assert "raw_probabilities" in d
    assert "matched_skills" in d
    assert "potential_skill_gaps" in d
    assert "skill_coverage_percentage" in d
    assert "job_required_skill_count" in d
    assert "resume_detected_skill_count" in d
    assert "matched_skill_count" in d
    assert "missing_skill_count" in d
    assert "signal_summary" in d
    assert "signal_flags" in d
    assert "explanation_summary" in d
    assert "skill_message" in d


def test_matcher_convenience_methods(matcher):
    """16. matcher.explain() and matcher.predict_and_explain() work as expected."""
    res = matcher.predict(GOOD_FIT_RESUME, GOOD_FIT_JD)
    exp1 = matcher.explain(res)
    assert isinstance(exp1, ExplainabilityResult)

    exp2 = matcher.predict_and_explain(GOOD_FIT_RESUME, GOOD_FIT_JD)
    assert isinstance(exp2, ExplainabilityResult)
    assert exp2.predicted_label == exp1.predicted_label
    assert exp2.confidence == exp1.confidence


def test_explainability_layer_never_alters_model_predictions(matcher):
    """17. CRITICAL: The explainability layer NEVER alters Model F predictions or probabilities."""
    for resume, jd in [
        (GOOD_FIT_RESUME, GOOD_FIT_JD),
        (POTENTIAL_FIT_RESUME, POTENTIAL_FIT_JD),
        (NO_FIT_RESUME, NO_FIT_JD),
        (NO_SKILLS_RESUME, NO_SKILLS_JD),
    ]:
        res = matcher.predict(resume, jd)
        exp = build_explanation(res)

        # Labels must be strictly identical
        assert exp.predicted_label == res.predicted_label
        # Confidences must be strictly identical
        assert exp.confidence == res.confidence
        # Calibrated probabilities must be strictly identical
        assert exp.calibrated_probabilities == res.class_probabilities
        # Raw probabilities must be strictly identical
        if res.raw_class_probabilities is not None:
            assert exp.raw_probabilities == res.raw_class_probabilities
