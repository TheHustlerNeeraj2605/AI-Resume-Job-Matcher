"""
Tests for Phase 11A and Phase 11B Streamlit Application.

Verifies:
1. Streamlit application script imports and initializes cleanly.
2. Native components (title, text area, button) are properly declared in AppTest.
3. Input validation warnings trigger when inputs are missing or invalid.
4. execute_match cleanly runs end-to-end inference and returns InferenceResult.
5. execute_match_with_explanation produces both InferenceResult and ExplainabilityResult.
6. Matched skills and potential skill gaps are accurately extracted and preserved.
7. Skill coverage behaves correctly: >0%, ==0.0%, and is None (when zero job skills).
8. Supporting signals (10 signals + signal flags) are present and properly typed.
9. Raw and calibrated probabilities are preserved without alteration.
10. Model F prediction outputs and probabilities are non-destructive and 100% invariant.
"""

import io
import pytest
import pypdf
from streamlit.testing.v1 import AppTest

from app.streamlit_app import (
    get_matcher,
    execute_match,
    execute_match_with_explanation,
    SIGNAL_NAME_MAP,
)
from src.inference import (
    ResumeJobMatcher,
    InferenceResult,
    ExplainabilityResult,
)


def create_test_pdf_bytes(text: str) -> bytes:
    """Create a minimal in-memory PDF with text content."""
    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    safe_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream_content = f"BT /F1 12 Tf 72 712 Td ({safe_text}) Tj ET"

    page_stream = pypdf.generic.DecodedStreamObject()
    page_stream.set_data(stream_content.encode("latin1", errors="replace"))

    font_dict = pypdf.generic.DictionaryObject({
        pypdf.generic.NameObject("/Type"): pypdf.generic.NameObject("/Font"),
        pypdf.generic.NameObject("/Subtype"): pypdf.generic.NameObject("/Type1"),
        pypdf.generic.NameObject("/BaseFont"): pypdf.generic.NameObject("/Helvetica")
    })
    res_dict = pypdf.generic.DictionaryObject({
        pypdf.generic.NameObject("/Font"): pypdf.generic.DictionaryObject({
            pypdf.generic.NameObject("/F1"): font_dict
        })
    })
    page[pypdf.generic.NameObject("/Resources")] = res_dict
    page[pypdf.generic.NameObject("/Contents")] = page_stream

    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


# =====================================================================
# Phase 11A Tests
# =====================================================================

def test_get_matcher_returns_production_instance():
    """Verify cached loader returns a valid ResumeJobMatcher."""
    matcher = get_matcher()
    assert isinstance(matcher, ResumeJobMatcher)
    assert matcher.model is not None


def test_streamlit_app_initial_render():
    """Verify Streamlit app loads and renders default elements via AppTest."""
    at = AppTest.from_file("app/streamlit_app.py")
    at.run(timeout=30)
    assert not at.exception

    # Check title and input controls exist
    assert len(at.title) >= 1
    assert "AI-Based Resume" in at.title[0].value
    assert len(at.text_area) == 1
    assert len(at.button) == 1
    assert at.button[0].label == "Analyze Match"


def test_validation_warning_when_no_pdf_uploaded():
    """Verify clicking Analyze without a PDF displays a user-facing warning."""
    at = AppTest.from_file("app/streamlit_app.py")
    at.run(timeout=30)
    at.button[0].click().run(timeout=30)
    assert not at.exception
    assert len(at.warning) >= 1
    assert "Please upload a candidate resume in PDF format" in at.warning[0].value


def test_execute_match_with_valid_pdf_and_jd():
    """Verify execute_match runs end-to-end inference and returns expected InferenceResult."""
    matcher = get_matcher()
    resume_text = (
        "Senior Machine Learning Engineer with 6 years experience in Python, PyTorch, "
        "Scikit-Learn, Docker, SQL, and AWS."
    )
    jd_text = (
        "Seeking Senior Machine Learning Engineer with Python, PyTorch, Scikit-Learn, "
        "Docker, and AWS cloud deployment experience."
    )
    pdf_bytes = create_test_pdf_bytes(resume_text)

    result = execute_match(pdf_bytes, jd_text, matcher=matcher)
    assert isinstance(result, InferenceResult)
    assert result.predicted_label in ["No Fit", "Potential Fit", "Good Fit"]
    assert result.confidence == max(result.class_probabilities.values())
    assert len(result.class_probabilities) == 3
    assert abs(sum(result.class_probabilities.values()) - 1.0) < 1e-4


def test_execute_match_invariance_with_direct_matcher():
    """Verify execute_match produces identical results to direct matcher call."""
    matcher = get_matcher()
    resume_text = (
        "Backend developer with experience in Java, Spring Boot, MySQL, and REST APIs."
    )
    jd_text = (
        "Looking for Java Backend Developer skilled in Spring Boot, MySQL, and microservices."
    )
    pdf_bytes = create_test_pdf_bytes(resume_text)

    result_app = execute_match(pdf_bytes, jd_text, matcher=matcher)
    result_direct = matcher.predict_from_pdf(pdf_bytes, jd_text)

    assert result_app.predicted_label == result_direct.predicted_label
    assert result_app.confidence == result_direct.confidence
    assert result_app.class_probabilities == result_direct.class_probabilities


def test_execute_match_invalid_pdf_raises_value_error():
    """Verify invalid PDF content raises ValueError handled by the UI."""
    matcher = get_matcher()
    jd_text = "Seeking Software Engineer proficient in Python and SQL databases."
    invalid_bytes = b"This is plain text, not a PDF document."

    with pytest.raises(ValueError):
        execute_match(invalid_bytes, jd_text, matcher=matcher)


def test_execute_match_short_jd_raises_value_error():
    """Verify too-short job description raises ValueError."""
    matcher = get_matcher()
    pdf_bytes = create_test_pdf_bytes("Senior Python Engineer with 5 years experience.")

    with pytest.raises(ValueError):
        execute_match(pdf_bytes, "too short", matcher=matcher)


# =====================================================================
# Phase 11B Tests: Skill Gap & Match Explanation UI
# =====================================================================

def test_execute_match_with_explanation_returns_both_objects():
    """Verify execute_match_with_explanation returns typed InferenceResult and ExplainabilityResult."""
    matcher = get_matcher()
    resume_text = (
        "Lead DevOps Specialist with 8 years managing Docker, Kubernetes, Linux, "
        "and AWS infrastructure pipelines."
    )
    jd_text = (
        "Hiring Senior DevOps Engineer with Docker, Kubernetes, AWS, and Linux administration."
    )
    pdf_bytes = create_test_pdf_bytes(resume_text)

    result, explanation = execute_match_with_explanation(pdf_bytes, jd_text, matcher=matcher)
    assert isinstance(result, InferenceResult)
    assert isinstance(explanation, ExplainabilityResult)
    assert explanation.predicted_label == result.predicted_label
    assert explanation.confidence == result.confidence
    assert explanation.calibrated_probabilities == result.class_probabilities


def test_matched_skills_and_potential_gaps_accuracy():
    """Verify matched skills and potential gaps are cleanly separated and non-empty."""
    matcher = get_matcher()
    resume_text = "Software Developer with Python, Docker, and SQL."
    jd_text = "Seeking Software Developer with Python, Docker, Kubernetes, and AWS."
    pdf_bytes = create_test_pdf_bytes(resume_text)

    _, explanation = execute_match_with_explanation(pdf_bytes, jd_text, matcher=matcher)

    assert "python" in explanation.matched_skills
    assert "docker" in explanation.matched_skills
    assert "kubernetes" in explanation.potential_skill_gaps
    assert "aws" in explanation.potential_skill_gaps
    assert explanation.matched_skill_count == len(explanation.matched_skills)
    assert explanation.missing_skill_count == len(explanation.potential_skill_gaps)


def test_skill_coverage_greater_than_zero():
    """Verify skill coverage percentage is correctly computed when skills match."""
    matcher = get_matcher()
    # JD requires 4 skills: python, docker, kubernetes, aws
    # Resume provides: python, docker
    resume_text = "Developer with Python and Docker experience."
    jd_text = "Hiring Engineer with Python, Docker, Kubernetes, and AWS skills."
    pdf_bytes = create_test_pdf_bytes(resume_text)

    _, explanation = execute_match_with_explanation(pdf_bytes, jd_text, matcher=matcher)

    assert explanation.job_required_skill_count == 4
    assert explanation.matched_skill_count == 2
    assert explanation.skill_coverage_percentage == 50.0


def test_skill_coverage_zero_when_no_skills_match():
    """Verify skill coverage is explicitly 0.0% (not None) when JD has skills but resume matches 0."""
    matcher = get_matcher()
    resume_text = "Registered Nurse with 10 years clinical hospital and patient care experience."
    jd_text = "Machine Learning Engineer with Python, PyTorch, and Docker experience."
    pdf_bytes = create_test_pdf_bytes(resume_text)

    _, explanation = execute_match_with_explanation(pdf_bytes, jd_text, matcher=matcher)

    assert explanation.job_required_skill_count > 0
    assert explanation.matched_skill_count == 0
    assert explanation.skill_coverage_percentage == 0.0
    assert explanation.skill_coverage_percentage is not None


def test_skill_coverage_is_none_when_no_job_skills_detected():
    """Verify skill coverage is explicitly None (NOT 0%) when JD contains zero taxonomy skills."""
    matcher = get_matcher()
    resume_text = "Experienced project manager with leadership, budget oversight, and team communication skills."
    jd_text = "Looking for a seasoned operational supervisor to lead team communication and workflow schedules."
    pdf_bytes = create_test_pdf_bytes(resume_text)

    _, explanation = execute_match_with_explanation(pdf_bytes, jd_text, matcher=matcher)

    assert explanation.job_required_skill_count == 0
    # Crucial semantic requirement: None rather than 0%
    assert explanation.skill_coverage_percentage is None


def test_supporting_signals_mapping_and_completeness():
    """Verify all 10 core model signals are present and mapped with friendly names."""
    matcher = get_matcher()
    resume_text = "Data Analyst proficient in SQL, Tableau, Pandas, and Python."
    jd_text = "Hiring Data Analyst with SQL, Tableau, and Python skills."
    pdf_bytes = create_test_pdf_bytes(resume_text)

    _, explanation = execute_match_with_explanation(pdf_bytes, jd_text, matcher=matcher)

    signals = explanation.signal_summary
    for signal_key in SIGNAL_NAME_MAP.keys():
        assert signal_key in signals, f"Missing expected signal: {signal_key}"
        assert isinstance(signals[signal_key], (float, int))


def test_signal_flags_are_boolean():
    """Verify signal flags are deterministic booleans without causal wording."""
    matcher = get_matcher()
    resume_text = "Python Developer with SQL experience."
    jd_text = "Python Developer with SQL and AWS experience."
    pdf_bytes = create_test_pdf_bytes(resume_text)

    _, explanation = execute_match_with_explanation(pdf_bytes, jd_text, matcher=matcher)

    flags = explanation.signal_flags
    for flag_name, flag_val in flags.items():
        assert isinstance(flag_val, bool)


def test_raw_vs_calibrated_probabilities_preservation():
    """Verify both raw and calibrated probabilities are preserved and valid."""
    matcher = get_matcher()
    resume_text = "Frontend Engineer skilled in JavaScript, HTML, CSS, and TypeScript."
    jd_text = "Senior Frontend Developer skilled in TypeScript, JavaScript, CSS, and React."
    pdf_bytes = create_test_pdf_bytes(resume_text)

    result, explanation = execute_match_with_explanation(pdf_bytes, jd_text, matcher=matcher)

    assert set(explanation.calibrated_probabilities.keys()) == {"No Fit", "Potential Fit", "Good Fit"}
    assert set(explanation.raw_probabilities.keys()) == {"No Fit", "Potential Fit", "Good Fit"}
    assert abs(sum(explanation.calibrated_probabilities.values()) - 1.0) < 1e-4
    assert abs(sum(explanation.raw_probabilities.values()) - 1.0) < 1e-4

    # Ensure calibrated probabilities match result.class_probabilities
    for k, v in result.class_probabilities.items():
        assert explanation.calibrated_probabilities[k] == v


def test_large_skill_lists_handling():
    """Verify robust handling of comprehensive skill lists."""
    matcher = get_matcher()
    resume_skills = (
        "Senior Polyglot Architect skilled in Python, Java, C++, Go, Rust, Scala, "
        "SQL, MongoDB, PostgreSQL, Redis, Docker, Kubernetes, AWS, GCP, Azure, Git, Linux."
    )
    jd_skills = (
        "Seeking Principal Systems Architect with Python, Go, Rust, Kubernetes, "
        "Docker, AWS, Snowflake, BigQuery, Kafka, Airflow."
    )
    pdf_bytes = create_test_pdf_bytes(resume_skills)

    _, explanation = execute_match_with_explanation(pdf_bytes, jd_skills, matcher=matcher)

    assert len(explanation.matched_skills) >= 5
    assert len(explanation.potential_skill_gaps) >= 3
    assert explanation.matched_skill_count == len(explanation.matched_skills)
    assert explanation.missing_skill_count == len(explanation.potential_skill_gaps)
    assert explanation.job_required_skill_count == (
        explanation.matched_skill_count + explanation.missing_skill_count
    )


def test_explainability_invariance_guarantee():
    """Verify that ExplainabilityResult NEVER alters Model F predictions or probabilities."""
    matcher = get_matcher()
    resume_text = "Senior Data Scientist with Python, Scikit-Learn, PyTorch, and NLP."
    jd_text = "Senior NLP Data Scientist with Python, PyTorch, and Scikit-Learn."
    pdf_bytes = create_test_pdf_bytes(resume_text)

    # 1. Direct prediction
    direct_res = matcher.predict_from_pdf(pdf_bytes, jd_text)

    # 2. Prediction + explanation
    res, exp = execute_match_with_explanation(pdf_bytes, jd_text, matcher=matcher)

    # Strict invariant checks
    assert exp.predicted_label == direct_res.predicted_label
    assert exp.confidence == direct_res.confidence
    assert exp.calibrated_probabilities == direct_res.class_probabilities
    if direct_res.raw_class_probabilities is not None:
        assert exp.raw_probabilities == direct_res.raw_class_probabilities
    assert exp.matched_skills == direct_res.matched_skills
    assert exp.potential_skill_gaps == direct_res.potential_skill_gaps
