import pytest
import numpy as np
import pandas as pd
from src.inference import ResumeJobMatcher, InferenceResult, MODEL_F_FEATURES


@pytest.fixture(scope="module")
def matcher():
    """Shared matcher instance to avoid reloading transformer weights multiple times."""
    return ResumeJobMatcher()


def test_pipeline_initializes_successfully(matcher):
    """1. Pipeline initializes successfully and loads artifacts."""
    assert matcher is not None
    assert matcher.model is not None
    assert matcher.feature_extractor is not None
    assert matcher.feature_extractor.vectorizer is not None
    assert matcher.feature_extractor.sbert_model is not None


def test_valid_input_produces_result(matcher):
    """2. Valid resume + JD produces an InferenceResult instance."""
    resume = "Experienced Senior Python Engineer with 6 years building microservices, SQL databases, Docker, and AWS."
    jd = "Seeking a Python Developer with experience in AWS, Docker, and PostgreSQL databases."
    result = matcher.predict(resume, jd)
    assert isinstance(result, InferenceResult)


def test_output_contains_all_three_class_probabilities(matcher):
    """3. Output contains all three class probabilities."""
    resume = "Data Scientist specializing in Python, Machine Learning, Scikit-Learn, and Pandas."
    jd = "Hiring Machine Learning Engineer with Python, Pandas, and Scikit-Learn experience."
    result = matcher.predict(resume, jd)
    probs = result.class_probabilities
    assert "No Fit" in probs
    assert "Potential Fit" in probs
    assert "Good Fit" in probs


def test_probabilities_sum_approximately_to_one(matcher):
    """4. Class probabilities sum approximately to 1."""
    resume = "Frontend Developer skilled in JavaScript, TypeScript, React, and CSS."
    jd = "Looking for Senior React and TypeScript Engineer with frontend architecture skills."
    result = matcher.predict(resume, jd)
    prob_sum = sum(result.class_probabilities.values())
    assert abs(prob_sum - 1.0) < 1e-4


def test_predicted_label_belongs_to_valid_classes(matcher):
    """5. Predicted label belongs to standard 3 classes."""
    resume = "DevOps Engineer with Kubernetes, Docker, CI/CD, and Linux administration."
    jd = "Need DevOps Specialist proficient in Docker, Kubernetes, and Linux."
    result = matcher.predict(resume, jd)
    assert result.predicted_label in ["No Fit", "Potential Fit", "Good Fit"]


def test_confidence_equals_maximum_probability(matcher):
    """6. Confidence equals the maximum class probability."""
    resume = "Backend software engineer working with Java, Spring Boot, and MySQL."
    jd = "Backend Java Developer wanted with MySQL and cloud experience."
    result = matcher.predict(resume, jd)
    max_p = max(result.class_probabilities.values())
    assert abs(result.confidence - max_p) < 1e-5


def test_matched_skills_returned(matcher):
    """7. Matched skills are detected and returned."""
    resume = "Machine Learning Specialist skilled in Python, PyTorch, SQL, and Docker."
    jd = "Looking for an engineer with Python, PyTorch, and Kubernetes experience."
    result = matcher.predict(resume, jd)
    assert "python" in result.matched_skills
    assert "pytorch" in result.matched_skills


def test_potential_skill_gaps_returned(matcher):
    """8. Potential skill gaps (required by JD but missing in resume) are returned."""
    resume = "Python developer experienced in Django and PostgreSQL."
    jd = "Requires Python, PostgreSQL, Docker, and Kubernetes."
    result = matcher.predict(resume, jd)
    assert "docker" in result.potential_skill_gaps
    assert "kubernetes" in result.potential_skill_gaps
    assert "python" not in result.potential_skill_gaps


def test_empty_resume_rejected(matcher):
    """9. Empty resume is rejected with ValueError."""
    jd = "Python Engineer needed with AWS experience."
    with pytest.raises(ValueError, match="Resume text"):
        matcher.predict("", jd)

    with pytest.raises(ValueError, match="Resume text"):
        matcher.predict("   ", jd)

    with pytest.raises(ValueError, match="Resume text"):
        matcher.predict(None, jd)


def test_empty_job_description_rejected(matcher):
    """10. Empty JD is rejected with ValueError."""
    resume = "Experienced Python developer with SQL."
    with pytest.raises(ValueError, match="Job description text"):
        matcher.predict(resume, "")

    with pytest.raises(ValueError, match="Job description text"):
        matcher.predict(resume, "   ")

    with pytest.raises(ValueError, match="Job description text"):
        matcher.predict(resume, None)


def test_short_inputs_rejected(matcher):
    """Extra: Extremely short inputs (< 3 words) are rejected."""
    with pytest.raises(ValueError, match="too short"):
        matcher.predict("Hi", "Looking for Python developer with AWS experience.")

    with pytest.raises(ValueError, match="too short"):
        matcher.predict("Senior Python Developer with 5 years experience.", "Job")


def test_feature_vector_count_is_exact(matcher):
    """11. Feature vector has exactly 18 features (Model F)."""
    resume = "Software Engineer with Python, Git, and Linux experience."
    jd = "Seeking Software Engineer with Python and Linux."
    features = matcher.feature_extractor.extract_features(resume, jd)
    vec = matcher.feature_extractor.to_feature_vector(features)
    assert len(features) == 18
    assert vec.shape == (1, 18)


def test_feature_order_matches_model_f_expectation(matcher):
    """12. Feature order matches MODEL_F_FEATURES exact sequence."""
    resume = "Data Analyst with SQL, Tableau, and Excel proficiency."
    jd = "Business Analyst with SQL, Tableau, and reporting experience."
    features = matcher.feature_extractor.extract_features(resume, jd)
    vec = matcher.feature_extractor.to_feature_vector(features)[0]

    for i, feature_name in enumerate(MODEL_F_FEATURES):
        assert abs(vec[i] - features[feature_name]) < 1e-6
