import os
import joblib
import pytest
import numpy as np

from src.inference import ResumeJobMatcher, InferenceResult


@pytest.fixture(scope="module")
def calibrated_matcher():
    """Matcher with production calibration enabled (default)."""
    return ResumeJobMatcher(apply_calibration=True)


@pytest.fixture(scope="module")
def uncalibrated_matcher():
    """Matcher with calibration disabled (raw probabilities)."""
    return ResumeJobMatcher(apply_calibration=False)


def test_calibration_artifact_loads_correctly():
    """1. Calibration artifact exists and loads with valid parameters."""
    cal_path = os.path.join("models", "model_f_calibrator.joblib")
    assert os.path.exists(cal_path)

    artifact = joblib.load(cal_path)
    assert artifact["method"] == "temperature_scaling"
    assert "temperature" in artifact
    assert artifact["temperature"] > 1.0  # T=1.42
    assert artifact["classes"] == ['No Fit', 'Potential Fit', 'Good Fit']


def test_calibrated_probabilities_have_three_classes(calibrated_matcher):
    """2. Calibrated probabilities contain all three semantic classes."""
    resume = "Experienced Senior Python Engineer with 6 years building microservices, SQL databases, Docker, and AWS."
    jd = "Seeking a Python Developer with experience in AWS, Docker, and PostgreSQL databases."
    result = calibrated_matcher.predict(resume, jd)

    assert "No Fit" in result.class_probabilities
    assert "Potential Fit" in result.class_probabilities
    assert "Good Fit" in result.class_probabilities
    assert len(result.class_probabilities) == 3


def test_probabilities_sum_to_one_and_in_range(calibrated_matcher):
    """3 & 4. Probabilities sum to 1.0 and each class probability is in [0, 1]."""
    resume = "Data Scientist with Python, Machine Learning, Scikit-Learn, and PyTorch expertise."
    jd = "Hiring Machine Learning Specialist with Python and PyTorch experience."
    result = calibrated_matcher.predict(resume, jd)

    prob_sum = sum(result.class_probabilities.values())
    assert abs(prob_sum - 1.0) < 1e-4

    for cls_name, p in result.class_probabilities.items():
        assert 0.0 <= p <= 1.0


def test_temperature_scaling_preserves_argmax_class(calibrated_matcher, uncalibrated_matcher):
    """5. Monotonicity: Temperature scaling preserves the exact predicted class from raw probabilities."""
    test_pairs = [
        ("Python Engineer with Docker and AWS.", "Python developer needed with AWS."),
        ("Senior Accountant with CPA and financial auditing.", "Senior Software Engineer with C++ and Linux."),
        ("Data Analyst with SQL, Tableau, and Excel.", "Business Intelligence Analyst with SQL and Tableau.")
    ]

    for resume, jd in test_pairs:
        cal_res = calibrated_matcher.predict(resume, jd)
        raw_res = uncalibrated_matcher.predict(resume, jd)

        # The predicted label must be identical because T > 0 preserves ranking
        assert cal_res.predicted_label == raw_res.predicted_label


def test_raw_probabilities_exposed_alongside_calibrated(calibrated_matcher):
    """6. Both raw and calibrated probabilities are transparently available."""
    resume = "DevOps Engineer with Kubernetes, Docker, CI/CD, and Linux administration."
    jd = "Need DevOps Specialist proficient in Docker, Kubernetes, and Linux."
    result = calibrated_matcher.predict(resume, jd)

    assert result.raw_class_probabilities is not None
    assert len(result.raw_class_probabilities) == 3
    assert abs(sum(result.raw_class_probabilities.values()) - 1.0) < 1e-4

    # Confidence should match the max calibrated probability
    max_cal_p = max(result.class_probabilities.values())
    assert abs(result.confidence - max_cal_p) < 1e-5


def test_uncalibrated_matcher_produces_raw_probabilities(uncalibrated_matcher):
    """7. Disabling calibration returns raw probabilities directly."""
    resume = "Software Engineer with Java, Spring Boot, and PostgreSQL."
    jd = "Java Developer wanted with Spring and PostgreSQL."
    result = uncalibrated_matcher.predict(resume, jd)

    assert result.metadata["calibration_applied"] is False
    assert result.confidence == max(result.class_probabilities.values())


def test_inference_determinism(calibrated_matcher):
    """8. Inference with calibration is completely deterministic."""
    resume = "Backend Developer specializing in Python, FastAPI, Redis, and PostgreSQL."
    jd = "Backend Engineer with Python and relational database experience."

    res1 = calibrated_matcher.predict(resume, jd)
    res2 = calibrated_matcher.predict(resume, jd)

    assert res1.predicted_label == res2.predicted_label
    assert res1.confidence == res2.confidence
    assert res1.class_probabilities == res2.class_probabilities
