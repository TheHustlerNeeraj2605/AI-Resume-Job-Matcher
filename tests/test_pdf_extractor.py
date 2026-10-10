import io
import os
import tempfile
import pytest
import pypdf

from src.inference import (
    PDFResumeExtractor,
    extract_resume_text,
    ResumeJobMatcher,
    InferenceResult
)


def create_synthetic_pdf(
    pages_text: list[str],
    password: str = None,
    empty_pages: int = 0
) -> bytes:
    """Helper to programmatically generate a deterministic in-memory PDF with text."""
    writer = pypdf.PdfWriter()

    for text in pages_text:
        page = writer.add_blank_page(width=612, height=792)
        # Construct PDF content stream
        # Escape parentheses in text for PDF literal string
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

    for _ in range(empty_pages):
        writer.add_blank_page(width=612, height=792)

    if password:
        writer.encrypt(password)

    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


@pytest.fixture
def extractor():
    return PDFResumeExtractor()


@pytest.fixture(scope="module")
def matcher():
    return ResumeJobMatcher()


# --- Unit Tests ---

def test_valid_pdf_extracts_text(extractor):
    """1. Valid PDF extracts text accurately."""
    content = "Senior Python Developer with 5 years building scalable web services, Docker, and AWS."
    pdf_bytes = create_synthetic_pdf([content])

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        text = extractor.extract(tmp_path)
        assert isinstance(text, str)
        assert "Senior Python Developer" in text
        assert "Docker" in text
        assert "AWS" in text
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_multi_page_pdf_extracts_all_pages(extractor):
    """2. Multi-page PDF extracts text from all pages."""
    p1 = "Page 1: Senior Software Engineer with Python and SQL experience."
    p2 = "Page 2: Leadership experience managing DevOps pipelines with Docker and Kubernetes."
    p3 = "Page 3: Education: B.S. in Computer Science. Certified AWS Solutions Architect."

    pdf_bytes = create_synthetic_pdf([p1, p2, p3])

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        meta = extractor.extract_with_metadata(tmp_path)
        assert meta["page_count"] == 3
        assert "Page 1" in meta["text"]
        assert "Page 2" in meta["text"]
        assert "Page 3" in meta["text"]
        assert "Kubernetes" in meta["text"]
        assert "AWS" in meta["text"]
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_missing_file_raises_error(extractor):
    """3. Missing file raises FileNotFoundError."""
    non_existent = "non_existent_file_12345.pdf"
    with pytest.raises(FileNotFoundError, match="PDF file not found"):
        extractor.extract(non_existent)


def test_non_pdf_file_raises_error(extractor):
    """4. Non-PDF input raises ValueError."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        tmp.write(b"This is a text file.")
        tmp_path = tmp.name

    try:
        with pytest.raises(ValueError, match="Invalid file extension"):
            extractor.extract(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_directory_path_raises_error(extractor):
    """Extra: Passing a directory instead of file raises ValueError."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        with pytest.raises(ValueError, match="path points to a directory"):
            extractor.extract(tmp_dir)


def test_empty_or_no_text_pdf_raises_error(extractor):
    """5. Empty/no-text PDF (e.g. blank or image-only) raises useful error."""
    # Blank PDF with no text stream
    pdf_bytes = create_synthetic_pdf([], empty_pages=2)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        with pytest.raises(ValueError, match="Could not extract text from PDF"):
            extractor.extract(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_zero_byte_file_raises_error(extractor):
    """Extra: 0-byte file raises ValueError."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with pytest.raises(ValueError, match="completely empty"):
            extractor.extract(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_password_protected_pdf_raises_error(extractor):
    """Extra: Encrypted PDF raises clear password error."""
    pdf_bytes = create_synthetic_pdf(["Secret resume content with Python and SQL"], password="mypassword")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        with pytest.raises(ValueError, match="password-protected"):
            extractor.extract(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_extracted_text_is_normalized(extractor):
    """6. Extracted text is normalized (whitespace collapsed, line-breaks preserved)."""
    messy_text = "Experienced   Software   Engineer   \n\n\n\nwith   Python    and    AWS."
    pdf_bytes = create_synthetic_pdf([messy_text])

    clean = extractor.extract(pdf_bytes)
    assert "   " not in clean
    assert "\n\n\n" not in clean
    assert "Experienced Software Engineer" in clean


def test_technical_terms_survive_extraction(extractor):
    """7. Important technical symbols and terms survive extraction intact."""
    technical_text = (
        "Proficient in C++, C#, .NET, Python, SQL, PostgreSQL, MongoDB, "
        "AWS, Azure, GCP, Docker, Kubernetes, and scikit-learn."
    )
    pdf_bytes = create_synthetic_pdf([technical_text])

    clean = extractor.extract(pdf_bytes)
    assert "C++" in clean
    assert "C#" in clean
    assert ".NET" in clean
    assert "scikit-learn" in clean
    assert "PostgreSQL" in clean
    assert "Kubernetes" in clean


def test_extractor_output_is_string(extractor):
    """8. Extractor output is a Python string."""
    pdf_bytes = create_synthetic_pdf(["Staff Machine Learning Engineer with PyTorch and NLP experience."])
    text = extractor.extract(pdf_bytes)
    assert isinstance(text, str)
    assert len(text) > 0


def test_file_stream_and_bytes_support(extractor):
    """Extra: Extractor supports bytes and io.BytesIO directly."""
    content = "Lead Data Scientist with Python, Pandas, NumPy, and Tableau experience."
    pdf_bytes = create_synthetic_pdf([content])

    # Test bytes
    text_from_bytes = extractor.extract(pdf_bytes)
    assert "Lead Data Scientist" in text_from_bytes

    # Test stream
    stream = io.BytesIO(pdf_bytes)
    text_from_stream = extractor.extract(stream)
    assert "Lead Data Scientist" in text_from_stream


# --- Integration Test (Step 7) ---

def test_pdf_to_matcher_integration(matcher):
    """
    9 & Step 7: End-to-end integration test demonstrating:
    PDF -> extract_resume_text() -> ResumeJobMatcher.predict() -> InferenceResult
    """
    resume_content = (
        "Senior Machine Learning Engineer with 6 years experience in Python, PyTorch, "
        "Scikit-Learn, Docker, SQL, and AWS. Built retrieval-augmented generation systems."
    )
    job_description = (
        "Seeking Senior ML Engineer proficient in Python, PyTorch, Docker, and AWS. "
        "Experience with Kubernetes and PostgreSQL is a plus."
    )

    pdf_bytes = create_synthetic_pdf([resume_content])

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        # 1. Extract text via convenience function
        extracted_text = extract_resume_text(tmp_path)
        assert isinstance(extracted_text, str)
        assert len(extracted_text) > 30

        # 2. Run prediction via matcher
        result = matcher.predict(extracted_text, job_description)

        # 3. Verify InferenceResult properties
        assert isinstance(result, InferenceResult)
        assert result.predicted_label in ["No Fit", "Potential Fit", "Good Fit"]
        assert len(result.class_probabilities) == 3
        assert abs(sum(result.class_probabilities.values()) - 1.0) < 1e-4
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.features) == 18

        # 4. Verify skill detection
        assert "python" in result.matched_skills
        assert "pytorch" in result.matched_skills
        assert "docker" in result.matched_skills
        assert "aws" in result.matched_skills
        assert "kubernetes" in result.potential_skill_gaps

        # 5. Also test matcher.predict_from_pdf() convenience wrapper
        result_direct = matcher.predict_from_pdf(tmp_path, job_description)
        assert result_direct.predicted_label == result.predicted_label
        assert abs(result_direct.confidence - result.confidence) < 1e-6
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
