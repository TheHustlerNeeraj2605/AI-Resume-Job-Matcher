import io
import os
import re
import unicodedata
from pathlib import Path
from typing import Union, BinaryIO, Dict, Any, List, Tuple

import pypdf


class PDFResumeExtractor:
    """
    Lightweight, robust PDF text extractor for digitally generated resumes.
    Preserves technical symbols, formatting integrity, and multi-page reading order.
    """

    def __init__(self, min_char_count: int = 20, min_word_count: int = 5):
        self.min_char_count = min_char_count
        self.min_word_count = min_word_count

    def clean_extracted_text(self, text: str) -> str:
        """
        Conservatively normalize extracted PDF text.
        Removes PDF artifacts and normalizes whitespace while strictly preserving
        technical keywords, symbols (C++, C#, .NET), punctuation, and casing.
        """
        if not text:
            return ""

        # 1. Normalize Unicode characters (e.g. smart quotes, non-standard hyphens/spaces)
        text = unicodedata.normalize("NFKC", text)

        # 2. Normalize newline characters
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # 3. Handle soft-hyphen word wrapping at line breaks (e.g. "deve-\nloper" -> "developer")
        # Be careful not to alter minus signs or hyphenated tools like "scikit-learn"
        text = re.sub(r'([a-zA-Z]{2,})-\n([a-zA-Z]{2,})', r'\1\2', text)

        # 4. Replace form-feed and non-printable control characters (except tabs and newlines)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', text)

        # 5. Normalize horizontal whitespace on each line (collapse multiple spaces/tabs)
        lines = []
        for line in text.split("\n"):
            line = re.sub(r'[^\S\n]+', ' ', line).strip()
            lines.append(line)

        cleaned_text = "\n".join(lines)

        # 6. Collapse excessive blank lines (more than two consecutive newlines)
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)

        return cleaned_text.strip()

    def _get_pdf_stream(self, pdf_source: Union[str, Path, BinaryIO, bytes]) -> Tuple[BinaryIO, str]:
        """
        Validate and convert source into a readable binary stream.
        Returns a tuple of (stream, source_identifier).
        """
        if isinstance(pdf_source, (str, Path)):
            path_str = str(pdf_source)
            if not os.path.exists(path_str):
                raise FileNotFoundError(f"PDF file not found at: {path_str}")

            if os.path.isdir(path_str):
                raise ValueError(f"Expected a PDF file, but path points to a directory: {path_str}")

            if not path_str.lower().endswith(".pdf"):
                raise ValueError(f"Invalid file extension. Expected a .pdf file, got: {path_str}")

            file_size = os.path.getsize(path_str)
            if file_size == 0:
                raise ValueError(f"PDF file is completely empty (0 bytes): {path_str}")

            stream = open(path_str, "rb")
            return stream, path_str

        elif isinstance(pdf_source, bytes):
            if len(pdf_source) == 0:
                raise ValueError("PDF byte buffer is empty (0 bytes).")
            if not pdf_source.startswith(b"%PDF"):
                raise ValueError("Invalid PDF content: Missing '%PDF' header signature.")
            return io.BytesIO(pdf_source), "<bytes_buffer>"

        elif hasattr(pdf_source, "read"):
            # File-like object (e.g. io.BytesIO, tempfile, or Streamlit UploadedFile)
            content = pdf_source.read()
            if len(content) == 0:
                raise ValueError("PDF stream is empty (0 bytes).")
            if not content.startswith(b"%PDF"):
                raise ValueError("Invalid PDF stream: Missing '%PDF' header signature.")
            return io.BytesIO(content), getattr(pdf_source, "name", "<file_stream>")

        else:
            raise TypeError(f"Unsupported PDF source type: {type(pdf_source)}. Expected path, bytes, or file stream.")

    def extract_with_metadata(self, pdf_source: Union[str, Path, BinaryIO, bytes]) -> Dict[str, Any]:
        """
        Extract text and detailed metadata across all pages of the PDF.
        """
        stream, source_name = self._get_pdf_stream(pdf_source)
        is_file_handle = isinstance(pdf_source, (str, Path))

        try:
            try:
                reader = pypdf.PdfReader(stream)
            except Exception as e:
                raise ValueError(f"Failed to parse PDF document ({source_name}): {str(e)}") from e

            # Check password protection
            if reader.is_encrypted:
                # Try empty password decryption
                try:
                    decrypted = reader.decrypt("")
                except Exception:
                    decrypted = 0

                if decrypted == 0:
                    raise ValueError(f"PDF document is encrypted/password-protected: {source_name}")

            total_pages = len(reader.pages)
            if total_pages == 0:
                raise ValueError(f"PDF document has 0 pages: {source_name}")

            raw_pages: List[str] = []
            cleaned_pages: List[str] = []

            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text() or ""
                except Exception as e:
                    page_text = ""

                cleaned_page = self.clean_extracted_text(page_text)
                raw_pages.append(page_text)
                cleaned_pages.append(cleaned_page)

            # Join all page texts
            full_text = "\n\n".join(p for p in cleaned_pages if p).strip()

            if not full_text:
                raise ValueError(
                    f"Could not extract text from PDF ({source_name}). "
                    "The file may be image-based, scanned, or empty."
                )

            word_count = len(full_text.split())
            char_count = len(full_text)

            if char_count < self.min_char_count or word_count < self.min_word_count:
                raise ValueError(
                    f"Extracted PDF text is too short ({char_count} chars, {word_count} words). "
                    "A valid resume must contain substantive text."
                )

            return {
                "text": full_text,
                "page_count": total_pages,
                "char_count": char_count,
                "word_count": word_count,
                "source": source_name,
                "pages": cleaned_pages
            }

        finally:
            if is_file_handle and hasattr(stream, "close"):
                stream.close()

    def extract(self, pdf_source: Union[str, Path, BinaryIO, bytes]) -> str:
        """
        Extract clean, normalized text from a PDF resume.
        """
        meta = self.extract_with_metadata(pdf_source)
        return meta["text"]


# Reusable default instance and functional shorthand
_DEFAULT_EXTRACTOR = PDFResumeExtractor()


def extract_resume_text(pdf_source: Union[str, Path, BinaryIO, bytes]) -> str:
    """
    Convenience function to extract clean text from a local PDF resume or byte stream.

    Parameters:
        pdf_source: File path (str or Path), bytes, or file-like binary stream.

    Returns:
        Clean, normalized plain text string ready for ResumeJobMatcher.predict().
    """
    return _DEFAULT_EXTRACTOR.extract(pdf_source)
