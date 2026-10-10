import argparse
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference import PDFResumeExtractor, ResumeJobMatcher


def main():
    parser = argparse.ArgumentParser(
        description="CLI utility to test PDF resume extraction and optional matching inference."
    )
    parser.add_argument("pdf_path", help="Path to the PDF resume file.")
    parser.add_argument(
        "--jd",
        "--job-description",
        dest="job_description",
        default=None,
        help="Optional job description text or path to a text file containing the job description."
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show a brief 200-character preview of extracted text (privacy-preserving)."
    )

    args = parser.parse_args()

    print("=" * 65)
    print("PDF RESUME INGESTION & INFERENCE SANITY CHECK")
    print("=" * 65)
    print(f"Target PDF: {args.pdf_path}")

    # 1. Extraction
    extractor = PDFResumeExtractor()
    try:
        metadata = extractor.extract_with_metadata(args.pdf_path)
    except Exception as e:
        print(f"\n[ERROR] Extraction failed: {e}")
        sys.exit(1)

    print("\n--- Extraction Summary ---")
    print(f"Total Pages:      {metadata['page_count']}")
    print(f"Total Words:      {metadata['word_count']:,}")
    print(f"Total Characters: {metadata['char_count']:,}")

    if args.preview:
        preview_text = metadata['text'][:200].replace('\n', ' ')
        print(f"Preview (first 200 chars): \"{preview_text}...\"")

    print("\nExtraction status: SUCCESS (Clean plain text ready for inference)")

    # 2. Optional matching inference
    if args.job_description:
        jd_text = args.job_description
        if os.path.exists(jd_text) and os.path.isfile(jd_text):
            with open(jd_text, "r", encoding="utf-8") as f:
                jd_text = f.read()

        print("\n--- Running Model F Inference ---")
        matcher = ResumeJobMatcher()
        result = matcher.predict(metadata["text"], jd_text)

        print(f"Predicted Class:   {result.predicted_label}")
        print(f"Model Confidence:  {result.confidence:.2%}")
        print("Class Probabilities:")
        for cls_name, p in result.class_probabilities.items():
            print(f"  - {cls_name:13s}: {p:.4f}")
        print(f"Matched Skills ({len(result.matched_skills)}):      {result.matched_skills}")
        print(f"Potential Skill Gaps ({len(result.potential_skill_gaps)}): {result.potential_skill_gaps}")

    print("=" * 65)


if __name__ == "__main__":
    main()
