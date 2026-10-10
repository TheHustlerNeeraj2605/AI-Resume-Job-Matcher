"""
Streamlit Application for AI-Based Resume-Job Matching & Skill Gap Analysis.
Phase 11C: Streamlit UI Polish & Visualization.

This module provides a polished, interactive decision-support interface connecting the
production inference pipeline (Model F + Temperature Scaling) and Explainability layer
to a presentation-ready Streamlit UI.
"""

import os
import sys
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import streamlit as st

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference import (
    ResumeJobMatcher,
    InferenceResult,
    ExplainabilityResult,
)


# Human-friendly mapping for supporting signals
SIGNAL_NAME_MAP: Dict[str, Tuple[str, str]] = {
    "tfidf_cosine_similarity": ("TF-IDF Similarity", "Score [0-1]"),
    "sbert_cosine_similarity": ("Semantic Similarity (SBERT)", "Cosine [-1 to 1]"),
    "skill_coverage": ("Skill Coverage Score", "Score [0-1]"),
    "jaccard_similarity": ("Jaccard Similarity", "Score [0-1]"),
    "resume_to_job_word_ratio": ("Resume / Job Word Ratio", "Ratio"),
    "resume_to_job_char_ratio": ("Resume / Job Character Ratio", "Ratio"),
    "num_job_skills": ("Job Skills Detected (Taxonomy)", "Count"),
    "num_resume_skills": ("Resume Skills Detected (Taxonomy)", "Count"),
    "num_matched_skills": ("Matched Skills Count", "Count"),
    "num_missing_skills": ("Potential Missing Skills Count", "Count"),
}

# Lightweight custom styling for professional polish
CUSTOM_CSS = """
<style>
/* Card containers */
.metric-card {
    background-color: var(--secondary-background-color, #f8fafc);
    border: 1px solid rgba(128, 128, 128, 0.2);
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 12px;
}
.fit-banner-good {
    background-color: rgba(34, 197, 94, 0.12);
    border-left: 5px solid #22c55e;
    border-radius: 6px;
    padding: 14px 18px;
    margin-bottom: 14px;
}
.fit-banner-potential {
    background-color: rgba(245, 158, 11, 0.12);
    border-left: 5px solid #f59e0b;
    border-radius: 6px;
    padding: 14px 18px;
    margin-bottom: 14px;
}
.fit-banner-no {
    background-color: rgba(100, 116, 139, 0.12);
    border-left: 5px solid #64748b;
    border-radius: 6px;
    padding: 14px 18px;
    margin-bottom: 14px;
}
/* Skill badges */
.skill-pill-matched {
    display: inline-block;
    background-color: rgba(14, 165, 233, 0.12);
    color: #0284c7;
    border: 1px solid rgba(14, 165, 233, 0.35);
    border-radius: 14px;
    padding: 4px 12px;
    margin: 3px 4px;
    font-size: 0.88rem;
    font-weight: 500;
}
.skill-pill-gap {
    display: inline-block;
    background-color: rgba(245, 158, 11, 0.12);
    color: #d97706;
    border: 1px solid rgba(245, 158, 11, 0.35);
    border-radius: 14px;
    padding: 4px 12px;
    margin: 3px 4px;
    font-size: 0.88rem;
    font-weight: 500;
}
/* Subtle disclaimers */
.subtle-disclaimer {
    font-size: 0.82rem;
    color: var(--text-color, #64748b);
    opacity: 0.85;
    margin-top: 4px;
}
</style>
"""


@st.cache_resource(show_spinner="Loading AI matching models...")
def get_matcher() -> ResumeJobMatcher:
    """
    Load and cache the production ResumeJobMatcher instance.
    Cached across user sessions to prevent redundant reloading of transformer models.
    """
    return ResumeJobMatcher()


def execute_match(
    pdf_source: Any,
    job_description_text: str,
    matcher: Optional[ResumeJobMatcher] = None
) -> InferenceResult:
    """
    Execute matching for a PDF source and job description using the production pipeline.
    """
    if matcher is None:
        matcher = get_matcher()
    return matcher.predict_from_pdf(pdf_source, job_description_text)


def execute_match_with_explanation(
    pdf_source: Any,
    job_description_text: str,
    matcher: Optional[ResumeJobMatcher] = None
) -> Tuple[InferenceResult, ExplainabilityResult]:
    """
    Execute matching and build structured explainability diagnostics.
    """
    if matcher is None:
        matcher = get_matcher()
    result = matcher.predict_from_pdf(pdf_source, job_description_text)
    explanation = matcher.explain(result)
    return result, explanation


def render_skill_badges(skills: list[str], pill_class: str) -> str:
    """Helper to render an array of skills as styled chip badges."""
    return "".join(f'<span class="{pill_class}">{s}</span>' for s in skills)


def main():
    st.set_page_config(
        page_title="AI-Based Resume–Job Matching & Skill Gap Analysis",
        page_icon="📄",
        layout="wide",
    )

    # Inject lightweight presentation CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ========================================================
    # Professional Header
    # ========================================================
    st.title("AI-Based Resume–Job Matching & Skill Gap Analysis")
    st.markdown(
        "**Semantic matching, skill-gap analysis, and explainable candidate–job fit assessment**"
    )
    st.caption(
        "Decision-support tool — not an automated hiring decision. Designed for recruiters and job seekers."
    )

    st.divider()

    # ========================================================
    # Section 1: Inputs
    # ========================================================
    st.subheader("1. Input Data")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Candidate Resume (PDF)**")
        uploaded_pdf = st.file_uploader(
            "Upload Resume",
            type=["pdf"],
            help="Upload a digitally generated, text-based PDF resume.",
            label_visibility="collapsed",
        )
        if uploaded_pdf is None:
            st.caption("Please upload a PDF resume to analyze.")
        else:
            st.caption(f"Loaded: `{uploaded_pdf.name}` ({len(uploaded_pdf.getvalue()):,} bytes)")

    with col2:
        st.markdown("**Target Job Description**")
        job_description = st.text_area(
            "Job Description Text",
            placeholder="Paste the target job description requirements, responsibilities, and qualifications here...",
            height=220,
            label_visibility="collapsed",
        )
        if not job_description or not job_description.strip():
            st.caption("Please paste a target job description.")
        else:
            word_count = len(job_description.strip().split())
            st.caption(f"Character count: {len(job_description.strip()):,} | Word count: {word_count:,}")

    st.markdown("")
    analyze_clicked = st.button("Analyze Match", type="primary", use_container_width=True)

    # ========================================================
    # Section 2: Inference & Diagnostic Presentation
    # ========================================================
    if analyze_clicked:
        if uploaded_pdf is None:
            st.warning("Please upload a candidate resume in PDF format.")
        elif not job_description or not job_description.strip():
            st.warning("Please provide a job description.")
        elif len(job_description.strip()) < 10 or len(job_description.strip().split()) < 3:
            st.warning("Job description is too short. Please provide substantive text (at least 3 words).")
        else:
            try:
                with st.spinner("Analyzing resume against job requirements..."):
                    matcher = get_matcher()
                    pdf_bytes = uploaded_pdf.getvalue()
                    result, explanation = execute_match_with_explanation(
                        pdf_bytes,
                        job_description.strip(),
                        matcher=matcher
                    )

                # ========================================================
                # 2. Prediction & Overall Fit
                # ========================================================
                st.divider()
                st.subheader("2. Prediction & Overall Fit")

                fit_class = explanation.predicted_label
                banner_class = (
                    "fit-banner-good" if fit_class == "Good Fit"
                    else "fit-banner-potential" if fit_class == "Potential Fit"
                    else "fit-banner-no"
                )

                # Visual Hierarchy Prediction Card
                st.markdown(
                    f"""
                    <div class="{banner_class}">
                        <div style="font-size:0.85rem; text-transform:uppercase; letter-spacing:0.05em; font-weight:600; opacity:0.8;">
                            Predicted Fit Classification
                        </div>
                        <div style="font-size:1.85rem; font-weight:700; margin:4px 0;">
                            {fit_class}
                        </div>
                        <div style="font-size:0.95rem;">
                            Model Confidence: <strong>{explanation.confidence * 100:.1f}%</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Metrics summary
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    st.metric(
                        label="Predicted Category",
                        value=explanation.predicted_label,
                    )
                with m_col2:
                    st.metric(
                        label="Model Confidence",
                        value=f"{explanation.confidence * 100:.1f}%",
                        help=(
                            "Maximum calibrated predicted class probability. "
                            "Reflects internal classification certainty among "
                            "[No Fit, Potential Fit, Good Fit] and is NOT a hiring "
                            "probability or interview guarantee."
                        ),
                    )

                st.caption(
                    "Model Confidence reflects the classifier's internal certainty across "
                    "the three fit classes. It is not a hiring probability or guarantee."
                )

                # Horizontal Calibrated Class Probabilities Visualization
                st.markdown("#### Calibrated Probability Distribution")
                st.caption(
                    "Post-hoc calibrated probabilities (Temperature Scaling T=1.42) reflecting model certainty."
                )

                for label, prob in explanation.calibrated_probabilities.items():
                    p_col1, p_col2 = st.columns([1, 4])
                    with p_col1:
                        st.markdown(f"**{label}**")
                        st.caption(f"{prob * 100:.1f}%")
                    with p_col2:
                        st.progress(float(prob))

                # ========================================================
                # 3. Match Explanation
                # ========================================================
                st.divider()
                st.subheader("3. Match Explanation")
                st.info(explanation.explanation_summary)
                if explanation.skill_message:
                    st.caption(f"Taxonomy alignment note: {explanation.skill_message}")

                # ========================================================
                # 4. Skill Analysis & Skill Gaps
                # ========================================================
                st.divider()
                st.subheader("4. Skill Analysis & Skill Gaps")

                # 4A. Compact Skill Overview Cards
                st.markdown("#### Skill Overview")
                s_col1, s_col2, s_col3, s_col4 = st.columns(4)
                with s_col1:
                    st.metric("Job Skills Detected", explanation.job_required_skill_count)
                with s_col2:
                    st.metric("Resume Skills Detected", explanation.resume_detected_skill_count)
                with s_col3:
                    st.metric("Matched Skills", explanation.matched_skill_count)
                with s_col4:
                    st.metric("Potential Skill Gaps", explanation.missing_skill_count)

                # 4B. Skill Coverage
                st.markdown("#### Skill Coverage")
                if explanation.skill_coverage_percentage is None:
                    st.info(
                        "Skill coverage is not available because no technical skills were "
                        "detected in the job description by the current taxonomy."
                    )
                elif explanation.skill_coverage_percentage == 0.0:
                    cov_col1, cov_col2 = st.columns([1, 4])
                    with cov_col1:
                        st.metric(
                            label="Coverage Rate",
                            value="0.0%",
                            help="Technical skills were detected in the job description, but none were matched in the resume."
                        )
                    with cov_col2:
                        st.caption(
                            "Technical skills were detected in the job description, but zero matching skills "
                            "were identified in the resume."
                        )
                        st.progress(0.0)
                else:
                    cov_col1, cov_col2 = st.columns([1, 4])
                    with cov_col1:
                        st.metric(
                            label="Coverage Rate",
                            value=f"{explanation.skill_coverage_percentage:.1f}%",
                            help="Percentage of detected job description skills that match the candidate resume."
                        )
                    with cov_col2:
                        st.caption(explanation.skill_message)
                        st.progress(min(1.0, max(0.0, explanation.skill_coverage_percentage / 100.0)))

                # 4C. Matched Skills vs Potential Skill Gaps Cards
                col_match, col_gap = st.columns(2)

                with col_match:
                    st.markdown("#### Matched Skills")
                    st.caption(
                        "Skills detected in both the resume and the job description by the current deterministic taxonomy."
                    )
                    if explanation.matched_skills:
                        badge_html = render_skill_badges(explanation.matched_skills, "skill-pill-matched")
                        st.markdown(badge_html, unsafe_allow_html=True)
                    else:
                        st.info("No overlapping technical skills detected between the resume and job description.")

                with col_gap:
                    st.markdown("#### Potential Skill Gaps")
                    st.caption(
                        "Skills mentioned in the job description but not detected in the resume by the current taxonomy. "
                        "These represent potential gaps, not confirmed deficiencies."
                    )
                    if explanation.potential_skill_gaps:
                        badge_html = render_skill_badges(explanation.potential_skill_gaps, "skill-pill-gap")
                        st.markdown(badge_html, unsafe_allow_html=True)
                    else:
                        st.info("No potential skill gaps identified based on the detected taxonomy.")

                # ========================================================
                # 5. Supporting Matching Signals
                # ========================================================
                st.divider()
                st.subheader("5. Supporting Matching Signals")
                st.caption(
                    "These are supporting model signals considered by the classifier. "
                    "They describe observed alignment and do not imply that any single signal caused the decision."
                )

                # High-level Signal Indicators
                if explanation.signal_flags:
                    st.markdown("##### Alignment Signal Indicators")
                    flag_cols = st.columns(len(explanation.signal_flags))
                    flag_meta = {
                        "has_strong_semantic_overlap": ("Semantic Alignment", "High semantic similarity detected between documents"),
                        "has_lexical_overlap": ("Lexical Overlap", "Keyword and vocabulary overlap detected"),
                        "has_detected_skill_match": ("Skill Match Signal", "At least one taxonomy skill match identified"),
                        "has_potential_skill_gaps": ("Potential Gaps Flag", "At least one required skill was not found in resume"),
                    }
                    for idx, (f_key, f_val) in enumerate(explanation.signal_flags.items()):
                        f_title, f_desc = flag_meta.get(f_key, (f_key, ""))
                        with flag_cols[idx]:
                            status_label = "Active" if f_val else "Not Active"
                            st.metric(label=f_title, value=status_label, help=f_desc)

                # Key Core Signals Display
                signals = explanation.signal_summary
                sig_c1, sig_c2, sig_c3, sig_c4 = st.columns(4)
                with sig_c1:
                    st.metric(
                        "Semantic Similarity",
                        f"{signals.get('sbert_cosine_similarity', 0.0):.4f}",
                        help="Dense contextual cosine similarity via Sentence-BERT (all-MiniLM-L6-v2)."
                    )
                with sig_c2:
                    st.metric(
                        "TF-IDF Similarity",
                        f"{signals.get('tfidf_cosine_similarity', 0.0):.4f}",
                        help="Lexical term overlap similarity score."
                    )
                with sig_c3:
                    st.metric(
                        "Jaccard Overlap",
                        f"{signals.get('jaccard_similarity', 0.0):.4f}",
                        help="Unique token set intersection over union."
                    )
                with sig_c4:
                    st.metric(
                        "Word Length Ratio",
                        f"{signals.get('resume_to_job_word_ratio', 0.0):.2f}",
                        help="Ratio of candidate resume word count to job description word count."
                    )

                # Expandable detailed signals table
                if signals:
                    with st.expander("Detailed Feature Values Considered by Classifier", expanded=False):
                        sig_records = []
                        for k, val in signals.items():
                            label, unit = SIGNAL_NAME_MAP.get(k, (k, "Value"))
                            formatted_val = f"{int(val)}" if unit == "Count" else f"{val:.4f}"
                            sig_records.append({
                                "Signal Name": label,
                                "Value": formatted_val,
                                "Measurement Type": unit
                            })
                        st.dataframe(pd.DataFrame(sig_records), use_container_width=True, hide_index=True)

                # ========================================================
                # 6. Technical Details & Auditability
                # ========================================================
                st.divider()
                with st.expander("Technical Details & Model Probabilities", expanded=False):
                    st.markdown("**Calibrated vs. Raw Probabilities**")
                    st.caption(
                        "Calibrated probabilities (Temperature Scaling T=1.42) are the primary user-facing confidence values. "
                        "Raw probabilities directly from LightGBM are preserved for auditability and transparency."
                    )

                    tech_data = [
                        {
                            "Fit Class": c_label,
                            "Calibrated Probability": f"{explanation.calibrated_probabilities[c_label] * 100:.2f}%",
                            "Raw Probability": f"{explanation.raw_probabilities[c_label] * 100:.2f}%",
                        }
                        for c_label in ["No Fit", "Potential Fit", "Good Fit"]
                    ]
                    st.dataframe(pd.DataFrame(tech_data), use_container_width=True, hide_index=True)

                    st.markdown("**Model & Pipeline Metadata**")
                    meta = explanation.metadata
                    st.json({
                        "Model Name": meta.get("model_name", "Model F (Ratio-Focused LightGBM)"),
                        "Architecture": meta.get("model_architecture", "18-Feature Gradient Boosted Trees"),
                        "Calibration": meta.get("confidence_type", "Temperature Scaling (T=1.42)"),
                        "Confidence Label": explanation.confidence_label,
                        "Guidance": meta.get(
                            "presentation_guidance",
                            "Confidence reflects classification certainty, not hiring probability."
                        ),
                        "Extraction Limitation": meta.get(
                            "skill_extraction_limitation",
                            "Deterministic regex matching over 55 canonical skills."
                        ),
                    })

            except ValueError as ve:
                st.error(f"Input Validation Error: {str(ve)}")
            except Exception as e:
                st.error(f"An unexpected error occurred during match analysis: {str(e)}")


if __name__ == "__main__":
    main()
