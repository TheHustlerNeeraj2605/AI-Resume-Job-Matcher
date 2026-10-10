import re
from typing import Dict, List, Set, Tuple, Any

# =====================================================================
# Phase 3 Skill Taxonomy (55 Canonical Skills)
# Used for exact Skill Coverage computation and interpretable gap reporting
# =====================================================================
SKILL_TAXONOMY_PHASE3: Dict[str, List[str]] = {
    # Programming Languages
    "python": ["python", "python3", "py"],
    "c++": ["c++", "cpp"],
    "java": ["java"],
    "javascript": ["javascript", "js", "ecmascript"],
    "typescript": ["typescript", "ts"],
    "r": ["r programming", "r language", "r-project", "rstudio"],
    "scala": ["scala"],
    "go": ["golang", "go language"],
    "rust": ["rust"],
    "c#": ["c#", "csharp", "c sharp"],
    "php": ["php"],
    "ruby": ["ruby"],
    "sql": ["sql"],

    # Data Science & Analytics Libraries
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "scipy": ["scipy"],
    "matplotlib": ["matplotlib"],
    "seaborn": ["seaborn"],
    "tableau": ["tableau"],
    "power bi": ["power bi", "powerbi"],

    # Classical ML & NLP
    "machine learning": ["machine learning"],
    "deep learning": ["deep learning"],
    "scikit-learn": ["scikit-learn", "scikit learn", "sklearn"],
    "nlp": ["nlp", "natural language processing"],
    "nltk": ["nltk"],
    "spacy": ["spacy", "spa cy"],
    "computer vision": ["computer vision"],
    "opencv": ["opencv", "open cv"],

    # Deep Learning Frameworks & Transformers
    "pytorch": ["pytorch", "py torch"],
    "tensorflow": ["tensorflow", "tensor flow"],
    "keras": ["keras"],
    "transformers": ["transformers"],
    "bert": ["bert"],
    "roberta": ["roberta"],
    "hugging face": ["hugging face", "huggingface"],
    "langchain": ["langchain"],
    "rag": ["rag", "retrieval augmented generation"],

    # Databases & Storage
    "mysql": ["mysql", "my sql"],
    "postgresql": ["postgresql", "postgres"],
    "mongodb": ["mongodb", "mongo db"],
    "redis": ["redis"],
    "snowflake": ["snowflake"],
    "bigquery": ["bigquery", "big query"],

    # Cloud & DevOps
    "aws": ["aws", "amazon web services"],
    "azure": ["azure", "microsoft azure"],
    "gcp": ["gcp", "google cloud", "google cloud platform"],
    "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s"],
    "git": ["git", "github", "gitlab"],
    "linux": ["linux", "unix"],

    # Big Data & Data Engineering
    "spark": ["spark", "apache spark", "pyspark"],
    "hadoop": ["hadoop", "apache hadoop"],
    "airflow": ["airflow", "apache airflow"],
    "kafka": ["kafka", "apache kafka"],
    "etl": ["etl", "extract transform load"]
}

def _compile_patterns_phase3() -> List[Tuple[re.Pattern, str]]:
    compiled = []
    for canonical, aliases in SKILL_TAXONOMY_PHASE3.items():
        for alias in aliases:
            if alias in ["c++", "cpp"]:
                pat = r'(?<![a-zA-Z0-9])(?:c\+\+|cpp)(?![a-zA-Z0-9\+])'
            elif alias in ["c#", "csharp", "c sharp"]:
                pat = r'(?<![a-zA-Z0-9])(?:c\#|csharp|c\s+sharp)(?![a-zA-Z0-9\#])'
            elif alias in ["power bi", "powerbi"]:
                pat = r'\bpower\s*bi\b'
            elif alias in ["r-project", "rstudio", "r programming", "r language"]:
                pat = r'\b(?:r\s+programming|r\s+language|r-project|rstudio)\b'
            else:
                pat = r'\b' + re.escape(alias) + r'\b'
            compiled.append((re.compile(pat, re.IGNORECASE), canonical))
    return compiled

COMPILED_PATTERNS_PHASE3 = _compile_patterns_phase3()

def extract_skills_phase3(text: str) -> List[str]:
    """Extract canonical skills using Phase 3 taxonomy for skill coverage and gap reporting."""
    if not isinstance(text, str) or not text.strip():
        return []
    cleaned = re.sub(r'\s+', ' ', text)
    detected = set()
    for reg, can in COMPILED_PATTERNS_PHASE3:
        if reg.search(cleaned):
            detected.add(can)
    return sorted(list(detected))


# =====================================================================
# Phase 6A Skill Taxonomy (50 Canonical Skills)
# Used strictly for Phase 6A skill count features (num_job_skills, etc.)
# =====================================================================
SKILL_TAXONOMY_PHASE6: Dict[str, List[str]] = {
    "python": ["python", "python3", "py"],
    "c++": ["c++", "cpp"],
    "java": ["java"],
    "javascript": ["javascript", "js", "ecmascript"],
    "typescript": ["typescript", "ts"],
    "r": ["r programming", "r language", "r-project", "rstudio"],
    "scala": ["scala"],
    "sql": ["sql", "structured query language"],
    "nosql": ["nosql"],
    "html": ["html", "html5"],
    "css": ["css", "css3"],
    "bash": ["bash", "shell scripting", "shell script"],
    "pytorch": ["pytorch", "torch"],
    "tensorflow": ["tensorflow", "tf"],
    "keras": ["keras"],
    "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "scipy": ["scipy"],
    "matplotlib": ["matplotlib"],
    "seaborn": ["seaborn"],
    "spark": ["spark", "pyspark", "apache spark"],
    "hadoop": ["hadoop", "apache hadoop"],
    "kafka": ["kafka", "apache kafka"],
    "airflow": ["airflow", "apache airflow"],
    "tableau": ["tableau"],
    "power bi": ["power bi", "powerbi"],
    "excel": ["excel", "ms excel", "microsoft excel"],
    "aws": ["aws", "amazon web services", "amazon ec2", "amazon s3"],
    "azure": ["azure", "microsoft azure"],
    "gcp": ["gcp", "google cloud", "google cloud platform"],
    "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s"],
    "git": ["git", "github", "gitlab"],
    "ci/cd": ["ci/cd", "cicd", "continuous integration", "continuous deployment"],
    "linux": ["linux", "unix", "ubuntu"],
    "rest api": ["rest api", "restful", "rest apis", "rest web services", "fastapi", "flask"],
    "graphql": ["graphql"],
    "mongodb": ["mongodb", "mongo"],
    "postgresql": ["postgresql", "postgres"],
    "mysql": ["mysql"],
    "redis": ["redis"],
    "nlp": ["nlp", "natural language processing"],
    "llm": ["llm", "llms", "large language model", "large language models"],
    "rag": ["rag", "retrieval augmented generation"],
    "computer vision": ["computer vision", "opencv"],
    "deep learning": ["deep learning", "neural networks", "neural network"],
    "machine learning": ["machine learning", "supervised learning", "unsupervised learning"],
    "bert": ["bert", "sentence-bert", "sbert"],
    "transformers": ["transformers", "huggingface", "hugging face"]
}

def _compile_patterns_phase6() -> List[Tuple[re.Pattern, str]]:
    compiled = []
    for canonical, aliases in SKILL_TAXONOMY_PHASE6.items():
        for alias in aliases:
            if alias in ["c++"]:
                pat = r'(?<![a-zA-Z0-9_])c\+\+(?![a-zA-Z0-9_\+])'
            elif alias in ["r"]:
                pat = r'(?<![a-zA-Z0-9_])r(?![a-zA-Z0-9_])'
            elif alias in ["c"]:
                pat = r'(?<![a-zA-Z0-9_])c(?![a-zA-Z0-9_\+])'
            else:
                pat = r'(?<![a-zA-Z0-9_])' + re.escape(alias) + r'(?![a-zA-Z0-9_])'
            compiled.append((re.compile(pat, re.IGNORECASE), canonical))
    return compiled

COMPILED_PATTERNS_PHASE6 = _compile_patterns_phase6()

def extract_skills_phase6(text: str) -> Set[str]:
    """Extract canonical skills using Phase 6A taxonomy for model feature counts."""
    if not isinstance(text, str) or not text.strip():
        return set()
    cleaned = re.sub(r'\s+', ' ', text)
    found = set()
    for reg, can in COMPILED_PATTERNS_PHASE6:
        if reg.search(cleaned):
            found.add(can)
    return found


def analyze_skill_gap(resume_text: str, job_description_text: str) -> Dict[str, Any]:
    """
    Performs deterministic skill extraction and gap analysis.
    Returns matched skills, potential skill gaps, and skill coverage.
    """
    res_skills = extract_skills_phase3(resume_text)
    job_skills = extract_skills_phase3(job_description_text)

    r_set = set(res_skills)
    j_set = set(job_skills)

    matched = sorted(list(r_set.intersection(j_set)))
    gaps = sorted(list(j_set.difference(r_set)))

    coverage = len(matched) / len(j_set) if len(j_set) > 0 else 0.0

    return {
        "resume_skills": res_skills,
        "job_skills": job_skills,
        "matched_skills": matched,
        "potential_skill_gaps": gaps,
        "skill_coverage": float(coverage)
    }
