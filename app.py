from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
from google.genai import types
import PyPDF2
import docx
import json
import io
import hashlib
import re

app = Flask(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
client = genai.Client(api_key=GEMINI_API_KEY)

results_cache = {}

TECHNICAL_SKILLS = [
    "python", "javascript", "typescript", "java", "c++", "c#", "go", "rust", "swift", "kotlin",
    "react", "angular", "vue", "node", "django", "flask", "fastapi", "spring", "express",
    "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "firebase",
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "jenkins", "github actions",
    "machine learning", "deep learning", "nlp", "computer vision", "tensorflow", "pytorch",
    "pandas", "numpy", "scikit-learn", "tableau", "power bi", "excel", "r",
    "html", "css", "sass", "tailwind", "bootstrap", "figma", "sketch",
    "rest", "graphql", "api", "microservices", "ci/cd", "devops", "agile", "scrum",
    "linux", "bash", "git", "selenium", "junit", "jest", "pytest"
]

SOFT_SKILLS = [
    "leadership", "communication", "teamwork", "collaboration", "problem solving",
    "critical thinking", "time management", "project management", "mentoring",
    "adaptability", "creativity", "analytical", "detail-oriented", "organized",
    "self-motivated", "initiative", "presentation", "negotiation", "strategic"
]

ACTION_VERBS = [
    "built", "developed", "designed", "implemented", "created", "launched", "led",
    "managed", "improved", "increased", "reduced", "optimized", "automated",
    "architected", "deployed", "integrated", "delivered", "achieved", "spearheaded",
    "collaborated", "mentored", "trained", "analyzed", "researched", "established"
]

CONTACT_PATTERNS = {
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "phone": r"(\+?\d[\d\s\-().]{7,}\d)",
    "linkedin": r"linkedin\.com/in/[a-zA-Z0-9\-]+",
    "github": r"github\.com/[a-zA-Z0-9\-]+"
}


def extract_keywords(text):
    """Resume se saare important keywords nikalta hai"""
    text_lower = text.lower()
    words = re.findall(r'\b[a-z][a-z0-9+#.\-]*\b', text_lower)
    word_set = set(words)
    bigrams = set()
    word_list = text_lower.split()
    for i in range(len(word_list) - 1):
        bigrams.add(word_list[i] + " " + word_list[i+1])

    found_technical = [s for s in TECHNICAL_SKILLS if s in word_set or s in bigrams]
    found_soft = [s for s in SOFT_SKILLS if s in word_set or s in bigrams]
    found_action = [v for v in ACTION_VERBS if v in word_set]

    return {
        "technical": found_technical,
        "soft": found_soft,
        "action_verbs": found_action
    }


def check_contact_info(text):
    """Contact details check karta hai"""
    found = {}
    for key, pattern in CONTACT_PATTERNS.items():
        match = re.search(pattern, text, re.IGNORECASE)
        found[key] = bool(match)
    return found


def check_quantified_achievements(text):
    """Numbers aur metrics dhundta hai — '40%', '$1M', '10x' etc."""
    patterns = [
        r'\d+\s*%',
        r'\$\s*\d+[\d,kmb]*',
        r'\d+\s*x\b',
        r'\d+\+\s*(users|clients|customers|teams|projects|employees)',
        r'(increased|decreased|reduced|improved|grew|saved)\s+\w+\s+by\s+\d+'
    ]
    achievements = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        achievements.extend(matches)
    return list(set(achievements))


def calculate_pre_score(text, keywords, contact, achievements):
    """
    TUMHARA APNA SCORING ALGORITHM
    AI se independent — pure rule-based logic
    """
    score = 0
    breakdown = {}

    # 1. Technical Skills 
    tech_count = len(keywords["technical"])
    tech_score = min(30, tech_count * 3)
    score += tech_score
    breakdown["technical_skills"] = {"score": tech_score, "max": 30, "found": tech_count}

    # 2. Soft Skills 
    soft_count = len(keywords["soft"])
    soft_score = min(15, soft_count * 3)
    score += soft_score
    breakdown["soft_skills"] = {"score": soft_score, "max": 15, "found": soft_count}

    # 3. Action Verbs
    action_count = len(keywords["action_verbs"])
    action_score = min(15, action_count * 2)
    score += action_score
    breakdown["action_verbs"] = {"score": action_score, "max": 15, "found": action_count}

    # 4. Quantified Achievements 
    ach_count = len(achievements)
    ach_score = min(20, ach_count * 5)
    score += ach_score
    breakdown["quantified_achievements"] = {"score": ach_score, "max": 20, "found": ach_count}

    # 5. Contact Info
    contact_score = sum([
        4 if contact.get("email") else 0,
        3 if contact.get("phone") else 0,
        2 if contact.get("linkedin") else 0,
        1 if contact.get("github") else 0
    ])
    score += contact_score
    breakdown["contact_info"] = {"score": contact_score, "max": 10, "details": contact}

    # 6. Resume Length
    word_count = len(text.split())
    if 300 <= word_count <= 800:
        length_score = 10
        length_note = "Ideal length"
    elif word_count < 300:
        length_score = max(0, word_count // 30)
        length_note = "Too short"
    else:
        length_score = max(5, 10 - (word_count - 800) // 100)
        length_note = "Too long"
    score += length_score
    breakdown["resume_length"] = {"score": length_score, "max": 10, "word_count": word_count, "note": length_note}

    return min(100, score), breakdown


# JOB DESCRIPTION MATCH ENGINE

def calculate_jd_match(resume_text, jd_text):
    """
    Resume aur Job Description ke beech match % calculate karta hai
    Tumhara apna keyword matching algorithm
    """
    resume_lower = resume_text.lower()
    jd_lower = jd_text.lower()

    #  Keywords from JD
    jd_words = set(re.findall(r'\b[a-z][a-z0-9+#.\-]*\b', jd_lower))
    jd_bigrams = set()
    jd_word_list = jd_lower.split()
    for i in range(len(jd_word_list) - 1):
        jd_bigrams.add(jd_word_list[i] + " " + jd_word_list[i+1])

    # Keywords from Resume
    resume_words = set(re.findall(r'\b[a-z][a-z0-9+#.\-]*\b', resume_lower))
    resume_bigrams = set()
    resume_word_list = resume_lower.split()
    for i in range(len(resume_word_list) - 1):
        resume_bigrams.add(resume_word_list[i] + " " + resume_word_list[i+1])

    # Important keywords from JD (technical + soft skills)
    all_keywords = TECHNICAL_SKILLS + SOFT_SKILLS
    jd_important = []
    for kw in all_keywords:
        if kw in jd_words or kw in jd_bigrams:
            jd_important.append(kw)

    if not jd_important:
        # Fallback: frequent meaningful words 
        stopwords = {"the", "and", "for", "are", "with", "you", "will", "have",
                     "this", "that", "from", "they", "been", "our", "your", "not",
                     "but", "all", "can", "was", "were", "their", "has", "its"}
        freq = {}
        for w in jd_word_list:
            if len(w) > 3 and w not in stopwords:
                freq[w] = freq.get(w, 0) + 1
        jd_important = sorted(freq, key=freq.get, reverse=True)[:20]

    # Match checking
    matched = []
    missing = []
    for kw in jd_important:
        in_resume = kw in resume_words or kw in resume_bigrams
        if in_resume:
            matched.append(kw)
        else:
            missing.append(kw)

    total = len(jd_important)
    match_pct = round((len(matched) / total * 100) if total > 0 else 0)

    return {
        "match_percentage": match_pct,
        "matched_keywords": matched,
        "missing_keywords": missing,
        "total_jd_keywords": total
    }


# FILE EXTRACTION

def extract_text_from_pdf(file_bytes):
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception:
        return None


def extract_text_from_docx(file_bytes):
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text.strip()
    except Exception:
        return None


def get_file_hash(file_bytes, job_role, jd_text):
    content = file_bytes + job_role.encode("utf-8") + jd_text.encode("utf-8")
    return hashlib.md5(content).hexdigest()


# AI ANALYSIS

def analyze_with_ai(resume_text, job_role, pre_score, keywords, achievements):
    role_context = f'Target job role: "{job_role}"' if job_role else "General job readiness analysis."

    prompt = f"""You are an expert resume reviewer. Analyze this resume and return ONLY valid JSON.
Be consistent — same resume must always get same response.

{role_context}

Pre-analysis data (use this to calibrate your scores):
- Rule-based pre-score: {pre_score}/100
- Technical skills found: {', '.join(keywords['technical'][:10]) or 'None'}
- Soft skills found: {', '.join(keywords['soft'][:5]) or 'None'}
- Action verbs found: {', '.join(keywords['action_verbs'][:5]) or 'None'}
- Quantified achievements: {len(achievements)} found

Resume:
\"\"\"
{resume_text[:3000]}
\"\"\"

Return exactly this JSON:
{{
  "overall_score": <integer, calibrate near {pre_score}>,
  "ats_score": <integer 0-100>,
  "impact_score": <integer 0-100>,
  "summary": "<2-3 sentence assessment>",
  "strengths": ["<strength 1>", "<strength 2>", "<strength 3>", "<strength 4>"],
  "weaknesses": ["<issue 1>", "<issue 2>", "<issue 3>", "<issue 4>"],
  "improvements": ["<tip 1>", "<tip 2>", "<tip 3>", "<tip 4>", "<tip 5>"],
  "present_skills": {json.dumps(keywords['technical'][:12])},
  "missing_skills": ["<important missing skill>", "<skill 2>", "<skill 3>", "<skill 4>"]
}}"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                top_p=1,
                top_k=1,
            )
        )
        raw = response.text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        return result, None
    except json.JSONDecodeError:
        return None, "Could not parse AI response. Please try again."
    except Exception as e:
        return None, f"AI error: {str(e)}"


# ROUTES

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "resume" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["resume"]
    job_role = request.form.get("job_role", "").strip()
    jd_text = request.form.get("jd_text", "").strip()

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    filename = file.filename.lower()
    file_bytes = file.read()
    resume_text = None

    if filename.endswith(".pdf"):
        resume_text = extract_text_from_pdf(file_bytes)
        if not resume_text:
            return jsonify({"error": "Could not extract text from PDF."}), 400
    elif filename.endswith(".docx"):
        resume_text = extract_text_from_docx(file_bytes)
        if not resume_text:
            return jsonify({"error": "Could not read DOCX file."}), 400
    elif filename.endswith(".txt"):
        try:
            resume_text = file_bytes.decode("utf-8")
        except Exception:
            return jsonify({"error": "Could not read TXT file."}), 400
    else:
        return jsonify({"error": "Only PDF, DOCX, or TXT files are supported."}), 400

    if len(resume_text.strip()) < 100:
        return jsonify({"error": "File has too little text. Please upload a proper resume."}), 400

    # Cache check
    cache_key = get_file_hash(file_bytes, job_role, jd_text)
    if cache_key in results_cache:
        print(f"Cache hit — {cache_key[:8]}")
        return jsonify(results_cache[cache_key])

    print(f"Cache miss — analyzing — {cache_key[:8]}")

    keywords = extract_keywords(resume_text)
    contact = check_contact_info(resume_text)
    achievements = check_quantified_achievements(resume_text)
    pre_score, breakdown = calculate_pre_score(resume_text, keywords, contact, achievements)

    # JD Match
    jd_match = None
    if jd_text and len(jd_text) > 50:
        jd_match = calculate_jd_match(resume_text, jd_text)

    # AI Analysis
    result, error = analyze_with_ai(resume_text, job_role, pre_score, keywords, achievements)
    if error:
        return jsonify({"error": error}), 500

    result["pre_score"] = pre_score
    result["score_breakdown"] = breakdown
    result["contact_info"] = contact
    result["quantified_achievements"] = achievements[:8]
    result["jd_match"] = jd_match

    results_cache[cache_key] = result
    return jsonify(result)


if __name__ == "__main__":
    print("=" * 50)
    print("  AI Resume Analyzer is running!")
    print("  Open in browser: http://127.0.0.1:5000")
    print("=" * 50)
import os
port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port, debug=False)
