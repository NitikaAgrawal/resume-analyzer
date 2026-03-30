import os
import io
import re
import json
import hashlib

from flask import Flask, request, jsonify, render_template
from google import genai
from google.genai import types
import PyPDF2
import docx

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
    text_lower = text.lower()
    words = set(re.findall(r'\b[a-z][a-z0-9+#.\-]*\b', text_lower))
    bigrams = set()
    word_list = text_lower.split()
    for i in range(len(word_list) - 1):
        bigrams.add(word_list[i] + " " + word_list[i+1])
    return {
        "technical": [s for s in TECHNICAL_SKILLS if s in words or s in bigrams],
        "soft": [s for s in SOFT_SKILLS if s in words or s in bigrams],
        "action_verbs": [v for v in ACTION_VERBS if v in words]
    }


def check_contact_info(text):
    return {k: bool(re.search(p, text, re.IGNORECASE)) for k, p in CONTACT_PATTERNS.items()}


def check_quantified_achievements(text):
    patterns = [
        r'\d+\s*%', r'\$\s*\d+[\d,kmb]*', r'\d+\s*x\b',
        r'\d+\+\s*(users|clients|customers|teams|projects|employees)',
        r'(increased|decreased|reduced|improved|grew|saved)\s+\w+\s+by\s+\d+'
    ]
    achievements = []
    for p in patterns:
        achievements.extend(re.findall(p, text, re.IGNORECASE))
    return list(set(str(a) for a in achievements))


def calculate_pre_score(text, keywords, contact, achievements):
    score = 0
    breakdown = {}
    tech_score = min(30, len(keywords["technical"]) * 3)
    score += tech_score
    breakdown["technical_skills"] = {"score": tech_score, "max": 30, "found": len(keywords["technical"])}
    soft_score = min(15, len(keywords["soft"]) * 3)
    score += soft_score
    breakdown["soft_skills"] = {"score": soft_score, "max": 15, "found": len(keywords["soft"])}
    action_score = min(15, len(keywords["action_verbs"]) * 2)
    score += action_score
    breakdown["action_verbs"] = {"score": action_score, "max": 15, "found": len(keywords["action_verbs"])}
    ach_score = min(20, len(achievements) * 5)
    score += ach_score
    breakdown["quantified_achievements"] = {"score": ach_score, "max": 20, "found": len(achievements)}
    contact_score = sum([4 if contact.get("email") else 0, 3 if contact.get("phone") else 0,
                         2 if contact.get("linkedin") else 0, 1 if contact.get("github") else 0])
    score += contact_score
    breakdown["contact_info"] = {"score": contact_score, "max": 10, "details": contact}
    word_count = len(text.split())
    if 300 <= word_count <= 800:
        length_score, length_note = 10, "Ideal length"
    elif word_count < 300:
        length_score, length_note = max(0, word_count // 30), "Too short"
    else:
        length_score, length_note = max(5, 10 - (word_count - 800) // 100), "Too long"
    score += length_score
    breakdown["resume_length"] = {"score": length_score, "max": 10, "word_count": word_count, "note": length_note}
    return min(100, score), breakdown


def calculate_jd_match(resume_text, jd_text):
    resume_lower = resume_text.lower()
    jd_lower = jd_text.lower()
    resume_words = set(re.findall(r'\b[a-z][a-z0-9+#.\-]*\b', resume_lower))
    jd_words = set(re.findall(r'\b[a-z][a-z0-9+#.\-]*\b', jd_lower))
    resume_bigrams, jd_bigrams = set(), set()
    rl, jl = resume_lower.split(), jd_lower.split()
    for i in range(len(rl)-1): resume_bigrams.add(rl[i]+" "+rl[i+1])
    for i in range(len(jl)-1): jd_bigrams.add(jl[i]+" "+jl[i+1])
    all_keywords = TECHNICAL_SKILLS + SOFT_SKILLS
    jd_important = [kw for kw in all_keywords if kw in jd_words or kw in jd_bigrams]
    if not jd_important:
        stopwords = {"the","and","for","are","with","you","will","have","this","that","from","they","been","our","your","not","but","all","can","was","were","their","has","its"}
        freq = {}
        for w in jl:
            if len(w) > 3 and w not in stopwords:
                freq[w] = freq.get(w, 0) + 1
        jd_important = sorted(freq, key=freq.get, reverse=True)[:20]
    matched = [kw for kw in jd_important if kw in resume_words or kw in resume_bigrams]
    missing = [kw for kw in jd_important if kw not in resume_words and kw not in resume_bigrams]
    total = len(jd_important)
    return {
        "match_percentage": round((len(matched)/total*100) if total > 0 else 0),
        "matched_keywords": matched,
        "missing_keywords": missing,
        "total_jd_keywords": total
    }


def extract_text_from_pdf(file_bytes):
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except Exception:
        return None


def extract_text_from_docx(file_bytes):
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs).strip()
    except Exception:
        return None


def analyze_with_ai(resume_text, job_role, pre_score, keywords, achievements):
    role_context = f'Target job role: "{job_role}"' if job_role else "General job readiness analysis."
    prompt = f"""You are an expert resume reviewer. Analyze this resume and return ONLY valid JSON. No markdown.
Be consistent — same resume must always get same response.
{role_context}
Pre-analysis: rule-based pre-score={pre_score}/100, technical skills found={len(keywords['technical'])}, achievements={len(achievements)}
Resume:
\"\"\"{resume_text[:3000]}\"\"\"
Return exactly:
{{
  "overall_score": <integer near {pre_score}>,
  "ats_score": <integer 0-100>,
  "impact_score": <integer 0-100>,
  "summary": "<2-3 sentence assessment>",
  "strengths": ["<s1>","<s2>","<s3>","<s4>"],
  "weaknesses": ["<w1>","<w2>","<w3>","<w4>"],
  "improvements": ["<t1>","<t2>","<t3>","<t4>","<t5>"],
  "present_skills": {json.dumps(keywords['technical'][:12])},
  "missing_skills": ["<m1>","<m2>","<m3>","<m4>"]
}}"""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0, top_p=1, top_k=1)
        )
        raw = response.text.strip().replace("```json","").replace("```","").strip()
        return json.loads(raw), None
    except json.JSONDecodeError:
        return None, "Could not parse AI response. Please try again."
    except Exception as e:
        return None, f"AI error: {str(e)}"


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
    cache_key = hashlib.md5((file_bytes + job_role.encode() + jd_text.encode())).hexdigest()
    if cache_key in results_cache:
        return jsonify(results_cache[cache_key])
    keywords = extract_keywords(resume_text)
    contact = check_contact_info(resume_text)
    achievements = check_quantified_achievements(resume_text)
    pre_score, breakdown = calculate_pre_score(resume_text, keywords, contact, achievements)
    jd_match = calculate_jd_match(resume_text, jd_text) if jd_text and len(jd_text) > 50 else None
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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
