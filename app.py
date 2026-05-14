import os, io, re, json, hashlib, tempfile, subprocess, shutil
from flask import Flask, request, jsonify, render_template, send_file
from google import genai
from google.genai import types
import PyPDF2
import docx

app = Flask(__name__)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
client = genai.Client(api_key=GEMINI_API_KEY)
results_cache = {}

TECHNICAL_SKILLS = [
    "python","javascript","typescript","java","c++","c#","go","rust","swift","kotlin",
    "react","angular","vue","node","django","flask","fastapi","spring","express",
    "sql","mysql","postgresql","mongodb","redis","elasticsearch","firebase",
    "aws","azure","gcp","docker","kubernetes","terraform","jenkins","github actions",
    "machine learning","deep learning","nlp","computer vision","tensorflow","pytorch",
    "pandas","numpy","scikit-learn","tableau","power bi","excel",
    "html","css","sass","tailwind","bootstrap","figma","sketch",
    "rest","graphql","api","microservices","ci/cd","devops","agile","scrum",
    "linux","bash","git","selenium","jest","pytest","r"
]
SOFT_SKILLS = [
    "leadership","communication","teamwork","collaboration","problem solving",
    "critical thinking","time management","project management","mentoring",
    "adaptability","creativity","analytical","detail-oriented","organized",
    "self-motivated","initiative","presentation","negotiation","strategic"
]
ACTION_VERBS = [
    "built","developed","designed","implemented","created","launched","led",
    "managed","improved","increased","reduced","optimized","automated",
    "architected","deployed","integrated","delivered","achieved","spearheaded",
    "collaborated","mentored","trained","analyzed","researched","established"
]
CONTACT_PATTERNS = {
    "email":    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "phone":    r"(\+?\d[\d\s\-().]{7,}\d)",
    "linkedin": r"(https?://(?:www\.)?linkedin\.com/in/[a-zA-Z0-9\-/]+)",
    "github":   r"(https?://github\.com/[a-zA-Z0-9\-]+)"
}
JOB_ROLES_DB = [
    {"role":"Frontend Developer",  "skills":["html","css","javascript","react","vue","angular","typescript","figma","tailwind","sass"]},
    {"role":"Backend Developer",   "skills":["python","node","java","django","flask","fastapi","express","spring","sql","postgresql","mongodb","rest","api"]},
    {"role":"Full Stack Developer","skills":["html","css","javascript","react","node","python","sql","mongodb","docker","git","api"]},
    {"role":"Data Scientist",      "skills":["python","machine learning","deep learning","pandas","numpy","scikit-learn","tensorflow","pytorch","sql","tableau","r"]},
    {"role":"Data Analyst",        "skills":["sql","python","excel","tableau","power bi","pandas","r","analytical"]},
    {"role":"DevOps Engineer",     "skills":["docker","kubernetes","aws","azure","gcp","ci/cd","jenkins","terraform","linux","bash","git"]},
    {"role":"ML Engineer",         "skills":["python","machine learning","deep learning","tensorflow","pytorch","nlp","computer vision","docker","aws","git"]},
    {"role":"UI/UX Designer",      "skills":["figma","sketch","css","html","javascript","analytical","creativity","presentation"]},
    {"role":"Cloud Architect",     "skills":["aws","azure","gcp","docker","kubernetes","terraform","microservices","ci/cd","linux"]},
    {"role":"Product Manager",     "skills":["agile","scrum","project management","analytical","communication","strategic","presentation","leadership"]},
]

# ── Find pdflatex automatically ──
def find_pdflatex():
    for path in ['/Library/TeX/texbin/pdflatex', '/usr/bin/pdflatex',
                 '/usr/local/bin/pdflatex', '/opt/homebrew/bin/pdflatex']:
        if os.path.exists(path):
            return path
    # Try which
    try:
        result = subprocess.run(['which', 'pdflatex'], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None

PDFLATEX = find_pdflatex()


# ─────────────────────────────────────────────
# LATEX HELPERS
# ─────────────────────────────────────────────
def e(text):
    """Escape special LaTeX characters. Safe for use inside LaTeX commands."""
    if not isinstance(text, str):
        return str(text)
    # Order matters — backslash must be first
    text = text.replace('\\', r'\textbackslash{}')
    text = text.replace('&',  r'\&')
    text = text.replace('%',  r'\%')
    text = text.replace('$',  r'\$')
    text = text.replace('#',  r'\#')
    text = text.replace('^',  r'\^{}')
    text = text.replace('_',  r'\_')
    text = text.replace('{',  r'\{')
    text = text.replace('}',  r'\}')
    text = text.replace('~',  r'\~{}')
    return text

def safe_bullet(text):
    """
    Safe escape for bullet text inside \\resumeItem{...}.
    Strategy: do NOT use e() here. Instead manually handle only
    the characters that are safe to escape, and replace braces
    with parentheses since curly braces break \\resumeItem{}.
    """
    if not isinstance(text, str):
        text = str(text)
    # Replace curly braces with parentheses BEFORE any escaping
    # (they will always break \resumeItem{...})
    text = text.replace('{', '(').replace('}', ')')
    # Now escape remaining special LaTeX chars
    text = text.replace('\\', '')          # strip backslashes
    text = text.replace('&',  r'\&')
    text = text.replace('%',  r'\%')
    text = text.replace('$',  r'\$')
    text = text.replace('#',  r'\#')
    text = text.replace('^',  r'\^{}')
    text = text.replace('_',  r'\_')
    text = text.replace('~',  r'\~{}')
    return text.strip()

def compile_latex(source):
    """Compile LaTeX source to PDF. Returns (pdf_bytes, error_string)."""
    if not PDFLATEX:
        return None, "pdflatex not found. Install MacTeX from tug.org/mactex"

    tmpdir = tempfile.mkdtemp()
    try:
        tex_path = os.path.join(tmpdir, 'doc.tex')
        pdf_path = os.path.join(tmpdir, 'doc.pdf')
        log_path = os.path.join(tmpdir, 'doc.log')

        with open(tex_path, 'w', encoding='utf-8') as f:
            f.write(source)

        # Compile twice for proper rendering
        for _ in range(2):
            subprocess.run(
                [PDFLATEX, '-interaction=nonstopmode', 'doc.tex'],
                cwd=tmpdir, capture_output=True, timeout=60
            )

        if os.path.exists(pdf_path):
            with open(pdf_path, 'rb') as f:
                return f.read(), None

        # Extract errors from log
        errors = []
        if os.path.exists(log_path):
            with open(log_path, encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if line.startswith('!'):
                        errors.append(line.strip())
        return None, "LaTeX error: " + "; ".join(errors[:3]) if errors else "LaTeX compilation failed"

    except subprocess.TimeoutExpired:
        return None, "LaTeX compilation timed out"
    except Exception as ex:
        return None, f"Compilation error: {str(ex)}"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ─────────────────────────────────────────────
# RESUME LaTeX BUILDER
# ─────────────────────────────────────────────
def build_resume_latex(d):
    # ── Experience ──
    exp_tex = ""
    for ex in d.get("experience", []):
        bullets = "\n".join(
            f"        \\resumeItem{{{safe_bullet(b)}}}"
            for b in ex.get("bullets", [])
        )
        ts = ex.get("techstack", "")
        if ts:
            bullets += f"\n        \\resumeItem{{\\textbf{{Tech Stack: {e(ts)}}}}}"
        exp_tex += f"""
\\resumeSubheading
      {{\\textbf{{\\textcolor{{mycolor}}{{\\textbf \\Large {e(ex['company'])}}}}}}}{{\\textcolor{{mycolor}}{{{e(ex['dates'])}}}}}
      {{\\textbf{{\\textcolor{{mycolor}}{{{e(ex['role'])}}}}}}}{{\\textcolor{{mycolor}}{{{e(ex['location'])}}}}}
      \\resumeItemListStart
{bullets}
      \\resumeItemListEnd"""

    # ── Projects ──
    proj_tex = ""
    for p in d.get("projects", []):
        tech = p.get("tech", "")
        tech_str = f"\\emph{{\\textbf{{\\textcolor{{mycolor}}{{{e(tech)}}}}}}}" if tech else ""
        heading = (f"\\textbf{{\\textcolor{{mycolor}}{{{e(p['title'])}}}}} \\textbar\\ {tech_str}"
                   if tech else f"\\textbf{{\\textcolor{{mycolor}}{{{e(p['title'])}}}}}")
        link = p.get("link", "")
        link_part = (f"\\href{{{link}}}{{\\underline{{\\textbf{{\\textcolor{{mycolor}}{{LINK}}}}}}}}"
                     if link else "")
        bullets = "\n".join(
            f"        \\resumeItem{{{safe_bullet(b)}}}"
            for b in p.get("bullets", [])
        )
        proj_tex += f"""
\\resumeProjectHeading
  {{{heading}}}
  {{{link_part}}}
      \\resumeItemListStart
{bullets}
      \\resumeItemListEnd"""

    # ── Education ──
    edu_tex = ""
    for ed in d.get("education", []):
        edu_tex += f"""
  \\resumeSubheading
    {{{e(ed['institution'])}}}{{{e(ed['dates'])}}}
    {{{e(ed['degree'])}}}{{{e(ed.get('detail', ''))}}}"""

    # ── Skills ──
    skills_tex = ""
    for cat, vals in d.get("skills", {}).items():
        skills_tex += f"\\textbf{{{e(cat)}:}} {e(vals)} \\\\\n"

    # ── Certificates ──
    certs_tex = ""
    for cert in d.get("certificates", []):
        url = cert.get("url", "")
        nm  = e(cert.get("name", ""))
        certs_tex += (f"\\href{{{url}}}{{\\textbf{{{nm}}}}} $|$\n"
                      if url else f"\\textbf{{{nm}}} $|$\n")

    name         = e(d.get("name", ""))
    phone        = e(d.get("phone", ""))
    email_val    = e(d.get("email", ""))
    linkedin_url = d.get("linkedin_url", "")
    github_url   = d.get("github_url", "")
    summary      = e(d.get("summary", ""))

    return (r"""\documentclass[letterpaper,11pt]{article}
\usepackage{latexsym}
\usepackage[empty]{fullpage}
\usepackage{titlesec}
\usepackage{marvosym}
\usepackage[usenames,dvipsnames]{color}
\usepackage{verbatim}
\usepackage{enumitem}
\usepackage[hidelinks]{hyperref}
\usepackage{fancyhdr}
\usepackage[english]{babel}
\usepackage{tabularx}
\usepackage{xcolor}
\definecolor{myblue}{HTML}{1A73E8}
\definecolor{mydarkblue}{HTML}{003366}
\definecolor{mycolor}{HTML}{0023F5}
\pagestyle{fancy}
\fancyhf{}
\fancyfoot{}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}
\addtolength{\oddsidemargin}{-0.5in}
\addtolength{\evensidemargin}{-0.5in}
\addtolength{\textwidth}{1in}
\addtolength{\topmargin}{-.5in}
\addtolength{\textheight}{1.0in}
\urlstyle{same}
\raggedbottom
\raggedright
\setlength{\tabcolsep}{0in}
\titleformat{\section}{\vspace{-4pt}\scshape\raggedright\large}{}{0em}{}[\color{black}\titlerule \vspace{-5pt}]
\newcommand{\resumeItem}[1]{\item\small{#1 \vspace{-2pt}}}
\newcommand{\resumeSubheading}[4]{
  \vspace{-2pt}\item
  \begin{tabular*}{0.97\textwidth}[t]{l@{\extracolsep{\fill}}r}
    \textbf{#1} & #2 \\
    \textit{\small#3} & \textit{\small #4} \\
  \end{tabular*}\vspace{-7pt}
}
\newcommand{\resumeSubSubheading}[2]{
  \item
  \begin{tabular*}{0.97\textwidth}{l@{\extracolsep{\fill}}r}
    \textit{\small#1} & \textit{\small #2} \\
  \end{tabular*}\vspace{-7pt}
}
\newcommand{\resumeProjectHeading}[2]{
  \item
  \begin{tabular*}{0.97\textwidth}{l@{\extracolsep{\fill}}r}
    \small#1 & #2 \\
  \end{tabular*}\vspace{-7pt}
}
\newcommand{\resumeSubItem}[1]{\resumeItem{#1}\vspace{-4pt}}
\renewcommand\labelitemii{$\vcenter{\hbox{\tiny$\bullet$}}$}
\newcommand{\resumeSubHeadingListStart}{\begin{itemize}[leftmargin=0.15in, label={}]}
\newcommand{\resumeSubHeadingListEnd}{\end{itemize}}
\newcommand{\resumeItemListStart}{\begin{itemize}}
\newcommand{\resumeItemListEnd}{\end{itemize}\vspace{-5pt}}
\begin{document}
\begin{center}
    \textbf{\Huge \scshape """ + name + r"""} \\ \vspace{1pt}
    \small """ + phone + r""" $|$
    \small """ + email_val + r""" $|$
    \href{""" + linkedin_url + r"""}{\underline{LinkedIn}} $|$
    \href{""" + github_url + r"""}{\underline{GitHub}}
\end{center}
\section{\textbf{\textcolor{myblue}{\Large Profile Summary}}}
""" + summary + r"""
\section{\textbf{\textcolor{myblue}{\Large Experience}}}
\resumeSubHeadingListStart""" + exp_tex + r"""
\resumeSubHeadingListEnd
\section{\textbf{\textcolor{myblue}{\Large Projects}}}
  \resumeSubHeadingListStart""" + proj_tex + r"""
  \resumeSubHeadingListEnd
\section{\textbf{\textcolor{myblue}{\Large Education}}}
\resumeSubHeadingListStart""" + edu_tex + r"""
\resumeSubHeadingListEnd
\section{\textbf{\textcolor{myblue}{\Large Technical Skills}}}
\begin{itemize}[leftmargin=0.15in, label={}]
\small{\item{
""" + skills_tex + r"""}}
\end{itemize}
\section{\textbf{\textcolor{myblue}{\Large Certificates}}}
\begin{itemize}[leftmargin=0.15in, label={}]
""" + certs_tex + r"""
\end{itemize}
\end{document}""")


# ─────────────────────────────────────────────
# COVER LETTER LaTeX BUILDER — Sid Lacy template
# ─────────────────────────────────────────────
def build_cover_letter_latex(d):
    name        = e(d.get("name", ""))
    email_raw   = d.get("email", "")
    email_esc   = e(email_raw)
    phone_raw   = d.get("phone", "").replace(" ", "")
    phone_esc   = e(d.get("phone", ""))
    linkedin_h  = d.get("linkedin_handle", "")
    location_v  = e(d.get("location", ""))
    recipient_v = e(d.get("recipient", "Hiring Manager"))
    company_v   = e(d.get("company", ""))
    title_v     = e(d.get("title", "Software Engineer"))

    paras    = d.get("body_paragraphs", [])
    # Each paragraph separated by blank line for proper LaTeX paragraph spacing
    body_tex = "\n\n".join(e(p) for p in paras)

    # Build company address block — only if provided
    street_v = e(d.get("street", ""))
    city_v   = e(d.get("city", ""))
    state_v  = e(d.get("state", ""))
    zip_v    = e(d.get("zip", ""))

    company_address = company_v + r"\\"
    if street_v:
        company_address += "\n" + street_v + r"\\"
    if city_v:
        company_address += "\n" + city_v
        if state_v: company_address += ", " + state_v
        if zip_v:   company_address += r"\ " + zip_v
        company_address += r"\\"

    return (r"""\documentclass[12pt]{letter}
\usepackage[utf8]{inputenc}
\usepackage[empty]{fullpage}
\usepackage[hidelinks]{hyperref}
\usepackage{eso-pic}
\usepackage{charter}
\usepackage{xcolor}
\addtolength{\topmargin}{-0.5in}
\addtolength{\textheight}{1.0in}
\definecolor{gr}{RGB}{225,225,225}
\begin{document}

\AddToShipoutPictureBG{%
\color{gr}
\AtPageUpperLeft{\rule[-1.3in]{\paperwidth}{1.3in}}
}

% ── Header ──
\begin{center}
{\fontsize{28}{0}\selectfont\scshape """ + name + r"""} \\ \vspace{4pt}
\small
\href{mailto:""" + email_raw + """}{""" + email_esc + r"""} \hspace{1em}
\href{https://linkedin.com/in/""" + linkedin_h + """}{linkedin.com/in/""" + e(linkedin_h) + r"""} \hspace{1em}
\href{tel:""" + phone_raw + """}{""" + phone_esc + r"""} \hspace{1em}
""" + location_v + r"""
\end{center}

\vspace{0.25in}

% ── Opening block ──
\today\\[4pt]
""" + recipient_v + r"""\\
""" + company_address + r"""
\vspace{0.1in}

Dear """ + recipient_v + r""",

% ── Body ──
\vspace{0.05in}
\setlength\parindent{24pt}

\noindent """ + body_tex + r"""

% ── Closer ──
\vspace{0.1in}
\vfill
\begin{flushright}
Sincerely,\\[16pt]
""" + name + r"""\\
""" + title_v + r"""
\end{flushright}

\end{document}""")


# ─────────────────────────────────────────────
# AI — EXTRACT + IMPROVE RESUME
# ─────────────────────────────────────────────
def ai_extract_and_improve(resume_text, improvements, missing_skills, job_role, exp_level):
    imp_str  = "\n".join(f"- {i}" for i in improvements)
    miss_str = ", ".join(missing_skills) if missing_skills else "none"

    prompt = f"""You are a professional resume editor. Study the original resume below carefully.

STRICT RULES:
1. Extract ALL data EXACTLY as written: name, phone, email, URLs, company names, dates, locations, project titles, education, grades, certificate names and URLs. Do NOT change or invent anything.
2. ONLY rewrite bullet points and summary using strong action verbs and quantified metrics.
3. Keep ALL projects, ALL experience entries, ALL education entries. Do not remove any.
4. Add missing skills naturally into existing skill categories.
5. NEVER use curly braces {{ or }} anywhere in bullet text or summary. Use regular parentheses () instead.
6. NEVER use backslash \\ anywhere in bullet text.
7. Preserve original dates exactly. Use -- for dash (e.g. "April 2021 -- Jan 2026").
8. Bullet text must be plain English sentences only. No code, no special symbols.

Resume:
\"\"\"
{resume_text[:5000]}
\"\"\"

Target role: {job_role or "general"}
Level: {exp_level or "mid"}

Suggestions to apply to bullets and summary:
{imp_str}

Missing skills to add: {miss_str}

Return ONLY valid JSON (no markdown):
{{
  "name": "<full name>",
  "phone": "<phone>",
  "email": "<email>",
  "linkedin_url": "<full LinkedIn URL>",
  "github_url": "<full GitHub URL>",
  "summary": "<improved 2-3 sentence summary, plain English>",
  "experience": [
    {{
      "company": "<exact>",
      "dates": "<exact, use -- for dash>",
      "role": "<exact>",
      "location": "<exact>",
      "bullets": ["<plain English bullet>"],
      "techstack": "<tech stack if present, else empty string>"
    }}
  ],
  "projects": [
    {{
      "title": "<exact>",
      "tech": "<tech display string>",
      "link": "<URL or empty>",
      "bullets": ["<plain English bullet>"]
    }}
  ],
  "education": [
    {{
      "institution": "<exact>",
      "dates": "<exact>",
      "degree": "<exact degree and grade>",
      "detail": ""
    }}
  ],
  "skills": {{
    "<Category>": "<comma-separated values>"
  }},
  "certificates": [
    {{"name": "<exact name>", "url": "<URL or empty>"}}
  ]
}}"""

    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash-lite", contents=prompt,
            config=types.GenerateContentConfig(temperature=0.1))
        raw = resp.text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(raw), None
    except json.JSONDecodeError as ex:
        return None, f"JSON parse error: {ex}"
    except Exception as ex:
        return None, f"AI error: {ex}"


# ─────────────────────────────────────────────
# AI — COVER LETTER BODY
# ─────────────────────────────────────────────
def ai_cover_letter_body(name, job_role, company, background, title):
    prompt = f"""Write a professional cover letter body.
Name: {name} — {title}
Role: {job_role} at {company}
Background: {background or "Experienced software professional"}

Write EXACTLY 3 paragraphs separated by a blank line:
- Para 1 (60 words): Genuine interest in this role at this company. Mention role and company name.
- Para 2 (100 words): 2-3 specific achievements with numbers. Concrete and relevant.
- Para 3 (50 words): Confident closing, thank them.

Rules: No salutation, no sign-off, plain text only, no markdown, under 250 words.
Return ONLY the 3 paragraphs."""

    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash-lite", contents=prompt,
            config=types.GenerateContentConfig(temperature=0.3))
        body  = resp.text.strip()
        paras = [p.strip() for p in re.split(r'\n{2,}', body) if p.strip()]
        return paras, None
    except Exception as ex:
        return None, str(ex)


# ─────────────────────────────────────────────
# ANALYSIS HELPERS
# ─────────────────────────────────────────────
def extract_keywords(text):
    tl = text.lower()
    words = set(re.findall(r'\b[a-z][a-z0-9+#.\-]*\b', tl))
    bigrams = set()
    wl = tl.split()
    for i in range(len(wl)-1):
        bigrams.add(wl[i] + " " + wl[i+1])
    return {
        "technical":    [s for s in TECHNICAL_SKILLS if s in words or s in bigrams],
        "soft":         [s for s in SOFT_SKILLS      if s in words or s in bigrams],
        "action_verbs": [v for v in ACTION_VERBS     if v in words]
    }

def check_contact_info(text):
    result = {}
    for k, p in CONTACT_PATTERNS.items():
        m = re.search(p, text, re.IGNORECASE)
        result[k] = m.group(0) if m else None
    return result

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
    score, breakdown = 0, {}
    ts = min(30, len(keywords["technical"]) * 3); score += ts
    breakdown["technical_skills"] = {"score": ts, "max": 30, "found": len(keywords["technical"])}
    ss = min(15, len(keywords["soft"]) * 3); score += ss
    breakdown["soft_skills"] = {"score": ss, "max": 15, "found": len(keywords["soft"])}
    av = min(15, len(keywords["action_verbs"]) * 2); score += av
    breakdown["action_verbs"] = {"score": av, "max": 15, "found": len(keywords["action_verbs"])}
    ach = min(20, len(achievements) * 5); score += ach
    breakdown["quantified_achievements"] = {"score": ach, "max": 20, "found": len(achievements)}
    cf = sum(1 for v in contact.values() if v)
    cs = min(10, cf * 3); score += cs
    breakdown["contact_info"] = {"score": cs, "max": 10, "details": {k: bool(v) for k, v in contact.items()}}
    wc = len(text.split())
    if 300 <= wc <= 800:   ls, ln = 10, "Ideal length"
    elif wc < 300:          ls, ln = max(0, wc // 30), "Too short"
    else:                   ls, ln = max(5, 10 - (wc - 800) // 100), "Too long"
    score += ls
    breakdown["resume_length"] = {"score": ls, "max": 10, "word_count": wc, "note": ln}
    return min(100, score), breakdown

def calculate_jd_match(resume_text, jd_text):
    rl, jl = resume_text.lower().split(), jd_text.lower().split()
    rw = set(re.findall(r'\b[a-z][a-z0-9+#.\-]*\b', resume_text.lower()))
    jw = set(re.findall(r'\b[a-z][a-z0-9+#.\-]*\b', jd_text.lower()))
    rb, jb = set(), set()
    for i in range(len(rl)-1): rb.add(rl[i] + " " + rl[i+1])
    for i in range(len(jl)-1): jb.add(jl[i] + " " + jl[i+1])
    important = [kw for kw in TECHNICAL_SKILLS + SOFT_SKILLS if kw in jw or kw in jb]
    if not important:
        sw = {"the","and","for","are","with","you","will","have","this","that","from","they",
              "been","our","your","not","but","all","can","was","were","their","has","its"}
        freq = {}
        for w in jl:
            if len(w) > 3 and w not in sw:
                freq[w] = freq.get(w, 0) + 1
        important = sorted(freq, key=freq.get, reverse=True)[:20]
    matched = [kw for kw in important if kw in rw or kw in rb]
    missing  = [kw for kw in important if kw not in rw and kw not in rb]
    total = len(important)
    return {
        "match_percentage": round(len(matched) / total * 100 if total else 0),
        "matched_keywords": matched,
        "missing_keywords": missing,
        "total_jd_keywords": total
    }

def calculate_role_matches(keywords):
    all_skills = set(keywords["technical"] + keywords["soft"])
    matches = []
    for job in JOB_ROLES_DB:
        js = set(job["skills"])
        matched = all_skills.intersection(js)
        matches.append({
            "role": job["role"],
            "match_percentage": round(len(matched) / len(js) * 100),
            "matched_skills": list(matched),
            "total_required": len(js)
        })
    matches.sort(key=lambda x: x["match_percentage"], reverse=True)
    return matches[:5]

def extract_text_from_pdf(file_bytes):
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except Exception:
        return None

def extract_text_from_docx_file(file_bytes):
    try:
        d = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in d.paragraphs).strip()
    except Exception:
        return None

def analyze_with_ai(resume_text, job_role, pre_score, keywords, achievements):
    role_context = f'Target job role: "{job_role}"' if job_role else "General job readiness."
    prompt = f"""Expert resume reviewer. Return ONLY valid JSON, no markdown.
{role_context}
pre-score={pre_score}/100, tech={len(keywords['technical'])}, achievements={len(achievements)}
Resume:\"\"\"{resume_text[:3000]}\"\"\"
Return:
{{
  "overall_score":<int>,"ats_score":<int>,"impact_score":<int>,
  "summary":"<2-3 sentences>",
  "strengths":["<s1>","<s2>","<s3>","<s4>"],
  "weaknesses":["<w1>","<w2>","<w3>","<w4>"],
  "improvements":["<t1>","<t2>","<t3>","<t4>","<t5>"],
  "present_skills":{json.dumps(keywords['technical'][:12])},
  "missing_skills":["<m1>","<m2>","<m3>","<m4>"],
  "experience_level":"<Fresher|Junior|Mid-level|Senior>",
  "top_roles":["<r1>","<r2>","<r3>"]
}}"""
    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash-lite", contents=prompt,
            config=types.GenerateContentConfig(temperature=0, top_p=1, top_k=1))
        raw = resp.text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(raw), None
    except json.JSONDecodeError:
        return None, "Could not parse AI response."
    except Exception as ex:
        return None, f"AI error: {ex}"


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "resume" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    file     = request.files["resume"]
    job_role = request.form.get("job_role", "").strip()
    jd_text  = request.form.get("jd_text", "").strip()
    if not file.filename:
        return jsonify({"error": "No file selected."}), 400

    fn = file.filename.lower()
    fb = file.read()
    rt = None
    if fn.endswith(".pdf"):
        rt = extract_text_from_pdf(fb)
        if not rt: return jsonify({"error": "Could not extract text from PDF."}), 400
    elif fn.endswith(".docx"):
        rt = extract_text_from_docx_file(fb)
        if not rt: return jsonify({"error": "Could not read DOCX."}), 400
    elif fn.endswith(".txt"):
        try: rt = fb.decode("utf-8")
        except: return jsonify({"error": "Could not read TXT."}), 400
    else:
        return jsonify({"error": "Only PDF, DOCX, or TXT supported."}), 400
    if len(rt.strip()) < 100:
        return jsonify({"error": "Too little text in file."}), 400

    ck = hashlib.md5((fb + job_role.encode() + jd_text.encode())).hexdigest()
    if ck in results_cache:
        return jsonify(results_cache[ck])

    kw   = extract_keywords(rt)
    ct   = check_contact_info(rt)
    ach  = check_quantified_achievements(rt)
    ps, bd = calculate_pre_score(rt, kw, ct, ach)
    jdm  = calculate_jd_match(rt, jd_text) if jd_text and len(jd_text) > 50 else None
    rm   = calculate_role_matches(kw)
    res, err = analyze_with_ai(rt, job_role, ps, kw, ach)
    if err: return jsonify({"error": err}), 500

    res.update({
        "pre_score": ps, "score_breakdown": bd,
        "contact_info": {k: bool(v) for k, v in ct.items()},
        "contact_values": {k: v for k, v in ct.items() if v},
        "quantified_achievements": ach[:8],
        "jd_match": jdm, "role_matches": rm,
        "resume_text_full": rt[:6000]
    })
    results_cache[ck] = res
    return jsonify(res)


@app.route("/generate-resume", methods=["POST"])
def generate_resume():
    data  = request.get_json()
    rtext = data.get("resume_text", "")
    imps  = data.get("improvements", [])
    miss  = data.get("missing_skills", [])
    role  = data.get("job_role", "")
    level = data.get("experience_level", "")

    if not rtext:
        return jsonify({"error": "No resume text. Analyze your resume first."}), 400

    structured, err = ai_extract_and_improve(rtext, imps, miss, role, level)
    if err:
        return jsonify({"error": err}), 500

    latex_src = build_resume_latex(structured)
    pdf_bytes, cerr = compile_latex(latex_src)
    if cerr:
        return jsonify({"error": cerr}), 500

    return send_file(io.BytesIO(pdf_bytes), mimetype="application/pdf",
                     as_attachment=True, download_name="improved_resume.pdf")


@app.route("/generate-cover-letter", methods=["POST"])
def generate_cover_letter():
    data     = request.get_json()
    job_role = data.get("job_role", "").strip()
    company  = data.get("company", "").strip()
    if not job_role or not company:
        return jsonify({"error": "Job role and company name are required."}), 400

    name     = data.get("name", "").strip()
    title    = data.get("title", "Software Engineer").strip()
    email_v  = data.get("email", "").strip()
    phone_v  = data.get("phone", "").strip()
    lin_url  = data.get("linkedin_url", "").strip()
    lin_handle = re.sub(r'https?://(www\.)?linkedin\.com/in/', '', lin_url).strip('/')
    location = data.get("location", "").strip()
    bg       = data.get("background", "").strip()
    street   = data.get("street", "").strip()
    city     = data.get("city", "").strip()
    state    = data.get("state", "").strip()
    zip_code = data.get("zip", "").strip()

    paras, err = ai_cover_letter_body(name, job_role, company, bg, title)
    if err:
        return jsonify({"error": err}), 500

    cl_data = {
        "name": name or "Applicant",
        "email": email_v,
        "phone": phone_v,
        "linkedin_url": lin_url,
        "linkedin_handle": lin_handle,
        "location": location,
        "recipient": "Hiring Manager",
        "company": company,
        "job_role": job_role,
        "title": title,
        "street": street,
        "city": city,
        "state": state,
        "zip": zip_code,
        "body_paragraphs": paras
    }

    latex_src = build_cover_letter_latex(cl_data)
    pdf_bytes, cerr = compile_latex(latex_src)
    if cerr:
        return jsonify({"error": cerr}), 500

    return send_file(io.BytesIO(pdf_bytes), mimetype="application/pdf",
                     as_attachment=True, download_name="cover_letter.pdf")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)