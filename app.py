import os
import io
import re
import json
import hashlib

from flask import Flask, request, jsonify, render_template, send_file
from google import genai
from google.genai import types
import PyPDF2
import docx
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm

app = Flask(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
client = genai.Client(api_key=GEMINI_API_KEY)
results_cache = {}

SECTION_KEYWORDS = {
    "SUMMARY", "SKILLS", "EXPERIENCE", "EDUCATION", "PROJECTS",
    "CERTIFICATIONS", "ACHIEVEMENTS", "OBJECTIVE", "WORK EXPERIENCE",
    "TECHNICAL SKILLS", "PROFESSIONAL SUMMARY", "KEY SKILLS"
}

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
    "linkedin": r"linkedin\.com/in/[a-zA-Z0-9\-]+",
    "github":   r"github\.com/[a-zA-Z0-9\-]+"
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


# ─────────────────────────────────────────────
# ANALYSIS HELPERS
# ─────────────────────────────────────────────
def _clean_md(text):
    """Remove markdown bold/italic markers"""
    return re.sub(r'\*+', '', text).strip()

def _is_section(line):
    return line.strip().rstrip(":").upper() in SECTION_KEYWORDS

def extract_keywords(text):
    tl = text.lower()
    words = set(re.findall(r'\b[a-z][a-z0-9+#.\-]*\b', tl))
    bigrams = set()
    wl = tl.split()
    for i in range(len(wl)-1): bigrams.add(wl[i]+" "+wl[i+1])
    return {
        "technical":    [s for s in TECHNICAL_SKILLS if s in words or s in bigrams],
        "soft":         [s for s in SOFT_SKILLS      if s in words or s in bigrams],
        "action_verbs": [v for v in ACTION_VERBS     if v in words]
    }

def check_contact_info(text):
    return {k: bool(re.search(p, text, re.IGNORECASE)) for k, p in CONTACT_PATTERNS.items()}

def check_quantified_achievements(text):
    patterns = [r'\d+\s*%', r'\$\s*\d+[\d,kmb]*', r'\d+\s*x\b',
                r'\d+\+\s*(users|clients|customers|teams|projects|employees)',
                r'(increased|decreased|reduced|improved|grew|saved)\s+\w+\s+by\s+\d+']
    achievements = []
    for p in patterns:
        achievements.extend(re.findall(p, text, re.IGNORECASE))
    return list(set(str(a) for a in achievements))

def calculate_pre_score(text, keywords, contact, achievements):
    score, breakdown = 0, {}
    ts = min(30, len(keywords["technical"])*3); score += ts
    breakdown["technical_skills"] = {"score":ts,"max":30,"found":len(keywords["technical"])}
    ss = min(15, len(keywords["soft"])*3); score += ss
    breakdown["soft_skills"] = {"score":ss,"max":15,"found":len(keywords["soft"])}
    av = min(15, len(keywords["action_verbs"])*2); score += av
    breakdown["action_verbs"] = {"score":av,"max":15,"found":len(keywords["action_verbs"])}
    ach = min(20, len(achievements)*5); score += ach
    breakdown["quantified_achievements"] = {"score":ach,"max":20,"found":len(achievements)}
    cs = sum([4 if contact.get("email") else 0, 3 if contact.get("phone") else 0,
              2 if contact.get("linkedin") else 0, 1 if contact.get("github") else 0])
    score += cs
    breakdown["contact_info"] = {"score":cs,"max":10,"details":contact}
    wc = len(text.split())
    if 300<=wc<=800:   ls,ln = 10,"Ideal length"
    elif wc<300:        ls,ln = max(0,wc//30),"Too short"
    else:               ls,ln = max(5,10-(wc-800)//100),"Too long"
    score += ls
    breakdown["resume_length"] = {"score":ls,"max":10,"word_count":wc,"note":ln}
    return min(100,score), breakdown

def calculate_jd_match(resume_text, jd_text):
    rl,jl = resume_text.lower().split(), jd_text.lower().split()
    rw = set(re.findall(r'\b[a-z][a-z0-9+#.\-]*\b', resume_text.lower()))
    jw = set(re.findall(r'\b[a-z][a-z0-9+#.\-]*\b', jd_text.lower()))
    rb,jb = set(),set()
    for i in range(len(rl)-1): rb.add(rl[i]+" "+rl[i+1])
    for i in range(len(jl)-1): jb.add(jl[i]+" "+jl[i+1])
    important = [kw for kw in TECHNICAL_SKILLS+SOFT_SKILLS if kw in jw or kw in jb]
    if not important:
        sw={"the","and","for","are","with","you","will","have","this","that","from","they","been","our","your","not","but","all","can","was","were","their","has","its"}
        freq={}
        for w in jl:
            if len(w)>3 and w not in sw: freq[w]=freq.get(w,0)+1
        important=sorted(freq,key=freq.get,reverse=True)[:20]
    matched=[kw for kw in important if kw in rw or kw in rb]
    missing=[kw for kw in important if kw not in rw and kw not in rb]
    total=len(important)
    return {"match_percentage":round(len(matched)/total*100 if total else 0),
            "matched_keywords":matched,"missing_keywords":missing,"total_jd_keywords":total}

def calculate_role_matches(keywords):
    all_skills=set(keywords["technical"]+keywords["soft"])
    matches=[]
    for job in JOB_ROLES_DB:
        js=set(job["skills"]); matched=all_skills.intersection(js)
        matches.append({"role":job["role"],"match_percentage":round(len(matched)/len(js)*100),
                        "matched_skills":list(matched),"total_required":len(js)})
    matches.sort(key=lambda x:x["match_percentage"],reverse=True)
    return matches[:5]

def extract_text_from_pdf(file_bytes):
    try:
        reader=PyPDF2.PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except: return None

def extract_text_from_docx_file(file_bytes):
    try:
        d=docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in d.paragraphs).strip()
    except: return None

def analyze_with_ai(resume_text, job_role, pre_score, keywords, achievements):
    role_context = f'Target job role: "{job_role}"' if job_role else "General job readiness analysis."
    prompt = f"""You are an expert resume reviewer. Return ONLY valid JSON. No markdown.
{role_context}
Pre-analysis: pre-score={pre_score}/100, tech_skills={len(keywords['technical'])}, achievements={len(achievements)}
Resume:\"\"\"{resume_text[:3000]}\"\"\"
Return ONLY:
{{
  "overall_score":<int near {pre_score}>,"ats_score":<int>,"impact_score":<int>,
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
        raw = resp.text.strip().replace("```json","").replace("```","").strip()
        return json.loads(raw), None
    except json.JSONDecodeError: return None, "Could not parse AI response."
    except Exception as e: return None, f"AI error: {str(e)}"


# ─────────────────────────────────────────────
# PDF GENERATOR — CANVAS BASED (FIXED)
# ─────────────────────────────────────────────
def build_resume_pdf(content_text):
    buf = io.BytesIO()
    w, h = A4
    c = rl_canvas.Canvas(buf, pagesize=A4)

    # Colors
    ACCENT = colors.HexColor("#4346a0")
    DARK   = colors.HexColor("#1a1a2e")
    BODY   = colors.HexColor("#222222")
    MUTED  = colors.HexColor("#555555")
    LGRAY  = colors.HexColor("#dddddd")

    LEFT  = 1.8 * cm
    RIGHT = w - 1.8 * cm
    MAX_W = RIGHT - LEFT

    y = h - 1.5 * cm  # start from top

    def new_page():
        nonlocal y
        c.showPage()
        y = h - 1.5 * cm

    def check_space(needed=16):
        if y < 2 * cm + needed:
            new_page()

    def draw_wrapped_text(text, x, start_y, font, size, color, indent=0):
        """Draw word-wrapped text, return final Y position"""
        nonlocal y
        avail = RIGHT - x
        c.setFont(font, size)
        c.setFillColor(color)
        words = text.split()
        line = ""
        cur_y = start_y
        for word in words:
            test = (line + " " + word).strip()
            if c.stringWidth(test, font, size) <= avail:
                line = test
            else:
                if line:
                    c.drawString(x, cur_y, line)
                    cur_y -= (size + 4)
                    check_space(size + 4)
                    x = LEFT + indent  # subsequent lines align left
                    avail = RIGHT - x
                line = word
        if line:
            c.drawString(x, cur_y, line)
            cur_y -= (size + 4)
        return cur_y

    lines = content_text.strip().split("\n")
    name_done = contact_done = False

    for line in lines:
        raw  = line.rstrip()
        cl   = _clean_md(raw).strip()

        # Empty line → small gap
        if not cl:
            if contact_done:
                y -= 4
            continue

        # ── NAME ──
        if not name_done:
            check_space(30)
            c.setFont("Helvetica-Bold", 20)
            c.setFillColor(DARK)
            c.drawCentredString(w / 2, y - 20, cl)
            y -= 30
            name_done = True
            continue

        # ── CONTACT LINE ──
        if name_done and not contact_done:
            check_space(30)
            c.setFont("Helvetica", 9.5)
            c.setFillColor(MUTED)
            c.drawCentredString(w / 2, y - 12, cl)
            y -= 20
            # Blue divider
            c.setStrokeColor(ACCENT)
            c.setLineWidth(2.5)
            c.line(LEFT, y - 4, RIGHT, y - 4)
            y -= 16
            contact_done = True
            continue

        # ── SECTION HEADER ──
        if _is_section(raw):
            check_space(30)
            y -= 8
            c.setFont("Helvetica-Bold", 11)
            c.setFillColor(ACCENT)
            c.drawString(LEFT, y - 12, cl.rstrip(":").upper())
            y -= 18
            c.setStrokeColor(LGRAY)
            c.setLineWidth(0.5)
            c.line(LEFT, y, RIGHT, y)
            y -= 7
            continue

        # ── BULLET POINT ──
        if raw.lstrip().startswith(("•","–","-","*","◦")):
            check_space(16)
            cb = re.sub(r'^[\s•–\-\*◦]+', '', raw).strip()
            cb = _clean_md(cb)
            # Draw bullet in accent color
            c.setFont("Helvetica-Bold", 11)
            c.setFillColor(ACCENT)
            c.drawString(LEFT + 6, y - 11, "•")
            # Draw text with wrap
            text_x = LEFT + 18
            c.setFont("Helvetica", 10)
            c.setFillColor(BODY)
            avail = RIGHT - text_x
            words = cb.split()
            cur_line = ""
            cur_y = y - 11
            for word in words:
                test = (cur_line + " " + word).strip()
                if c.stringWidth(test, "Helvetica", 10) <= avail:
                    cur_line = test
                else:
                    if cur_line:
                        c.drawString(text_x, cur_y, cur_line)
                        cur_y -= 14
                        check_space(14)
                        text_x = LEFT + 18  # maintain indent
                    cur_line = word
            if cur_line:
                c.drawString(text_x, cur_y, cur_line)
                cur_y -= 14
            y = cur_y - 2
            continue

        # ── JOB/COMPANY line with | ──
        if "|" in cl and len(cl) < 200:
            check_space(20)
            y -= 5
            parts = [_clean_md(p).strip() for p in cl.split("|")]
            cur_x = LEFT
            # First part: bold dark
            c.setFont("Helvetica-Bold", 10.5)
            c.setFillColor(DARK)
            c.drawString(cur_x, y - 12, parts[0])
            cur_x += c.stringWidth(parts[0], "Helvetica-Bold", 10.5)
            for part in parts[1:]:
                # Separator
                c.setFont("Helvetica", 10)
                c.setFillColor(ACCENT)
                sep = "  |  "
                c.drawString(cur_x, y - 12, sep)
                cur_x += c.stringWidth(sep, "Helvetica", 10)
                # Part
                c.setFillColor(MUTED)
                c.drawString(cur_x, y - 12, part)
                cur_x += c.stringWidth(part, "Helvetica", 10)
            y -= 18
            continue

        # ── SKILLS line: Label: values ──
        if ":" in cl and len(cl) < 250 and not cl.endswith(":"):
            check_space(16)
            colon_i = cl.index(":")
            lbl = cl[:colon_i].strip()
            val = cl[colon_i+1:].strip()
            # Label bold accent
            c.setFont("Helvetica-Bold", 10)
            c.setFillColor(ACCENT)
            lbl_str = lbl + ":  "
            c.drawString(LEFT, y - 11, lbl_str)
            lbl_w = c.stringWidth(lbl_str, "Helvetica-Bold", 10)
            # Value wrapped
            c.setFont("Helvetica", 10)
            c.setFillColor(BODY)
            text_x = LEFT + lbl_w
            avail = RIGHT - text_x
            words = val.split()
            cur_line = ""
            cur_y = y - 11
            first = True
            for word in words:
                test = (cur_line + " " + word).strip()
                lim = avail if first else MAX_W
                if c.stringWidth(test, "Helvetica", 10) <= lim:
                    cur_line = test
                else:
                    if cur_line:
                        c.drawString(text_x if first else LEFT, cur_y, cur_line)
                        cur_y -= 14
                        check_space(14)
                        first = False
                        text_x = LEFT
                    cur_line = word
            if cur_line:
                c.drawString(text_x if first else LEFT, cur_y, cur_line)
                cur_y -= 14
            y = cur_y - 2
            continue

        # ── REGULAR TEXT (wrapped) ──
        check_space(14)
        c.setFont("Helvetica", 10)
        c.setFillColor(BODY)
        words = cl.split()
        cur_line = ""
        cur_y = y - 11
        for word in words:
            test = (cur_line + " " + word).strip()
            if c.stringWidth(test, "Helvetica", 10) <= MAX_W:
                cur_line = test
            else:
                if cur_line:
                    c.drawString(LEFT, cur_y, cur_line)
                    cur_y -= 14
                    check_space(14)
                cur_line = word
        if cur_line:
            c.drawString(LEFT, cur_y, cur_line)
            cur_y -= 14
        y = cur_y - 2

    c.save()
    buf.seek(0)
    return buf


# ─────────────────────────────────────────────
# DOCX GENERATOR — CLEAN PROFESSIONAL
# ─────────────────────────────────────────────
def _add_border(paragraph, color_hex="4f46e5", size="6"):
    pPr = paragraph._p.get_or_add_pPr()
    pBdr = OxmlElement('w:pBdr')
    bottom = OxmlElement('w:bottom')
    bottom.set(qn('w:val'), 'single')
    bottom.set(qn('w:sz'), size)
    bottom.set(qn('w:space'), '1')
    bottom.set(qn('w:color'), color_hex)
    pBdr.append(bottom)
    pPr.append(pBdr)

def build_resume_docx(content_text):
    doc = Document()
    for sec in doc.sections:
        sec.top_margin    = Inches(0.75)
        sec.bottom_margin = Inches(0.75)
        sec.left_margin   = Inches(0.9)
        sec.right_margin  = Inches(0.9)
    norm = doc.styles['Normal']
    norm.paragraph_format.space_before = Pt(0)
    norm.paragraph_format.space_after  = Pt(0)

    lines = content_text.strip().split("\n")
    name_done = contact_done = False

    for line in lines:
        raw  = line.rstrip()
        cl   = _clean_md(raw).strip()

        if not cl:
            if name_done:
                sp = doc.add_paragraph()
                sp.paragraph_format.space_after = Pt(2)
            continue

        # NAME
        if not name_done:
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(3)
            r = p.add_run(cl)
            r.bold = True
            r.font.size = Pt(22)
            r.font.color.rgb = RGBColor(0x1a, 0x1a, 0x2e)
            name_done = True
            continue

        # CONTACT
        if name_done and not contact_done:
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(2)
            r = p.add_run(cl)
            r.font.size = Pt(9.5)
            r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
            # Blue divider
            div = doc.add_paragraph()
            div.paragraph_format.space_before = Pt(3)
            div.paragraph_format.space_after  = Pt(5)
            _add_border(div, "4346a0", "14")
            contact_done = True
            continue

        # SECTION HEADER
        if _is_section(raw):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after  = Pt(3)
            r = p.add_run(cl.rstrip(":").upper())
            r.bold = True
            r.font.size = Pt(11)
            r.font.color.rgb = RGBColor(0x43, 0x46, 0xa0)
            _add_border(p, "4346a0", "6")
            continue

        # BULLET
        if raw.lstrip().startswith(("•","–","-","*","◦")):
            cb = re.sub(r'^[\s•–\-\*◦]+', '', raw).strip()
            cb = _clean_md(cb)
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after  = Pt(1)
            p.paragraph_format.left_indent  = Inches(0.18)
            dot = p.add_run("• ")
            dot.font.size = Pt(10)
            dot.font.color.rgb = RGBColor(0x43, 0x46, 0xa0)
            txt = p.add_run(cb)
            txt.font.size = Pt(10)
            txt.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
            continue

        # JOB/COMPANY with |
        if "|" in cl and len(cl) < 200:
            parts = [_clean_md(p).strip() for p in cl.split("|")]
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after  = Pt(1)
            for j, part in enumerate(parts):
                if j > 0:
                    sep = p.add_run("  |  ")
                    sep.font.size = Pt(10)
                    sep.font.color.rgb = RGBColor(0x43, 0x46, 0xa0)
                r = p.add_run(part)
                r.font.size = Pt(10.5)
                if j == 0:
                    r.bold = True
                    r.font.color.rgb = RGBColor(0x1a, 0x1a, 0x2e)
                else:
                    r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
            continue

        # SKILLS line: Label: values
        if ":" in cl and len(cl) < 250 and not cl.endswith(":"):
            colon_i = cl.index(":")
            lbl = cl[:colon_i].strip()
            val = cl[colon_i+1:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after  = Pt(1)
            rl = p.add_run(lbl + ": ")
            rl.bold = True
            rl.font.size = Pt(10)
            rl.font.color.rgb = RGBColor(0x43, 0x46, 0xa0)
            rv = p.add_run(val)
            rv.font.size = Pt(10)
            rv.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
            continue

        # REGULAR TEXT
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after  = Pt(1)
        r = p.add_run(cl)
        r.font.size = Pt(10)
        r.font.color.rgb = RGBColor(0x22, 0x22, 0x22)

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf


# ─────────────────────────────────────────────
# COVER LETTER PDF
# ─────────────────────────────────────────────
def build_cover_letter_pdf(content_text, job_role="", company=""):
    buf = io.BytesIO()
    w, h = A4
    c = rl_canvas.Canvas(buf, pagesize=A4)

    ACCENT = colors.HexColor("#4346a0")
    DARK   = colors.HexColor("#1a1a2e")
    BODY   = colors.HexColor("#222222")
    MUTED  = colors.HexColor("#888888")
    LEFT   = 2.2 * cm
    RIGHT  = w - 2.2 * cm
    MAX_W  = RIGHT - LEFT

    y = h - 2.0 * cm

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.setFillColor(ACCENT)
    c.drawCentredString(w/2, y - 18, "Cover Letter")
    y -= 28

    # Subtitle
    if job_role or company:
        sub = f"{job_role} at {company}" if job_role and company else (job_role or company)
        c.setFont("Helvetica", 11)
        c.setFillColor(MUTED)
        c.drawCentredString(w/2, y - 12, sub)
        y -= 22

    # Blue line
    c.setStrokeColor(ACCENT)
    c.setLineWidth(2)
    c.line(LEFT, y - 6, RIGHT, y - 6)
    y -= 22

    # Body paragraphs
    for para in content_text.strip().split("\n\n"):
        para = para.strip()
        if not para:
            continue
        words = para.replace("\n", " ").split()
        c.setFont("Helvetica", 11)
        c.setFillColor(BODY)
        cur_line = ""
        cur_y = y - 14
        for word in words:
            test = (cur_line + " " + word).strip()
            if c.stringWidth(test, "Helvetica", 11) <= MAX_W:
                cur_line = test
            else:
                if cur_line:
                    c.drawString(LEFT, cur_y, cur_line)
                    cur_y -= 17
                    if cur_y < 2*cm:
                        c.showPage()
                        cur_y = h - 2*cm
                cur_line = word
        if cur_line:
            c.drawString(LEFT, cur_y, cur_line)
            cur_y -= 17
        y = cur_y - 10

    c.save()
    buf.seek(0)
    return buf


# ─────────────────────────────────────────────
# COVER LETTER DOCX
# ─────────────────────────────────────────────
def build_cover_letter_docx(content_text, job_role="", company=""):
    doc = Document()
    for sec in doc.sections:
        sec.top_margin    = Inches(1.0)
        sec.bottom_margin = Inches(1.0)
        sec.left_margin   = Inches(1.1)
        sec.right_margin  = Inches(1.1)
    norm = doc.styles['Normal']
    norm.paragraph_format.space_before = Pt(0)
    norm.paragraph_format.space_after  = Pt(0)

    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    t.paragraph_format.space_after = Pt(4)
    tr = t.add_run("Cover Letter")
    tr.bold = True; tr.font.size = Pt(18)
    tr.font.color.rgb = RGBColor(0x43, 0x46, 0xa0)

    if job_role or company:
        sub_text = f"{job_role} at {company}" if job_role and company else (job_role or company)
        s = doc.add_paragraph()
        s.alignment = WD_ALIGN_PARAGRAPH.CENTER
        s.paragraph_format.space_after = Pt(2)
        sr = s.add_run(sub_text)
        sr.font.size = Pt(11)
        sr.font.color.rgb = RGBColor(0x77, 0x77, 0x77)

    div = doc.add_paragraph()
    div.paragraph_format.space_before = Pt(4)
    div.paragraph_format.space_after  = Pt(12)
    _add_border(div, "4346a0", "12")

    for para in content_text.strip().split("\n\n"):
        para = para.strip()
        if not para:
            continue
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(10)
        r = p.add_run(para.replace("\n"," "))
        r.font.size = Pt(11)
        r.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
        p.paragraph_format.line_spacing = Pt(17)

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf


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
    job_role = request.form.get("job_role","").strip()
    jd_text  = request.form.get("jd_text","").strip()
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400
    filename   = file.filename.lower()
    file_bytes = file.read()
    resume_text = None
    if filename.endswith(".pdf"):
        resume_text = extract_text_from_pdf(file_bytes)
        if not resume_text: return jsonify({"error":"Could not extract text from PDF."}),400
    elif filename.endswith(".docx"):
        resume_text = extract_text_from_docx_file(file_bytes)
        if not resume_text: return jsonify({"error":"Could not read DOCX file."}),400
    elif filename.endswith(".txt"):
        try: resume_text = file_bytes.decode("utf-8")
        except: return jsonify({"error":"Could not read TXT file."}),400
    else:
        return jsonify({"error":"Only PDF, DOCX, or TXT files are supported."}),400
    if len(resume_text.strip()) < 100:
        return jsonify({"error":"File has too little text."}),400
    cache_key = hashlib.md5((file_bytes+job_role.encode()+jd_text.encode())).hexdigest()
    if cache_key in results_cache:
        return jsonify(results_cache[cache_key])
    keywords     = extract_keywords(resume_text)
    contact      = check_contact_info(resume_text)
    achievements = check_quantified_achievements(resume_text)
    pre_score, breakdown = calculate_pre_score(resume_text, keywords, contact, achievements)
    jd_match     = calculate_jd_match(resume_text, jd_text) if jd_text and len(jd_text)>50 else None
    role_matches = calculate_role_matches(keywords)
    result, error = analyze_with_ai(resume_text, job_role, pre_score, keywords, achievements)
    if error: return jsonify({"error":error}),500
    result.update({
        "pre_score":pre_score,"score_breakdown":breakdown,
        "contact_info":contact,"quantified_achievements":achievements[:8],
        "jd_match":jd_match,"role_matches":role_matches,
        "resume_text_preview": resume_text[:1500]
    })
    results_cache[cache_key] = result
    return jsonify(result)


@app.route("/generate-resume", methods=["POST"])
def generate_resume():
    data             = request.get_json()
    resume_preview   = data.get("resume_preview","")
    improvements     = data.get("improvements",[])
    missing_skills   = data.get("missing_skills",[])
    job_role         = data.get("job_role","")
    experience_level = data.get("experience_level","")
    fmt              = data.get("format","pdf").lower()
    if not resume_preview:
        return jsonify({"error":"No resume content found. Analyze your resume first."}),400
    prompt = f"""You are a professional resume writer. Rewrite the resume applying ALL improvements.

Original Resume:
\"\"\"{resume_preview}\"\"\"

Target Role: {job_role or "Not specified"}
Level: {experience_level or "Not specified"}

Improvements to apply:
{chr(10).join(f"- {i}" for i in improvements)}

Missing skills to add in Skills section: {", ".join(missing_skills) or "None"}

STRICT FORMATTING — follow exactly:
1. Line 1: Full Name only
2. Line 2: phone | email | LinkedIn | GitHub
3. Blank line
4. Sections in order: SUMMARY, SKILLS, EXPERIENCE, PROJECTS, EDUCATION
5. Section headers: ALL CAPS on their own line
6. Job entries: Company Name | Role | Start - End Date
7. Bullets: start with "- " then strong action verb
8. Add realistic numbers/percentages in bullets
9. Skills: Label: comma-separated values (e.g. "Frontend: React, Vue, TypeScript")
10. NO markdown (**bold**), NO [placeholders], NO emoji
11. Max 650 words, plain text only

Return ONLY the resume text."""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite", contents=prompt,
            config=types.GenerateContentConfig(temperature=0.1))
        content = re.sub(r'\*+', '', response.text.strip())
        content = re.sub(r'#{1,6}\s*', '', content)
    except Exception as e:
        return jsonify({"error":f"AI generation failed: {str(e)}"}),500
    if fmt == "docx":
        buf = build_resume_docx(content)
        return send_file(buf,
            mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            as_attachment=True, download_name="improved_resume.docx")
    else:
        buf = build_resume_pdf(content)
        return send_file(buf, mimetype="application/pdf",
            as_attachment=True, download_name="improved_resume.pdf")


@app.route("/generate-cover-letter", methods=["POST"])
def generate_cover_letter():
    data     = request.get_json()
    job_role = data.get("job_role","").strip()
    company  = data.get("company","").strip()
    bg       = data.get("resume_summary","").strip()
    fmt      = data.get("format","pdf").lower()
    if not job_role or not company:
        return jsonify({"error":"Job role and company name are required."}),400
    prompt = f"""Write a professional cover letter.
Job Role: {job_role}
Company: {company}
Background: {bg or "A skilled professional"}

Rules:
- Exactly 3 paragraphs: intro, value proposition, confident closing
- Under 280 words, professional tone
- No placeholders, no salutation/sign-off, plain text only
Return ONLY the 3 paragraphs."""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite", contents=prompt,
            config=types.GenerateContentConfig(temperature=0.3))
        content = re.sub(r'\*+', '', response.text.strip())
    except Exception as e:
        return jsonify({"error":f"AI generation failed: {str(e)}"}),500
    if fmt == "docx":
        buf = build_cover_letter_docx(content, job_role, company)
        return send_file(buf,
            mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            as_attachment=True, download_name="cover_letter.docx")
    else:
        buf = build_cover_letter_pdf(content, job_role, company)
        return send_file(buf, mimetype="application/pdf",
            as_attachment=True, download_name="cover_letter.pdf")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)