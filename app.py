"""
Residential Electrical Apprentice Screening Test
Hosted version with live Grok (xAI) AI grading
2023 NEC Edition – Practical Field Assessment
"""

import os
import json
import httpx
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv
from fpdf import FPDF
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from io import BytesIO

load_dotenv()

app = FastAPI(title="Residential Electrical Apprentice Screening Test", version="1.0")

if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

XAI_API_KEY = os.getenv("XAI_API_KEY")
XAI_BASE_URL = "https://api.x.ai/v1"
GROK_MODEL = os.getenv("GROK_MODEL", "grok-4.6")
RESULTS_EMAIL_TO = os.getenv("RESULTS_EMAIL_TO", "Mark@keywestlights.com")
GMAIL_USER = os.getenv("GMAIL_USER", "")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")


class AnswerItem(BaseModel):
    id: str
    year: int
    section: str
    question: str
    answer: str
    seconds: int = 0


class GradeRequest(BaseModel):
    applicant: dict
    answers: list[AnswerItem]
    claimed_year: int


GRADING_SYSTEM_PROMPT = """
You are an expert electrical contractor, field supervisor, NEC 2023 specialist, and hiring manager.

You are grading an INTERNAL EMPLOYEE SKILL-LEVEL & PLACEMENT ASSESSMENT.

The candidate answered a mixed set of questions spanning first-year fundamentals through lead-level judgment (safety, theory, NEC, calculations, troubleshooting, mechanical aptitude, job-site judgment, and leadership). They did NOT select a year level - you must determine their actual level from performance.

OVERALL LEVELS (choose exactly one):
- Below Year 1 / Helper
- Year 1 Apprentice
- Year 2 Apprentice
- Year 3 Apprentice
- Year 4 Apprentice
- Lead Electrician
- Not a fit for electrical fieldwork (major safety or fundamental failures)

Also produce:
- skill_level: plain-English summary of what they actually know and their strongest areas
- mechanical_aptitude: hands-on / installation / recognition of bad workmanship
- pay_grade_band: appropriate pay band for a residential/commercial electrical contractor
- project_placement: what work they can safely be assigned now
- hire_recommendation: clear recommendation - e.g. "Hire and place as Year 2; strong safety and trainable", "Hire as helper only; major gaps", "Do not hire - unsafe judgment / fundamental confusion on neutral vs ground", "Strong lead candidate; can train apprentices"
- experience_mismatch_note: if claimed experience does not match demonstrated ability

CATEGORY SCORES (0-100 each):
safety, fundamentals, nec, calculations, troubleshooting, field_judgment, leadership, mechanical_aptitude, limitations


TIMING SIGNAL (secondary, use carefully):
Each answer may include time spent in seconds. Use this as supporting evidence only:
- Very fast correct answers on basic code/safety facts suggest knowledge "in the head."
- Very long times on simple True/False or short code facts (e.g. 90+ seconds) may indicate codebook lookup or uncertainty - note this, but do not alone fail a strong written performer.
- Calculations, scenarios, and written explanations legitimately take longer - do not penalize those for time.
- Interruptions happen; treat timing as a soft signal, not primary scoring.
Mention notable timing patterns in strengths/weaknesses or summary when relevant (e.g. "fast on fundamentals, slow on code facts").

CRITICAL RULES:
- Safety and neutral-vs-ground / testing-before-work are non-negotiable. Major failures here → low level and/or do-not-hire regardless of other scores.
- Reward correct process and good judgment even if wording is imperfect.
- Calculations: partial credit for correct formula even if arithmetic is off; zero if formula is wrong or omitted when required.
- Willingness to say "I don't know / I would ask" is POSITIVE for apprentices; guessing on safety is a red flag.
- Someone cannot be placed as Lead if they fail critical safety or show willingness to work energized without procedure, or confuse neutral and ground.
- Be rigorous. This assessment is used for placement and pay.

Return ONLY valid JSON (no markdown):

{
  "overall_score_percent": 0-100,
  "level": "one of the levels above",
  "level_description": "2-4 sentences why this level",
  "skill_level": "string",
  "mechanical_aptitude": "string",
  "pay_grade_band": "string",
  "project_placement": "string",
  "hire_recommendation": "string",
  "experience_mismatch_note": "string or empty",
  "category_scores": {
    "safety": 0-100,
    "fundamentals": 0-100,
    "nec": 0-100,
    "calculations": 0-100,
    "troubleshooting": 0-100,
    "field_judgment": 0-100,
    "leadership": 0-100,
    "mechanical_aptitude": 0-100,
    "limitations": 0-100
  },
  "strengths": ["3-6 items"],
  "weaknesses": ["3-6 items"],
  "red_flags": ["critical safety or judgment issues, or empty list"],
  "summary_for_hiring_manager": "4-7 sentence overall assessment including hire/train recommendation"
}
"""




def build_user_prompt(applicant: dict, answers: list, claimed_year: int) -> str:
    lines = []
    lines.append("=== APPLICANT ===")
    lines.append(f"Name: {applicant.get('name', 'Unknown')}")
    lines.append(f"Years claimed: {applicant.get('years', 'N/A')}")
    lines.append(f"Background: {applicant.get('experience', 'N/A')}")
    lines.append("")
    lines.append("NOTE: Candidate did not select a year. Determine actual year level from performance.")
    lines.append("Format below: ID | Category | Answer (questions are mostly True/False or A/B multiple choice)")
    lines.append("")
    lines.append("=== ANSWERS ===")
    for a in answers:
        ans = (a.answer or "").strip().replace("\n", " ")[:200]
        # Keep question short for MC - first line only
        q_short = (a.question or "").split("\n")[0][:120]
        secs = getattr(a, "seconds", 0) or 0
        time_note = f" | time={secs}s"
        if secs >= 90 and a.section in ("Code", "Fundamentals", "Safety"):
            time_note += " [SLOW for fact/code - possible lookup]"
        elif secs > 0 and secs <= 15 and a.section in ("Code", "Fundamentals", "Safety"):
            time_note += " [FAST]"
        lines.append(f"{a.id} [{a.section}] Q: {q_short} | A: {ans if ans else '(blank)'}{time_note}")
    lines.append("")
    lines.append("Evaluate all answers. Return ONLY the required JSON.")
    return "\n".join(lines)


async def call_grok(system_prompt: str, user_prompt: str) -> dict:
    if not XAI_API_KEY:
        raise HTTPException(status_code=500, detail="XAI_API_KEY is not configured on the server.")

    url = f"{XAI_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 4000,
    }

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=30.0)) as client:
            resp = await client.post(url, headers=headers, json=payload)
    except httpx.ReadTimeout:
        raise HTTPException(
            status_code=504,
            detail="Grok took too long to respond (timeout). Please try again, or contact the hiring manager."
        )
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Could not reach Grok API: {type(e).__name__}")

    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Grok API error {resp.status_code}: {resp.text[:400]}"
        )

    try:
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError) as e:
        raise HTTPException(status_code=502, detail=f"Unexpected Grok response format: {e}")

    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        if content.endswith("```"):
            content = content.rsplit("```", 1)[0]
        content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(content[start:end])
            except Exception:
                pass
        raise HTTPException(status_code=502, detail="Grok did not return valid JSON. Try submitting again.")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})






def _safe(text, limit=2000):
    if text is None:
        return "-"
    s = str(text)
    for a, b in [
        ("\u2014", "-"), ("\u2013", "-"), ("\u2018", "'"), ("\u2019", "'"),
        ("\u201c", '"'), ("\u201d", '"'), ("\u2022", "*"), ("\u2026", "..."),
        ("\r", " "), ("\t", " "),
    ]:
        s = s.replace(a, b)
    # Keep newlines for paragraphs but normalize weird whitespace
    s = "\n".join(line.strip() for line in s.replace("\r", "").split("\n"))
    s = "".join(ch if (ch == "\n" or 32 <= ord(ch) < 127) else "?" for ch in s)
    s = s.strip()
    return s[:limit] if s else "-"


def build_results_pdf(applicant: dict, grade: dict) -> bytes:
    """Clean, readable assessment PDF."""
    pdf = FPDF(format="Letter", unit="mm")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.set_left_margin(18)
    pdf.set_right_margin(18)
    pdf.set_xy(18, 18)

    page_w = pdf.w - 36  # usable width

    def h1(txt):
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_x(18)
        pdf.multi_cell(page_w, 8, _safe(txt, 120))
        pdf.ln(2)

    def h2(txt):
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_x(18)
        pdf.multi_cell(page_w, 7, _safe(txt, 120))
        pdf.ln(1)

    def body(txt):
        pdf.set_font("Helvetica", "", 10)
        pdf.set_x(18)
        pdf.multi_cell(page_w, 5.5, _safe(txt, 2500))
        pdf.ln(1)

    def kv(label, value):
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_x(18)
        pdf.multi_cell(page_w, 5.5, _safe(label, 80))
        pdf.set_font("Helvetica", "", 10)
        pdf.set_x(18)
        pdf.multi_cell(page_w, 5.5, _safe(value, 2500))
        pdf.ln(1.5)

    def bullets(items):
        pdf.set_font("Helvetica", "", 10)
        if not items:
            body("-")
            return
        for item in items:
            pdf.set_x(18)
            pdf.multi_cell(page_w, 5.5, "- " + _safe(item, 400))
        pdf.ln(1)

    h1("Key West Lights - Assessment Report")
    body("Confidential hiring document")

    h2("Candidate")
    kv("Name", applicant.get("name") or "-")
    kv("Years claimed", applicant.get("years") or "-")
    kv("Residential years", applicant.get("resYears") or applicant.get("years") or "-")
    kv("Commercial years", applicant.get("comYears") or "-")
    kv("Largest project", applicant.get("largestProject") or "-")
    kv("Largest crew", applicant.get("largestCrew") or "-")
    kv("J-Card / License", applicant.get("jcard") or "-")

    h2("Placement Result")
    kv("Level", grade.get("level") or "-")
    kv("Overall score", str(grade.get("overall_score_percent", "-")) + "%")
    kv("Level description", grade.get("level_description") or "-")
    kv("Skill / strong suits", grade.get("skill_level") or "-")
    if grade.get("mechanical_aptitude"):
        kv("Mechanical aptitude", grade.get("mechanical_aptitude"))
    kv("Pay grade band", grade.get("pay_grade_band") or "-")
    kv("Project placement", grade.get("project_placement") or "-")
    kv("Hire recommendation", grade.get("hire_recommendation") or "-")
    if grade.get("experience_mismatch_note"):
        kv("Experience mismatch", grade.get("experience_mismatch_note"))

    h2("Category Scores")
    cat = grade.get("category_scores") or {}
    if cat:
        lines = []
        for k, v in cat.items():
            lines.append(f"{k}: {v}%")
        body("\n".join(lines))
    else:
        body("-")

    h2("Strengths")
    bullets(grade.get("strengths") or [])

    h2("Weaknesses / Development")
    bullets(grade.get("weaknesses") or [])

    flags = grade.get("red_flags") or []
    if flags:
        h2("RED FLAGS")
        bullets(flags)

    h2("Hiring Manager Summary")
    body(grade.get("summary_for_hiring_manager") or "-")

    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_x(18)
    pdf.multi_cell(page_w, 4, "Confidential - Key West Lights internal use only. Not for distribution without management approval.")

    out = BytesIO()
    pdf.output(out)
    return out.getvalue()


def send_results_email(applicant: dict, grade: dict, pdf_bytes: bytes) -> str:
    """Send PDF via Resend (HTTPS) preferred; SMTP Gmail fallback."""
    to_addr = (RESULTS_EMAIL_TO or "").strip()
    if not to_addr:
        return "Email skipped: RESULTS_EMAIL_TO not set"

    name = applicant.get("name") or "Candidate"
    level = grade.get("level") or "Unknown"
    score = grade.get("overall_score_percent", "-")
    subject = f"Assessment: {name} - {level} ({score}%)"
    body = (
        f"Assessment results\\n\\n"
        f"Candidate: {name}\\n"
        f"Level: {level}\\n"
        f"Score: {score}%\\n"
        f"Hire recommendation: {grade.get('hire_recommendation', '-')}\\n\\n"
        f"PDF report attached.\\n\\n"
        f"- Key West Lights hiring system\\n"
    )
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:40]
    filename = f"Assessment_{safe_name}.pdf"

    resend_key = (os.getenv("RESEND_API_KEY") or "").strip()
    resend_from = (os.getenv("RESEND_FROM") or "").strip() or "Key West Lights <onboarding@resend.dev>"

    # Preferred: Resend HTTPS API (works on Railway)
    if resend_key:
        import base64
        import httpx as _httpx
        payload = {
            "from": resend_from,
            "to": [to_addr],
            "subject": subject,
            "text": body,
            "attachments": [
                {
                    "filename": filename,
                    "content": base64.b64encode(pdf_bytes).decode("ascii"),
                }
            ],
        }
        headers = {
            "Authorization": f"Bearer {resend_key}",
            "Content-Type": "application/json",
        }
        with _httpx.Client(timeout=60.0) as client:
            resp = client.post("https://api.resend.com/emails", headers=headers, json=payload)
        if resp.status_code >= 400:
            return f"Email failed (Resend {resp.status_code}): {resp.text[:300]}"
        return f"Email sent to {to_addr} via Resend"

    # Fallback: Gmail SMTP (often blocked on Railway)
    user = (GMAIL_USER or "").strip()
    password = (GMAIL_APP_PASSWORD or "").strip().replace(" ", "")
    if not user or not password:
        return (
            "Email skipped: set RESEND_API_KEY (recommended on Railway) "
            "or GMAIL_USER + GMAIL_APP_PASSWORD"
        )

    msg = MIMEMultipart()
    msg["From"] = user
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    part = MIMEBase("application", "pdf")
    part.set_payload(pdf_bytes)
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", "attachment", filename=filename)
    msg.attach(part)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as server:
            server.login(user, password)
            server.sendmail(user, [to_addr], msg.as_string())
        return f"Email sent to {to_addr} via Gmail SMTP"
    except OSError as e:
        return (
            f"Email failed: {e}. "
            "Railway often blocks SMTP. Add RESEND_API_KEY for HTTPS email."
        )



@app.post("/api/grade")
async def grade_test(payload: GradeRequest):
    print(f"GRADE request: {len(payload.answers)} answers, name={payload.applicant.get('name')}")
    user_prompt = build_user_prompt(payload.applicant, payload.answers, payload.claimed_year)
    result = await call_grok(GRADING_SYSTEM_PROMPT, user_prompt)

    result["_meta"] = {
        "graded_at": datetime.utcnow().isoformat() + "Z",
        "model": GROK_MODEL,
        "applicant_name": payload.applicant.get("name"),
        "claimed_year": payload.claimed_year,
    }

    email_status = "not attempted"
    try:
        pdf_bytes = build_results_pdf(payload.applicant, result)
        email_status = send_results_email(payload.applicant, result, pdf_bytes)
        print(email_status)
    except Exception as e:
        email_status = f"Email failed: {type(e).__name__}: {e}"
        print(email_status)

    result["_meta"]["email_status"] = email_status
    return JSONResponse(content=result)


@app.get("/health")
async def health():
    return {"status": "ok", "model": GROK_MODEL, "api_key_configured": bool(XAI_API_KEY)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
