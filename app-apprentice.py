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





def _safe(text, limit=500):
    if text is None:
        return "-"
    s = str(text)
    for a, b in [("—", "-"), ("–", "-"), ("‘", "'"), ("’", "'"), ("“", '"'), ("”", '"'), ("•", "*"), ("…", "..."), ("\r", " "), ("\n", " "), ("\t", " ")]:
        s = s.replace(a, b)
    s = "".join(ch if 32 <= ord(ch) < 127 else "?" for ch in s).strip()
    if not s:
        return "-"
    # force break long unbroken strings
    out = []
    for word in s.split(" "):
        while len(word) > 60:
            out.append(word[:60])
            word = word[60:]
        if word:
            out.append(word)
    return " ".join(out)[:limit]


def build_results_pdf(applicant: dict, grade: dict) -> bytes:
    """Minimal PDF writer - avoids FPDF layout crashes."""
    pdf = FPDF()
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(True, 15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    w = max(pdf.epw, 160)

    def line(txt, bold=False):
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", "B" if bold else "", 11 if bold else 10)
        text = _safe(txt, 1000)
        # write in chunks of ~90 chars to avoid width bugs
        while text:
            chunk = text[:90]
            text = text[90:]
            try:
                pdf.multi_cell(w, 6, chunk)
            except Exception:
                try:
                    pdf.cell(0, 6, chunk[:40], ln=True)
                except Exception:
                    pdf.ln(6)
        pdf.set_x(pdf.l_margin)

    line("Key West Lights - Assessment Report", bold=True)
    line("Confidential hiring document")
    line("")
    line("Candidate: " + _safe(applicant.get("name"), 80), bold=True)
    line("Years: " + _safe(applicant.get("years"), 40))
    line("Residential: " + _safe(applicant.get("resYears"), 20) + "  Commercial: " + _safe(applicant.get("comYears"), 20))
    line("Largest project: " + _safe(applicant.get("largestProject"), 100))
    line("Largest crew: " + _safe(applicant.get("largestCrew"), 40))
    line("J-Card: " + _safe(applicant.get("jcard"), 40))
    line("")
    line("LEVEL: " + _safe(grade.get("level"), 60), bold=True)
    line("Score: " + _safe(str(grade.get("overall_score_percent", "-"))) + "%")
    line("")
    line("Level description:", bold=True)
    line(grade.get("level_description") or "-")
    line("")
    line("Skill / strong suits:", bold=True)
    line(grade.get("skill_level") or "-")
    line("")
    line("Pay grade band:", bold=True)
    line(grade.get("pay_grade_band") or "-")
    line("")
    line("Project placement:", bold=True)
    line(grade.get("project_placement") or "-")
    line("")
    line("Hire recommendation:", bold=True)
    line(grade.get("hire_recommendation") or "-")
    line("")
    if grade.get("experience_mismatch_note"):
        line("Experience mismatch:", bold=True)
        line(grade.get("experience_mismatch_note"))
        line("")
    line("Category scores:", bold=True)
    cat = grade.get("category_scores") or {}
    if cat:
        for k, v in cat.items():
            line("  " + _safe(str(k)) + ": " + _safe(str(v)) + "%")
    else:
        line("  -")
    line("")
    line("Strengths:", bold=True)
    for s in (grade.get("strengths") or ["-"]):
        line("  - " + _safe(s, 200))
    line("")
    line("Weaknesses:", bold=True)
    for s in (grade.get("weaknesses") or ["-"]):
        line("  - " + _safe(s, 200))
    flags = grade.get("red_flags") or []
    if flags:
        line("")
        line("RED FLAGS:", bold=True)
        for s in flags:
            line("  ! " + _safe(s, 200))
    line("")
    line("Hiring manager summary:", bold=True)
    line(grade.get("summary_for_hiring_manager") or "-")
    line("")
    line("Confidential - Key West Lights internal use only.")

    out = BytesIO()
    pdf.output(out)
    return out.getvalue()


def send_results_email(applicant: dict, grade: dict, pdf_bytes: bytes) -> str:
    """Send PDF to RESULTS_EMAIL_TO via Gmail. Returns status message."""
    to_addr = (RESULTS_EMAIL_TO or "").strip()
    user = (GMAIL_USER or "").strip()
    password = (GMAIL_APP_PASSWORD or "").strip().replace(" ", "")

    if not to_addr:
        return "Email skipped: RESULTS_EMAIL_TO not set"
    if not user or not password:
        return "Email skipped: GMAIL_USER or GMAIL_APP_PASSWORD not set"

    name = applicant.get("name") or "Candidate"
    level = grade.get("level") or "Unknown level"
    score = grade.get("overall_score_percent", "-")

    msg = MIMEMultipart()
    msg["From"] = user
    msg["To"] = to_addr
    msg["Subject"] = f"Placement Assessment: {name} - {level} ({score}%)"

    body = f"""Electrical Skill-Level & Placement Assessment results

Candidate: {name}
Level: {level}
Overall score: {score}%
Hire recommendation: {grade.get('hire_recommendation', '-')}

PDF report attached.

- Key West Lights hiring system (automated)
"""
    msg.attach(MIMEText(body, "plain"))

    part = MIMEBase("application", "pdf")
    part.set_payload(pdf_bytes)
    encoders.encode_base64(part)
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:40]
    part.add_header("Content-Disposition", "attachment", filename=f"Placement_{safe_name}.pdf")
    msg.attach(part)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as server:
        server.login(user, password)
        server.sendmail(user, [to_addr], msg.as_string())

    return f"Email sent to {to_addr}"



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
