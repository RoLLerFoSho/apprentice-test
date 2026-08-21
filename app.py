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

load_dotenv()

app = FastAPI(title="Residential Electrical Apprentice Screening Test", version="1.0")

if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

XAI_API_KEY = os.getenv("XAI_API_KEY")
XAI_BASE_URL = "https://api.x.ai/v1"
GROK_MODEL = os.getenv("GROK_MODEL", "grok-4.6")


class AnswerItem(BaseModel):
    id: str
    year: int
    section: str
    question: str
    answer: str


class GradeRequest(BaseModel):
    applicant: dict
    answers: list[AnswerItem]
    claimed_year: int


GRADING_SYSTEM_PROMPT = """
You are an expert residential electrical contractor, journeyman electrician, electrical instructor, and hiring manager with 25+ years of experience.

You are grading a Residential Electrical Apprentice Screening Test based on the 2023 National Electrical Code (NEC).

IMPORTANT: The candidate did NOT choose a year level. They answered a mixed set of questions spanning Year 1 through Year 4 material. Your job is to determine their ACTUAL year level from the quality of their answers.

Evaluate on these categories:
1. Safety Judgment & Safe Work Practices (critical weight)
2. Electrical Fundamentals (voltage, current, neutral vs ground, etc.)
3. Residential Installation Knowledge
4. Troubleshooting Approach
5. 2023 NEC Knowledge (practical application)
6. Practical Field Judgment
7. Understanding of Limitations / Willingness to Ask for Help
8. Mechanical Aptitude (how they approach physical/installation problems, tool use, recognition of bad workmanship, ability to visualize and solve hands-on issues)

OVERALL LEVELS (choose exactly one based on demonstrated performance):
- Below Year 1
- Year 1 Apprentice
- Year 2 Apprentice
- Year 3 Apprentice
- Year 4 Apprentice
- Ready for Journeyman-level responsibility

Also provide:
- skill_level: A short plain-English description (e.g. "Solid Year 2 with strong safety habits but weak on multiwire and subpanels")
- mechanical_aptitude: Assessment of hands-on / mechanical problem-solving ability
- pay_grade_band: Appropriate residential apprentice pay band
- project_placement: What type of work they can safely be assigned right now
- experience_mismatch_note: If they claimed experience that does not match their performance, note it clearly

CRITICAL RULES:
- Reward correct process and good safety judgment even if wording differs.
- Heavily penalize unsafe practices (working without testing, confusing neutral/ground, thinking breakers protect people from shock, etc.).
- An experienced person who explains correctly in their own words scores high.
- Vague, incomplete, or dangerous answers score low.
- Willingness to say "I don't know / I would ask the journeyman" is a POSITIVE trait for apprentices.
- Be fair but rigorous. Safety is non-negotiable.
- Base the year level on the highest level they can consistently perform at, not on a single correct advanced answer.

Return ONLY valid JSON in exactly this structure (no markdown, no extra text):

{
  "overall_score_percent": 0-100,
  "level": "one of the six levels above",
  "level_description": "2-3 sentence explanation of why this level was chosen",
  "skill_level": "short plain-English skill description",
  "mechanical_aptitude": "assessment of mechanical / hands-on aptitude",
  "pay_grade_band": "string",
  "project_placement": "string",
  "experience_mismatch_note": "string or empty if none",
  "category_scores": {
    "safety": 0-100,
    "fundamentals": 0-100,
    "installation": 0-100,
    "troubleshooting": 0-100,
    "nec": 0-100,
    "field_judgment": 0-100,
    "limitations": 0-100,
    "mechanical_aptitude": 0-100
  },
  "strengths": ["list of 3-6 specific strengths"],
  "weaknesses": ["list of 3-6 specific weaknesses or gaps"],
  "red_flags": ["list any serious safety or judgment concerns, or empty list"],
  "summary_for_hiring_manager": "4-6 sentence overall assessment and recommendation"
}
"""



def build_user_prompt(applicant: dict, answers: list, claimed_year: int) -> str:
    lines = []
    lines.append("=== APPLICANT INFORMATION ===")
    lines.append(f"Name: {applicant.get('name', 'Unknown')}")
    lines.append(f"Years of Experience Claimed: {applicant.get('years', 'N/A')}")
    lines.append(f"Previous Employers / Experience: {applicant.get('experience', 'N/A')}")
    lines.append("")
    lines.append("NOTE: Candidate did not select a year level. Determine their actual year level from performance.")
    lines.append("")
    lines.append("=== CANDIDATE ANSWERS ===")
    lines.append("")

    for a in answers:
        lines.append(f"--- {a.id} (Year {a.year} material – {a.section}) ---")
        lines.append(f"Q: {a.question}")
        lines.append(f"A: {a.answer.strip() if a.answer else '(No answer provided)'}")
        lines.append("")

    lines.append("Evaluate all answers thoroughly. Determine the actual year level, skill level, mechanical aptitude, and placement. Return the JSON result.")
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

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(url, headers=headers, json=payload)
        if resp.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=f"Grok API error {resp.status_code}: {resp.text[:500]}"
            )
        data = resp.json()

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise HTTPException(status_code=502, detail="Unexpected response format from Grok API")

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
        raise HTTPException(status_code=502, detail="Grok did not return valid JSON.")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/grade")
async def grade_test(payload: GradeRequest):
    user_prompt = build_user_prompt(payload.applicant, payload.answers, payload.claimed_year)
    result = await call_grok(GRADING_SYSTEM_PROMPT, user_prompt)

    result["_meta"] = {
        "graded_at": datetime.utcnow().isoformat() + "Z",
        "model": GROK_MODEL,
        "applicant_name": payload.applicant.get("name"),
        "claimed_year": payload.claimed_year,
    }
    return JSONResponse(content=result)


@app.get("/health")
async def health():
    return {"status": "ok", "model": GROK_MODEL, "api_key_configured": bool(XAI_API_KEY)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
