# Residential Electrical Apprentice Screening Test
## Live Grok (xAI) AI Grading – Hosted Version

2023 NEC Edition – Practical Field Assessment

This is a professional web application for screening residential electrical apprentice applicants (Year 1–4).

When the candidate finishes, **Grok (xAI)** evaluates every answer with real understanding and returns:

- Overall level (Below Year 1 → Ready for Journeyman)
- Whether they match / exceed / fall below the year they claimed
- Recommended pay-grade band
- Project placement recommendation
- Category scores (Safety, Fundamentals, Installation, Troubleshooting, NEC, Field Judgment, Limitations)
- Strengths, weaknesses, and red flags

---

## Deploy the same way as the Superintendent test

1. Create a new empty project on Railway
2. Connect a new GitHub repository
3. Upload these files
4. Set environment variables:
   - `XAI_API_KEY` = your xAI key
   - `GROK_MODEL` = `grok-4.6`
5. Set Start Command:
   ```
   uvicorn app:app --host 0.0.0.0 --port $PORT
   ```
6. Generate Domain

You can reuse the same xAI API key you already created.
