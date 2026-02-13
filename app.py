## this code will help in summaries and extract fields from any document linking it to google sheets, can do part of a admin job in some companies 
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pypdf import PdfReader
import io
import json
import re
import os

# ====== SET YOUR GROQ KEY HERE ======
os.environ["GROQ_API_KEY"] = "YOUR_GROQ_KEY_HERE"

client = Groq(api_key=os.environ["GROQ_API_KEY"])

app = FastAPI()

# ====== CORS (ALLOW GOOGLE SHEETS) ======
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== HEALTH CHECK ======
@app.get("/")
def health():
    return {"status": "Document Analyzer API running"}

# ====== PDF TEXT EXTRACTION ======
def extract_pdf_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

# ====== CLEAN RAW MODEL OUTPUT TO EXTRACT JSON ======
def extract_json(raw: str):
    # Remove markdown code fences
    raw = re.sub(r"```.*?```", "", raw, flags=re.DOTALL)
    raw = raw.strip()

    # Extract JSON object
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        return match.group(0)
    return raw

# ====== GROQ ANALYSIS ======
def analyze_text_with_groq(text: str):
    prompt = f"""
You are an expert document analysis AI.

You MUST return ONLY valid JSON.
No explanations.
No markdown.
No code fences.
No extra text.

Analyze the document text and return EXACTLY this structure:

{{
  "document_type": "",
  "fields": {{}},
  "summary": []
}}

Rules:
- "document_type" must be a short phrase (e.g., "Invoice", "Contract", "Report").
- "fields" must contain ONLY the key information relevant to the detected document type.
- "summary" must be a list of 1 to 5 bullet points (max 5).
- Bullet points must be short, clear, and factual.
- DO NOT include markdown or hyphens. Just plain text strings.

Document text:
{text}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    raw = response.choices[0].message.content
    cleaned = extract_json(raw)

    try:
        return json.loads(cleaned)
    except:
        return {
            "error": "Groq returned invalid JSON",
            "raw_response": raw,
            "cleaned_attempt": cleaned
        }

# ====== MAIN API ENDPOINT ======
@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)):
    pdf_bytes = await file.read()

    text = extract_pdf_text(pdf_bytes)
    result = analyze_text_with_groq(text)

    return result
