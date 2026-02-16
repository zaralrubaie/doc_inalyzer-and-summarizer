from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
import google.generativeai as genai
import io
import json
import re
import os

# ====== CONFIGURE GEMINI ======
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("models/gemini-2.5-pro")
#
app = FastAPI()

# ====== CORS ======
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
    return {"status": "Document Analyzer API running (Gemini version)"}

# ====== PDF TEXT EXTRACTION ======
def extract_pdf_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text.strip()

# ====== CLEAN RAW MODEL OUTPUT ======
def extract_json(raw: str):
    raw = re.sub(r"```.*?```", "", raw, flags=re.DOTALL).strip()
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    return match.group(0) if match else raw

# ====== GEMINI ANALYSIS ======
def analyze_text_with_gemini(text: str, filename: str):
    prompt = f"""
You are an expert document analysis AI.

Your job is to classify the document type, extract key fields, and generate a summary.
If the PDF text is incomplete, noisy, or missing, you MUST use the filename and keywords.

You MUST return ONLY valid JSON. No explanations. No markdown.

Return EXACTLY this structure:

{{
  "document_type": "",
  "fields": {{}},
  "summary": []
}}

===========================
CLASSIFICATION RULES
===========================
Use filename + text.

- If filename or text contains: invoice, tax, bill, receipt → "Invoice"
- If contains: contract, agreement → "Contract"
- If contains: certificate, cert, attestation → "Certificate"
- If contains: report, analysis, assessment → "Report"
- If contains: payment, statement → "Financial Statement"
- If contains: application, request → "Application"
- If contains: letter, correspondence → "Letter"
- If contains: shipping, delivery, airway, cargo → "Shipping Document"
- If unsure, choose the closest match. Avoid "Unknown" unless absolutely no clues exist.

===========================
FIELD EXTRACTION RULES
===========================
- Extract key-value pairs relevant to the document type.
- If text is weak, infer fields from filename patterns.
- Keep fields short, factual, and structured.
- Examples:
  - Invoice: invoice number, date, amount, supplier, customer
  - Certificate: certificate number, issue date, expiry date, holder
  - Contract: parties, start date, end date, contract number
  - Report: report title, date, subject

===========================
SUMMARY RULES
===========================
- 1 to 5 bullet points.
- No hyphens. Just plain text strings.
- Summaries must be factual and concise.

===========================
INPUTS
===========================
FILENAME:
{filename}

DOCUMENT TEXT:
{text}
"""

    try:
        response = model.generate_content(prompt)
        raw = response.text
        cleaned = extract_json(raw)
        return json.loads(cleaned)

    except Exception as e:
        return {
            "error": "Gemini returned invalid JSON",
            "raw_response": str(response.text if 'response' in locals() else ''),
            "exception": str(e)
        }

# ====== MAIN API ENDPOINT ======
@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    text = extract_pdf_text(pdf_bytes)
    result = analyze_text_with_gemini(text, file.filename)
    return result
