from fastapi import FastAPI, UploadFile, File
from groq import Groq
from pypdf import PdfReader
import io
import os
import uvicorn

app = FastAPI()

client = Groq(api_key=os.environ["GROQ_API_KEY"])

def extract_pdf_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def analyze_text_with_groq(text: str) -> str:
    prompt = f"""
You are an expert document analysis AI.
Given the following document text, do 3 things:
1. Identify the document type (invoice, receipt, contract, report, certificate, etc.)
2. Extract key fields in JSON format.
3. Provide a short summary.

Document text:
{text}

Return your answer in this JSON structure:
{{
  "document_type": "",
  "fields": {{}},
  "summary": ""
}}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content

@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    text = extract_pdf_text(pdf_bytes)
    result = analyze_text_with_groq(text)
    return {"result": result}

@app.get("/")
def home():
    return {"message": "Document Analyzer API is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
