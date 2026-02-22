# 📬 AI Document Analyzer & Auto‑Routing System  
A full end‑to‑end automation pipeline that reads PDFs from Google Drive, analyzes them using a FastAPI + Gemini backend, classifies the document type, extracts structured fields, generates summaries, and routes the results into the correct Google Sheet.

This system removes manual document processing and turns it into a fully automated workflow.

---

## 🚀 Project Overview

This project consists of **two connected components**:

### **1. FastAPI Backend (Python + Gemini 2.5 Flash)**
- Accepts PDF uploads  
- Extracts text using `pypdf`  
- Sends the text + filename to Gemini  
- Receives structured JSON  
- Returns the result to the caller  

### **2. Google Apps Script Automation**
- Monitors a Google Drive folder  
- Sends each PDF to the FastAPI backend  
- Routes the result to the correct sheet  
- Saves a JSON summary file  
- Moves the PDF to a processed folder  
- Logs errors  

Together, they form a complete AI‑powered document‑processing system.

---

## 🧠 Features

### ✔ Automatic PDF ingestion  
Drop a PDF into the **RAW** folder → the system picks it up instantly.

### ✔ AI‑powered document classification  
Gemini identifies whether the document is an:
- Invoice  
- Contract  
- Certificate  
- Report  
- Financial Statement  
- Application  
- Letter  
- Shipping Document  
- Or closest match  

### ✔ Field extraction  
Depending on the document type, the AI extracts:
- Invoice numbers  
- Dates  
- Parties  
- Amounts  
- Certificate details  
- Contract metadata  
- Report titles  
- And more  

### ✔ Clean structured JSON output  
Every response follows this exact format:

```json
{
  "document_type": "",
  "fields": {},
  "summary": []
}
````
### Smart routing to Google Sheets
Documents are automatically routed into:
- ACCOUNTING
- CERTIFICATIONS
- DOCUMENTS
### ✔ JSON summary file saved
Each processed PDF gets a .json summary stored in a dedicated folder.
 Automatic file movement
Processed PDFs are moved from RAW → PROCESSED.
✔ Error logging
Any failure creates a log file in the LOGS folder.

## 🏗️ System Architecture
                ┌──────────────────────────────┐
                │     Google Drive (RAW)       │
                └───────────────┬──────────────┘
                                │
                                ▼
                    Apps Script Trigger
                                │
                                ▼
                ┌────────────────────────────────┐
                │  FastAPI Backend (Python)      │
                │  /analyze                      │
                │  Gemini 2.5 Flash              │
                └────────────────────────────────┘
                                │
                                ▼
                ┌────────────────────────────────┐
                │   JSON Result Returned          │
                └────────────────────────────────┘
                                │
                                ▼
                ┌────────────────────────────────┐
                │ Google Sheets (3‑Sheet System) │
                └────────────────────────────────┘
                                │
                                ▼
                ┌────────────────────────────────┐
                │ JSON Summary Saved             │
                │ PDF moved to PROCESSED         │
                └────────────────────────────────┘



## 🔌 FastAPI Endpoint
The deployed API:
POST https://doc-inalyzer-and-summarizer.onrender.com/analyze

## Request
- Multipart form upload
- Field name: file
- Value: PDF blob
## Response
Structured JSON with:
- document_type
- fields
- summary

## 🧩 FastAPI Code (Core Logic)
The backend:
- Extracts PDF text
- Builds a structured prompt
- Sends it to Gemini
- Cleans the output
- Returns valid JSON
## Key components:
- extract_pdf_text()
- analyze_text_with_gemini()
- extract_json()
- /analyze endpoint

## 📊 Google Apps Script Logic
The Apps Script:
- Reads next PDF from RAW folder
- Sends it to your FastAPI endpoint
- Parses the JSON
- Routes to correct sheet
- Saves .json summary
- Moves PDF to PROCESSED
- Logs errors
Routing logic:

```
if (/(invoice|bill|receipt|statement|financial|payment)/.test(type))
    return "ACCOUNTING";

if (/(certificate|certification|attestation)/.test(type))
    return "CERTIFICATIONS";

return "DOCUMENTS";
```
## 📁 Folder Structure (Google Drive)
````
RAW_FOLDER_ID
PROCESSED_FOLDER_ID
SUMMARIES_FOLDER_ID
LOGS_FOLDER_ID
````
## Each folder has a specific role in the pipeline.

### 🧪 Example Output
````
{
  "document_type": "Invoice",
  "fields": {
    "invoice_number": "BW-INV-2024-117",
    "supplier": "BrightWave Solutions LLC",
    "customer": "Horizon Tech Trading",
    "issue_date": "12 February 2024",
    "due_date": "26 February 2024",
    "amount": "920 AED"
  },
  "summary": [
    "Invoice issued by BrightWave Solutions LLC",
    "Billed to Horizon Tech Trading",
    "Includes Cloud Storage, API Usage, Support & Maintenance",
    "Total amount due is 920 AED"
  ]
}
````
## 🛠️ Tech Stack
### Backend
- Python
- FastAPI
- pypdf
- Google Gemini 2.5 Flash
- Render (deployment)
Frontend / Automation
- Google Apps Script
- Google Drive
- Google Sheets

## 📈 Future Enhancements
- Add RAG for document comparison
- Add duplicate invoice detection
- Add vendor‑specific extraction rules
- Add Slack/Teams notifications
- Add dashboard analytics

🏁 Summary
This project turns a manual, repetitive document‑processing workflow into a fully automated AI pipeline.
It’s fast, scalable, and easy to extend — perfect for accounting, HR, operations, and document‑heavy teams.

---



