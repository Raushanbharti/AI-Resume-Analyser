import io
import json
import os
import shutil
import zipfile
from pathlib import Path

import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from docx import Document
from openai import OpenAI

load_dotenv()

app = Flask(__name__)
# Allow your deployed frontend to access backend APIs
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "https://ai-resume-analyser-2-qxbn.onrender.com"
        ]
    }
})
@app.route("/", methods=["GET"])
def home():
    return {
        "status": "success",
        "message": "ResumeIQ AI Resume Analyzer Backend is running!",
        "api_health": "/api/health",
        "api_analyze": "/api/analyze",
        "model": "gpt-4o-mini"
    }, 200

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

FRONTEND_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "https://ai-resume-analyser-2-qxbn.onrender.com"
    ).split(",")
    if origin.strip()
]
CORS(app, resources={r"/api/*": {"origins": FRONTEND_ORIGINS}})

# Optional overrides for non-standard server installations.
if os.getenv("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_CMD"]


def ocr_image(image):
    """Run OCR on a PIL image and return extracted text."""
    try:
        return pytesseract.image_to_string(image, lang=os.getenv("OCR_LANG", "eng")).strip()
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "Tesseract OCR is not installed on the backend. Deploy the backend with the supplied Dockerfile."
        ) from exc


def extract_pdf_text(file_bytes):
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                raise ValueError("This PDF is password-protected. Please upload an unlocked PDF.")

        page_texts = []
        methods = set()
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            # OCR pages with no text OR suspiciously little text. This catches
            # scanned pages that contain a few selectable header/footer words.
            if len(text) < 40:
                try:
                    images = convert_from_bytes(
                        file_bytes,
                        dpi=250,
                        first_page=page_number,
                        last_page=page_number,
                        fmt="png",
                    )
                    if images:
                        ocr_text = ocr_image(images[0])
                        if len(ocr_text) > len(text):
                            text = ocr_text
                            methods.add("OCR")
                except Exception as exc:
                    message = str(exc).lower()
                    if "poppler" in message or "pdfinfo" in message:
                        raise RuntimeError(
                            "PDF OCR needs Poppler. Deploy the backend using the supplied Dockerfile (it installs Poppler)."
                        ) from exc
                    raise
            else:
                methods.add("PDF text extraction")
            if text:
                page_texts.append(f"[Page {page_number}]\n{text}")

        result = "\n\n".join(page_texts).strip()
        if not result:
            raise ValueError(
                "No readable text was found in this PDF. Try a clearer scan or a text-based PDF."
            )
        if "OCR" in methods and "PDF text extraction" in methods:
            method = "PDF text extraction + OCR"
        elif "OCR" in methods:
            method = "OCR"
        else:
            method = "PDF text extraction"
        return result, method
    except ValueError:
        raise
    except Exception as exc:
        if "poppler" in str(exc).lower() or "pdfinfo" in str(exc).lower():
            raise RuntimeError(
                "PDF processing needs Poppler. Deploy the backend using the supplied Dockerfile."
            ) from exc
        raise ValueError(f"Could not read this PDF: {exc}") from exc


def extract_docx_text(file_bytes):
    try:
        doc = Document(io.BytesIO(file_bytes))
        chunks = []

        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                chunks.append(paragraph.text.strip())

        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    chunks.append(row_text)

        # Also capture common header/footer content.
        for section in doc.sections:
            for paragraph in section.header.paragraphs:
                if paragraph.text.strip():
                    chunks.append(paragraph.text.strip())
            for paragraph in section.footer.paragraphs:
                if paragraph.text.strip():
                    chunks.append(paragraph.text.strip())

        text = "\n".join(chunks).strip()
        methods = ["DOCX text extraction"] if text else []

        # OCR pictures embedded in DOCX files (e.g. a scanned resume pasted
        # into Word). python-docx does not extract text from those pictures.
        image_texts = []
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as archive:
            media_files = [
                name for name in archive.namelist()
                if name.startswith("word/media/") and not name.endswith("/")
            ]
            for media_name in media_files:
                try:
                    image_bytes = archive.read(media_name)
                    with Image.open(io.BytesIO(image_bytes)) as image:
                        image_text = ocr_image(image.convert("RGB"))
                    if image_text:
                        image_texts.append(image_text)
                except (OSError, ValueError):
                    # Ignore unsupported/non-image media entries.
                    continue

        if image_texts:
            text = "\n\n".join(part for part in [text, *image_texts] if part).strip()
            methods.append("OCR of embedded DOCX images")

        if not text:
            raise ValueError(
                "No readable text was found in this DOCX. If it contains a scanned page, make sure the scan is inserted as an image and try again."
            )

        return text, " + ".join(methods)


    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not read this DOCX: {exc}") from exc

def extract_text(uploaded_file):
    filename = (uploaded_file.filename or "").lower()
    file_bytes = uploaded_file.read()

    if not file_bytes:
        raise ValueError("The uploaded file is empty.")

    if filename.endswith(".pdf"):
        return extract_pdf_text(file_bytes)
    if filename.endswith(".docx"):
        return extract_docx_text(file_bytes)

    raise ValueError("Please upload a PDF or DOCX resume.")


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "ocr": {
            "tesseract_installed": bool(shutil.which("tesseract")),
            "poppler_installed": bool(shutil.which("pdfinfo") and shutil.which("pdftoppm")),
        },
        "openai_key_configured": bool(os.getenv("OPENAI_API_KEY")),
    })


@app.post("/api/analyze")
def analyze():
    uploaded = request.files.get("resume")
    job_description = request.form.get("job_description", "").strip()

    if not uploaded:
        return jsonify({"error": "Upload a resume first."}), 400

    try:
        resume_text, extraction_method = extract_text(uploaded)
        if not resume_text.strip():
            return jsonify({"error": "No readable text was extracted from this resume."}), 400

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return jsonify({
                "error": "OPENAI_API_KEY is missing in the Render backend Environment settings. Add it there and redeploy."
            }), 503

        client = OpenAI(api_key=api_key)
        prompt = f"""
Analyze this resume as a career coach. Be constructive, evidence-based, and never invent experience.
Return ONLY valid JSON using this schema:
{{
  "score": integer 0-100,
  "score_breakdown": {{"ats": integer, "grammar": integer, "formatting": integer, "keywords": integer, "experience": integer}},
  "strengths": [string],
  "weaknesses": [string],
  "suggestions": [string],
  "missing_skills": [string],
  "grammar_issues": [{{"issue": string, "suggestion": string}}],
  "formatting_issues": [string],
  "hr_questions": [string],
  "technical_questions": [string],
  "behavioral_questions": [string],
  "project_questions": [string],
  "coach": [{{"question": string, "sample_answer": string}}],
  "confidence_tips": [string],
  "communication_tips": [string]
}}
Use the job description to assess keyword relevance if supplied. If none, evaluate for a general software/technology role and clearly avoid claiming job-specific matching.
RESUME:
{resume_text[:24000]}
JOB DESCRIPTION:
{job_description[:12000] or "Not supplied"}
"""
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "You are a careful ATS resume reviewer and interview coach. Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.35,
        )
        data = json.loads(response.choices[0].message.content)
        data["extracted_text"] = resume_text[:24000]
        data["extraction_method"] = extraction_method
        data["filename"] = uploaded.filename
        return jsonify(data)

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        # Do not expose credentials or full internal traces to website visitors.
        app.logger.exception("Resume analysis failed")
        return jsonify({
            "error": "Analysis failed. Check the backend Render logs and verify your OpenAI API key, billing, and model settings."
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
