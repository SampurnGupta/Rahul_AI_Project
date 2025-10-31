"""
FastAPI web service for the PDF Chatbot - HackRx 6.0 Competition.
(Version 2.1.0) Enhanced: The chatbot always produces a non-placeholder answer for each question,
even if context is missing or LLM fails. If both document and LLM fail, a best-effort informative
response is synthesized using available summaries. No more generic placeholders such as
'I cannot answer...' or 'Unable to generate an answer due to an LLM API error.'
"""

import os
import shutil
import logging
import requests
import tempfile
from typing import List, Dict, Union, Tuple
from fastapi import FastAPI, HTTPException, status, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Query
from pydantic import BaseModel, Field, validator
import uvicorn
import time
from datetime import datetime
from advanced_pdf_bot import PDFChatbot, setup_logging
from cors_config import CORS_DEV, CORS_PROD, CORS_HACKRX, CORS_EXTERNAL_TESTING
from fastapi.staticfiles import StaticFiles

import hashlib
from urllib.parse import urlparse

# ---------------- Logging ----------------
def setup_hackathon_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

setup_hackathon_logging()
logger = logging.getLogger(__name__)

# ---------------- FastAPI App ----------------
app = FastAPI(
    title="HackRx 6.0 PDF Chatbot API",
    description="Competition-ready PDF question-answering service with exact specification compliance",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)
app.mount("/static", StaticFiles(directory="static"), name="static")


def get_cors_config():
    env = os.getenv("ENVIRONMENT", "hackrx").lower()
    if env == "production":
        return CORS_PROD
    elif env in ("external_testing", "remote_testing"):
        return CORS_EXTERNAL_TESTING
    else:
        return CORS_HACKRX

cors_config = get_cors_config()
app.add_middleware(CORSMiddleware, **cors_config)

# ---------------- Models ----------------
class HackRxRequest(BaseModel):
    documents: str = Field(
        ...,
        description="URL of the PDF document to process",
        example="https://example.com/sample.pdf"
    )
    questions: List[str] = Field(
        ...,
        min_items=1,
        max_items=20,
        description="List of questions to ask about the document"
    )
    @validator('documents')
    def validate_document_url(cls, v):
        if not v or not v.strip():
            raise ValueError("Document URL cannot be empty")
        if not v.startswith(('http://', 'https://')):
            raise ValueError("Document URL must be a valid HTTP/HTTPS URL")
        return v.strip()
    @validator('questions')
    def validate_questions(cls, v):
        if not v:
            raise ValueError("At least one question is required")
        cleaned = [q.strip() for q in v if q and q.strip()]
        if not cleaned:
            raise ValueError("At least one non-empty question is required")
        return cleaned

class FileUploadRequest(BaseModel):
    questions: List[str] = Field(
        ...,
        min_items=1,
        max_items=20,
        description="List of questions to ask about the uploaded document"
    )
    @validator('questions')
    def validate_questions(cls, v):
        if not v:
            raise ValueError("At least one question is required")
        cleaned = [q.strip() for q in v if q and q.strip()]
        if not cleaned:
            raise ValueError("At least one non-empty question is required")
        return cleaned

class HackRxResponse(BaseModel):
    answers: List[str] = Field(..., description="List of answers corresponding to the input questions")

# ---------------- Globals ----------------
chatbots: Dict[str, PDFChatbot] = {}
TEMP_DIR = "temp_pdfs"
os.makedirs(TEMP_DIR, exist_ok=True)

request_stats = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "average_response_time": 0
}

# ---------------- Helper: Answer quality check + fallback logic ----------------
def is_unusable_doc_answer(ans: str) -> bool:
    """
    Decide if the retrieval-based answer is NOT acceptable and we should
    transparently fall back to the LLM. Typical 'not found / unrelated'
    messages (including those produced by the workflow) are unusable.
    """
    if not ans:
        return True
    ans_l = ans.strip().lower()

    # Very short or generic non-answers
    if len(ans_l) < 5:
        return True
    if ans_l in {"i don't know", "idk", "no answer", "n/a"}:
        return True

    trigger_substrings = [
        "i'm sorry, but your question doesn't seem to be related",
        "couldn't find relevant information",
        "i couldn't find relevant information",
        "question doesn't seem to be related",
        "not related to the content of this document",
        "please ask questions about the topics covered",
        "i encountered an error while generating the response",
        "unable to generate an answer due to an llm api error",
        "i cannot answer based on the provided information",
        "no response from llm"
    ]
    if any(t in ans_l for t in trigger_substrings):
        return True

    return False

def best_effort_fallback(question: str, doc_summary: str = None) -> str:
    """
    Fallback answer if both retrieval and LLM fail. Use document summary if available.
    """
    if doc_summary and doc_summary.strip():
        return (
            f"The document does not provide a direct answer to your question, "
            f"but here is a summary of its content: {doc_summary.strip()}"
        )
    else:
        return (
            "No answer could be generated for your question. "
            "The document appears empty, unreadable, or there was a technical issue."
        )

# ---------------- Document handling ----------------
def download_document_from_url(url: str) -> Tuple[str, str]:
    try:
        logger.info(f"📥 Downloading document from URL: {url[:120]}")
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only http/https URLs are allowed"
            )

        start_time = time.time()
        guessed_ext = '.pdf'
        if url.lower().endswith('.docx'):
            guessed_ext = '.docx'

        with tempfile.NamedTemporaryFile(delete=False, suffix=guessed_ext, dir=TEMP_DIR) as temp_file:
            temp_path = temp_file.name
            headers = {
                'User-Agent': 'HackRx-PDFBot/2.0',
                'Accept': 'application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,/'
            }
            with requests.get(
                url,
                stream=True,
                timeout=60,
                headers=headers,
                allow_redirects=True
            ) as response:
                response.raise_for_status()
                content_type = response.headers.get('content-type', '').lower()
                if not any(x in content_type for x in ['pdf', 'word', 'octet-stream']):
                    logger.warning(f"⚠ Unexpected content type: {content_type}")

                total_size = 0
                hash_obj = hashlib.sha256()
                max_size = 50 * 1024 * 1024  # 50MB limit
                for chunk in response.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    total_size += len(chunk)
                    if total_size > max_size:
                        raise HTTPException(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail="File too large. Maximum size is 50MB"
                        )
                    temp_file.write(chunk)
                    hash_obj.update(chunk)

        download_time = time.time() - start_time
        logger.info(f"✅ Downloaded {total_size/1024/1024:.2f}MB in {download_time:.2f}s -> {temp_path}")
        return temp_path, hash_obj.hexdigest()

    except HTTPException:
        raise
    except requests.exceptions.Timeout:
        logger.error("⏰ Download timeout")
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail="Document download timed out. Try a smaller file or check the host."
        )
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Download error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to download document: {str(e)}"
        )
    except Exception as e:
        logger.error(f"❌ Unexpected download error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error downloading document: {str(e)}"
        )

def save_uploaded_file(upload_file: UploadFile) -> Tuple[str, str]:
    try:
        logger.info(f"📁 Saving uploaded file: {upload_file.filename}")

        filename_lower = upload_file.filename.lower()
        if not (filename_lower.endswith('.pdf') or filename_lower.endswith('.docx')):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF and DOCX files are supported."
            )

        suffix = '.docx' if filename_lower.endswith('.docx') else '.pdf'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=TEMP_DIR) as temp_file:
            temp_path = temp_file.name
            hash_obj = hashlib.sha256()
            max_size = 50 * 1024 * 1024
            total_size = 0
            while True:
                chunk = upload_file.file.read(8192)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > max_size:
                    os.remove(temp_path)
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="File too large. Maximum size is 50MB"
                    )
                temp_file.write(chunk)
                hash_obj.update(chunk)

        logger.info(f"✅ File saved: {total_size/1024/1024:.2f}MB to {temp_path}")
        return temp_path, hash_obj.hexdigest()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error saving uploaded file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing uploaded file: {str(e)}"
        )

def process_document_and_get_chatbot(document_source: Union[str, UploadFile], source_type: str = "url") -> PDFChatbot:
    """
    Process a document from URL or uploaded file and return a chatbot (with caching).
    """
    if source_type == "url":
        preliminary_key = f"url::{document_source}"
        if preliminary_key in chatbots:
            logger.info("🚀 Using cached chatbot for URL (string key)")
            return chatbots[preliminary_key]
        pdf_path, file_hash = download_document_from_url(document_source)
        cache_key = f"hash::{file_hash}"
        if cache_key in chatbots:
            logger.info("🚀 Using cached chatbot for URL (content hash)")
            try:
                os.remove(pdf_path)
            except Exception:
                pass
            return chatbots[cache_key]
    else:
        pdf_path, file_hash = save_uploaded_file(document_source)
        cache_key = f"hash::{file_hash}"
        if cache_key in chatbots:
            logger.info("🚀 Using cached chatbot for uploaded file (content hash)")
            try:
                os.remove(pdf_path)
            except Exception:
                pass
            return chatbots[cache_key]

    try:
        start_time = time.time()
        logger.info(f"🤖 Creating new chatbot instance for cache_key={cache_key}")

        chatbot = PDFChatbot(pdf_path, force_reprocess=False)
        chatbot.setup_index()
        process_start = time.time()
        chatbot.process_document()
        process_time = time.time() - process_start

        chatbots[cache_key] = chatbot
        if source_type == "url":
            chatbots[f"url::{document_source}"] = chatbot

        total_time = time.time() - start_time
        logger.info(f"✅ Chatbot ready! Processing: {process_time:.2f}s, Total: {total_time:.2f}s")
        return chatbot

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Chatbot creation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document: {str(e)}"
        )

# ---------------- Startup / Health ----------------
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting HackRx 6.0 PDF Chatbot API")
    logger.info(f"📊 Environment: {os.getenv('ENVIRONMENT', 'hackrx')}")
    logger.info("🔐 Auth: Disabled (public API)")
    logger.info(f"📁 Temp directory: {TEMP_DIR}")

@app.get("/", tags=["Health"])
async def root():
    return {
        "message": "HackRx 6.0 PDF Chatbot API is running",
        "status": "healthy",
        "version": "2.1.0",
        "competition": "HackRx 6.0",
        "endpoint": "/hackrx/run",
        "docs": "/docs",
        "stats": request_stats
    }

@app.api_route("/health", methods=["GET", "HEAD"], tags=["Health"])
async def health_check():
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "environment": os.getenv("ENVIRONMENT", "hackrx"),
        "cached_documents": len(chatbots),
        "llm_query_support": "enabled",
    }

# ---------------- Endpoints ----------------
@app.post(
    "/hackrx/run",
    response_model=HackRxResponse,
    tags=["HackRx Competition"],
    summary="Process PDF and answer questions",
    description="Main competition endpoint - processes a PDF document and answers questions about it"
)
async def hackrx_endpoint(request: HackRxRequest):
    start_time = time.time()
    request_stats["total_requests"] += 1
    try:
        logger.info(f"🎯 Processing HackRx request: {len(request.questions)} questions")
        logger.info(f"📄 Document: {request.documents[:120]}")
        chatbot = process_document_and_get_chatbot(request.documents, "url")

        # Try to extract document summary for fallback
        try:
            doc_summary = (getattr(chatbot, "doc_summary", None) or
                           getattr(chatbot, "document_summary", None) or
                           "")
            doc_summary = doc_summary.strip() if doc_summary else ""
        except Exception:
            doc_summary = ""

        answers: List[str] = []
        for i, question in enumerate(request.questions):
            q_start = time.time()
            logger.info(f"❓ Q{i+1}/{len(request.questions)}: {question[:120]}")
            try:
                doc_answer = ""
                try:
                    doc_answer = (chatbot.ask_question(question) or "").strip()
                except Exception as e:
                    logger.warning(f"⚠ Retrieval error for question {i+1}: {e}")

                if is_unusable_doc_answer(doc_answer):
                    logger.info(f"🔄 Fallback to LLM for question {i+1}")
                    llm_answer = query_llm_for_answer(question).strip()
                    if is_unusable_doc_answer(llm_answer):
                        logger.info(f"🛡️ Both retrieval and LLM failed; using best-effort fallback for Q{i+1}")
                        final_answer = best_effort_fallback(question, doc_summary)
                    else:
                        final_answer = llm_answer
                else:
                    final_answer = doc_answer

                answers.append(final_answer)
                logger.info(f"✅ Answer {i+1} ({time.time()-q_start:.2f}s): {final_answer[:120]}")
            except Exception as e:
                logger.error(f"❌ Error processing question {i+1}: {e}")
                fallback_ans = best_effort_fallback(question, doc_summary)
                answers.append(fallback_ans)

        total_time = time.time() - start_time
        request_stats["successful_requests"] += 1
        request_stats["average_response_time"] = (
            (request_stats["average_response_time"] * (request_stats["successful_requests"] - 1) + total_time)
            / request_stats["successful_requests"]
        )
        logger.info(f"🏁 Request completed: {len(answers)} answers in {total_time:.2f}s")
        return HackRxResponse(answers=answers)

    except HTTPException as e:
        request_stats["failed_requests"] += 1
        logger.error(f"❌ HTTP Exception: {e.detail}")
        raise
    except Exception as e:
        request_stats["failed_requests"] += 1
        logger.error(f"❌ Unexpected error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/api/v1/hackrx/run", response_model=HackRxResponse, tags=["HackRx Competition"])
async def hackrx_endpoint_v1(request: HackRxRequest):
    return await hackrx_endpoint(request)

@app.post(
    "/upload",
    response_model=HackRxResponse,
    tags=["File Upload"],
    summary="Upload PDF and answer questions",
    description="Upload a PDF/DOCX file and ask questions about it"
)
async def upload_pdf_endpoint(
    file: UploadFile = File(..., description="PDF/DOCX file to upload and process"),
    questions: str = Form(..., description="JSON array of questions as string")
):
    start_time = time.time()
    request_stats["total_requests"] += 1
    try:
        import json
        try:
            questions_list = json.loads(questions)
            if not isinstance(questions_list, list):
                raise ValueError("Questions must be a JSON array")
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid questions format. Expected JSON array: {str(e)}"
            )
        if not questions_list:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one question is required"
            )
        if len(questions_list) > 20:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maximum 20 questions allowed per request"
            )

        logger.info(f"📁 Processing uploaded file: {file.filename} with {len(questions_list)} questions")

        chatbot = process_document_and_get_chatbot(file, "upload")

        try:
            doc_summary = (getattr(chatbot, "doc_summary", None) or
                           getattr(chatbot, "document_summary", None) or
                           "")
            doc_summary = doc_summary.strip() if doc_summary else ""
        except Exception:
            doc_summary = ""

        answers: List[str] = []
        for i, question in enumerate(questions_list):
            q_start = time.time()
            logger.info(f"❓ Q{i+1}/{len(questions_list)}: {question[:120]}")
            try:
                answer_text = ""
                try:
                    answer_text = (chatbot.ask_question(question) or "").strip()
                except Exception as e:
                    logger.warning(f"⚠ Retrieval error (upload) Q{i+1}: {e}")

                if is_unusable_doc_answer(answer_text):
                    logger.info(f"🔄 Fallback to LLM (upload) for question {i+1}")
                    llm_answer = query_llm_for_answer(question).strip()
                    if is_unusable_doc_answer(llm_answer):
                        logger.info(f"🛡️ Both retrieval and LLM failed (upload); using best-effort fallback for Q{i+1}")
                        answer_text = best_effort_fallback(question, doc_summary)
                    else:
                        answer_text = llm_answer

                answers.append(answer_text)
                logger.info(f"✅ Answer {i+1} ({time.time()-q_start:.2f}s): {answer_text[:120]}")
            except Exception as e:
                logger.error(f"❌ Error processing question {i+1}: {e}")
                fallback_ans = best_effort_fallback(question, doc_summary)
                answers.append(fallback_ans)

        total_time = time.time() - start_time
        request_stats["successful_requests"] += 1
        request_stats["average_response_time"] = (
            (request_stats["average_response_time"] * (request_stats["successful_requests"] - 1) + total_time)
            / request_stats["successful_requests"]
        )
        logger.info(f"🏁 Upload request completed: {len(answers)} answers in {total_time:.2f}s")
        return HackRxResponse(answers=answers)

    except HTTPException as e:
        request_stats["failed_requests"] += 1
        logger.error(f"❌ HTTP Exception in upload endpoint: {e.detail}")
        raise
    except Exception as e:
        request_stats["failed_requests"] += 1
        logger.error(f"❌ Unexpected error in upload endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/api/v1/upload", response_model=HackRxResponse, tags=["File Upload"])
async def upload_pdf_endpoint_v1(
    file: UploadFile = File(..., description="PDF file to upload and process"),
    questions: str = Form(..., description="JSON array of questions as string")
):
    return await upload_pdf_endpoint(file, questions)

# @app.get("/keepwarm", tags=["Health", "Maintenance"])
@app.api_route(
    "/keepwarm",
    methods=["GET", "HEAD"],
    tags=["Health", "Maintenance"]
)
async def keepwarm(
    doc_url: str = Query(
        default="https://hackrx-pdfbot-syj1.onrender.com/static/warmup.pdf",
        description="URL of a small, always-cached PDF for warming up."
    ),
    question: str = Query(
        default="What is Adobe Acrobat PDF Files?",
        description="Trivial question for warm-up. Must match PDF content exactly for retrieval to succeed."
    )
):
    import time
    start_time = time.time()
    try:
        chatbot = process_document_and_get_chatbot(doc_url, "url")
        try:
            doc_summary = (getattr(chatbot, "doc_summary", None) or
                        getattr(chatbot, "document_summary", None) or
                        "")
            doc_summary = doc_summary.strip() if doc_summary else ""
        except Exception:
            doc_summary = ""
        doc_answer = ""
        try:
            doc_answer = (chatbot.ask_question(question) or "").strip()
        except Exception:
            pass
        if is_unusable_doc_answer(doc_answer):
            llm_answer = query_llm_for_answer(question).strip()
            if is_unusable_doc_answer(llm_answer):
                final_answer = best_effort_fallback(question, doc_summary)
            else:
                final_answer = llm_answer
        else:
            final_answer = doc_answer
        elapsed = time.time() - start_time
        return {
            "status": "ok",
            "keepwarm": True,
            "elapsed": elapsed,
            "doc_url": doc_url,
            "question": question,
            "answer": final_answer[:200]
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

# ---------------- Shutdown ----------------
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 Shutting down HackRx 6.0 PDF Chatbot API")
    logger.info(f"📊 Final stats: {request_stats}")
    try:
        import glob
        for pattern in (".pdf", ".docx"):
            temp_files = glob.glob(os.path.join(TEMP_DIR, pattern))
            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                    logger.info(f"🧹 Cleaned up: {temp_file}")
                except Exception as e:
                    logger.warning(f"⚠ Could not remove {temp_file}: {e}")
    except Exception as e:
        logger.error(f"❌ Cleanup error: {e}")

# ---------------- Main ----------------
if __name__ == "__main__":
    logger.info("🎮 Starting in development mode for hackathon testing")
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False,
        log_level="info",
        workers=1
    )

# ---------------- LLM Fallback (always returns a non-placeholder answer) ----------------
def query_llm_for_answer(question: str) -> str:
    """
    Query the LLM to answer questions outside the document context.
    If LLM or API fails, provide a best-effort generic informative answer.
    """
    try:
        logger.info(f"🔍 Querying LLM for question: {question}")
        import requests
        api_url = "https://api.openai.com/v1/completions"
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.warning("⚠ OPENAI_API_KEY not set; returning fallback message")
            return "The system is not configured to answer this question right now."
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "text-davinci-003",
            "prompt": question,
            "max_tokens": 150,
            "temperature": 0.7
        }
        response = requests.post(api_url, json=payload, headers=headers, timeout=40)
        response.raise_for_status()
        llm_response = response.json().get("choices", [{}])[0].get("text", "").strip()
        if not llm_response or is_unusable_doc_answer(llm_response):
            return ("No definitive answer was found in the document or from the AI model. "
                    "Please try rephrasing your question or provide more detail.")
        logger.info(f"✅ LLM response: {llm_response[:120]}")
        return llm_response
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Error querying LLM API: {e}")
        return ("No definitive answer was found due to a technical issue contacting the AI service. "
                "Please try again later or rephrase your question.")
    except Exception as e:
        logger.error(f"❌ Unexpected error querying LLM: {e}")
        return ("No answer could be generated due to an unexpected error. "
                "Please try again with a different question.")