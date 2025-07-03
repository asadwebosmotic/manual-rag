from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict
import os
from data_ingestion.parse_and_chunk import parse_file, chunk_pdfplumber_parsed_data
from data_ingestion.embed_and_store import embed_and_store_pdf
from chat.chat_session import SlidingWindowSession
from llm.llm_operations import GeminiChat
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
import threading
import uuid

app = FastAPI(title="RAG Chat Bot with CRUD operation API",
              description="API for uploading pdf files in Qdrant DB that shows chunks, Gemini 2.5 flash LLm powered chat bot.",
              version="1.0.0")

embed_model = SentenceTransformer('intfloat/e5-base-v2')
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
qdrant_client = QdrantClient(host="localhost", port=6333)  # Adjust as needed
session_memory: Dict[str, SlidingWindowSession] = {}
session_lock = threading.Lock()  # Thread-safe session management

@app.on_event("startup")
async def startup_event():
    """Initialize resources on app startup."""
    # Any additional startup logic can go here
    pass

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Qdrant client
qdrant_client = QdrantClient(host="localhost", port=6333)

# --- 1. Upload PDF and get chunks ---
@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith(".pdf"): #validate if the uploaded file is pdf
            raise HTTPException(status_code=400, detail="Only PDF files are supported.") #if not pdf it will throw error with msg
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        # Parse and chunk
        pages = parse_file(file_path)
        chunks = chunk_pdfplumber_parsed_data(pages)
        stored_file = embed_and_store_pdf(chunks)
        return {"filename": file.filename, "chunks": chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

# --- 2. Chat with LLM (context from PDF + memory) ---
session_memory = {}

@app.post("/chat/")
async def chat_with_llm(user_msg: str, request: Request, pdf_name: Optional[str] = Query(None)):
    try:
        # Use a session ID (e.g., from a header or cookie) instead of client IP
        session_id = request.headers.get("X-Session-ID", str(uuid.uuid4()))
        
        # Thread-safe session access
        with session_lock:
            session = session_memory.setdefault(session_id, SlidingWindowSession(max_turns=10))

        # Validate input
        if not user_msg.strip():
            raise HTTPException(status_code=400, detail="User message cannot be empty")

        # Embed user query
        query_vector = embed_model.encode(user_msg).tolist()

        # Prepare Qdrant filter
        qdrant_filter = None
        if pdf_name and pdf_name.strip():
            qdrant_filter = Filter(must=[FieldCondition(key="source", match=MatchValue(value=pdf_name))])

        # Search Qdrant
        results = qdrant_client.search(
            collection_name="rag_chunks",
            query_vector=query_vector,
            limit=20,
            with_payload=True,
            query_filter=qdrant_filter
        )

        # Rerank results
        context = None
        if results:
            pairs = [(user_msg, r.payload['text']) for r in results]
            scores = reranker.predict(pairs)
            ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
            top_chunks = [r.payload['text'] for r, _ in ranked[:5]]
            context = "\n\n".join(top_chunks)

        # Generate response
        gemini = GeminiChat()
        response = gemini.generate_response(
            user_input=user_msg,
            context=context,
            chat_history=session.get_messages({"USER": "user", "ASSISTANT": "assistant"})
        )

        # Store in session
        with session_lock:
            session.add_turn(user_msg, response)

        return {"response": response.strip()}

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# --- 3. List PDFs in DB ---
@app.get("/pdfs/")
async def list_pdfs():
    try:
        res = qdrant_client.scroll(
            collection_name="rag_chunks",
            limit=1000,
            with_payload=True  # This is the key you're missing
        )
        pdfs = set()
        for point in res[0]:
            source = point.payload.get("source")
            if source:
                pdfs.add(source)
        return {"pdfs": list(pdfs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 3. Delete PDFs in DB ---
@app.delete("/pdfs/{pdf_name}")
async def delete_pdf(pdf_name: str):
    try:
        # Delete all points with this source
        qdrant_client.delete(
            collection_name="rag_chunks",
            points_selector=Filter(must=[FieldCondition(key="source", match=MatchValue(value=pdf_name))])
        )
        # Optionally, delete the file from disk
        file_path = os.path.join(UPLOAD_DIR, pdf_name)
        if os.path.exists(file_path):
            os.remove(file_path)
        return {"message": f"Deleted PDF and its chunks: {pdf_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete PDF: {str(e)}")

# --- 4. Global error handler ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": str(exc)})
