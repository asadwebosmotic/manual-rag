from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict
import os
from data_ingestion.parse_and_chunk import parse_file, chunk_pdfplumber_parsed_data
from data_ingestion.embed_and_store import embed_and_store_pdf, embed_model
from chat.chat_session import SlidingWindowSession
from llm.llm_operations import GeminiChat
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, PointStruct
from pathlib import Path
import threading
import uuid

app = FastAPI(title="RAG Chat Bot with CRUD operation API",
              description="API for uploading pdf files in Qdrant DB that shows chunks, Gemini 2.5 flash LLm powered chat bot.",
              version="1.0.0")

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
qdrant_client = QdrantClient(host="localhost", port=6333) 
session_memory: Dict[str, SlidingWindowSession] = {}
session_lock = threading.Lock()  # Thread-safe session management

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
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        Path(UPLOAD_DIR).mkdir(exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        # Parse and chunk
        pages = parse_file(file_path)
        chunks = chunk_pdfplumber_parsed_data(pages)
        
        # Embed and store in Qdrant
        points = []
        for chunk in chunks:
            vector = embed_model.encode(chunk["page_content"]).tolist()
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": chunk["page_content"],
                    "source": chunk["metadata"]["source"],
                    "page": chunk["metadata"]["page_number"],
                    "type": chunk["metadata"]["type"]
                }
            )
            points.append(point)
        qdrant_client.upsert(collection_name="rag_chunks", points=points)
        
        return {
            "filename": file.filename,
            "chunks": chunks,
            "message": f"Successfully uploaded and processed {file.filename}",
            "chunks_stored": len(points)
        }
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
            session = session_memory.setdefault(session_id, SlidingWindowSession(max_turns=20))

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
        top_chunks = []
        references = []
        context = ""
        # Rerank results if any
        if results:
            pairs = [(user_msg, r.payload["text"]) for r in results]
            scores = reranker.predict(pairs)
            ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
            
            Threshold = 0.3
            for (r, score) in ranked[:5]:
                if score < Threshold:
                    continue
                text = r.payload["text"]
                source = r.payload.get("source", "unknown_pdf")
                page = r.payload.get("page", "unknown_page")
                top_chunks.append(f"{text}\n(Source: {source}, Page: {page})")
                references.append(f"{source} (Page {page})")

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

        return {
            "response": response.strip(),
            "source": references
            }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# --- 3. List PDFs in DB ---
@app.get("/pdfs/")
async def list_pdfs():
    '''Gets the name of Pdf files uploaded and stored in qdrant db.

    Args:  None.

    Return: Dict of list of pdf files.
    '''
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
async def delete_pdf(pdf_name: str) -> Dict[str, str]:
    """
    Delete a PDF from local storage and its associated chunks from Qdrant.
    
    Args:
        pdf_name: Name of the PDF file to delete (e.g., 'document.pdf').
    
    Returns:
        Dict with a message indicating the result and number of chunks deleted.
    
    Raises:
        HTTPException: If deletion fails or the PDF/chunks are not found.
    """
    try:
        # Normalize pdf_name (remove path prefixes, ensure consistent extension)
        pdf_name = os.path.basename(pdf_name.strip())
        if not pdf_name:
            raise HTTPException(status_code=400, detail="Invalid PDF name")

        # Check if PDF exists in local storage
        file_path = os.path.join(UPLOAD_DIR, pdf_name)
        file_exists = os.path.exists(file_path)

        # Check if chunks exist in Qdrant
        search_result = qdrant_client.scroll(
            collection_name="rag_chunks",
            scroll_filter=Filter(must=[FieldCondition(key="source", match=MatchValue(value=pdf_name))]),
            limit=1  # Just check if any points exist
        )
        chunks_exist = len(search_result[0]) > 0

        if not file_exists and not chunks_exist:
            raise HTTPException(status_code=404, detail=f"No PDF or chunks found for: {pdf_name}")

        # Delete chunks from Qdrant
        deleted_count = 0
        if chunks_exist:
            delete_response = qdrant_client.delete(
                collection_name="rag_chunks",
                points_selector=Filter(must=[FieldCondition(key="source", match=MatchValue(value=pdf_name))])
            )
            # Qdrant doesn't return deleted count directly, so we estimate by re-checking
            post_delete_check = qdrant_client.scroll(
                collection_name="rag_chunks",
                scroll_filter=Filter(must=[FieldCondition(key="source", match=MatchValue(value=pdf_name))]),
                limit=1
            )
            deleted_count = "unknown" if post_delete_check[0] else "all"

        # Delete file from local storage if it exists
        if file_exists:
            os.remove(file_path)

        return {
            "message": f"Successfully deleted PDF '{pdf_name}' from local storage and its chunks from Qdrant",
            "chunks_deleted": str(deleted_count)
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete PDF or chunks: {str(e)}")

# --- 4. Global error handler ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": str(exc)})
