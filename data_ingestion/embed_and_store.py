import logging
import os
import uuid
from data_ingestion.parse_and_chunk import parse_file, chunk_pdfplumber_parsed_data
from qdrant_client import QdrantClient
from llm.llm_operations import GeminiChat
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer, CrossEncoder
from config import settings
from fastapi import APIRouter, HTTPException

router = APIRouter()

# === Configure logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Initialize Qdrant client and models ===
client = QdrantClient(host = "localhost", port = 6333)
embed_model = SentenceTransformer("intfloat/e5-base-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
model = GeminiChat()

# === Recreate collection if not exists ===
if not client.collection_exists("rag_chunks"):
    client.create_collection(
        collection_name="rag_chunks",
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
)

# === Parse + Chunk PDF ===
script_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(script_dir, "..", "chunking", "tables.pdf")
pages = parse_file(pdf_path)
chunks = chunk_pdfplumber_parsed_data(pages)

# === Create Embeddings ===
data = [chunk.get("page_content", "") for chunk in chunks]
embeddings = embed_model.encode(data).tolist()

# === Build + Upsert Points to Qdrant ===
points = []
for emb, chunk, text in zip(embeddings, chunks, data):
    if text.strip():
        metadata = chunk.get("metadata", {})
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=emb,
            payload={
                "text": text,
                "page": metadata.get("page_number", 1),
                "source": os.path.basename(metadata.get("source", "tables.pdf")),
                "type": metadata.get("type", "text")
            }
        )
        points.append(point)
        logger.info(f"Upserted point: {text[:50]}... with metadata: {point.payload}")

client.upsert(collection_name='rag_chunks', points=points)
print("✅ All chunks uploaded to Qdrant.")
