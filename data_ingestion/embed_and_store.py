import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import os

def setup_qdrant():
    client = QdrantClient(host="localhost", port=6333)
    if not client.collection_exists("rag_chunks"):
        client.create_collection(
            collection_name="rag_chunks",
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
    return client

def embed_and_store(chunks, client, embed_model):
    data = [chunk.get("page_content", "") for chunk in chunks]
    embeddings = embed_model.encode(data).tolist()
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

    client.upsert(collection_name='rag_chunks', points=points)
    return len(points)