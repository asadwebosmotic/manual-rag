# --- Imports ---
import time
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from src.chunking.chunker import chunks  # Should return list of dicts like {'text': ..., 'metadata': {...}}

# --- Initialize ChromaDB in-memory ---
client = chromadb.Client(Settings())

# --- Embedding model ---
model = SentenceTransformer('intfloat/e5-base-v2')

# --- Preprocess chunks: Add page number & filter valid metadata ---
valid_data = []
for i, chunk in enumerate(chunks):
    text = chunk['text']
    metadata = chunk.get('metadata', {})
    
    # Attach page number if available, else -1
    page = metadata.get('page', -1)
    metadata['page'] = page

    # Skip empty metadata
    if isinstance(metadata, dict) and len(metadata) > 0:
        valid_data.append((text, metadata, f"chunk_{i}"))

# --- Unpack valid data ---
texts, metadatas, ids = zip(*valid_data)
embeddings = model.encode(texts).tolist()

# --- Create collection ---
collection = client.get_or_create_collection(name="rag_chunks")

# --- Add data to ChromaDB ---
collection.add(
    documents=list(texts),
    ids=list(ids),
    metadatas=list(metadatas),
    embeddings=embeddings
)

# Start timing query
start_time = time.time()

# --- Querying ChromaDB ---
query = "tell me about The taxonomy identifier if the tag is a standard tag, otherwise adsh"
query_embedding = model.encode([query]).tolist()[0]

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

# End timing
elapsed_time = time.time() - start_time

# --- Output retrieved chunks ---
retrieved_chunks = results['documents'][0]
retrieved_metadata = results['metadatas'][0]

# --- Display retrieved results with page info ---
for i, (chunk, meta) in enumerate(zip(retrieved_chunks, retrieved_metadata)):
    print(f"\n🔍 Query completed in {elapsed_time:.4f} seconds")
    print(f"\n--- Retrieved Chunk {i+1} (Page: {meta.get('page', 'N/A')}) ---")
    print(chunk)
