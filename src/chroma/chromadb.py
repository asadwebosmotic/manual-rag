import chromadb
import time
from sentence_transformers import SentenceTransformer
from src.chunking.plumber_chunking import parse_file, chunk_pdfplumber_parsed_data
import uuid

chroma_client = chromadb.Client()
model = SentenceTransformer('intfloat/e5-base-v2')

pages = parse_file("src/chunking/tables.pdf")
chunks = chunk_pdfplumber_parsed_data(pages)

data = [chunk['page_content'] for chunk in chunks]

collection = chroma_client.create_collection(name="rag_chunks")

for chunk in chunks:
    text = chunk['page_content']
    embedding = model.encode(f"passage: {text}")
    metadata = {
        "text": text,
        "page": chunk.get("page", -1),
        "source": "tables.pdf",
        "type": chunk.get("type", "text")
    }
    collection.add(
        documents=[text],
        metadatas=[metadata],
        ids=[chunk.get("chunk_id", str(uuid.uuid4()))]
    )
    print(f"Added chunk {chunk.get('chunk_id', 'unknown')}")

print("✅ All chunks inserted into Chroma successfully.")

# Start timing query
start_time = time.time()

results = collection.query(
    query_texts=["tell me about common size statement"],
    n_results=2 # how many results to return
)

# End timing
elapsed_time = time.time() - start_time

# Output results
print(f"\n🔍 Query completed in {elapsed_time:.4f} seconds")
print(results)