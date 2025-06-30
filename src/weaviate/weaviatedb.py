import weaviate
import time
import uuid
import os
import google.generativeai as genai
from weaviate.classes.config import Configure, Property, DataType
from src.chunking.plumber_chunking import parse_file, chunk_pdfplumber_parsed_data
from sentence_transformers import SentenceTransformer, CrossEncoder

# === Configure Gemini ===
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))  # Or hardcode it for dev

model_gemini = genai.GenerativeModel("gemini-2.5-flash")

# === Connect to Weaviate ===
client = weaviate.connect_to_local()

# === Load embedding models ===
retriever_model = SentenceTransformer('intfloat/e5-base-v2')
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# === Chunking ===
pages = parse_file("src/chunking/tables.pdf")
chunks = chunk_pdfplumber_parsed_data(pages)

# === Filter and clean chunks ===
clean_chunks = [chunk for chunk in chunks if len(chunk["page_content"].split()) >= 5 and "|" not in chunk["page_content"][:10]]
data = [chunk['page_content'] for chunk in clean_chunks]
embeddings = retriever_model.encode([f"passage: {d}" for d in data]).tolist()

# === Create collection if not exists ===
if not client.collections.exists("rag_chunks"):
    client.collections.create(
        name="rag_chunks",
        properties=[
            Property(name="text", data_type=DataType.TEXT),
            Property(name="page", data_type=DataType.INT),
            Property(name="source", data_type=DataType.TEXT),
            Property(name="type", data_type=DataType.TEXT)
        ],
        vectorizer_config=Configure.Vectorizer.none()
    )

collection = client.collections.get("rag_chunks")

# === Insert chunks ===
for idx, chunk in enumerate(clean_chunks):
    metadata = {
        "text": chunk["page_content"],
        "page": chunk.get("page", 0),
        "source": "tables.pdf",
        "type": chunk.get("type", "text")
    }
    collection.data.insert(uuid=str(uuid.uuid4()), properties=metadata, vector=embeddings[idx])

print("✅ All cleaned chunks inserted into Weaviate.")

# === SEMANTIC SEARCH WITH RERANKING ===
query = "give me the paragraph from page number 182 for 4.6 common size statement"
query_vector = retriever_model.encode(f"query: {query}")

start_time = time.time()
results = collection.query.near_vector(query_vector, limit=10).objects
elapsed_time = time.time() - start_time

# === Prepare rerank candidates ===
candidates = [(query, obj.properties["text"]) for obj in results]
scores = reranker.predict(candidates)

# === Sort and show top 3 results ===
ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)[:3]

print(f"\n🔍 Top Results for Query (Reranked, retrieved in {elapsed_time:.2f} sec):")
context_chunks = []
for obj, score in ranked:
    text = obj.properties["text"]
    page = obj.properties.get("page", "?")
    src = obj.properties.get("source", "?")
    chunk_type = obj.properties.get("type", "?")
    context_chunks.append(text)
    print(f"\n📄 Source: {src} | 📄 Page: {page} | 🔖 Type: {chunk_type} | 🧠 Score: {score:.4f}")
    print(f"---\n{text.strip()[:]}...\n---")

# === Gemini Answer Generation ===
context = "\n\n".join(context_chunks)
gemini_prompt = f"""
You are a helpful assistant. Based on the following extracted chunks from a PDF document, answer the user query.

User Query:
{query}

Extracted Chunks:
{context}
"""

response = model_gemini.generate_content(gemini_prompt)
print("\n💬 Gemini Answer:\n", response.text.strip())

client.close()
