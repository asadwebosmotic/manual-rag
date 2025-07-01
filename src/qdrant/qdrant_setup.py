import time
import uuid
import os
import torch
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.chunking.plumber_chunking import parse_file, chunk_pdfplumber_parsed_data
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Qdrant + Embedding Setup ===
client = QdrantClient(host="localhost", port=6333)
embed_model = SentenceTransformer('intfloat/e5-base-v2')
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# === Load Phi 3 mini 4k instruct (Instruction-Tuned) ===
phi_model_name = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(phi_model_name)
phi_model = AutoModelForCausalLM.from_pretrained(
    phi_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
phi_model.eval()

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

# === Query and RAG Flow ===
query = str(input('Enter your query:'))
query_vector = embed_model.encode(query).tolist()

start_time = time.time()
results = client.query_points(
    collection_name='rag_chunks',
    query=query_vector,
    limit=20,
    score_threshold=0.7
)
elapsed_time = time.time() - start_time
print(f"\n🔍 Query completed in {elapsed_time:.4f} seconds")

# === Filter and rerank results ===
pairs = [(query, res.payload["text"]) for res in results.points]
scores = reranker.predict(pairs)
ranked_results = sorted(zip(results.points, scores), key=lambda x: x[1], reverse=True)
seen_texts = set()
unique_results = []
for result, score in ranked_results:
    text = result.payload.get("text", "").strip()
    if text and text not in seen_texts:
        seen_texts.add(text)
        unique_results.append(result)
results = unique_results[:5]
logger.info(f"Retrieved {len(results)} points, filtered to {len(unique_results)} unique results")

# === Prepare top chunks for answer generation ===
top_chunks = [res.payload["text"] for res in results]
context = "\n\n".join(top_chunks)

# === Build prompt for Phi 3 mini instruct (chat-style) ===
prompt = f"""<start_of_turn>user
You are a helpful assistant. Based on the following extracted document chunks, provide a detailed explanation of the {query}, including its definition, purpose, how it is created, and its significance. Include specific examples from the context if available.

Query:
{query}

Context:
{context}
<end_of_turn>"""

# === Generate Answer ===
input_ids = tokenizer(prompt, return_tensors="pt").to(phi_model.device)

with torch.no_grad():
    output_ids = phi_model.generate(
        **input_ids,
        max_new_tokens=1000,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id
    )

answer = tokenizer.decode(output_ids[0][input_ids['input_ids'].shape[1]:], skip_special_tokens=True)
logger.info(f"Input tokens: {input_ids['input_ids'].shape[1]}, Generated tokens: {output_ids.shape[1]}")

# === Print Results ===
for idx, result in enumerate(results, 1):
    payload = result.payload
    print(f"\nResult #{idx}")
    print(f"Score: {result.score:.4f}")
    print(f"Page: {payload.get('page', 'N/A')}")
    print(f"Type: {payload.get('type', 'N/A')}")
    print(f"Source: {payload.get('source', 'N/A')}")
    print(f"Text:\n{payload.get('text', '')[:]}...\n")

print("\n💬 Microsoft Phi-3-mini Answer:\n")
print(answer.strip())
