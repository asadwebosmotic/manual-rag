import os
from sentence_transformers import SentenceTransformer, CrossEncoder
from data_ingestion.parse_and_chunk import parse_and_chunk
from data_ingestion.embed_and_store import setup_qdrant, embed_and_store
from rag_engine.retriever import retrieve_relevant_chunks
from rag_engine.reranker import rerank_results
from rag_engine.generator import load_phi_model, generate_response
from chat.chat_session import ChatSession

# === Setup ===
embed_model = SentenceTransformer('intfloat/e5-base-v2')
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
client = setup_qdrant()
tokenizer, phi_model = load_phi_model("microsoft/Phi-3-mini-4k-instruct")
chat = ChatSession()

# === Embed PDF once ===
script_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(script_dir, "data_ingestion", "tables.pdf")
chunks = parse_and_chunk(pdf_path)
embed_and_store(chunks, client, embed_model)

# === Interactive Chat ===
while True:
    query = input("\n👤 You: ")
    if query.lower() in ["exit", "quit"]:
        break

    results = retrieve_relevant_chunks(client, embed_model, query)
    results = rerank_results(reranker, query, results)
    context = "\n\n".join([res.payload["text"] for res in results])
    answer = generate_response(tokenizer, phi_model, query, context, chat.get_history())
    chat.add_turn(query, answer)

    print(f"\n🤖 Phi: {answer}\n")