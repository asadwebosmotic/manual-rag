def retrieve_relevant_chunks(client, embed_model, query, top_k=20, threshold=0.7):
    query_vector = embed_model.encode(query).tolist()
    results = client.query_points(
        collection_name='rag_chunks',
        query=query_vector,
        limit=top_k,
        score_threshold=threshold
    )
    return results.points