def rerank_results(reranker, query, points):
    pairs = [(query, point.payload["text"]) for point in points]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(points, scores), key=lambda x: x[1], reverse=True)
    seen = set()
    filtered = []
    for point, _ in ranked:
        text = point.payload["text"].strip()
        if text and text not in seen:
            seen.add(text)
            filtered.append(point)
    return filtered[:5]