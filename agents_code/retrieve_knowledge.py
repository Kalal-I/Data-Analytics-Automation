
import json
import faiss
import numpy as np
from pathlib import Path
from fastembed import TextEmbedding

BASE_DIR = Path(__file__).parent
INDEX_FILE = BASE_DIR / "knowledge.index"
METADATA_FILE = BASE_DIR / "knowledge_metadata.json"

model = TextEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

index = faiss.read_index(str(INDEX_FILE))

with open(METADATA_FILE, "r", encoding="utf-8") as file:
    paragraphs = json.load(file)


def retrieve_knowledge(query, top_k=5):
    query_embedding = np.array(
        list(model.embed([query])),
        dtype=np.float32
    )

    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(
        query_embedding,
        min(top_k, len(paragraphs))
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        results.append({
            "id": paragraphs[idx]["id"],
            "title": paragraphs[idx]["title"],
            "text": paragraphs[idx]["text"],
            "score": float(score)
        })

    return results


if __name__ == "__main__":
    test_query = (
        "The Age column has missing numerical values, "
        "and the Income column has missing values and "
        "different numerical scales."
    )

    results = retrieve_knowledge(test_query)

    for result in results:
        print(f"\n{result['id']} - {result['title']}")
        print(f"Similarity: {result['score']:.4f}")
        print(result["text"])
