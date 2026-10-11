
import json
import faiss
import numpy as np
from pathlib import Path
from fastembed import TextEmbedding

BASE_DIR = Path(__file__).parent
KNOWLEDGE_FILE = BASE_DIR / "knowledge_base.json"
INDEX_FILE = BASE_DIR / "knowledge.index"
METADATA_FILE = BASE_DIR / "knowledge_metadata.json"

# Load knowledge paragraphs
with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as file:
    paragraphs = json.load(file)

if not paragraphs:
    raise ValueError("Knowledge base is empty.")

# Create the embedding model
model = TextEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Embed each paragraph's title, keywords, and text
documents = []

for paragraph in paragraphs:
    document = (
        paragraph["title"] + ". "
        + " ".join(paragraph["keywords"]) + ". "
        + paragraph["text"]
    )
    documents.append(document)

embeddings = np.array(
    list(model.embed(documents)),
    dtype=np.float32
)

# Normalize vectors for cosine similarity
faiss.normalize_L2(embeddings)

# Build and save the FAISS index
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, str(INDEX_FILE))

# Save metadata in the same order as the index
with open(METADATA_FILE, "w", encoding="utf-8") as file:
    json.dump(paragraphs, file, indent=2, ensure_ascii=False)

print("Knowledge index created successfully.")
print("Paragraphs indexed:", len(paragraphs))
print("Embedding dimensions:", embeddings.shape[1])
