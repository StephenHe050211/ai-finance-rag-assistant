from pathlib import Path
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"


def load_chunks():
    chunks = []
    for path in DATA_DIR.glob("*.txt"):
        text = path.read_text().split("\n\n")
        for i, paragraph in enumerate(text, start=1):
            paragraph = paragraph.strip()
            if paragraph:
                chunks.append({"source": path.name, "chunk_id": i, "text": paragraph})
    return chunks


def retrieve(query, chunks, k=3):
    corpus = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(corpus + [query])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return ranked[:k]


def answer(query):
    chunks = load_chunks()
    results = retrieve(query, chunks)
    print(f"Question: {query}\n")
    print("Retrieved evidence:")
    for chunk, score in results:
        print(f"- {chunk['source']}#{chunk['chunk_id']} | score={score:.3f}")
        print(f"  {chunk['text']}\n")
    print("Draft answer:")
    print("Based on the retrieved evidence, the system should answer only using the cited snippets above. Replace this baseline printout with an LLM call when deploying in a private environment.")


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "What should be deducted before AE rebate is calculated?"
    answer(query)
