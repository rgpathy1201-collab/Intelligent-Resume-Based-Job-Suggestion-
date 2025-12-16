import os
from collections import defaultdict
from dotenv import load_dotenv

from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------- LOAD ENV ----------
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

# ---------- MONGO ----------
client = MongoClient(MONGO_URI)
db = client["resume_db"]

# This is the collection where Lambda stored chunks
chunks_col = db["resumes"]          # rename to "resumes_chunks" if you changed it
# This will be the new collection with one doc per resume
resumes_col = db["resumes_meta"]    # you can name it "resumes_master" etc.

# ---------- STEP 2: LOAD CHUNKS AND BUILD FULL TEXT ----------
def load_resumes_text():
    grouped = defaultdict(list)

    for doc in chunks_col.find():
        # expect fields: resume_id, chunk_index, text, bucket, key, timestamp
        rid = doc["resume_id"]
        grouped[rid].append(
            (
                doc.get("chunk_index", 0),
                doc.get("text", ""),
                doc.get("bucket", ""),
                doc.get("key", ""),
                doc.get("timestamp", "")
            )
        )

    resumes = []
    for rid, items in grouped.items():
        items_sorted = sorted(items, key=lambda x: x[0])
        full_text = "".join(text for _, text, _, _, _ in items_sorted)
        _, _, bucket, key, timestamp = items_sorted[0]
        resumes.append({
            "resume_id": rid,
            "full_text": full_text,
            "bucket": bucket,
            "key": key,
            "timestamp": timestamp
        })
    return resumes

# ---------- STEP 3: SKILLS EXTRACTION ----------
KNOWN_SKILLS = [
    "python", "java", "c++", "c#", "sql", "mysql", "postgresql", "mongodb",
    "aws", "azure", "gcp", "docker", "kubernetes",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
    "machine learning", "deep learning", "data analysis", "data engineering",
    "spark", "hadoop", "excel", "power bi", "tableau",
]

def extract_skills(text: str):
    text_low = text.lower()
    found = []
    for skill in KNOWN_SKILLS:
        if skill.lower() in text_low:
            found.append(skill.lower())
    return sorted(set(found))

# ---------- STEP 4: SIMPLE SUMMARY ----------
def simple_summary(text: str, max_chars: int = 600):
    text = text.strip().replace("\n", " ")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."

# ---------- STEP 5: TF-IDF VECTORS ----------
def build_tfidf_for_resumes(resumes):
    texts = [r["full_text"] for r in resumes]
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english"
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    return vectorizer, tfidf_matrix

# ---------- MAIN (STEPS 1–6) ----------
if __name__ == "__main__":
    # Step 2: load and group chunks
    resumes = load_resumes_text()
    print(f"Loaded {len(resumes)} resumes from chunks")

    if not resumes:
        print("No resumes found. Ensure Lambda has inserted chunk documents.")
        exit(0)

    # Step 5: build TF-IDF
    vectorizer, tfidf_matrix = build_tfidf_for_resumes(resumes)
    print("Built TF-IDF matrix with shape:", tfidf_matrix.shape)

    # Optionally save vectorizer for jobs later
    import pickle
    with open("tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    print("Saved TF-IDF vectorizer to tfidf_vectorizer.pkl")

    # Step 3,4,6: compute skills, summary, and store doc per resume
    docs = []
    for i, r in enumerate(resumes):
        full_text = r["full_text"]
        skills = extract_skills(full_text)
        summary = simple_summary(full_text)

        vec_dense = tfidf_matrix[i].toarray()[0]
        vec_list = vec_dense.tolist()

        doc = {
            "resume_id": r["resume_id"],
            "bucket": r["bucket"],
            "key": r["key"],
            "full_text": full_text,
            "skills": skills,
            "summary": summary,
            "vector": vec_list,
            "timestamp": r["timestamp"]
        }
        docs.append(doc)

    # Write to Mongo
    if docs:
        resumes_col.drop()  # optional, clears old data
        resumes_col.insert_many(docs)
        print(f"Inserted {len(docs)} resume-level documents into 'resumes_meta'")
    else:
        print("No documents to insert.")
