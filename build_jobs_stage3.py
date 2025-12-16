import os
import requests
from dotenv import load_dotenv
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from datetime import datetime

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY")
ADZUNA_COUNTRY = os.getenv("ADZUNA_COUNTRY", "in")

client = MongoClient(MONGO_URI)
db = client["resume_db"]
jobs_raw_col = db["jobs_raw"]
jobs_col = db["jobs"]

# Load the same TF-IDF vectorizer used for resumes
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer: TfidfVectorizer = pickle.load(f)

# Skills list reused from Stage 2
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

def fetch_jobs_from_adzuna(query="data scientist", results_per_page=20, page=1):
    base_url = f"https://api.adzuna.com/v1/api/jobs/{ADZUNA_COUNTRY}/search/{page}"
    params = {
        "app_id": ADZUNA_APP_ID,
        "app_key": ADZUNA_APP_KEY,
        "what": query,
        "results_per_page": results_per_page,
        "content-type": "application/json"
    }
    resp = requests.get(base_url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data.get("results", [])

def store_jobs_raw(jobs):
    if not jobs:
        return
    jobs_raw_col.insert_many(jobs)
    print(f"Inserted {len(jobs)} raw job documents into 'jobs_raw'")

def normalize_job(raw):
    title = raw.get("title", "")
    company = (raw.get("company") or {}).get("display_name", "")
    location = (raw.get("location") or {}).get("display_name", "")
    description = raw.get("description", "")
    url = raw.get("redirect_url", "")
    created = raw.get("created", "")

    # Convert created to timestamp
    try:
        posted_at = datetime.fromisoformat(created.replace("Z", "+00:00"))
        posted_ts = posted_at.timestamp()
    except Exception:
        posted_ts = None

    required_skills = extract_skills(description)

    return {
        "title": title,
        "company": company,
        "location": location,
        "description": description,
        "url": url,
        "posted_at": posted_ts,
        "required_skills": required_skills
    }

def build_jobs_documents():
    docs = []
    for raw in jobs_raw_col.find():
        docs.append(normalize_job(raw))
    return docs

from scipy.sparse import csr_matrix
import numpy as np

def add_vectors_to_jobs(jobs_docs):
    texts = [j["description"] for j in jobs_docs]
    tfidf_matrix: csr_matrix = vectorizer.transform(texts)
    for i, j in enumerate(jobs_docs):
        vec_dense = tfidf_matrix[i].toarray()[0]
        j["vector"] = vec_dense.tolist()
    return jobs_docs

if __name__ == "__main__":
    # 1) Fetch jobs from API
    print("Fetching jobs from Adzuna...")
    jobs = fetch_jobs_from_adzuna(query="data scientist", results_per_page=25, page=1)
    print(f"Fetched {len(jobs)} jobs")

    if not jobs:
        print("No jobs fetched. Check API key / query / country.")
        exit(0)

    # 2) Store raw jobs
    store_jobs_raw(jobs)

    # 3) Build normalized jobs
    jobs_docs = build_jobs_documents()
    print(f"Normalized {len(jobs_docs)} jobs")

    if not jobs_docs:
        print("No jobs_docs to process.")
        exit(0)

    # 4) Add TF-IDF vectors
    jobs_docs = add_vectors_to_jobs(jobs_docs)
    print("Added TF-IDF vectors to jobs")

    # 5) Save into 'jobs' collection
    jobs_col.drop()  # optional clear
    jobs_col.insert_many(jobs_docs)
    print(f"Inserted {len(jobs_docs)} documents into 'jobs'")
