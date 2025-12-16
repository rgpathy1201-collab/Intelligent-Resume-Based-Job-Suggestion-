import os
from dotenv import load_dotenv
from pymongo import MongoClient
import numpy as np

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI)
db = client["resume_db"]
resumes_col = db["resumes_meta"]
jobs_col = db["jobs"]

# ---- helpers ----

def cosine_sim(a, b):
    a = np.array(a)
    b = np.array(b)
    if a.shape != b.shape:
        # pad or early return; for now just handle mismatch
        min_len = min(len(a), len(b))
        a = a[:min_len]
        b = b[:min_len]
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def keyword_overlap(resume_skills, job_skills):
    rs = set(resume_skills or [])
    js = set(job_skills or [])
    if not rs or not js:
        return 0.0
    return len(rs & js) / len(rs | js)

import time
import math

def recency_weight(posted_ts):
    if not posted_ts:
        return 0.5
    days = max(0, (time.time() - posted_ts) / 86400.0)
    return math.exp(-days / 30.0)  # recent jobs ≈ 1, old jobs → 0

def final_score(sem, kw, rec, pop=0.5):
    return 0.55*sem + 0.25*kw + 0.10*rec + 0.10*pop

if __name__ == "__main__":
    # 1) pick latest resume for now
    resume = resumes_col.find_one(sort=[("timestamp", -1)])
    if not resume:
        print("No resumes in resumes_meta")
        exit(0)

    print("Using resume_id:", resume["resume_id"])
    resume_vec = resume["vector"]
    resume_skills = resume.get("skills", [])

    # 2) get all jobs
    jobs = list(jobs_col.find())
    if not jobs:
        print("No jobs in jobs collection")
        exit(0)

    matches = []
    for job in jobs:
        job_vec = job.get("vector")
        if not job_vec:
            continue
        sem = cosine_sim(resume_vec, job_vec)
        kw = keyword_overlap(resume_skills, job.get("required_skills", []))
        rec = recency_weight(job.get("posted_at"))
        score = final_score(sem, kw, rec, pop=0.5)

        matches.append({
            "title": job.get("title", ""),
            "company": job.get("company", ""),
            "location": job.get("location", ""),
            "url": job.get("url", ""),
            "score": score,
            "semantic_similarity": sem,
            "keyword_overlap": kw
        })

    # 3) sort and show top 10
    matches_sorted = sorted(matches, key=lambda x: x["score"], reverse=True)[:10]

    print("\nTop 10 matches:")
    for m in matches_sorted:
        print(f"\n{m['title']} @ {m['company']}")
        print(f"Location: {m['location']}")
        print(f"Score: {m['score']:.3f} | semantic={m['semantic_similarity']:.3f} | kw={m['keyword_overlap']:.3f}")
        print(f"URL: {m['url']}")
