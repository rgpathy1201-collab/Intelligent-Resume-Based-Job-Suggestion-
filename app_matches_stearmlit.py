import os
import time
import math
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient
import boto3

# ---------- INITIAL SETUP ----------
load_dotenv()

# Mongo
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["resume_db"]
resumes_col = db["resumes_meta"]   # one doc per resume
jobs_col = db["jobs"]              # normalized jobs with vectors

# S3
s3 = boto3.client("s3")
BUCKET_NAME = "resume-upload-guvi"   # your existing bucket

# ---------- BACKGROUND IMAGE (CSS) ----------
# Put an image file (e.g., bg.jpg) in the same folder and set its URL via base64 or static hosting.
# For simplicity here, use a remote image URL. Replace with your own if needed.
BACKGROUND_URL = "https://images.pexels.com/photos/1181671/pexels-photo-1181671.jpeg"  # example tech/office image

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{BACKGROUND_URL}");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }}
    .block-container {{
        background-color: rgba(255, 255, 255, 0.85);
        padding: 1.5rem 1.5rem 3rem 1.5rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.set_page_config(page_title="Resume Matcher", layout="wide")

st.title("End-to-End Resume Matcher")

tab1, tab2 = st.tabs(["📤 Upload Resume", "📊 Matches & Insights"])

# ---------- HELPERS (for matching) ----------

def cosine_sim(a, b):
    a = np.array(a)
    b = np.array(b)
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

def recency_weight(posted_ts):
    if not posted_ts:
        return 0.5
    days = max(0, (time.time() - posted_ts) / 86400.0)
    return math.exp(-days / 30.0)

def final_score(sem, kw, rec, pop=0.5):
    return 0.55*sem + 0.25*kw + 0.10*rec + 0.10*pop

def explain_match(resume_skills, job_skills, score):
    rs = set(resume_skills or [])
    js = set(job_skills or [])
    common = sorted(rs & js)
    missing = sorted(js - rs)

    lines = [f"Score: {score:.2f}"]
    if common:
        lines.append("Common: " + ", ".join(common))
    if missing:
        lines.append("Learn: " + ", ".join(missing[:5]))
    return " | ".join(lines)

COURSE_MAP = {
    "python": ["Coursera: Python for Everybody"],
    "sql": ["Coursera: SQL for Data Science"],
    "machine learning": ["Coursera: Machine Learning (Andrew Ng)"],
    "deep learning": ["Coursera: Deep Learning Specialization"],
    "aws": ["Coursera: AWS Fundamentals"],
}

def course_recommendations(missing_skills):
    rows = []
    for skill in missing_skills:
        for course in COURSE_MAP.get(skill, []):
            rows.append({"Skill": skill, "Course": course})
    return rows

# ---------- TAB 1: UPLOAD RESUME ----------

with tab1:
    st.subheader("Upload your resume to S3")

    uploaded_file = st.file_uploader("Upload your resume", type=["pdf", "docx"])

    if uploaded_file is not None:
        if st.button("Upload to S3"):
            try:
                user_id = str(uuid.uuid4())
                timestamp = datetime.now().isoformat()

                s3.upload_fileobj(
                    uploaded_file,
                    BUCKET_NAME,
                    f"resumes/{user_id}_{uploaded_file.name}",
                    ExtraArgs={
                        "Metadata": {
                            "user_id": user_id,
                            "timestamp": timestamp,
                            "file_type": uploaded_file.type
                        }
                    }
                )

                st.success("Resume uploaded successfully to S3!")
                st.info(
                    "Lambda will pick this file from S3, extract text, and store chunks in MongoDB."
                )
            except Exception as e:
                st.error(f"Upload failed: {e}")

# ---------- TAB 2: MATCHES & INSIGHTS ----------

with tab2:
    st.subheader("View job matches and insights for processed resumes")

    resume_docs = list(resumes_col.find())
    if not resume_docs:
        st.warning("No processed resumes found in 'resumes_meta'.")
        st.info("Make sure Lambda + Stage 2 scripts have run at least once.")
    else:
        resume_options = {f"{r['resume_id'][:8]}...": r for r in resume_docs}
        selected_label = st.selectbox("Select a processed resume", list(resume_options.keys()))
        selected_resume = resume_options[selected_label]

        resume_skills = selected_resume.get("skills", [])
        st.markdown("### Selected Resume Overview")
        st.write(f"**Resume ID:** {selected_resume['resume_id']}")
        st.write("**Skills detected:**", ", ".join(resume_skills) if resume_skills else "None")

        if st.checkbox("Show resume summary", key="summary"):
            st.write(selected_resume.get("summary", "")[:2000])

        if st.button("Find top job matches", key="find_matches"):
            jobs = list(jobs_col.find())
            if not jobs:
                st.error("No jobs found in 'jobs'. Run Stage 3 script first.")
            else:
                resume_vec = selected_resume["vector"]

                matches = []
                all_job_skills = []

                for job in jobs:
                    job_vec = job.get("vector")
                    if not job_vec:
                        continue

                    job_skills = job.get("required_skills", [])
                    all_job_skills.extend(job_skills)

                    sem = cosine_sim(resume_vec, job_vec)
                    kw = keyword_overlap(resume_skills, job_skills)
                    rec = recency_weight(job.get("posted_at"))
                    score = final_score(sem, kw, rec, pop=0.5)

                    reason = explain_match(resume_skills, job_skills, score)

                    matches.append({
                        "Title": job.get("title", ""),
                        "Company": job.get("company", ""),
                        "Location": job.get("location", ""),
                        "Score": round(score, 3),
                        "SemanticSim": round(sem, 3),
                        "KeywordOverlap": round(kw, 3),
                        "Reason": reason,
                        "URL": job.get("url", "")
                    })

                if not matches:
                    st.warning("No matches could be computed (check vectors in jobs collection).")
                else:
                    matches_sorted = sorted(matches, key=lambda x: x["Score"], reverse=True)[:20]

                    st.markdown("## Top 20 Job Matches")
                    st.dataframe(matches_sorted)

                    # Skill gap heat map
                    st.markdown("## Skill Gap Heat Map")

                    if all_job_skills:
                        job_skill_counts = pd.Series(all_job_skills).value_counts()
                        resume_skill_set = set(resume_skills)
                        missing_skills = [s for s in job_skill_counts.index if s not in resume_skill_set]
                        missing_counts = job_skill_counts[missing_skills]

                        if not missing_counts.empty:
                            fig, ax = plt.subplots(figsize=(8, 5))
                            missing_counts.head(20).plot(kind="barh", ax=ax)
                            ax.set_xlabel("Frequency in job postings")
                            ax.set_ylabel("Skill")
                            ax.invert_yaxis()
                            st.pyplot(fig)
                        else:
                            st.info("No missing skills detected based on your KNOWN_SKILLS list.")
                    else:
                        st.info("No job skills available to build heat map.")

                    # Recommended courses
                    st.markdown("## Recommended Courses (static demo)")

                    if all_job_skills:
                        resume_skill_set = set(resume_skills)
                        missing_skills = sorted(set(all_job_skills) - resume_skill_set)
                        rec_rows = course_recommendations(missing_skills)
                        if rec_rows:
                            st.table(pd.DataFrame(rec_rows))
                        else:
                            st.info("No course recommendations for current missing skills (update COURSE_MAP).")
                    else:
                        st.info("No skills found in jobs to recommend courses from.")
