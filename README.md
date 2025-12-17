 🎯 Intelligent Resume–Job Matcher (AWS + MongoDB + Streamlit)

An end‑to‑end system that:

- Ingests resumes via a Streamlit UI → uploads to **AWS S3**
- Parses resumes serverlessly with **AWS Lambda** → stores chunks in **MongoDB Atlas**
- Fetches live jobs from a public **jobs API** (Adzuna)
- Uses **TF‑IDF + cosine similarity** (no paid LLMs) to match resumes to jobs
- Visualizes matches, skill gaps, and course recommendations in a single Streamlit app

---

## 🚀 Quick Demo (What You Can Do)

1. **Upload your resume** (PDF/DOCX) from the browser.
2. Wait a few seconds for AWS Lambda to process it.
3. Open the **Matches & Insights** tab:
   - Pick your resume.
   - See **top 20 job matches** with scores and human‑readable “reasons”.
   - Explore a **skill‑gap heat map** (missing skills vs job market).
   - See a simple **Recommended Courses** list for missing skills.

This repo is built to be **student‑friendly**: everything runs with free AWS + free libraries, and is easy to extend for your own experiments.

---

## 🧩 Architecture at a Glance

flowchart LR
A[Streamlit: Upload Resume] --> B[S3 Bucket: resume-upload-guvi]
B --> C[Lambda: resume-main-processor]
C --> D[MongoDB: resumes (chunks)]
D --> E[Local Script: build_resume_stage2.py]
E --> F[MongoDB: resumes_meta (summary + skills + vector)]

text
G[Local Script: build_jobs_stage3.py] --> H[Job API (Adzuna)]
H --> I[MongoDB: jobs_raw]
I --> J[MongoDB: jobs (clean + vector)]

F --> K[Streamlit: Matches & Insights]
J --> K
text

**Key ideas (no Bedrock / Claude):**

- Text extraction: **PyPDF2** in Lambda  
- Semantics: **TF‑IDF** + cosine similarity (scikit‑learn)  
- Skills: Simple keyword matching  
- Explanations: Rule‑based text (no LLM cost)

---

## 📂 Project Structure

.
├── app.py # Streamlit app (Upload + Matches)
├── build_resume_stage2.py # Offline Stage 2 (resume -> summary, skills, TF-IDF vector)
├── build_jobs_stage3.py # Offline Stage 3 (jobs API -> normalized jobs + vectors)
├── match_stage4.py # CLI-based matcher for debugging
├── requirements.txt # Python dependencies
├── .env # Local secrets (NOT committed)
├── .gitignore
└── lambda/ # (optional) Lambda source files

text

MongoDB **database**: `resume_db`

- `resumes`      → chunks of resume text from Lambda
- `resumes_meta` → one document per resume (summary, skills, vector)
- `jobs_raw`     → raw job API responses
- `jobs`         → cleaned job docs (skills + vector)

---

## 🛠 Prerequisites

- Python 3.9+
- AWS account (S3 + Lambda free tier)
- MongoDB Atlas free cluster
- Adzuna (or similar) job API key
- Git, pip, and a terminal

---

## ⚙️ 1. Local Setup

Clone and enter the repo:

git clone https://github.com/<your-username>/resume-matcher-aws.git
cd resume-matcher-aws

text

Create a virtual environment:

python -m venv .venv

Windows (Git Bash / PowerShell)
source .venv/Scripts/activate

macOS / Linux
source .venv/bin/activate

text

Install dependencies:

pip install -r requirements.txt

or, if missing:
pip install streamlit pymongo scikit-learn python-dotenv requests numpy pandas matplotlib boto3 scipy
text

---

## 🔑 2. Environment Variables (.env)

Create a file named `.env` in the project root:

MongoDB Atlas
MONGO_URI=mongodb+srv://<user>:<password>@cluster0.daaq9qs.mongodb.net/?retryWrites=true&w=majority

Jobs API (Adzuna example)
ADZUNA_APP_ID=your_app_id_here
ADZUNA_APP_KEY=your_app_key_here
ADZUNA_COUNTRY=in

text

- Replace `<user>` / `<password>` with your Atlas user credentials.
- Get `ADZUNA_APP_ID` and `ADZUNA_APP_KEY` from https://developer.adzuna.com (My Apps).

> `.env` is in `.gitignore` → it stays local and is **never** pushed to GitHub.

---

## ☁️ 3. AWS: S3 + Lambda (Ingestion)

### S3 Bucket

- Create a bucket: `resume-upload-guvi` (or change in `app.py`).
- Block public access.
- This is where Streamlit will upload files: `resumes/<uuid>_<filename>.pdf`.

### Lambda (`resume-main-processor`)

You deploy a Python Lambda that:

- Gets triggered with `{ "bucket": "...", "key": "resumes/.....pdf" }`.
- Downloads the PDF from S3.
- Extracts text using PyPDF2.
- Splits into chunks and inserts into MongoDB `resumes` collection.

(Your existing Lambda code from the project goes here; see `lambda_handler` in the repo.)

---

## 🧮 4. Stage 2 (Offline) – Build Resume Metadata

Run **after** at least one resume has been processed by Lambda into `resumes`.

python build_resume_stage2.py

text

This script:

- Reads all chunks per `resume_id` from `resumes`.
- Concatenates them to `full_text`.
- Extracts skills via a keyword list.
- Generates a simple summary (first N characters).
- Builds a **TF‑IDF vector** for each resume and saves it in `resumes_meta`.
- Saves the trained TF‑IDF vectorizer to `tfidf_vectorizer.pkl` for reuse.

You should see console output like:

- `Loaded X resumes from chunks`
- `Built TF-IDF matrix with shape: ...`
- `Inserted X resume-level documents into 'resumes_meta'`

---

## 🌐 5. Stage 3 – Fetch Jobs & Vectorize

Fetch jobs from Adzuna and store them:

python build_jobs_stage3.py

text

What this does:

1. Calls Adzuna jobs API (e.g., “data scientist” in `ADZUNA_COUNTRY`).
2. Stores raw responses in `jobs_raw`.
3. Normalizes into `jobs` (title, company, location, description, url, posted_at).
4. Extracts `required_skills` using the same keyword list as resumes.
5. Loads `tfidf_vectorizer.pkl` and computes `vector` for each job description.

Expected output:

- `Fetched X jobs`
- `Inserted X raw job documents into 'jobs_raw'`
- `Normalized X jobs`
- `Added TF-IDF vectors to jobs`
- `Inserted X documents into 'jobs'`

---

## 🧠 6. Stage 4 – Matching Logic (Concept)

For a chosen resume and each job:

- Compute **semantic similarity** (cosine of resume vs job vector).
- Compute **keyword overlap** (resume skills vs job required skills).
- Compute **recency_weight** based on job posting age.
- Combine into final score:

score = 0.55 * semantic_similarity
+ 0.25 * keyword_overlap
+ 0.10 * recency_weight
+ 0.10 * popularity_score # currently constant

text

You can test this with a CLI script:

python match_stage4.py

text

And see top matches printed in the terminal.

---

## 📊 7. Stage 5 – Streamlit UI (Upload + Matching)

Run the app:

python -m streamlit run app.py

text

### Tab 1 – Upload Resume

- Choose a PDF/DOCX.
- Click “Upload to S3”.
- The file is stored in S3 at `resumes/<uuid>_<filename>.pdf` with metadata.
- AWS Lambda will pick it up and write chunks to MongoDB.

### Tab 2 – Matches & Insights

- Select a processed resume from `resumes_meta`.
- See:
  - **Skills detected** in the resume.
  - (Optional) a **summary** of the resume.
- Click **“Find top job matches”**:
  - Top 20 jobs with:
    - Title, company, location
    - Score, semantic similarity, keyword overlap
    - Explanation: common skills + missing skills
    - Job URL
- View **skill-gap heat map**:
  - Frequencies of missing skills across job postings.
- View static **Recommended Courses**:
  - Hard-coded mapping from missing skills → Coursera course names.

---

## 🔄 8. Extending the Project

Some ideas to go further:

- **Real embeddings** when you get Bedrock access (Titan + Claude).
- Better **summarization** using open-source NLP models (e.g., transformers).
- Real **Coursera API** integration instead of static mapping.
- Feedback collection in Mongo (`likes`, `dislikes`) → adjust weights in `final_score`.
- Deploy Streamlit to:
  - EC2
  - Streamlit Community Cloud
  - Or a container platform (ECS / Kubernetes)

---

## 🔐 Security & Good Practices

- Do **not** commit `.env` or any secrets.
- Rotate API keys if they are ever exposed on GitHub.
- Use IAM roles for Lambda to access S3 (don’t hard-code AWS keys).

---

## 💡 Why This Project Is Useful

- Shows a realistic **cloud ML pipeline** without paid LLMs.
- Uses **serverless architecture** (Lambda + S3) plus **offline ML** (TF‑IDF, matching).
- Good for portfolios: combines **AWS**, **data processing**, **vector search concepts**, and **UI** in one project.

If you fork/clone this repo, change names/keys and customize the skills list and job queries to suit your own domain (e.g., data science, web dev, cloud, etc.).
