# astro-bot

This repository contains a FastAPI-based RAG app that uses ChromaDB for retrieval and (optionally) Gemini for generation.

Quick local run

1. Create & activate virtualenv:

```powershell
python -m venv .venv
& .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Start locally (development):

```powershell
uvicorn main:app --reload
```

3. Test endpoints (PowerShell examples):

```powershell
Invoke-RestMethod -Method Get -Uri 'http://127.0.0.1:8000/'
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/ingest' -ContentType 'application/json' -Body (@{ id='1'; text='Example text'; metadata=@{source='manual'} } | ConvertTo-Json)
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/query' -ContentType 'application/json' -Body (@{ query = 'astrology signs'; top_k = 3 } | ConvertTo-Json)
```

Deploy to Render

1. Create a new GitHub repo and push this project (see commands below).

2. Create a new Web Service on Render and connect it to your GitHub repo. For a Python FastAPI app use the default Build Command `pip install -r requirements.txt` and the Start Command:

```
web: gunicorn -k uvicorn.workers.UvicornWorker main:app --log-file -
```

3. Add environment variables in Render:
- `GEMINI_API_KEY` (optional)
- `GEMINI_MODEL` (optional, default `gemini-pro`)

Git / GitHub push commands

```powershell
# from repo root
git init
git add .
git commit -m "Initial commit: prepare for Render deployment"
# create repo on GitHub (use GitHub CLI or create via web)
git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```

If you want, I can create `render.yaml` with service specs, or update the Procfile/start command to your preference.
