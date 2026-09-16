# Student Stress Prediction Backend

FastAPI backend for the Student Stress Prediction System.

## Folder Structure

```text
backend/
  app/
    api/
      routes.py              # API endpoints
    core/
      config.py              # Paths and CORS settings
    schemas/
      prediction.py          # Request and response validation models
      chat.py                # Gemini assistant request/response models
    services/
      model_service.py       # Model loading, anxiety calculation, prediction
      gemini_service.py      # Safe Gemini-powered result assistant
    main.py                  # FastAPI app setup
  models/
    decision_tree_model.pkl  # Add your trained model here
    scaler.pkl               # Add your saved scaler here
    label_encoder.pkl        # Add your saved label encoder here
  main.py                    # Uvicorn entry point
  requirements.txt
```

## Run

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

## Gemini Assistant

Set your Gemini API key on the backend only. Do not expose it in the frontend.

Create `backend/.env`:

```env
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3.8-flash
```

You can also set variables directly in PowerShell for a temporary session:

```powershell
$env:GEMINI_API_KEY="your_api_key_here"
$env:GEMINI_MODEL="gemini-3.8-flash"
```

If no Gemini key is configured, the `/chat` endpoint returns a safe local fallback response so the research prototype remains usable.

## CORS for Deployment

Set `CORS_ORIGINS` on Render to the exact frontend URL that calls the backend:

```env
CORS_ORIGINS=http://localhost:3000,https://student-stress-level-frontend-9vf6oj8ms-sewmini-s-projects.vercel.app
```

For Vercel preview URLs, you can also set a regex:

```env
CORS_ORIGIN_REGEX=https://.*\.vercel\.app
```

After changing Render environment variables, redeploy or restart the backend service.
