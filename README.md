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
    services/
      model_service.py       # Model loading, anxiety calculation, prediction
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
