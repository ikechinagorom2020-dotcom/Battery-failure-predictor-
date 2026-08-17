# Battery Failure Predictor

FastAPI service that predicts EV battery failure risk from telemetry
features, using the PyTorch model trained in the accompanying notebook.

## Project structure

```
battery-failure-predictor/
├── app/
│   ├── main.py          # FastAPI app (routes: /, /health, /features, /predict)
│   ├── model.py          # Model architecture (must match training)
│   ├── inference.py       # Preprocessing + prediction pipeline
│   └── static/
│       └── index.html    # Simple web form for manual testing
├── artifacts/             # Trained model + preprocessing objects (see below)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .gitignore
```

## 1. Add your trained artifacts

Run Cells 21-23 in your training notebook to produce
`deployment_artifacts.zip`. Unzip it and place these 6 files into the
`artifacts/` folder here:

- `best_model.pt`
- `scaler.pkl`
- `num_imputer.pkl`
- `cat_imputer.pkl`
- `label_encoders.pkl`
- `feature_config.json`

## 2. Run locally with Docker

```bash
docker compose up --build
```

Then open **http://localhost:8000** in your browser for the web form,
or use the API directly:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": {"battery_capacity_kwh": 75.3, "odometer_km": 42000, "vehicle_brand": "Tesla", ...}}'
```

## 3. Run locally without Docker (optional)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## API Reference

| Method | Path        | Description                                      |
|--------|-------------|---------------------------------------------------|
| GET    | `/`         | Web interface                                     |
| GET    | `/health`   | Health check                                      |
| GET    | `/features` | Returns expected feature names + category options |
| POST   | `/predict`  | Body: `{"data": {...feature values...}}`          |

`/predict` response:

```json
{
  "failure_probability": 0.9236,
  "prediction": 1,
  "label": "Failure",
  "threshold_used": 0.863
}
```

## Notes

- Missing fields in a `/predict` request are imputed the same way as
  during training (median for numeric, most-frequent for categorical).
- Unknown categorical values fall back to a known category rather than
  raising an error, keeping the API robust to unexpected input.
- The decision threshold (`threshold_used`) comes from the tuned value
  saved in `feature_config.json`, not a hardcoded 0.5.
