# Fraud Detection using Big Data Analytics in Banking

A full-stack Flask web app that detects potentially fraudulent transactions from uploaded CSV files using a pre-trained RandomForest model. It can also load a PyTorch `.pt` model as a fallback if no `.pkl` is present.

## Features
- Upload a CSV (Kaggle Credit Card Fraud schema: `Time`, `V1..V28`, `Amount`).
- Robust preprocessing and validation.
- Inference via a pre-trained `RandomForestClassifier`.
- Interactive charts (Chart.js): Pie (Fraud vs Non-Fraud), Bar (Top 10 suspicious amounts).
- Download predictions as CSV.

## Project Structure
```
fraud_detection_web/
  app/
    static/
      css/
      js/
      sample/sample_transactions.csv
    templates/
      index.html
      upload.html
      result.html
    __init__.py
    routes.py
    model/
      fraud_model.pkl   # PLACE YOUR TRAINED MODEL HERE (or place a .pt torch model)
    utils/
      preprocessing.py
      prediction.py
  data/
    sample_transactions.csv
    uploads/
    predictions/
  app.py
  requirements.txt
  README.md
```

## Setup (Windows PowerShell)

```powershell
# 1) Create and activate a virtual environment (recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Install dependencies
pip install -r requirements.txt

# 3) Place the trained model
# Option A (preferred): copy your RandomForest model file to: app/model/fraud_model.pkl
# Option B: copy a PyTorch model file (*.pt) into app/model/

# 4) Run the app
python app.py
# App will be available at http://127.0.0.1:5000/
```

## CSV Format
- Required columns: `Time`, `V1..V28`, `Amount`.  `Class` is optional and ignored for prediction.
- The Kaggle dataset uses PCA-transformed features `V1..V28`.

Example header:
```
Time,V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,V21,V22,V23,V24,V25,V26,V27,V28,Amount,Class
```

## One-time Model Training
Use the Kaggle "Credit Card Fraud Detection" dataset. Train a single strong model (RandomForest), save it as `app/model/fraud_model.pkl`, and do not retrain on each request.

Example training snippet:
```python
# train_model.py (example)
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

CSV = 'path/to/creditcard.csv'  # Kaggle dataset

df = pd.read_csv(CSV)
features = ['Time'] + [f'V{i}' for i in range(1,29)] + ['Amount']
X = df[features]
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

clf = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    n_jobs=-1,
    class_weight='balanced',
    random_state=42,
)
clf.fit(X_train, y_train)

print(classification_report(y_test, clf.predict(X_test)))

joblib.dump(clf, 'app/model/fraud_model.pkl')
print('Saved model to app/model/fraud_model.pkl')
```

### Using a PyTorch `.pt` model
Place a `.pt` file in `app/model/`. The app will attempt to load it on CPU and infer fraud probabilities from model outputs. Input feature order must be:
`Time, V1, V2, ..., V28, Amount` (30 features). If the model outputs a single logit, we apply sigmoid; if it outputs 2 logits, we apply softmax.

You can control the decision threshold via an environment variable (default 0.3):

```powershell
$env:FRAUD_THRESHOLD = "0.3"; python app.py
```

## Big Data Handling (Optional)
- For very large CSVs, consider chunked reading with pandas or using PySpark. This demo uses pandas and is optimized for single-machine processing. You can extend `utils/preprocessing.py` to stream chunks, run batch predictions, and concatenate outputs.

## Notes
- Files are stored locally under `data/uploads` and `data/predictions`.
- If `app/model/fraud_model.pkl` is missing or empty, the upload page will show a friendly error.
 - Alternatively, place a `.pt` model in `app/model/` (ensure `torch` is installed).

## License
This project is for educational/demo purposes. Ensure you comply with Kaggle's terms when using the dataset.
