import os
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SAMPLE = os.path.join(ROOT, 'app', 'static', 'sample', 'sample_transactions.csv')
MODEL_OUT = os.path.join(ROOT, 'app', 'model', 'model.pkl')
       
# Align with app/predict.py preprocessing
sys.path.append(ROOT)
from app.predict import preprocess_data

def main():
    df = pd.read_csv(SAMPLE)
    df_processed = preprocess_data(df.copy())
    if df_processed.empty:
        raise RuntimeError('No numeric columns after preprocessing')

# Labels: rows 5 and 9 (1-based) are fraud
    n = len(df_processed)
    y = [1 if (i+1) in (5, 9) else 0 for i in range(n)]

    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    clf.fit(df_processed, y)

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(clf, MODEL_OUT)
    print(f'Saved model to {MODEL_OUT}')

if __name__ == '__main__':
    main()
