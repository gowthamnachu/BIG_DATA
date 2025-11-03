import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(ROOT, 'app', 'model', 'fraud_model.pkl')

FEATURES = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']


def generate_synthetic(n_samples: int = 2000, fraud_ratio: float = 0.02):
    rng = np.random.default_rng(42)
    # Time: 0..172800 seconds (2 days)
    Time = rng.uniform(0, 172800, size=n_samples)
    # PCA-like features V1..V28: standard normal
    Vs = rng.normal(0, 1, size=(n_samples, 28))
    # Amount: log-normal-ish amounts
    Amount = np.round(np.exp(rng.normal(3.5, 1.0, size=n_samples)), 2)

    # Labels: rare frauds
    y = (rng.uniform(0, 1, size=n_samples) < fraud_ratio).astype(int)

    # Introduce slight signal: frauds tend to have larger Amount and certain V's skewed
    Amount = Amount * (1 + y * rng.uniform(1.5, 3.0, size=n_samples))
    Vs[:, 1] += y * rng.normal(2.0, 0.5, size=n_samples)
    Vs[:, 14] -= y * rng.normal(2.5, 0.7, size=n_samples)

    # Build X matrix in FEATURE order
    X = np.column_stack([Time, Vs, Amount])
    return X, y


def main():
    # Generate synthetic training data matching the expected schema
    X, y = generate_synthetic()

    clf = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1,
    )
    clf.fit(X, y)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    size = os.path.getsize(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH} ({size} bytes)")


if __name__ == '__main__':
    main()
