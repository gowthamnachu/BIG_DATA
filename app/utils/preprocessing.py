import os
from typing import Tuple, Dict, List, Optional

import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Expected columns according to Kaggle Credit Card Fraud dataset
EXPECTED_COLUMNS: List[str] = (
    ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
)

OPTIONAL_LABEL = 'Class'  # If present, kept only for reference, not used for prediction

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model'))


def _load_preprocessor() -> Tuple[Optional[object], Optional[str]]:
    """Try to load a saved sklearn preprocessor/scaler from model dir.
    Looks for 'preprocessor.pkl' or 'scaler.pkl'. Returns (obj, path) or (None, None).
    """
    candidates = ['preprocessor.pkl', 'scaler.pkl']
    for name in candidates:
        path = os.path.join(MODEL_DIR, name)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            try:
                obj = joblib.load(path)
                return obj, path
            except Exception:
                continue
    return None, None


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in EXPECTED_COLUMNS + [OPTIONAL_LABEL]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def load_and_prepare(csv_path: str, standardize: bool = False) -> Tuple[pd.DataFrame, Dict]:
    """
    Read CSV, validate schema, handle missing values, and return features DataFrame.

    Returns
    -------
    df_features: pd.DataFrame
        DataFrame with only expected feature columns, in order.
    meta: dict
        Metadata such as original_row_count and label_presence.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError('Uploaded file not found on server.')

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f'Unable to read CSV: {e}')

    if df.empty:
        raise ValueError('CSV appears to be empty.')

    df = _coerce_numeric(df)
    _validate_columns(df)

    original_rows = len(df)

    # Handle missing values: simple imputation with median for features
    for c in EXPECTED_COLUMNS:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    # Keep features in correct order
    features = df[EXPECTED_COLUMNS].copy()

    # Clip extreme values of Amount to reduce outlier impact (RF robust but helps stability)
    if 'Amount' in features.columns:
        q99 = features['Amount'].quantile(0.99)
        features['Amount'] = features['Amount'].clip(lower=0, upper=q99)

    # Apply preprocessor if available; else optionally standardize
    preproc_info = {}
    preproc, preproc_path = _load_preprocessor()
    if preproc is not None and hasattr(preproc, 'transform'):
        try:
            transformed = preproc.transform(features)
            features = pd.DataFrame(transformed, columns=EXPECTED_COLUMNS, index=features.index)
            preproc_info = {'type': 'preprocessor_file', 'path': preproc_path}
        except Exception as e:
            # If preprocessor fails, proceed without it
            preproc_info = {'type': 'preprocessor_error', 'path': preproc_path, 'error': str(e)}
    elif standardize:
        try:
            scaler = StandardScaler()
            transformed = scaler.fit_transform(features)
            features = pd.DataFrame(transformed, columns=EXPECTED_COLUMNS, index=features.index)
            preproc_info = {'type': 'computed_standard_scaler'}
        except Exception as e:
            preproc_info = {'type': 'standardize_error', 'error': str(e)}

    meta = {
        'original_row_count': original_rows,
        'label_present': OPTIONAL_LABEL in df.columns,
        'preprocessor': preproc_info,
    }
    return features, meta
