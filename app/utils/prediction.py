import os
import glob
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model'))
DEFAULT_SKLEARN_MODEL = os.path.join(MODEL_DIR, 'fraud_model.pkl')


class TorchModelWrapper:
    def __init__(self, torch_model):
        import torch  # type: ignore
        self.torch = torch
        self.model = torch_model.eval()
        self.fd_model_info = {'type': 'torch', 'path': None}

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # Convert to float32 tensor
        with self.torch.no_grad():
            x_np = X.to_numpy(dtype=np.float32)
            x_t = self.torch.from_numpy(x_np)
            out = self.model(x_t)
            # Handle various output shapes
            if out.dim() == 1:
                # shape: (N,) logits or probs
                probs1 = self.torch.sigmoid(out).cpu().numpy()
            elif out.dim() == 2 and out.shape[1] == 1:
                # shape: (N,1)
                probs1 = self.torch.sigmoid(out.squeeze(1)).cpu().numpy()
            elif out.dim() == 2 and out.shape[1] == 2:
                # shape: (N,2) logits -> softmax
                probs1 = self.torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            else:
                raise ValueError(f"Unsupported torch model output shape: {tuple(out.shape)}")

        probs1 = probs1.astype(np.float64)
        probs0 = 1.0 - probs1
        return np.stack([probs0, probs1], axis=1)


def _try_load_sklearn(path: str):
    try:
        model = joblib.load(path)
        try:
            setattr(model, 'fd_model_info', {'type': 'sklearn', 'path': path})
        except Exception:
            pass
        return model
    except Exception as e:
        raise RuntimeError(f'Failed to load sklearn model from {path}: {e}')


def _try_load_torch(pt_path: str):
    try:
        import torch  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "PyTorch is not installed. Install torch to use a .pt model. Error: " + str(e)
        )

    try:
        model = torch.load(pt_path, map_location='cpu')
    except Exception as e:
        raise RuntimeError(f'Failed to load torch model from {pt_path}: {e}')
    wrapper = TorchModelWrapper(model)
    wrapper.fd_model_info['path'] = pt_path
    return wrapper


def load_model():
    """
    Load a pre-trained model.
    Preference order (auto): .pt (PyTorch) > .pkl (scikit-learn)
    Override with env FRAUD_MODEL_TYPE = 'pt' | 'pkl' | 'auto'
    """
    prefer = os.environ.get('FRAUD_MODEL_TYPE', 'auto').lower()

    def find_pt():
        pts = sorted(
            glob.glob(os.path.join(MODEL_DIR, '*.pt')),
            key=lambda p: os.path.getsize(p) if os.path.exists(p) else 0,
            reverse=True,
        )
        for pt in pts:
            if os.path.getsize(pt) > 0:
                return pt
        return None

    chosen = None
    if prefer == 'pt':
        chosen = ('pt', find_pt())
    elif prefer == 'pkl':
        if os.path.exists(DEFAULT_SKLEARN_MODEL) and os.path.getsize(DEFAULT_SKLEARN_MODEL) > 0:
            chosen = ('pkl', DEFAULT_SKLEARN_MODEL)
    else:  # auto
        pt_path = find_pt()
        if pt_path:
            chosen = ('pt', pt_path)
        elif os.path.exists(DEFAULT_SKLEARN_MODEL) and os.path.getsize(DEFAULT_SKLEARN_MODEL) > 0:
            chosen = ('pkl', DEFAULT_SKLEARN_MODEL)

    if not chosen or not chosen[1]:
        raise FileNotFoundError(
            'No valid pre-trained model found. Place a PyTorch .pt file in app/model/ '
            'or a scikit-learn .pkl at app/model/fraud_model.pkl. See README for instructions.'
        )

    kind, path = chosen
    print(f"[model] Loading {kind.upper()} model from {path}")
    if kind == 'pt':
        return _try_load_torch(path)
    else:
        return _try_load_sklearn(path)


def predict_dataframe(model, df_features: pd.DataFrame, meta: Dict, threshold: Optional[float] = None) -> Tuple[pd.DataFrame, Dict]:
    if not hasattr(model, 'predict_proba'):
        raise ValueError('Loaded model does not support predict_proba.')

    # Predict probabilities for the positive class (fraud = 1)
    proba = model.predict_proba(df_features)[:, 1]
    if threshold is None:
        threshold = float(os.environ.get('FRAUD_THRESHOLD', '0.1'))

    out = df_features.copy()
    out['fraud_probability'] = proba
    out['predicted_label'] = (proba >= threshold).astype(int)

    fraud_count = int((out['predicted_label'] == 1).sum())
    nonfraud_count = int((out['predicted_label'] == 0).sum())

    summary = {
        'fraud_count': fraud_count,
        'nonfraud_count': nonfraud_count,
        'original_row_count': meta.get('original_row_count', len(out)),
        'fraud_percentage': (fraud_count / max(1, len(out))) * 100,
        'threshold': threshold,
        'min_prob': float(np.min(proba)) if len(proba) else 0.0,
        'mean_prob': float(np.mean(proba)) if len(proba) else 0.0,
        'max_prob': float(np.max(proba)) if len(proba) else 0.0,
    }

    return out, summary
