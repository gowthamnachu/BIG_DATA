import os
import pandas as pd
import numpy as np
import joblib


def preprocess_data(df: pd.DataFrame, scaler=None) -> tuple:
    """
    Preprocess uploaded CSV for fraud prediction.
    Returns: (processed_df, scaler_used)
    """
    # Drop irrelevant columns
    drop_cols = [col for col in df.columns if col.lower() in ['transactionid', 'timestamp', 'id', 'class']]
    df = df.drop(columns=drop_cols, errors='ignore')

    # Expected features: Amount + V1-V28
    expected_features = ['Amount'] + [f'V{i}' for i in range(1, 29)]
    
    # Fill missing values
    df = df.fillna(df.mean(numeric_only=True)).fillna(0)

    # Keep only numeric columns
    df = df.select_dtypes(include=['float64', 'int64', 'float32', 'int32'])
    
    # Align to expected features
    df = df.reindex(columns=expected_features, fill_value=0)
    
    # Apply scaling if scaler provided
    if scaler is not None:
        df_scaled = pd.DataFrame(
            scaler.transform(df),
            columns=df.columns,
            index=df.index
        )
        return df_scaled, scaler
    
    return df, None


def predict_with_model(df: pd.DataFrame):
    """
    Predict fraud using trained model.pkl and scaler.pkl.
    Returns summary with total, fraud_count, fraud_ratio, fraud_indices, and HTML table.
    """
    model_dir = os.path.join(os.path.dirname(__file__), 'model')
    model_path = os.path.join(model_dir, 'model.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    
    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    model = joblib.load(model_path)
    
    # Load scaler if available
    scaler = None
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    
    # Preprocess
    processed, scaler_used = preprocess_data(df.copy(), scaler=scaler)
    if processed.empty:
        raise ValueError('No numeric columns to predict on after preprocessing.')

    # Align features to model's expected columns
    if hasattr(model, 'feature_names_in_'):
        feature_cols = list(model.feature_names_in_)
        X = processed.reindex(columns=feature_cols, fill_value=0)
    else:
        X = processed

    # Predict classes
    preds = model.predict(X)
    
    # Get probabilities if available
    proba = None
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)[:, 1]  # Probability of fraud class

    # Build output
    out = df.copy()
    out['Predicted_Class'] = preds
    if proba is not None:
        out['Fraud_Probability'] = np.round(proba, 4)

    fraud_mask = out['Predicted_Class'] == 1
    fraud_indices = out.index[fraud_mask].tolist()
    total = len(out)
    fraud_count = int(fraud_mask.sum())
    fraud_ratio = round((fraud_count / total) * 100, 2) if total else 0.0

    # Generate summary table (top 20 rows with key columns)
    display_cols = []
    if 'Amount' in out.columns:
        display_cols.append('Amount')
    for col in [f'V{i}' for i in range(1, 6)]:  # Show first 5 V features
        if col in out.columns:
            display_cols.append(col)
    display_cols.extend(['Predicted_Class'])
    if 'Fraud_Probability' in out.columns:
        display_cols.append('Fraud_Probability')
    
    table_df = out[display_cols].head(20)
    table_html = table_df.to_html(classes='table table-striped table-sm', index=True, index_names=['Row'])

    # Top 10 suspicious transactions for bar chart
    # Sort by fraud probability (if available) and then by Amount
    bar_labels = []
    bar_values = []
    
    if 'Amount' in out.columns:
        if proba is not None and 'Fraud_Probability' in out.columns:
            # Sort by probability then amount
            top10 = out.nlargest(10, ['Fraud_Probability', 'Amount'])
        else:
            # Sort by predicted class (frauds first) then amount
            top10 = out.sort_values(['Predicted_Class', 'Amount'], ascending=[False, False]).head(10)
        
        bar_labels = [f"Txn {i+1}" for i in range(len(top10))]
        bar_values = top10['Amount'].round(2).tolist()

    results = {
        "total": total,
        "fraud_count": fraud_count,
        "fraud_ratio": fraud_ratio,
        "fraud_indices": [i + 1 for i in fraud_indices],  # 1-based index
        "table": table_html,
        "scaler_used": scaler_used is not None,
        "bar_labels": bar_labels,
        "bar_values": bar_values,
        "probabilities": {
            "min": float(np.min(proba)) if proba is not None else None,
            "mean": float(np.mean(proba)) if proba is not None else None,
            "max": float(np.max(proba)) if proba is not None else None,
        }
    }
    return results
