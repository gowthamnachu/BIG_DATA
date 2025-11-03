"""
Train RandomForestClassifier on synthetic fraud data.
Saves model.pkl and scaler.pkl for production use.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

def train_fraud_model():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, 'data', 'fraud_training_data.csv')
    model_dir = os.path.join(project_root, 'app', 'model')
    os.makedirs(model_dir, exist_ok=True)
    
    # Load data
    print(f"Loading training data from: {data_path}")
    if not os.path.exists(data_path):
        print(f"ERROR: Training data not found at {data_path}")
        print("Please run: python scripts/generate_fraud_dataset.py first")
        sys.exit(1)
    
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} transactions")
    print(f"  Fraud: {df['Class'].sum()}, Normal: {len(df) - df['Class'].sum()}")
    print(f"  Fraud ratio: {df['Class'].mean()*100:.2f}%")
    
    # Prepare features and target
    feature_cols = ['Amount'] + [f'V{i}' for i in range(1, 29)]
    X = df[feature_cols].copy()
    y = df['Class'].copy()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n✓ Split data: {len(X_train)} train, {len(X_test)} test")
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train RandomForest
    print("\nTraining RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight='balanced',  # Handle imbalanced data
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    print("✓ Model trained")
    
    # Evaluate
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                 Predicted")
    print(f"               Normal  Fraud")
    print(f"Actual Normal  {cm[0,0]:6d}  {cm[0,1]:5d}")
    print(f"       Fraud   {cm[1,0]:6d}  {cm[1,1]:5d}")
    
    print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    
    # Analyze probability distribution
    fraud_probs = y_proba[y_test == 1]
    normal_probs = y_proba[y_test == 0]
    print(f"\nProbability Distribution:")
    print(f"  Fraud transactions   - min: {fraud_probs.min():.3f}, mean: {fraud_probs.mean():.3f}, max: {fraud_probs.max():.3f}")
    print(f"  Normal transactions  - min: {normal_probs.min():.3f}, mean: {normal_probs.mean():.3f}, max: {normal_probs.max():.3f}")
    
    # Recommended threshold
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_threshold = 0.5
    best_f1 = 0
    for thresh in thresholds:
        y_pred_thresh = (y_proba >= thresh).astype(int)
        from sklearn.metrics import f1_score
        f1 = f1_score(y_test, y_pred_thresh)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    print(f"\n✓ Recommended threshold: {best_threshold:.2f} (F1={best_f1:.3f})")
    
    # Save model and scaler
    model_path = os.path.join(model_dir, 'model.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\n{'='*60}")
    print(f"✓ Model saved: {model_path}")
    print(f"✓ Scaler saved: {scaler_path}")
    print(f"{'='*60}")
    
    # Save feature names for reference
    feature_names_path = os.path.join(model_dir, 'feature_names.txt')
    with open(feature_names_path, 'w') as f:
        f.write('\n'.join(feature_cols))
    print(f"✓ Feature names saved: {feature_names_path}")
    
    return model, scaler, best_threshold

if __name__ == '__main__':
    model, scaler, threshold = train_fraud_model()
    print(f"\n✅ Training complete! Use threshold={threshold:.2f} for predictions.")
