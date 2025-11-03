"""
Fine-tune the fraud detection model with improved hyperparameters.
Trains on realistic banking transaction data with comprehensive evaluation.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    precision_recall_curve,
    f1_score,
    accuracy_score
)
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_dataset(data_path):
    """Load and validate the dataset."""
    print(f"Loading dataset from: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"ERROR: Dataset not found at {data_path}")
        print("Available data files:")
        data_dir = os.path.dirname(data_path)
        if os.path.exists(data_dir):
            for f in os.listdir(data_dir):
                if f.endswith('.csv'):
                    print(f"  - {f}")
        sys.exit(1)
    
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} transactions")
    
    # Validate columns
    required_features = ['Amount'] + [f'V{i}' for i in range(1, 29)]
    missing = [col for col in required_features if col not in df.columns]
    
    if missing:
        print(f"WARNING: Missing columns: {missing}")
        print(f"Available columns: {df.columns.tolist()}")
    
    if 'Class' not in df.columns:
        print("ERROR: 'Class' column not found. Cannot train without labels.")
        sys.exit(1)
    
    return df

def prepare_data(df):
    """Separate features and labels, handle missing values."""
    print("\nPreparing data...")
    
    # Drop non-feature columns
    drop_cols = ['TransactionID', 'Time', 'Timestamp', 'ID', 'index']
    df_clean = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
    
    # Separate features and target
    if 'Class' not in df_clean.columns:
        raise ValueError("Class column not found")
    
    y = df_clean['Class'].values
    X = df_clean.drop(columns=['Class'])
    
    # Ensure we have Amount + V1-V28
    expected_features = ['Amount'] + [f'V{i}' for i in range(1, 29)]
    X = X.reindex(columns=expected_features, fill_value=0)
    
    # Fill any remaining missing values
    X = X.fillna(X.mean()).fillna(0)
    
    print(f"✓ Features: {X.shape[1]} columns")
    print(f"✓ Labels: {len(y)} samples")
    print(f"  - Class 0 (Normal): {(y == 0).sum()} ({(y == 0).mean()*100:.2f}%)")
    print(f"  - Class 1 (Fraud):  {(y == 1).sum()} ({(y == 1).mean()*100:.2f}%)")
    
    return X, y

def train_model(X_train, y_train, X_test, y_test):
    """Train RandomForestClassifier with fine-tuned hyperparameters."""
    print("\n" + "="*70)
    print("TRAINING FINE-TUNED MODEL")
    print("="*70)
    
    # Initialize StandardScaler
    print("\nApplying StandardScaler normalization...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("✓ Features normalized (mean=0, std=1)")
    
    # Initialize model with improved hyperparameters
    print("\nInitializing RandomForestClassifier...")
    print("  n_estimators: 300 (increased from 100)")
    print("  max_depth: 10")
    print("  min_samples_split: 10")
    print("  min_samples_leaf: 4")
    print("  class_weight: balanced")
    print("  random_state: 42")
    
    model = RandomForestClassifier(
        n_estimators=300,          # More trees for better accuracy
        max_depth=10,              # Prevent overfitting
        min_samples_split=10,      # Minimum samples to split
        min_samples_leaf=4,        # Minimum samples per leaf
        class_weight='balanced',   # Handle imbalanced classes
        random_state=42,
        n_jobs=-1,                 # Use all CPU cores
        verbose=0
    )
    
    # Train the model
    print("\nTraining model...")
    model.fit(X_train_scaled, y_train)
    print("✓ Model training complete")
    
    return model, scaler, X_train_scaled, X_test_scaled

def evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test):
    """Comprehensive model evaluation."""
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Training set metrics
    print("\n📊 TRAINING SET PERFORMANCE:")
    print(f"  Accuracy:  {accuracy_score(y_train, y_train_pred)*100:.2f}%")
    print(f"  Precision: {classification_report(y_train, y_train_pred, output_dict=True)['1']['precision']*100:.2f}%")
    print(f"  Recall:    {classification_report(y_train, y_train_pred, output_dict=True)['1']['recall']*100:.2f}%")
    print(f"  F1-Score:  {f1_score(y_train, y_train_pred)*100:.2f}%")
    print(f"  ROC-AUC:   {roc_auc_score(y_train, y_train_proba):.4f}")
    
    # Test set metrics
    print("\n📊 TEST SET PERFORMANCE (IMPORTANT):")
    print(f"  Accuracy:  {accuracy_score(y_test, y_test_pred)*100:.2f}%")
    print(f"  Precision: {classification_report(y_test, y_test_pred, output_dict=True)['1']['precision']*100:.2f}%")
    print(f"  Recall:    {classification_report(y_test, y_test_pred, output_dict=True)['1']['recall']*100:.2f}%")
    print(f"  F1-Score:  {f1_score(y_test, y_test_pred)*100:.2f}%")
    print(f"  ROC-AUC:   {roc_auc_score(y_test, y_test_proba):.4f}")
    
    # Detailed classification report
    print("\n📋 DETAILED CLASSIFICATION REPORT (Test Set):")
    print(classification_report(y_test, y_test_pred, target_names=['Normal', 'Fraud'], digits=4))
    
    # Confusion matrix
    print("\n📈 CONFUSION MATRIX (Test Set):")
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"                   Predicted")
    print(f"                 Normal    Fraud")
    print(f"Actual Normal  {cm[0,0]:7d}  {cm[0,1]:7d}")
    print(f"       Fraud   {cm[1,0]:7d}  {cm[1,1]:7d}")
    
    # Calculate metrics from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  True Negatives:  {tn:5d}  (Correctly identified normal)")
    print(f"  False Positives: {fp:5d}  (False alarms)")
    print(f"  False Negatives: {fn:5d}  (Missed fraud)")
    print(f"  True Positives:  {tp:5d}  (Correctly detected fraud)")
    
    # Probability distribution analysis
    fraud_probs = y_test_proba[y_test == 1]
    normal_probs = y_test_proba[y_test == 0]
    
    print(f"\n📊 PROBABILITY DISTRIBUTION (Test Set):")
    print(f"  Fraud transactions:")
    print(f"    Min:  {fraud_probs.min():.4f}")
    print(f"    Mean: {fraud_probs.mean():.4f}")
    print(f"    Max:  {fraud_probs.max():.4f}")
    print(f"  Normal transactions:")
    print(f"    Min:  {normal_probs.min():.4f}")
    print(f"    Mean: {normal_probs.mean():.4f}")
    print(f"    Max:  {normal_probs.max():.4f}")
    
    # Find optimal threshold
    print("\n🎯 THRESHOLD OPTIMIZATION:")
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_test_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    print(f"  Optimal threshold: {optimal_threshold:.3f}")
    print(f"  Precision at optimal: {precisions[optimal_idx]:.4f}")
    print(f"  Recall at optimal:    {recalls[optimal_idx]:.4f}")
    print(f"  F1-Score at optimal:  {f1_scores[optimal_idx]:.4f}")
    
    # Feature importance
    print("\n🔍 TOP 10 MOST IMPORTANT FEATURES:")
    feature_names = ['Amount'] + [f'V{i}' for i in range(1, 29)]
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    
    for i, idx in enumerate(indices, 1):
        print(f"  {i:2d}. {feature_names[idx]:8s}: {importances[idx]:.4f}")
    
    return optimal_threshold

def save_model(model, scaler, model_dir='app/model'):
    """Save the fine-tuned model and scaler."""
    print("\n" + "="*70)
    print("SAVING MODEL")
    print("="*70)
    
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'model.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    model_size = os.path.getsize(model_path) / 1024
    scaler_size = os.path.getsize(scaler_path) / 1024
    
    print(f"✓ Model saved: {model_path} ({model_size:.1f} KB)")
    print(f"✓ Scaler saved: {scaler_path} ({scaler_size:.1f} KB)")
    
    # Save feature names
    feature_names = ['Amount'] + [f'V{i}' for i in range(1, 29)]
    feature_names_path = os.path.join(model_dir, 'feature_names.txt')
    with open(feature_names_path, 'w') as f:
        f.write('\n'.join(feature_names))
    print(f"✓ Feature names saved: {feature_names_path}")
    
    return model_path, scaler_path

def main():
    """Main fine-tuning pipeline."""
    print("="*70)
    print("FRAUD DETECTION MODEL FINE-TUNING")
    print("="*70)
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Try multiple dataset names
    possible_datasets = [
        'realistic_fraud_transactions.csv',
        'fraud_training_data.csv',
        'training_data.csv'
    ]
    
    data_path = None
    for dataset_name in possible_datasets:
        test_path = os.path.join(project_root, 'data', dataset_name)
        if os.path.exists(test_path):
            data_path = test_path
            break
    
    if not data_path:
        print("\nNo training dataset found. Generating synthetic data...")
        import subprocess
        gen_script = os.path.join(script_dir, 'generate_fraud_dataset.py')
        subprocess.run([sys.executable, gen_script], check=True)
        data_path = os.path.join(project_root, 'data', 'fraud_training_data.csv')
    
    # Load data
    df = load_dataset(data_path)
    
    # Prepare features and labels
    X, y = prepare_data(df)
    
    # Split data (80/20)
    print("\nSplitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    print(f"✓ Training set: {len(X_train)} samples")
    print(f"✓ Test set:     {len(X_test)} samples")
    
    # Train model
    model, scaler, X_train_scaled, X_test_scaled = train_model(
        X_train, y_train, X_test, y_test
    )
    
    # Evaluate
    optimal_threshold = evaluate_model(
        model, X_train_scaled, y_train, X_test_scaled, y_test
    )
    
    # Save model
    model_path, scaler_path = save_model(model, scaler)
    
    # Final summary
    print("\n" + "="*70)
    print("✅ FINE-TUNING COMPLETE!")
    print("="*70)
    print(f"\n📦 Model artifacts:")
    print(f"  - model.pkl:  {model_path}")
    print(f"  - scaler.pkl: {scaler_path}")
    print(f"\n🎯 Recommended settings:")
    print(f"  - Decision threshold: {optimal_threshold:.3f}")
    print(f"  - Model: RandomForestClassifier (300 trees)")
    print(f"  - Preprocessing: StandardScaler")
    print(f"\n🚀 Ready to use! Run the web app with: python app.py")
    print("="*70)

if __name__ == '__main__':
    main()
