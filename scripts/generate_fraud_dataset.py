"""
Generate synthetic credit card fraud dataset with realistic patterns.
Creates a CSV with Amount, V1-V28 features, and Class labels.
"""
import numpy as np
import pandas as pd
import os

np.random.seed(42)

def generate_fraud_dataset(n_samples=1000, fraud_ratio=0.15):
    """
    Generate synthetic fraud detection dataset.
    
    Fraud patterns:
    - High amounts (>5000)
    - Extreme V values (|V| > 2.5)
    - Unusual combinations
    """
    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud
    
    # Normal transactions
    normal_amounts = np.random.lognormal(mean=4.5, sigma=1.2, size=n_normal)
    normal_amounts = np.clip(normal_amounts, 1, 3000)  # Clip to reasonable range
    
    # V features for normal transactions (centered around 0)
    normal_v = np.random.randn(n_normal, 28) * 0.8  # Lower variance for normal
    
    # Fraudulent transactions
    # Mix of different fraud patterns
    n_high = n_fraud // 3
    n_low = n_fraud // 3
    n_medium = n_fraud - n_high - n_low
    
    fraud_amounts = np.concatenate([
        np.random.uniform(5000, 15000, n_high),     # Very high amounts
        np.random.uniform(0.01, 5, n_low),          # Very low amounts
        np.random.uniform(1000, 4000, n_medium)     # Medium-high
    ])
    
    # V features for fraud (more extreme values)
    fraud_v = np.random.randn(n_fraud, 28) * 2.0  # Higher variance
    # Add some extreme outliers
    extreme_mask = np.random.rand(n_fraud, 28) < 0.3
    fraud_v[extreme_mask] = np.random.choice([-1, 1], size=extreme_mask.sum()) * np.random.uniform(2.5, 5, size=extreme_mask.sum())
    
    # Combine
    amounts = np.concatenate([normal_amounts, fraud_amounts])
    v_features = np.vstack([normal_v, fraud_v])
    classes = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)])
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    amounts = amounts[indices]
    v_features = v_features[indices]
    classes = classes[indices]
    
    # Create DataFrame
    data = {'Amount': amounts}
    for i in range(28):
        data[f'V{i+1}'] = v_features[:, i]
    data['Class'] = classes.astype(int)
    
    df = pd.DataFrame(data)
    return df

if __name__ == '__main__':
    # Generate training dataset
    print("Generating synthetic fraud dataset...")
    df_train = generate_fraud_dataset(n_samples=5000, fraud_ratio=0.15)
    
    # Save to data folder
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    train_path = os.path.join(data_dir, 'fraud_training_data.csv')
    df_train.to_csv(train_path, index=False)
    print(f"✓ Saved training data: {train_path}")
    print(f"  Total: {len(df_train)}, Fraud: {df_train['Class'].sum()}, Normal: {len(df_train) - df_train['Class'].sum()}")
    print(f"  Fraud ratio: {df_train['Class'].mean()*100:.2f}%")
    
    # Generate smaller test dataset
    df_test = generate_fraud_dataset(n_samples=200, fraud_ratio=0.20)
    test_path = os.path.join(data_dir, 'test_transactions.csv')
    # Remove Class column for test data (users will upload without labels)
    df_test_no_labels = df_test.drop(columns=['Class'])
    df_test_no_labels.to_csv(test_path, index=False)
    print(f"✓ Saved test data (no labels): {test_path}")
    
    # Save test with labels for validation
    test_labeled_path = os.path.join(data_dir, 'test_transactions_labeled.csv')
    df_test.to_csv(test_labeled_path, index=False)
    print(f"✓ Saved test data (with labels): {test_labeled_path}")
    print(f"  Total: {len(df_test)}, Expected fraud: {df_test['Class'].sum()}")
