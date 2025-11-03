"""
Test the fraud detection pipeline with the generated test data.
"""
import os
import sys
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.predict import predict_with_model

def test_predictions():
    # Load test data (without labels)
    test_path = 'data/test_transactions.csv'
    test_labeled_path = 'data/test_transactions_labeled.csv'
    
    print("="*60)
    print("TESTING FRAUD DETECTION PREDICTIONS")
    print("="*60)
    
    # Load test data
    df_test = pd.read_csv(test_path)
    df_labeled = pd.read_csv(test_labeled_path)
    
    print(f"\nTest dataset: {len(df_test)} transactions")
    print(f"Actual fraud count: {df_labeled['Class'].sum()}")
    print(f"Actual fraud ratio: {df_labeled['Class'].mean()*100:.2f}%")
    
    # Run predictions
    print("\nRunning predictions...")
    results = predict_with_model(df_test)
    
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"Total Transactions: {results['total']}")
    print(f"Predicted Frauds: {results['fraud_count']}")
    print(f"Fraud Ratio: {results['fraud_ratio']}%")
    print(f"\nFraud Indices (first 20): {results['fraud_indices'][:20]}")
    
    probs = results['probabilities']
    if probs['min'] is not None:
        print(f"\nProbability Distribution:")
        print(f"  Min:  {probs['min']:.4f}")
        print(f"  Mean: {probs['mean']:.4f}")
        print(f"  Max:  {probs['max']:.4f}")
    
    # Compare with actual labels
    predicted_fraud_indices = set(results['fraud_indices'])
    actual_fraud_indices = set((df_labeled[df_labeled['Class'] == 1].index + 1).tolist())
    
    true_positives = len(predicted_fraud_indices & actual_fraud_indices)
    false_positives = len(predicted_fraud_indices - actual_fraud_indices)
    false_negatives = len(actual_fraud_indices - predicted_fraud_indices)
    true_negatives = results['total'] - true_positives - false_positives - false_negatives
    
    print("\n" + "="*60)
    print("ACCURACY METRICS")
    print("="*60)
    print(f"True Positives (correctly detected fraud):  {true_positives}")
    print(f"False Positives (false alarms):            {false_positives}")
    print(f"False Negatives (missed fraud):            {false_negatives}")
    print(f"True Negatives (correctly identified normal): {true_negatives}")
    
    accuracy = (true_positives + true_negatives) / results['total']
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nAccuracy:  {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1-Score:  {f1*100:.2f}%")
    
    # Check high-amount fraud detection
    print("\n" + "="*60)
    print("HIGH-AMOUNT FRAUD DETECTION CHECK")
    print("="*60)
    high_amount_threshold = 5000
    high_amount_txns = df_test[df_test['Amount'] > high_amount_threshold].index + 1
    high_amount_predicted = [idx for idx in high_amount_txns if idx in predicted_fraud_indices]
    
    print(f"Transactions with Amount > ${high_amount_threshold}: {len(high_amount_txns)}")
    print(f"Detected as fraud: {len(high_amount_predicted)}")
    if len(high_amount_txns) > 0:
        print(f"Detection rate: {len(high_amount_predicted)/len(high_amount_txns)*100:.2f}%")
    
    if len(high_amount_predicted) < len(high_amount_txns):
        missed = [idx for idx in high_amount_txns if idx not in predicted_fraud_indices]
        print(f"Missed high-amount transactions (rows): {missed[:5]}")
    
    print("\n✅ Test complete!")

if __name__ == '__main__':
    test_predictions()
