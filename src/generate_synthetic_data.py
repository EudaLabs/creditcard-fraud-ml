import pandas as pd
import numpy as np
import os

def generate_synthetic_transactions(n_samples=1000, fraud_ratio=0.002):
    """
    Generate synthetic credit card transactions.
    
    Args:
        n_samples: Number of transactions to generate
        fraud_ratio: Ratio of fraudulent transactions
    """
    # Load original data statistics
    original_data = pd.read_csv('data/creditcard.csv')
    
    # Calculate mean and std for each feature from original data
    feature_stats = {}
    for column in original_data.columns:
        if column != 'Class':
            feature_stats[column] = {
                'mean': original_data[column].mean(),
                'std': original_data[column].std()
            }
    
    # Generate synthetic data
    synthetic_data = {}
    
    # Time feature (simulating transactions over 24 hours)
    synthetic_data['Time'] = np.random.uniform(0, 24*3600, n_samples)
    
    # V1-V28 features (using normal distribution with similar statistics)
    for i in range(1, 29):
        column = f'V{i}'
        stats = feature_stats[column]
        synthetic_data[column] = np.random.normal(
            loc=stats['mean'],
            scale=stats['std'],
            size=n_samples
        )
    
    # Amount feature (using log-normal distribution to simulate realistic amounts)
    amount_mean = np.log(feature_stats['Amount']['mean'])
    amount_std = np.log(feature_stats['Amount']['std'])
    synthetic_data['Amount'] = np.random.lognormal(
        mean=amount_mean,
        sigma=amount_std,
        size=n_samples
    )
    
    # Create DataFrame
    df = pd.DataFrame(synthetic_data)
    
    # Generate fraud labels
    n_fraud = int(n_samples * fraud_ratio)
    fraud_indices = np.random.choice(n_samples, n_fraud, replace=False)
    labels = np.zeros(n_samples)
    labels[fraud_indices] = 1
    
    # For fraudulent transactions, modify the features to make them look suspicious
    for idx in fraud_indices:
        # Modify some key features for fraudulent transactions
        df.loc[idx, 'V1':'V28'] += np.random.normal(0, 2, 28)  # Add some noise
        df.loc[idx, 'Amount'] *= np.random.uniform(1.5, 3)  # Higher amounts
    
    return df

def main():
    print("Generating synthetic transaction data...")
    
    # Generate two sets of data:
    # 1. A small set for immediate testing
    # 2. A larger set for more extensive testing
    
    # Small test set
    test_transactions = generate_synthetic_transactions(n_samples=10, fraud_ratio=0.2)
    os.makedirs('data', exist_ok=True)
    test_transactions.to_csv('data/new_transactions.csv', index=False)
    print(f"Created test set with {len(test_transactions)} transactions")
    print(f"Including {test_transactions['Amount'].sum():.2f} total transaction amount")
    
    # Larger test set
    large_test = generate_synthetic_transactions(n_samples=1000, fraud_ratio=0.002)
    large_test.to_csv('data/synthetic_transactions_large.csv', index=False)
    print(f"\nCreated large test set with {len(large_test)} transactions")
    print(f"Including {large_test['Amount'].sum():.2f} total transaction amount")
    
    print("\nFiles saved:")
    print("- data/new_transactions.csv (small test set)")
    print("- data/synthetic_transactions_large.csv (large test set)")

if __name__ == "__main__":
    main() 