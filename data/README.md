# Dataset Directory

This directory contains the credit card transaction dataset used for fraud detection.

## Dataset Files

- `creditcard.csv`: The main dataset file (not included in repository)
  - Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/code/realhamzanet/credit-card-fraud-detection)
  - Size: 284,807 transactions
  - Features: 30 columns (Time, V1-V28, Amount, Class)

## File Structure

The `creditcard.csv` file contains the following columns:

1. `Time`: Seconds elapsed between each transaction and the first transaction
2. `V1` to `V28`: Principal components obtained with PCA transformation
3. `Amount`: Transaction amount
4. `Class`: Target variable
   - 0: Normal transaction
   - 1: Fraudulent transaction

## Dataset Statistics

- Total transactions: 284,807
- Normal transactions: 284,315 (99.828%)
- Fraudulent transactions: 492 (0.172%)
- Time period: 2 days

## Generated Files

The following files are generated during model training/testing:

- `new_transactions.csv`: Small synthetic dataset for quick testing
- `synthetic_transactions_large.csv`: Larger synthetic dataset for extensive testing

## Note

The main dataset file (`creditcard.csv`) is not included in the repository due to size and licensing restrictions. Please download it from Kaggle and place it in this directory before running the training script.
