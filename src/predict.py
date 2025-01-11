import torch
import pandas as pd
import numpy as np
from fraud_detection import FraudDetector, check_gpu

def load_model(model_path, input_size, device):
    """Load the trained model."""
    model = FraudDetector(input_size).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model

def predict_transaction(model, scaler, transaction_data, device):
    """Make prediction for a single transaction or batch of transactions."""
    # Convert to numpy array if it's a DataFrame
    if isinstance(transaction_data, pd.DataFrame):
        transaction_data = transaction_data.values
    
    # Scale the data using the same scaler used during training
    scaled_data = scaler.transform(transaction_data)
    
    # Convert to PyTorch tensor
    tensor_data = torch.FloatTensor(scaled_data).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(tensor_data).squeeze()
        probabilities = outputs.cpu().numpy()
        predictions = (outputs > 0.5).float().cpu().numpy()
    
    return predictions, probabilities

def format_results(transaction_data, predictions, probabilities):
    """Format the prediction results."""
    results = pd.DataFrame(transaction_data)
    results['Fraud_Prediction'] = predictions
    results['Fraud_Probability'] = probabilities
    results['Risk_Level'] = pd.cut(results['Fraud_Probability'], 
                                 bins=[0, 0.2, 0.4, 0.6, 0.8, 1], 
                                 labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    return results

def main():
    # Set up device
    device = check_gpu()
    
    # Load the trained model
    model_path = 'models/best_model.pth'
    input_size = 30  # Number of features in the credit card dataset
    model = load_model(model_path, input_size, device)
    
    # Load the scaler
    import joblib
    scaler = joblib.load('models/scaler.save')
    
    # Example: Load new transactions to predict
    # You can modify this part to load your actual transaction data
    new_transactions = pd.read_csv('data/new_transactions.csv')  # Replace with your data file
    
    # Make predictions
    predictions, probabilities = predict_transaction(model, scaler, new_transactions, device)
    
    # Format and display results
    results = format_results(new_transactions, predictions, probabilities)
    
    # Display summary
    print("\nPrediction Summary:")
    print("-" * 50)
    print(f"Total Transactions: {len(results)}")
    print(f"Flagged as Fraud: {predictions.sum()}")
    print("\nRisk Level Distribution:")
    print(results['Risk_Level'].value_counts().sort_index())
    
    # Save results
    results.to_csv('prediction_results_new.csv', index=False)
    print("\nDetailed results saved to 'prediction_results_new.csv'")

if __name__ == "__main__":
    main() 