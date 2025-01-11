import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_recall_curve, auc, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import os
import joblib

def check_gpu():
    print("\nChecking GPU availability...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA build available: {'+cu' in torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if hasattr(torch.version, 'cuda'):
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("CUDA version: Not found")
        
    if hasattr(torch.backends, 'cudnn'):
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
    
    # Try to get NVIDIA GPU info using system command
    try:
        import subprocess
        nvidia_smi = subprocess.check_output(['nvidia-smi']).decode('utf-8')
        print("\nNVIDIA System Management Interface output:")
        print(nvidia_smi)
    except:
        print("Could not get NVIDIA GPU information from system")
    
    if not torch.cuda.is_available():
        print("No GPU available, using CPU")
        return torch.device('cpu')
    
    try:
        # Force CUDA initialization
        torch.cuda.init()
        current_device = torch.cuda.current_device()
        
        print(f"\nGPU Information:")
        print(f"Device: {torch.cuda.get_device_name(current_device)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {current_device}")
        print(f"GPU Capability: {torch.cuda.get_device_capability(current_device)}")
        
        # Memory Information
        print(f"\nGPU Memory Usage:")
        print(f" - Total: {torch.cuda.get_device_properties(current_device).total_memory / 1024**2:.0f} MB")
        print(f" - Allocated: {torch.cuda.memory_allocated(current_device) / 1024**2:.1f} MB")
        print(f" - Cached: {torch.cuda.memory_reserved(current_device) / 1024**2:.1f} MB")
        
        device = torch.device('cuda')
        
        # Test GPU with a more comprehensive operation
        print("\nTesting GPU...")
        try:
            x = torch.randn(1000, 1000, device=device)
            y = torch.matmul(x, x.t())
            del x, y  # Clean up test tensors
            torch.cuda.empty_cache()  # Clear GPU cache
            print("✓ GPU test passed successfully")
        except Exception as e:
            print(f"! GPU computation test failed: {e}")
            raise
            
        return device
        
    except Exception as e:
        print(f"\n! GPU initialization failed: {e}")
        print("Falling back to CPU")
        return torch.device('cpu')

class FraudDataset(Dataset):
    def __init__(self, X, y):
        # Convert pandas Series/DataFrame to numpy arrays if needed
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values
            
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class FraudDetector(nn.Module):
    def __init__(self, input_size):
        super(FraudDetector, self).__init__()
        
        self.batch_norm1 = nn.BatchNorm1d(input_size)
        
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.layer4 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.output = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.output(x)

def load_and_preprocess_data():
    print("Loading and preprocessing data...")
    # Load the dataset
    df = pd.read_csv('data/creditcard.csv')
    
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.save')
    
    # Calculate class weights for balanced training
    n_samples = len(y)
    n_classes = len(y.unique())
    class_counts = y.value_counts()
    pos_weight = class_counts[0] / class_counts[1]  # Weight for positive class
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create datasets
    train_dataset = FraudDataset(X_train, y_train)
    test_dataset = FraudDataset(X_test, y_test)
    
    # Create dataloaders with larger batch size for GPU
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    return train_loader, test_loader, pos_weight, X_train.shape[1], (X_test, y_test), scaler

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_model(train_loader, test_loader, pos_weight, input_size, device, model_save_path='models'):
    print(f"Training on: {device}")
    model = FraudDetector(input_size).to(device)
    
    # Use weighted BCELoss for better handling of class imbalance
    criterion = nn.BCELoss(weight=torch.tensor([pos_weight]).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.1, 
        patience=3,
        min_lr=1e-6
    )
    
    # Early stopping based on validation F1
    early_stopping = EarlyStopping(patience=5)
    
    # Training loop
    epochs = 50
    best_val_f1 = 0.0
    
    # Create model directory if it doesn't exist
    os.makedirs(model_save_path, exist_ok=True)
    
    print("\nTraining Progress:")
    print("Epoch  Train Loss  Train F1  Val F1    LR")
    print("-" * 50)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        train_y_true = []
        train_y_pred = []
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
            # Collect predictions for F1 score
            predictions = (outputs > 0.5).float()
            train_y_true.extend(batch_y.cpu().numpy())
            train_y_pred.extend(predictions.cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        train_f1 = f1_score(train_y_true, train_y_pred)
        
        # Validation phase
        model.eval()
        val_y_true = []
        val_y_pred = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:  # Using test_loader as validation
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X).squeeze()
                predictions = (outputs > 0.5).float()
                val_y_true.extend(batch_y.cpu().numpy())
                val_y_pred.extend(predictions.cpu().numpy())
        
        val_f1 = f1_score(val_y_true, val_y_pred)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"{epoch+1:5d}  {avg_loss:.6f}  {train_f1:.4f}  {val_f1:.4f}  {current_lr:.2e}")
        
        # Learning rate scheduling based on validation F1
        scheduler.step(-val_f1)  # Negative because scheduler is in 'min' mode
        
        # Early stopping check based on validation F1
        early_stopping(-val_f1)  # Negative because early stopping expects loss-like metric
        
        # Save best model based on validation F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            # Save only the model state dict
            torch.save(model.state_dict(), os.path.join(model_save_path, 'best_model.pth'))
            # Save metadata separately
            metadata = {
                'epoch': epoch,
                'train_f1': train_f1,
                'val_f1': val_f1,
            }
            torch.save(metadata, os.path.join(model_save_path, 'metadata.pth'))
        
        if early_stopping.early_stop:
            print("\nEarly stopping triggered")
            break
    
    print("\nTraining completed!")
    print(f"Best Validation F1 Score: {best_val_f1:.4f}")
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(model_save_path, 'best_model.pth'), weights_only=True))
    return model

def evaluate_model(model, test_loader, test_data, device):
    model.eval()
    X_test, y_test = test_data
    
    # Make predictions
    y_pred = []
    y_pred_proba = []
    
    print("\nMaking predictions...")
    with torch.no_grad():
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X).squeeze()
            predictions = (outputs > 0.5).float()
            y_pred.extend(predictions.cpu().numpy())
            y_pred_proba.extend(outputs.cpu().numpy())
    
    y_pred = np.array(y_pred)
    y_pred_proba = np.array(y_pred_proba)
    
    # Calculate metrics
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    f1 = f1_score(y_test, y_pred)
    
    # Print detailed evaluation
    print("\nModel Performance Metrics:")
    print("-" * 30)
    print(f"F1 Score: {f1:.4f}")
    print(f"PR-AUC Score: {pr_auc:.4f}")
    
    # Print classification report with zero_division parameter
    print("\nDetailed Classification Report:")
    print("-" * 30)
    print(classification_report(y_test, y_pred, zero_division=1))
    
    # Plot precision-recall curve
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    
    # Add threshold points
    thresholds = [0.3, 0.5, 0.7]
    for threshold in thresholds:
        y_pred_th = (y_pred_proba >= threshold).astype(int)
        prec = precision_score(y_test, y_pred_th, zero_division=1)
        rec = recall_score(y_test, y_pred_th, zero_division=1)
        plt.plot(rec, prec, 'o', label=f'Threshold = {threshold}')
    
    plt.legend()
    plt.savefig('fraud_detection_pr_curve.png')
    plt.close()
    
    # Save predictions to CSV for further analysis
    results_df = pd.DataFrame({
        'True_Label': y_test,
        'Predicted_Probability': y_pred_proba,
        'Predicted_Label': y_pred
    })
    results_df.to_csv('prediction_results.csv', index=False)
    print("\nPrediction results saved to 'prediction_results.csv'")
    
    return pr_auc, f1

def load_trained_model(model_path, input_size, device):
    model = FraudDetector(input_size).to(device)
    try:
        # Try loading with weights_only first
        model.load_state_dict(torch.load(model_path, weights_only=True))
    except RuntimeError:
        # If that fails, try loading old format
        try:
            # First try with weights_only=True
            checkpoint = torch.load(model_path, weights_only=True)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        except Exception:
            # If that fails, try one last time without weights_only
            checkpoint = torch.load(model_path)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                raise RuntimeError("Could not load model: incompatible format")
    
    model.eval()  # Set to evaluation mode
    return model

def main():
    # Set device and check GPU
    device = check_gpu()
    
    # Load and preprocess data
    train_loader, test_loader, pos_weight, input_size, test_data, scaler = load_and_preprocess_data()
    
    # Train or load model
    model_path = 'models/best_model.pth'
    if os.path.exists(model_path):
        print("Loading existing model...")
        model = load_trained_model(model_path, input_size, device)
    else:
        print("Training new model...")
        model = train_model(train_loader, test_loader, pos_weight, input_size, device)
    
    print("Evaluating model...")
    pr_auc, f1 = evaluate_model(model, test_loader, test_data, device)
    print(f"\nArea Under Precision-Recall Curve: {pr_auc:.3f}")
    print(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main() 