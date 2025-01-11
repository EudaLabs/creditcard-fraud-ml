# Credit Card Fraud Detection System

[sincerely, EudaLabs](https://github.com/eudalabs)

A deep learning-based credit card fraud detection system developed by EudaLabs. This system utilizes PyTorch and advanced neural network architectures to identify fraudulent transactions with high precision and recall.

## 🎯 Dataset

This project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/code/realhamzanet/credit-card-fraud-detection) from Kaggle. The dataset contains credit card transactions made in September 2013 by European cardholders, with the following characteristics:

- 284,807 transactions, of which 492 (0.172%) are fraudulent
- 30 features: 28 PCA-transformed features (V1-V28), 'Time', and 'Amount'
- Highly imbalanced dataset reflecting real-world fraud detection scenarios
- All features are numerical and PCA-transformed for confidentiality

## 🎯 Features

- Real-time fraud detection using GPU-accelerated deep learning
- High precision in fraud detection (94% precision on test data)
- Balanced handling of highly imbalanced dataset (0.172% fraud cases)
- Support for both batch and real-time predictions
- Comprehensive evaluation metrics and visualization
- Synthetic data generation for testing

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- CUDA-capable GPU (NVIDIA GeForce RTX 3060 or better)
- CUDA 12.1+ and cuDNN
- Kaggle account (to download the dataset)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/eudalabs/creditcard-fraud-ml.git
cd creditcard-fraud-ml
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

3. Install dependencies:

```bash
uv pip install -r requirements.txt
```

4. Download the dataset:
   - Visit [Kaggle Dataset](https://www.kaggle.com/code/realhamzanet/credit-card-fraud-detection)
   - Download `creditcard.csv`
   - Place it in the `data/` directory

## 💡 Usage

### Training the Model

```bash
python src/fraud_detection.py
```

This will:

- Load and preprocess the credit card transaction dataset
- Train a deep neural network using GPU acceleration
- Save the best model based on validation F1-score
- Generate performance metrics and visualizations

### Making Predictions

```bash
# Generate synthetic data for testing
python src/generate_synthetic_data.py

# Run predictions on new transactions
python src/predict.py
```

## 📊 Model Performance

- F1 Score: 0.837
- Precision: 0.94 (Fraud Detection)
- Recall: 0.77 (Fraud Detection)
- Area Under Precision-Recall Curve: 0.865

## 🏗️ Project Structure

```
creditcard-fraud-ml/
├── data/                  # Data directory (excluded from git)
│   └── creditcard.csv    # Dataset from Kaggle (not included)
├── models/               # Saved models directory
├── src/
│   ├── fraud_detection.py # Main training script
│   ├── predict.py         # Prediction script
│   └── generate_synthetic_data.py # Synthetic data generator
├── requirements.txt      # Project dependencies
└── README.md            # This file
```

## 📝 License

This project is proprietary and confidential. © 2024 EudaLabs. All rights reserved.

The dataset used in this project is sourced from Kaggle and is subject to Kaggle's terms of use.

## 👥 Contributors

- [byigitt](https://github.com/byigitt)

## 🙏 Acknowledgments

- Credit Card Fraud Detection dataset from [Kaggle](https://www.kaggle.com/code/realhamzanet/credit-card-fraud-detection)
- Original dataset creators: Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi
- The Machine Learning Group at ULB (Université Libre de Bruxelles) for the dataset
