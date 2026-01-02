""" Module for training the churn prediction model with text using DistilBERT and XGBoost.
Training parameters are stored in 'params.yaml'.
Run in CLI example: 'python train.py'
"""

import os
import sys
import json
import yaml
import joblib
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import precision_recall_curve
import xgboost
from xgboost import XGBClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
from transformers import DistilBertTokenizer, DistilBertModel


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Load configuration parameters
with open("../model/params.yaml", "r") as params_file:
    params = yaml.safe_load(params_file)

# Define model directory from configuration
model_dir = params['model_dir']

class DistilBertTextClassifier(nn.Module):
    """Text classifier using DistilBERT for feature extraction"""
    def __init__(self, num_classes=1):
        super(DistilBertTextClassifier, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_ids, attention_mask):
        """Forward pass through the text model"""
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        output = self.classifier(pooled_output)
        return output

class TabularModel(nn.Module):
    """Neural network for tabular features"""
    def __init__(self, input_size):
        super(TabularModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass through tabular model"""
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class CombinedModel(nn.Module):
    """Fusion model combining text and tabular features"""
    def __init__(self, tabular_input_size, text_model, tabular_model):
        super(CombinedModel, self).__init__()
        self.text_model = text_model
        self.tabular_model = tabular_model
        self.fusion_layer = nn.Linear(2, 1)  # Fusion of text and tabular outputs

    def forward(self, tabular_data, input_ids, attention_mask):
        """Forward pass through the combined model"""
        text_output = torch.sigmoid(self.text_model(input_ids, attention_mask))
        tabular_output = torch.sigmoid(self.tabular_model(tabular_data))
        
        combined = torch.cat([text_output, tabular_output], dim=1)
        fused_output = self.fusion_layer(combined)
        
        return fused_output, text_output, tabular_output

def preprocess2(X_train, X_test):
    # Preprocess tabular data: handle categorical features, numerical features, and missing values
    
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    # Identify numerical and categorical columns
    numeric_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = X_train.select_dtypes(include=['object']).columns.tolist()
    
    # Handle categorical features with LabelEncoder
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        all_values = pd.concat([X_train[col], X_test[col]]).astype(str)
        le.fit(all_values)
        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        label_encoders[col] = le
    
    # Save column info
    all_columns = X_train.columns.tolist()  # 顺序很重要！
    
    # Scale ONLY numeric columns
    scaler = StandardScaler()
    X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
    X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])
    
    # Impute ALL columns
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    return (
        X_train_imputed, X_test_imputed, 
        label_encoders, scaler, imputer,
        numeric_columns, categorical_columns, all_columns
    )

def train_model(X, y, X_test, y_test, text_train, text_test):
    """Train the combined model using tabular and text features"""
    batch_size = params['batch_size']
    batch_size_test = params['batch_size_test']
    epochs = params['epochs']
    pos_weight = params['pos_weight']
    lr = params['lr']
    
    # Preprocess tabular data
    X_train_processed, X_test_processed, label_encoders, scaler, imputer, numeric_columns, categorical_columns, all_columns = preprocess2(pd.DataFrame(X), pd.DataFrame(X_test))

    # Save preprocessing objects for later use
    joblib.dump(label_encoders, Path(model_dir, 'label_encoders.joblib'))
    joblib.dump(scaler, Path(model_dir, 'scaler.joblib'))
    joblib.dump(imputer, Path(model_dir, 'imputer.joblib'))
    joblib.dump(numeric_columns, Path(model_dir, 'numeric_columns.joblib'))
    joblib.dump(categorical_columns, Path(model_dir, 'categorical_columns.joblib'))
    joblib.dump(all_columns, Path(model_dir, 'all_columns.joblib'))

    # Tokenize text data
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    def tokenize_text(texts):
        return tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
    
    train_encodings = tokenize_text(text_train)
    test_encodings = tokenize_text(text_test)
    
    # Convert to numpy arrays for tensor dataset
    X = np.array(X_train_processed, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    X_test = np.array(X_test_processed, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)
    
    # Prepare data loaders
    train_dataset = TensorDataset(
        Tensor(X),
        torch.tensor(y),
        train_encodings['input_ids'],
        train_encodings['attention_mask']
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    test_dataset = TensorDataset(
        Tensor(X_test),
        torch.tensor(y_test),
        test_encodings['input_ids'],
        test_encodings['attention_mask']
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False, num_workers=0)
    
    # Initialize models
    text_model = DistilBertTextClassifier()
    tabular_model = TabularModel(X.shape[1])
    combined_model = CombinedModel(X.shape[1], text_model, tabular_model)
    
    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(pos_weight, dtype=torch.float))
    optimizer = optim.Adam(combined_model.parameters(), lr=lr)
    
    # Training loop
    train_scores = []
    test_scores = []
    
    for epoch in range(1, epochs + 1):
        combined_model.train()
        print(f"Starting epoch: {epoch}")
        

        for batch_idx, (tabular_data, targets, input_ids, attention_mask) in enumerate(train_loader, 1):
            optimizer.zero_grad()
            outputs, _, _ = combined_model(tabular_data, input_ids, attention_mask)
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)  # [batch_size] → [batch_size, 1]
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Evaluate training performance
        train_score = evaluate_model(combined_model, train_loader)
        train_scores.append(train_score)
        logger.info(f'Train Epoch: {epoch}, train-auc-score: {train_score:.4f}')

        # Evaluate test performance
        test_score = evaluate_model(combined_model, test_loader)
        test_scores.append(test_score)
    
    # Save training scores
    scores_df = pd.DataFrame({
        'train_scores': train_scores,
        'test_scores': test_scores
    })
    scores_df.to_csv(Path(model_dir, 'training_scores.csv'), index=False)
    
    # Save the combined model
    torch.save({
        'combined_model_state_dict': combined_model.state_dict(),
    }, Path(model_dir, 'combined_model.pth'))

    # Train XGBoost for tabular features interpretation
    xgb_model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=123,
        n_jobs=-1
        )
    
    xgb_model.fit(X_train_processed, y)

    # Save XGBoost model
    joblib.dump(xgb_model, Path(model_dir, 'xgboost_model.joblib'))
    feature_cols = pd.DataFrame(X).columns.tolist()
    joblib.dump(feature_cols, Path(model_dir, 'feature_columns.joblib'))

    return combined_model, xgb_model, test_loader

def evaluate_model(model, data_loader):
    """Evaluate model performance on a dataset"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for tabular_data, targets, input_ids, attention_mask in data_loader:
            outputs, _, _ = model(tabular_data, input_ids, attention_mask)
            preds = torch.sigmoid(outputs).squeeze()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Handle case with only one class
    if len(np.unique(all_targets)) < 2:
        return 0.5
    
    return roc_auc_score(all_targets, all_preds)

def predict(tabular_features, text_features, model_path=None):
    """Predict using the trained combined model"""
    if model_path is None:
        model_path = Path(model_dir, 'combined_model.pth')
    
    checkpoint = torch.load(model_path)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Load preprocessing objects
    label_encoders = joblib.load(Path(model_dir, 'label_encoders.joblib'))
    scaler = joblib.load(Path(model_dir, 'scaler.joblib'))
    
    # Preprocess tabular features
    tabular_df = pd.DataFrame(tabular_features)
    numeric_columns = tabular_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = tabular_df.select_dtypes(include=['object']).columns.tolist()
    
    # Transform categorical features
    for col in categorical_columns:
        if col in label_encoders:
            tabular_df[col] = label_encoders[col].transform(tabular_df[col].astype(str))

    imputer = joblib.load(Path(model_dir, 'imputer.joblib'))
    tabular_df_imputed = imputer.transform(tabular_df)
    tabular_df = pd.DataFrame(tabular_df_imputed, columns=tabular_df.columns)

    # Then scale numerical features
    tabular_df[numeric_columns] = scaler.transform(tabular_df[numeric_columns])
    
    tabular_features = tabular_df.values.astype(np.float32)
    tabular_input_size = tabular_features.shape[1]
    
    # Initialize models
    text_model = DistilBertTextClassifier()
    tabular_model = TabularModel(tabular_input_size)
    model = CombinedModel(tabular_input_size, text_model, tabular_model)
    
    # Load model weights
    model.load_state_dict(checkpoint['combined_model_state_dict'])
    model.eval()
    
    # Tokenize text features
    encodings = tokenizer(
        list(text_features),
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Make predictions
    with torch.no_grad():
        outputs, _, _ = model(
            torch.tensor(tabular_features, dtype=torch.float32),
            encodings['input_ids'],
            encodings['attention_mask']
        )
        preds = torch.sigmoid(outputs).squeeze()
    
    return preds.numpy()

def predict_with_xgboost(tabular_features):
    """Predict using XGBoost model trained on tabular features only"""
    xgb_model = joblib.load(Path(model_dir, 'xgboost_model.joblib'))
    
    # Preprocess tabular features
    tabular_df = pd.DataFrame(tabular_features)
    numeric_columns = tabular_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = tabular_df.select_dtypes(include=['object']).columns.tolist()
    
    # Load preprocessing objects
    label_encoders = joblib.load(Path(model_dir, 'label_encoders.joblib'))
    scaler = joblib.load(Path(model_dir, 'scaler.joblib'))
    
    # Transform categorical features
    for col in categorical_columns:
        if col in label_encoders:
            tabular_df[col] = label_encoders[col].transform(tabular_df[col].astype(str))

    imputer = joblib.load(Path(model_dir, 'imputer.joblib'))
    tabular_df = pd.DataFrame(
        imputer.transform(tabular_df),
        columns=tabular_df.columns
    )
    
    # Scale numerical features
    tabular_df[numeric_columns] = scaler.transform(tabular_df[numeric_columns])
    
    tabular_features_processed = tabular_df.values.astype(np.float32)
    
    return xgb_model.predict_proba(tabular_features_processed)[:, 1]

def plot_roc_curve(tabular_features, text_features, labels):
    """Plot ROC curve for the combined model"""
    preds = predict(tabular_features, text_features)
    fpr, tpr, _ = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.title('ROC Curve')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(Path(model_dir, 'roc_curve.png'))
    plt.close()

def plot_pr_curve(tabular_features, text_features, labels):
    """Plot Precision-Recall curve for the combined model"""
    preds = predict(tabular_features, text_features)
    precisions, recalls, _ = precision_recall_curve(labels, preds)
    roc_auc = roc_auc_score(labels, preds)

    plt.figure(figsize=(6, 5))
    plt.plot(recalls, precisions, label='Model: AUC = %0.2f' % roc_auc)
    plt.plot([1, 0], [0, 1], '--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left')
    plt.title('Precision-Recall Curve')
    plt.savefig(Path(model_dir, 'pr_curve.png'))
    plt.close()

   
