# -*- coding: utf-8 -*-
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
from sklearn.model_selection import GroupKFold, train_test_split
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

with open("../model/params.yaml", "r") as params_file:
    params = yaml.safe_load(params_file)

model_dir = params['model_dir']
    
class DistilBertTextClassifier(nn.Module):
    def __init__(self, output_dim=32):
        super(DistilBertTextClassifier, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)
        
class TabularModel(nn.Module):
    def __init__(self, input_size, output_dim=32):
        super(TabularModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.network(x)


class CombinedModel(nn.Module):
    def __init__(self, tabular_input_size, text_model, tabular_model):
        super(CombinedModel, self).__init__()
        self.text_model = text_model
        self.tabular_model = tabular_model
        self.fusion_layer = nn.Sequential(
            nn.Linear(32 + 32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1) 
        )

    def forward(self, tabular_data, input_ids, attention_mask):
        text_output = self.text_model(input_ids, attention_mask)
        tabular_output = self.tabular_model(tabular_data)
        combined = torch.cat([text_output, tabular_output], dim=1)
        fused_output = self.fusion_layer(combined)
        return fused_output, text_output, tabular_output

def preprocess2(X_train, X_test):
    """Preprocess tabular data: handle categorical features, numerical features, and missing values"""
    
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    numeric_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = X_train.select_dtypes(include=['object']).columns.tolist()
    
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        all_values = pd.concat([X_train[col], X_test[col]]).astype(str)
        le.fit(all_values)
        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        label_encoders[col] = le
    
    all_columns = X_train.columns.tolist()
    
    scaler = StandardScaler()
    X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
    X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])
    
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    joblib.dump({
        "label_encoders": label_encoders,
        "scaler": scaler,
        "imputer": imputer,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "all_columns": all_columns
    }, "../model/preprocess_artifacts.joblib")
    
    return (
        X_train_imputed, X_test_imputed, 
        label_encoders, scaler, imputer,
        numeric_columns, categorical_columns, all_columns
    )

def train_model(X_train, y_train, X_test, y_test, text_train, text_test):
    batch_size = params['batch_size']
    batch_size_test = params['batch_size_test']
    epochs = params['epochs']
    pos_weight = params['pos_weight']
    base_lr = params.get('lr', 0.001) 
    bert_lr = 2e-5 
    
    X_train_processed, X_test_processed, label_encoders, scaler, imputer, numeric_columns, categorical_columns, all_columns = preprocess2(
        pd.DataFrame(X_train), pd.DataFrame(X_test)
    )

    X_train_array = np.array(X_train_processed, dtype=np.float32)
    y_train_array = np.array(y_train, dtype=np.float32)
    X_test_array = np.array(X_test_processed, dtype=np.float32)
    y_test_array = np.array(y_test, dtype=np.float32)

    joblib.dump(label_encoders, Path(model_dir, 'label_encoders.joblib'))
    joblib.dump(scaler, Path(model_dir, 'scaler.joblib'))
    joblib.dump(imputer, Path(model_dir, 'imputer.joblib'))
    joblib.dump(all_columns, Path(model_dir, 'all_columns.joblib'))
    joblib.dump(numeric_columns, Path(model_dir, 'numeric_columns.joblib'))
    joblib.dump(categorical_columns, Path(model_dir, 'categorical_columns.joblib'))

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    train_encodings = tokenizer(list(text_train), padding=True, truncation=True, max_length=512, return_tensors="pt")
    test_encodings = tokenizer(list(text_test), padding=True, truncation=True, max_length=512, return_tensors="pt")

    train_dataset = TensorDataset(Tensor(X_train_array), torch.tensor(y_train_array), 
                                train_encodings['input_ids'], train_encodings['attention_mask'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = TensorDataset(Tensor(X_test_array), torch.tensor(y_test_array), 
                               test_encodings['input_ids'], test_encodings['attention_mask'])
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)

    text_model = DistilBertTextClassifier(output_dim=32)
    tabular_model = TabularModel(X_train_array.shape[1], output_dim=32)
    combined_model = CombinedModel(X_train_array.shape[1], text_model, tabular_model)
    
    bert_params = list(combined_model.text_model.distilbert.parameters())
    other_params = [p for n, p in combined_model.named_parameters() if 'distilbert' not in n]
    
    optimizer = optim.AdamW([
        {'params': bert_params, 'lr': bert_lr},
        {'params': other_params, 'lr': base_lr}
    ], weight_decay=0.01)    
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(pos_weight, dtype=torch.float))
    
    train_scores = []
    test_scores = []
    
    for epoch in range(1, epochs + 1):
        combined_model.train()
        epoch_losses = []
        
        for batch_idx, (tabular_data, targets, input_ids, attention_mask) in enumerate(train_loader, 1):
            optimizer.zero_grad()
            
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            
            outputs, _, _ = combined_model(tabular_data, input_ids, attention_mask)
            loss = criterion(outputs, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(combined_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if batch_idx % 10 == 0:
                weights = combined_model.fusion_layer[0].weight.data
                text_w = weights[:, :32].abs().mean().item()
                tabular_w = weights[:, 32:].abs().mean().item()
                print(f"Epoch {epoch} Batch {batch_idx} | Avg Weights -> Text: {text_w:.4f}, Tabular: {tabular_w:.4f}")
            
            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        train_score = evaluate_model(combined_model, train_loader)
        test_score = evaluate_model(combined_model, test_loader)
        
        train_scores.append(train_score)
        test_scores.append(test_score)
        
        logger.info(f'Epoch {epoch}: Loss={avg_loss:.4f}, TrainAUC={train_score:.4f}, TestAUC={test_score:.4f}')


    torch.save({'combined_model_state_dict': combined_model.state_dict()}, Path(model_dir, 'combined_model.pth'))


    X_train_named = pd.DataFrame(X_train_processed, columns=all_columns)
    xgb_model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=123,
        n_jobs=-1
    )
    xgb_model.fit(X_train_named, y_train)

    joblib.dump(xgb_model, Path(model_dir, 'xgboost_model.joblib'))
    
    return combined_model, xgb_model, test_loader


def evaluate_model(model, data_loader):
    """Evaluate model performance on a dataset"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for tabular_data, targets, input_ids, attention_mask in data_loader:
            outputs, _, _ = model(tabular_data, input_ids, attention_mask)
            # FIXED: Apply sigmoid here to get probabilities
            preds = torch.sigmoid(outputs).squeeze()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    if len(np.unique(all_targets)) < 2:
        return 0.5
    
    return roc_auc_score(all_targets, all_preds)

def predict(tabular_features, text_features, model_path=None):
    if model_path is None:
        model_path = Path(model_dir, 'combined_model.pth')
    
    all_columns = joblib.load(Path(model_dir, 'all_columns.joblib'))
    label_encoders = joblib.load(Path(model_dir, 'label_encoders.joblib'))
    scaler = joblib.load(Path(model_dir, 'scaler.joblib'))
    imputer = joblib.load(Path(model_dir, 'imputer.joblib'))
    numeric_columns = joblib.load(Path(model_dir, 'numeric_columns.joblib'))
    categorical_columns = joblib.load(Path(model_dir, 'categorical_columns.joblib'))

    tabular_df = pd.DataFrame(tabular_features)[all_columns].copy()
    for col in categorical_columns:
        if col in label_encoders:
            tabular_df[col] = label_encoders[col].transform(tabular_df[col].astype(str))
    
    tabular_df_imputed = imputer.transform(tabular_df)
    tabular_df = pd.DataFrame(tabular_df_imputed, columns=all_columns)
    tabular_df[numeric_columns] = scaler.transform(tabular_df[numeric_columns])
    tabular_processed = tabular_df.values.astype(np.float32)
    
    text_model = DistilBertTextClassifier(output_dim=32)
    tabular_model = TabularModel(tabular_processed.shape[1], output_dim=32)
    model = CombinedModel(tabular_processed.shape[1], text_model, tabular_model)
    
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['combined_model_state_dict'])
    model.eval()

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    encodings = tokenizer(list(text_features), padding=True, truncation=True, max_length=512, return_tensors="pt")

    with torch.no_grad():
        outputs, text_feats, tab_feats = model(torch.tensor(tabular_processed), 
                                          encodings['input_ids'], 
                                          encodings['attention_mask'])   
        preds = torch.sigmoid(outputs).numpy().flatten() 
    
    return preds


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

# Main execution
if __name__ == "__main__":
    # Load data
    data_dir = Path("../data")
    train_df = pd.read_csv(data_dir / "train2.csv")
    test_df = pd.read_csv(data_dir / "test2.csv")
    
    # Identify features
    numerical_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols = [c for c in numerical_cols if c != 'churn']
    
    categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c != 'chat_log']
    
    # Prepare data
    X_train = train_df[numerical_cols + categorical_cols]
    y_train = train_df['churn'].values
    text_train = train_df['chat_log'].values
    
    X_test = test_df[numerical_cols + categorical_cols]
    y_test = test_df['churn'].values
    text_test = test_df['chat_log'].values
    
    # Train
    print("Starting training...")
    combined_model, xgb_model, test_loader = train_model(
        X_train, y_train, X_test, y_test, text_train, text_test
    )
    
    print("Training complete!")
    
    # Plot curves
    plot_roc_curve(X_test, text_test, y_test)
    plot_pr_curve(X_test, text_test, y_test)