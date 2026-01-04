# -*- coding: utf-8 -*-
"""
Ablation Study Module
- Tabular-only model training and evaluation
- Text-only model training and evaluation
- Compare with combined model to identify leakage source
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from transformers import DistilBertTokenizer, DistilBertModel
from pathlib import Path
import joblib
import yaml

# Load config
with open("../model/params.yaml", "r") as f:
    params = yaml.safe_load(f)

model_dir = Path(params['model_dir'])
data_dir = Path(params['data_dir'])


class TextOnlyModel(nn.Module):
    """DistilBERT-based text classifier without tabular features"""
    
    def __init__(self, output_dim=1):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.classifier = nn.Sequential(
            nn.Linear(self.distilbert.config.hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # CLS token
        return self.classifier(pooled)


def preprocess_tabular(X_train, X_test):
    """Preprocess tabular data for ablation study"""
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    
    # Label encode categorical
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        combined = pd.concat([X_train[col], X_test[col]]).astype(str)
        le.fit(combined)
        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        label_encoders[col] = le
    
    # Scale numeric
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    # Impute missing
    imputer = SimpleImputer(strategy='mean')
    X_train_arr = imputer.fit_transform(X_train)
    X_test_arr = imputer.transform(X_test)
    
    return X_train_arr, X_test_arr


def train_tabular_only(X_train, y_train, X_test, y_test):
    """
    Train XGBoost model using only tabular features
    
    This ablation tests whether tabular features alone can predict churn.
    If AUC is much lower than combined model, text adds real value.
    If AUC is similar, text may not be contributing or may be leaking.
    """
    print("=" * 60)
    print("ABLATION: TABULAR FEATURES ONLY (XGBoost)")
    print("=" * 60)
    
    # Preprocess
    X_train_arr, X_test_arr = preprocess_tabular(X_train, X_test)
    
    # Train XGBoost
    model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_arr, y_train)
    
    # Evaluate
    train_pred = model.predict_proba(X_train_arr)[:, 1]
    test_pred = model.predict_proba(X_test_arr)[:, 1]
    
    train_auc = roc_auc_score(y_train, train_pred)
    test_auc = roc_auc_score(y_test, test_pred)
    
    print(f"\nResults:")
    print(f"  Train AUC: {train_auc:.4f}")
    print(f"  Test AUC:  {test_auc:.4f}")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 5 Important Features:")
    for _, row in importance.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    print("=" * 60)
    
    return {
        'model': model,
        'train_auc': train_auc,
        'test_auc': test_auc,
        'feature_importance': importance
    }


def train_text_only(text_train, y_train, text_test, y_test, epochs=3, batch_size=16):
    """
    Train DistilBERT model using only text features
    
    This ablation tests whether text alone can predict churn.
    If AUC ≈ 1.0, text contains label information (leakage).
    If AUC ≈ 0.5, text has no predictive power.
    """
    print("=" * 60)
    print("ABLATION: TEXT FEATURES ONLY (DistilBERT)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Tokenize
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    print("Tokenizing texts...")
    train_enc = tokenizer(
        list(text_train), 
        padding=True, 
        truncation=True, 
        max_length=256,
        return_tensors='pt'
    )
    test_enc = tokenizer(
        list(text_test), 
        padding=True, 
        truncation=True, 
        max_length=256,
        return_tensors='pt'
    )
    
    # Create datasets
    train_dataset = TensorDataset(
        train_enc['input_ids'],
        train_enc['attention_mask'],
        torch.tensor(y_train, dtype=torch.float)
    )
    test_dataset = TensorDataset(
        test_enc['input_ids'],
        test_enc['attention_mask'],
        torch.tensor(y_test, dtype=torch.float)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    model = TextOnlyModel(output_dim=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        
        for batch_idx, (input_ids, attention_mask, labels) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # Evaluate
        train_auc = evaluate_text_model(model, train_loader, device)
        test_auc = evaluate_text_model(model, test_loader, device)
        
        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Train AUC={train_auc:.4f}, Test AUC={test_auc:.4f}")
    
    # Final evaluation
    final_train_auc = evaluate_text_model(model, train_loader, device)
    final_test_auc = evaluate_text_model(model, test_loader, device)
    
    print(f"\nFinal Results:")
    print(f"  Train AUC: {final_train_auc:.4f}")
    print(f"  Test AUC:  {final_test_auc:.4f}")
    
    # Interpretation
    print(f"\nInterpretation:")
    if final_test_auc > 0.95:
        print("  [WARNING] Test AUC > 0.95 suggests text contains label information!")
        print("  This indicates potential data leakage in text generation.")
    elif final_test_auc > 0.7:
        print("  [INFO] Text has moderate predictive power.")
    else:
        print("  [INFO] Text has limited predictive power for churn.")
    
    print("=" * 60)
    
    return {
        'model': model,
        'train_auc': final_train_auc,
        'test_auc': final_test_auc
    }


def evaluate_text_model(model, data_loader, device):
    """Evaluate text-only model and return AUC"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for input_ids, attention_mask, labels in data_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            outputs = model(input_ids, attention_mask)
            preds = torch.sigmoid(outputs).squeeze().cpu().numpy()
            
            if preds.ndim == 0:
                preds = np.array([preds])
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    if len(np.unique(all_labels)) < 2:
        return 0.5
    
    return roc_auc_score(all_labels, all_preds)


def run_ablation_study(train_df, test_df):
    """
    Run complete ablation study comparing tabular-only, text-only, and combined models
    
    Args:
        train_df: Training DataFrame with 'churn', 'chat_log', and tabular columns
        test_df: Test DataFrame with same columns
    
    Returns:
        dict with all results
    """
    print("\n" + "=" * 70)
    print("ABLATION STUDY: DIAGNOSING DATA LEAKAGE SOURCE")
    print("=" * 70 + "\n")
    
    # Prepare data
    numerical_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols = [c for c in numerical_cols if c != 'churn']
    
    categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c != 'chat_log']
    
    X_train = train_df[numerical_cols + categorical_cols]
    y_train = train_df['churn'].values
    text_train = train_df['chat_log'].values
    
    X_test = test_df[numerical_cols + categorical_cols]
    y_test = test_df['churn'].values
    text_test = test_df['chat_log'].values
    
    print(f"Data shapes:")
    print(f"  Tabular: {X_train.shape[1]} features")
    print(f"  Text: {len(text_train)} samples")
    print(f"  Train/Test: {len(train_df)}/{len(test_df)}\n")
    
    # Run ablations
    tabular_results = train_tabular_only(X_train, y_train, X_test, y_test)
    text_results = train_text_only(text_train, y_train, text_test, y_test, epochs=3)
    
    # Summary
    print("\n" + "=" * 70)
    print("ABLATION STUDY SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<25} {'Train AUC':<12} {'Test AUC':<12}")
    print("-" * 50)
    print(f"{'Tabular Only (XGBoost)':<25} {tabular_results['train_auc']:<12.4f} {tabular_results['test_auc']:<12.4f}")
    print(f"{'Text Only (DistilBERT)':<25} {text_results['train_auc']:<12.4f} {text_results['test_auc']:<12.4f}")
    
    # Diagnosis
    print("\n" + "-" * 50)
    print("DIAGNOSIS:")
    
    if text_results['test_auc'] > 0.95:
        print("  [LEAKAGE SOURCE: TEXT]")
        print("  Text alone achieves near-perfect AUC.")
        print("  This strongly suggests text was generated with label-aware patterns.")
        print("  Recommended: Regenerate text data without label conditioning.")
    elif tabular_results['test_auc'] > 0.95:
        print("  [LEAKAGE SOURCE: TABULAR]")
        print("  Tabular features alone achieve near-perfect AUC.")
        print("  Check for features that directly encode the label.")
    elif text_results['test_auc'] > tabular_results['test_auc'] + 0.1:
        print("  [TEXT ADDS VALUE]")
        print("  Text significantly improves prediction over tabular alone.")
        print("  This is expected behavior if text captures customer sentiment.")
    else:
        print("  [TEXT ADDS MINIMAL VALUE]")
        print("  Text does not significantly improve over tabular features.")
        print("  Consider whether text features are necessary.")
    
    print("=" * 70)
    
    return {
        'tabular': tabular_results,
        'text': text_results
    }


def quick_leakage_test(text_train, y_train, text_test, y_test):
    """
    Quick test for text leakage using simple keyword matching
    
    This is faster than training DistilBERT and can detect obvious leakage.
    """
    print("=" * 60)
    print("QUICK TEXT LEAKAGE TEST")
    print("=" * 60)
    
    # Define sentiment keywords
    negative_keywords = ['apologize', 'sorry', 'refund', 'cancel', 'complaint', 
                        'problem', 'issue', 'delay', 'frustrated', 'disappointed']
    positive_keywords = ['thank', 'great', 'excellent', 'resolved', 'happy',
                        'satisfied', 'helpful', 'appreciate', 'wonderful']
    
    def score_text(text):
        if not isinstance(text, str):
            return 0
        text_lower = text.lower()
        neg_count = sum(1 for kw in negative_keywords if kw in text_lower)
        pos_count = sum(1 for kw in positive_keywords if kw in text_lower)
        return neg_count - pos_count  # Higher = more negative
    
    # Score all texts
    train_scores = np.array([score_text(t) for t in text_train])
    test_scores = np.array([score_text(t) for t in text_test])
    
    # Simple threshold classifier
    threshold = np.median(train_scores)
    train_pred = (train_scores > threshold).astype(int)
    test_pred = (test_scores > threshold).astype(int)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    # Correlation
    train_corr = np.corrcoef(train_scores, y_train)[0, 1]
    test_corr = np.corrcoef(test_scores, y_test)[0, 1]
    
    print(f"\nKeyword-based sentiment scores:")
    print(f"  Threshold: {threshold:.2f}")
    print(f"  Train accuracy: {train_acc:.2%}")
    print(f"  Test accuracy:  {test_acc:.2%}")
    print(f"  Train correlation: {train_corr:.4f}")
    print(f"  Test correlation:  {test_corr:.4f}")
    
    # Interpretation
    if test_corr > 0.5:
        print(f"\n  [WARNING] High correlation ({test_corr:.2f}) suggests text sentiment")
        print("  is strongly linked to churn label - likely leakage!")
    elif test_corr > 0.3:
        print(f"\n  [CAUTION] Moderate correlation ({test_corr:.2f}) detected.")
    else:
        print(f"\n  [OK] Low correlation ({test_corr:.2f}) - sentiment keywords alone")
        print("  do not predict churn well.")
    
    print("=" * 60)
    
    return {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_corr': train_corr,
        'test_corr': test_corr
    }


if __name__ == "__main__":
    # Load data
    train_df = pd.read_csv(data_dir / "train2.csv")
    test_df = pd.read_csv(data_dir / "test2.csv")
    
    # Quick test first
    quick_leakage_test(
        train_df['chat_log'].values,
        train_df['churn'].values,
        test_df['chat_log'].values,
        test_df['churn'].values
    )
    
    # Full ablation study
    results = run_ablation_study(train_df, test_df)
