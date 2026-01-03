# -*- coding: utf-8 -*-
"""
Model Interpretation Module
- Text: Attention-based + Gradient methods
- Tabular: SHAP + Feature Importance
- Keyword Analysis: Extract and analyze important keywords with context
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import shap
from pathlib import Path
import joblib
from transformers import DistilBertTokenizer
import yaml
from collections import defaultdict, Counter
import re
import pickle

with open("../model/params.yaml", "r") as f:
    params = yaml.safe_load(f)

model_dir = Path(params['model_dir'])
data_dir = Path(params['data_dir'])

os.makedirs(model_dir, exist_ok=True)

# Comprehensive stopwords list
STOPWORDS = {
    # Articles, prepositions, conjunctions
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'as', 'by', 'with', 'from', 'which', 'that', 'this', 'these', 'those',
    'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'between', 'under', 'over', 'out', 'up', 'down',
    
    # Verbs
    'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'done',
    'will', 'would', 'shall', 'should', 'may', 'might', 'must', 'can', 'could',
    'go', 'get', 'make', 'know', 'think', 'see', 'come', 'take', 'use',
    'find', 'give', 'tell', 'work', 'call', 'try', 'ask', 'need', 'feel',
    
    # Pronouns
    'i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves',
    'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
    'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    
    # Common words
    'what', 'when', 'where', 'why', 'how', 'who', 'whom', 'whose',
    'not', 'no', 'yes', 'now', 'then', 'here', 'there', 'just', 'only',
    'very', 'so', 'also', 'well', 'even', 'still', 'back', 'much', 'any',
    'all', 'some', 'such', 'each', 'every', 'both', 'few', 'more', 'most',
    'other', 'own', 'same', 'than', 'too',
    
    # Customer service terms
    'hi', 'hello', 'hey', 'thanks', 'thank', 'please', 'sorry',
    'ok', 'okay', 'sure', 'yeah', 'yep', 'customer', 'agent', 'service', 'help', 'want',
    
    # Special tokens (all variants)
    '[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]',
    'cls', 'sep', 'pad', 'unk', 'mask',
    '[cls]', '[sep]', '[pad]', '[unk]', '[mask]',
    
    # Punctuation
    '.', ',', '!', '?', ';', ':', '-', '--', '...', "'", '"',
    '(', ')', '[', ']', '{', '}', '/', '\\', '|',
}


class DistilBertTextClassifierWithAttention(nn.Module):
    """Text classifier that exposes attention weights for interpretation"""
    
    def __init__(self, num_classes=32):
        super().__init__()
        from transformers import DistilBertModel
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_ids=None, attention_mask=None, output_attentions=True):
        input_ids = input_ids.long()
        attention_mask = attention_mask.long()
        
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )
        
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits, outputs.attentions if output_attentions else None


def compute_attention_importance(attentions, attention_mask):
    """Compute token importance based on attention to CLS token"""
    cls_attentions = []
    
    for layer_attention in attentions:
        cls_attention = layer_attention[0, :, 0, :].mean(dim=0)
        cls_attentions.append(cls_attention)
    
    avg_attention = torch.stack(cls_attentions).mean(dim=0)
    avg_attention = avg_attention * attention_mask[0].float()
    
    return avg_attention.cpu().numpy()


def compute_gradient_x_input(model, input_ids, attention_mask, embedding_layer):
    """Compute Gradient x Input saliency"""
    model.train()
    input_embeds = embedding_layer(input_ids)
    input_embeds.requires_grad_(True)
    input_embeds.retain_grad()
    
    outputs = model.distilbert(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        output_attentions=False
    )
    
    pooled_output = outputs.last_hidden_state[:, 0]
    pooled_output = model.dropout(pooled_output)
    logits = model.classifier(pooled_output)
    score = torch.sigmoid(logits).mean()
    
    score.backward()
    
    if input_embeds.grad is not None:
        importance = (input_embeds * input_embeds.grad).sum(dim=-1).squeeze(0)
        model.eval()
        return importance.abs().detach().cpu().numpy()
    
    model.eval()
    return np.zeros(input_ids.shape[1])


def filter_stopwords(tokens, scores, top_k=20):
    """Filter stopwords and return top-k important tokens"""
    token_score_pairs = []
    
    for token, score in zip(tokens, scores):
        token_clean = token.lower().replace('##', '')
        
        # Multi-level stopword check
        is_stopword = (
            token_clean in STOPWORDS or
            token.lower() in STOPWORDS or
            token in STOPWORDS
        )
        
        # Filter conditions
        if (not is_stopword and 
            len(token_clean) > 1 and 
            abs(score) > 1e-8 and
            not token_clean.isdigit()):
            token_score_pairs.append((token, score))
    
    # Fallback if no tokens pass filter
    if len(token_score_pairs) == 0:
        valid_pairs = [(t, s) for t, s in zip(tokens, scores) if abs(s) > 1e-8]
        token_score_pairs = valid_pairs[:top_k] if valid_pairs else list(zip(tokens[:top_k], scores[:top_k]))
    
    sorted_pairs = sorted(token_score_pairs, key=lambda x: abs(x[1]), reverse=True)
    return sorted_pairs[:top_k]


def save_explanation_html(tokens, scores, output_path, method="attention", top_k=20):
    """Save token importance visualization as HTML"""
    
    top_pairs = filter_stopwords(tokens, scores, top_k)
    
    if len(top_pairs) == 0:
        top_pairs = [(tokens[0], scores[0])]
    
    display_tokens, display_scores = zip(*top_pairs)
    abs_scores = np.abs(display_scores)
    max_score = abs_scores.max() if abs_scores.max() > 1e-8 else 1.0
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{ font-family: Arial, sans-serif; padding: 25px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 12px; box-shadow: 0 10px 40px rgba(0,0,0,0.2); }}
        .token {{ display: inline-block; margin: 5px; padding: 10px 16px; border-radius: 6px; font-size: 15px; font-weight: 600; transition: all 0.3s; }}
        .token:hover {{ transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }}
        .header {{ font-size: 28px; font-weight: bold; margin-bottom: 20px; color: #2d3748; }}
        .method-badge {{ display: inline-block; padding: 6px 14px; background: #48bb78; color: white; border-radius: 20px; font-size: 13px; margin-left: 12px; }}
        .legend {{ margin: 25px 0; padding: 20px; background: #f7fafc; border-radius: 8px; border-left: 4px solid #667eea; }}
        .stats {{ margin: 20px 0; padding: 15px; background: #edf2f7; border-radius: 6px; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            Token Importance Analysis
            <span class="method-badge">{method.upper()}</span>
        </div>
        <div class="legend">
            <strong>Method:</strong> {'Attention-based importance (average attention to CLS token)' if method == 'attention' else 'Gradient x Input (saliency map)'}<br>
            <strong>Interpretation:</strong> Darker colors indicate higher importance for prediction.
        </div>
        <div class="stats">
            <strong>Displayed Tokens:</strong> {len(display_tokens)} (stopwords filtered)<br>
            <strong>Score Range:</strong> [{min(abs_scores):.6f}, {max(abs_scores):.6f}]
        </div>
        <div style="line-height: 2.8; margin-top: 20px;">
"""
    
    # Add colored tokens
    for token, score in zip(display_tokens, display_scores):
        intensity = min(abs(score) / max_score, 1.0)
        color_r = int(255 * (1 - intensity * 0.7))
        color_g = int(255 * (1 - intensity * 0.5))
        color_b = 255
        bgcolor = f"rgb({color_r}, {color_g}, {color_b})"
        
        html_content += f'<span class="token" style="background-color: {bgcolor};">{token}</span>'
    
    html_content += """
        </div>
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def explain_text_with_attention(max_samples=5):
    """
    Generate text explanations using attention and gradient methods
    Returns results for keyword analysis
    """
    
    print("=" * 70)
    print("TEXT INTERPRETATION")
    print("=" * 70)
    
    # Load model
    print("\nLoading model...")
    device = torch.device('cpu')
    combined_model = torch.load(model_dir / "combined_model.pth", map_location=device)
    text_model = combined_model.text_model
    text_model.eval()
    
    # Load tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Load test data
    test_df = pd.read_csv(data_dir / "test2.csv")
    sample_texts = test_df['chat_log'].dropna().tolist()[:max_samples]
    
    print(f"Processing {len(sample_texts)} samples...")
    
    output_dir = model_dir / "text_explanations"
    output_dir.mkdir(exist_ok=True)
    
    # Store results for keyword analysis
    attention_results = {}
    gradient_results = {}
    
    for i, text in enumerate(sample_texts):
        print(f"\nSample {i+1}/{len(sample_texts)}")
        
        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=128
        )
        
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        # Get actual length (excluding padding)
        actual_length = attention_mask.sum().item()
        
        # Attention-based importance
        with torch.no_grad():
            _, attentions = text_model(input_ids, attention_mask, output_attentions=True)
        
        attention_scores = compute_attention_importance(attentions, attention_mask)
        attention_scores = attention_scores[:actual_length]
        
        # Gradient-based importance
        embedding_layer = text_model.distilbert.embeddings.word_embeddings
        grad_scores = compute_gradient_x_input(text_model, input_ids, attention_mask, embedding_layer)
        grad_scores = grad_scores[:actual_length]
        
        # Get tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0][:actual_length])
        
        # Store for keyword analysis
        sample_id = f"sample_{i}"
        attention_results[sample_id] = dict(zip(tokens, attention_scores.tolist()))
        gradient_results[sample_id] = dict(zip(tokens, grad_scores.tolist()))
        
        # Save HTML visualizations
        save_explanation_html(
            tokens=tokens,
            scores=attention_scores,
            output_path=output_dir / f"sample_{i}_attention.html",
            method="attention",
            top_k=20
        )
        
        save_explanation_html(
            tokens=tokens,
            scores=grad_scores,
            output_path=output_dir / f"sample_{i}_gradient.html",
            method="gradient",
            top_k=20
        )
    
    # Save results
    results_data = {
        'attention': attention_results,
        'gradient': gradient_results,
        'metadata': {
            'num_samples': len(sample_texts),
            'max_length': 128,
            'model': 'DistilBERT'
        }
    }
    
    results_path = output_dir / "interpretation_results.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(results_data, f)
    
    print(f"\n{'='*70}")
    print(f"Explanations saved to: {output_dir}")
    print(f"Results saved to: {results_path}")
    print("=" * 70)
    
    return str(output_dir), attention_results, gradient_results


def get_important_keywords(
    attention_results=None,
    gradient_results=None,
    top_k=50,
    min_count=2,
    combine_method='weighted_average'
):
    """Extract important keywords from interpretation results"""
    
    print("=" * 70)
    print("KEYWORD ANALYSIS")
    print("=" * 70)
    
    token_data = defaultdict(lambda: {
        'attention_scores': [],
        'gradient_scores': [],
        'count': 0
    })
    
    # Aggregate attention scores
    if attention_results:
        print(f"Processing {len(attention_results)} attention results")
        for sample_id, token_scores in attention_results.items():
            for token, score in token_scores.items():
                token_data[token]['attention_scores'].append(score)
                token_data[token]['count'] += 1
    
    # Aggregate gradient scores
    if gradient_results:
        print(f"Processing {len(gradient_results)} gradient results")
        for sample_id, token_scores in gradient_results.items():
            for token, score in token_scores.items():
                token_data[token]['gradient_scores'].append(score)
    
    # Build results with stopword filtering
    results = []
    filtered_count = 0
    
    for token, data in token_data.items():
        if data['count'] < min_count:
            continue
        
        # Apply stopword filter
        token_clean = token.lower().replace('##', '')
        is_stopword = (
            token_clean in STOPWORDS or
            token.lower() in STOPWORDS or
            token in STOPWORDS
        )
        
        # Skip stopwords and invalid tokens
        if is_stopword or len(token_clean) <= 1 or token_clean.isdigit():
            filtered_count += 1
            continue
        
        avg_attention = np.mean(data['attention_scores']) if data['attention_scores'] else 0.0
        avg_gradient = np.mean(data['gradient_scores']) if data['gradient_scores'] else 0.0
        max_attention = np.max(data['attention_scores']) if data['attention_scores'] else 0.0
        max_gradient = np.max(data['gradient_scores']) if data['gradient_scores'] else 0.0
        
        # Combine scores (attention weighted 60%, gradient 40%)
        if combine_method == 'weighted_average':
            combined_score = 0.6 * avg_attention + 0.4 * avg_gradient
        elif combine_method == 'max':
            combined_score = max(max_attention, max_gradient)
        else:
            combined_score = avg_attention + avg_gradient
        
        # Boost by frequency (log scale)
        frequency_boost = np.log1p(data['count'])
        combined_score_with_freq = combined_score * frequency_boost
        
        results.append({
            'keyword': token_clean,  # Use cleaned version
            'frequency': data['count'],
            'avg_attention': avg_attention,
            'max_attention': max_attention,
            'avg_gradient': avg_gradient,
            'max_gradient': max_gradient,
            'combined_score': combined_score,
            'combined_score_with_freq': combined_score_with_freq
        })
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        print("No keywords found after filtering")
        return pd.DataFrame()
    
    results_df = results_df.sort_values('combined_score_with_freq', ascending=False)
    results_df['rank'] = range(1, len(results_df) + 1)
    
    results_df = results_df[[
        'rank', 'keyword', 'frequency',
        'avg_attention', 'max_attention',
        'avg_gradient', 'max_gradient',
        'combined_score', 'combined_score_with_freq'
    ]]
    
    print(f"Filtered {filtered_count} stopwords/invalid tokens")
    print(f"Identified {len(results_df)} valid keywords")
    print(f"Returning top {min(top_k, len(results_df))}\n")
    
    return results_df.head(top_k)


def obtain_context(
    chats_list,
    keyword,
    window_size=50,
    max_examples=10,
    highlight=True
):
    """Find and display contexts where keyword appears"""
    
    print("=" * 70)
    print(f"CONTEXT ANALYSIS: '{keyword}'")
    print("=" * 70)
    
    # Clean keyword (remove ## and lowercase)
    keyword_lower = keyword.lower().replace('##', '').strip()
    
    is_stopword = (
        keyword_lower in STOPWORDS or
        keyword.lower() in STOPWORDS or
        keyword in STOPWORDS
    )
    
    if is_stopword:
        print(f"Warning: '{keyword}' is a stopword, results may not be meaningful")
    
    contexts = []
    
    for idx, chat in enumerate(chats_list):
        if not isinstance(chat, str):
            continue
        
        chat_lower = chat.lower()
        pattern = re.compile(re.escape(keyword_lower), re.IGNORECASE)
        matches = list(pattern.finditer(chat_lower))
        
        for match in matches:
            start_pos = match.start()
            end_pos = match.end()
            
            context_start = max(0, start_pos - window_size)
            context_end = min(len(chat), end_pos + window_size)
            
            before = chat[context_start:start_pos].strip()
            keyword_text = chat[start_pos:end_pos]
            after = chat[end_pos:context_end].strip()
            
            contexts.append({
                'chat_id': idx,
                'position': start_pos,
                'before': before,
                'keyword': keyword_text,
                'after': after,
                'full_text': chat
            })
            
            if len(contexts) >= max_examples:
                break
        
        if len(contexts) >= max_examples:
            break
    
    print(f"\nFound {len(contexts)} occurrences")
    
    if len(contexts) == 0:
        print(f"Keyword '{keyword}' not found")
        return contexts
    
    print(f"Showing {min(max_examples, len(contexts))} examples:\n")
    
    for i, ctx in enumerate(contexts[:max_examples], 1):
        print(f"[Example {i}] Chat ID: {ctx['chat_id']}, Position: {ctx['position']}")
        
        if highlight:
            # ANSI color highlighting for terminal
            print(f"  ...{ctx['before']} \033[1;91m{ctx['keyword']}\033[0m {ctx['after']}...")
        else:
            print(f"  ...{ctx['before']} {ctx['keyword']} {ctx['after']}...")
        print()
    
    return contexts


def visualize_keyword_importance(results_df, top_n=20, save_path=None):
    """Create visualization of keyword importance"""
    
    print("=" * 70)
    print("KEYWORD VISUALIZATION")
    print("=" * 70)
    
    top_keywords = results_df.head(top_n)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Combined score bar chart
    ax1 = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_keywords)))
    ax1.barh(top_keywords['keyword'], top_keywords['combined_score'], color=colors)
    ax1.set_xlabel('Combined Importance Score', fontsize=12)
    ax1.set_title(f'Top {top_n} Keywords by Combined Score', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    
    # 2. Frequency vs importance scatter
    ax2 = axes[0, 1]
    scatter = ax2.scatter(
        top_keywords['frequency'],
        top_keywords['combined_score'],
        s=top_keywords['combined_score_with_freq'] * 100,
        c=range(len(top_keywords)),
        cmap='plasma',
        alpha=0.6,
        edgecolors='black',
        linewidth=1
    )
    ax2.set_xlabel('Frequency', fontsize=12)
    ax2.set_ylabel('Combined Score', fontsize=12)
    ax2.set_title('Frequency vs Importance', fontsize=14, fontweight='bold')
    
    # Add labels for top 5
    for idx, row in top_keywords.head(5).iterrows():
        ax2.annotate(
            row['keyword'],
            (row['frequency'], row['combined_score']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9
        )
    
    # 3. Attention vs gradient comparison
    ax3 = axes[1, 0]
    x = np.arange(len(top_keywords))
    width = 0.35
    
    ax3.bar(x - width/2, top_keywords['avg_attention'], width,
            label='Avg Attention', color='skyblue', alpha=0.8)
    ax3.bar(x + width/2, top_keywords['avg_gradient'], width,
            label='Avg Gradient', color='salmon', alpha=0.8)
    
    ax3.set_xlabel('Keywords', fontsize=12)
    ax3.set_ylabel('Score', fontsize=12)
    ax3.set_title('Attention vs Gradient', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(top_keywords['keyword'], rotation=45, ha='right')
    ax3.legend()
    
    # 4. Cumulative importance
    ax4 = axes[1, 1]
    cumulative = top_keywords['combined_score'].cumsum() / top_keywords['combined_score'].sum() * 100
    ax4.plot(range(1, len(cumulative) + 1), cumulative,
             marker='o', linewidth=2, markersize=6, color='green')
    ax4.axhline(y=80, color='r', linestyle='--', label='80% threshold')
    ax4.fill_between(range(1, len(cumulative) + 1), cumulative, alpha=0.3, color='green')
    ax4.set_xlabel('Number of Keywords', fontsize=12)
    ax4.set_ylabel('Cumulative Importance (%)', fontsize=12)
    ax4.set_title('Cumulative Keyword Importance', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")
    
    plt.show()
    print("=" * 70)


def interpret_tabular_model():
    """Interpret tabular model using SHAP and feature importance"""
    
    print("=" * 70)
    print("TABULAR MODEL INTERPRETATION")
    print("=" * 70)
    
    # Load XGBoost model
    xgb_model = joblib.load(model_dir / "xgb_model.joblib")
    
    # Load test data
    test_df = pd.read_csv(data_dir / "test2.csv")
    
    # Get feature columns (same as used in training)
    feature_cols = [col for col in test_df.columns if col not in ['chat_log', 'Churn']]
    X_test = test_df[feature_cols]
    
    print(f"\nAnalyzing {len(X_test)} samples with {len(feature_cols)} features...")
    
    # Feature importance
    print("\nComputing feature importance...")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Save feature importance
    feature_importance.to_csv(model_dir / 'xgb_feature_importance.csv', index=False)
    print(f"Top 10 important features:")
    print(feature_importance.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(20)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title('Top 20 Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(model_dir / 'xgb_feature_importance.png', dpi=150)
    print("Feature importance plot saved")
    
    # SHAP analysis
    print("\nComputing SHAP values...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test.iloc[:100])  # Use subset for speed
    
    # SHAP summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test.iloc[:100], show=False)
    plt.tight_layout()
    plt.savefig(model_dir / 'shap_summary.png', dpi=150, bbox_inches='tight')
    print("SHAP summary plot saved")
    
    print("=" * 70)
    
    return feature_importance, shap_values


if __name__ == "__main__":
    # Example usage
    print("Running comprehensive interpretation pipeline...\n")
    
    # Text interpretation
    output_dir, attention_results, gradient_results = explain_text_with_attention(max_samples=10)
    
    # Keyword analysis
    results_df = get_important_keywords(
        attention_results=attention_results,
        gradient_results=gradient_results,
        top_k=30
    )
    
    print("\nTop 20 Keywords:")
    print(results_df.head(20))
    
    # Context for top keywords
    test_df = pd.read_csv(data_dir / "test2.csv")
    chats = test_df['chat_log'].dropna().tolist()
    
    top_keyword = results_df.iloc[0]['keyword'] if len(results_df) > 0 else 'payment'
    contexts = obtain_context(chats, top_keyword, max_examples=5)
    
    # Visualize
    visualize_keyword_importance(results_df, top_n=20, save_path=model_dir / 'keyword_importance.png')
    
    # Tabular interpretation
    interpret_tabular_model()
    
    print("\nInterpretation complete!")
