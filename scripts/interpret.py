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
from itertools import combinations

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

def apply_preprocess2(X, artifacts):
    X = X.copy()

    label_encoders = artifacts["label_encoders"]
    scaler = artifacts["scaler"]
    imputer = artifacts["imputer"]
    numeric_columns = artifacts["numeric_columns"]
    categorical_columns = artifacts["categorical_columns"]
    all_columns = artifacts["all_columns"]

    X = X[all_columns]

    # Label encoding
    for col in categorical_columns:
        le = label_encoders[col]
        X[col] = le.transform(X[col].astype(str))

    # Scaling
    X[numeric_columns] = scaler.transform(X[numeric_columns])

    # Impute
    X_imputed = imputer.transform(X)

    return X_imputed

POSITIVE_WORDS = {
    'thank', 'thanks', 'appreciate', 'grateful', 'pleased', 'satisfied',
    'happy', 'great', 'excellent', 'wonderful', 'amazing', 'fantastic',
    'perfect', 'love', 'best', 'good', 'nice', 'helpful', 'friendly',
    'professional', 'quick', 'fast', 'efficient', 'resolved', 'fixed',
    'working', 'success', 'successful', 'recommend', 'awesome',

    'confirm', 'confirmed', 'complete', 'completed', 'delivered',
    'received', 'solved', 'done', 'ready', 'available', 'approved',
}

NEGATIVE_WORDS = {
    'sorry', 'apologize', 'apology', 'unfortunately', 'regret',
    'disappointed', 'frustrat', 'angry', 'upset', 'annoyed', 'terrible',
    'awful', 'horrible', 'worst', 'bad', 'poor', 'unacceptable',
    'ridiculous', 'outrageous', 'disgusted', 'furious',

    'problem', 'issue', 'error', 'wrong', 'mistake', 'fail', 'failed',
    'broken', 'damaged', 'missing', 'lost', 'late', 'delay', 'delayed',
    'cancel', 'cancelled', 'refund', 'return', 'complain', 'complaint',

    'not', 'never', 'nobody', 'nothing', 'nowhere', 'neither',
    "can't", "won't", "don't", "didn't", "couldn't", "wouldn't",
    'unable', 'impossible', 'difficult', 'hard', 'confusing',

    'leave', 'leaving', 'quit', 'stop', 'unsubscribe', 'close',
    'closing', 'terminate', 'switch', 'competitor', 'alternative',
}

INTENSIFIERS = {
    'very': 1.5, 'really': 1.5, 'extremely': 2.0, 'absolutely': 2.0,
    'completely': 1.8, 'totally': 1.8, 'highly': 1.5, 'so': 1.3,
    'too': 1.3, 'quite': 1.2, 'fairly': 1.1, 'incredibly': 2.0,
}

NEGATORS = {'not', "n't", 'no', 'never', 'neither', 'nobody', 'nothing'}


def analyze_sentiment(text, return_details=False):
    if not isinstance(text, str):
        return 0.0, 'neutral', {} if return_details else (0.0, 'neutral')
    
    words = text.lower().split()
    
    positive_count = 0
    negative_count = 0
    positive_matches = []
    negative_matches = []

    for i, word in enumerate(words):
        word_clean = ''.join(c for c in word if c.isalnum())
        
        has_negator = any(
            w in NEGATORS 
            for w in words[max(0, i-3):i]
        )
        
        intensifier = 1.0
        for w in words[max(0, i-2):i]:
            w_clean = ''.join(c for c in w if c.isalnum())
            if w_clean in INTENSIFIERS:
                intensifier = INTENSIFIERS[w_clean]
                break
        
        is_positive = any(word_clean.startswith(pw) or pw in word_clean for pw in POSITIVE_WORDS)
        is_negative = any(word_clean.startswith(nw) or nw in word_clean for nw in NEGATIVE_WORDS)
        
        if is_positive:
            if has_negator:
                negative_count += intensifier
                negative_matches.append(f"NOT {word}")
            else:
                positive_count += intensifier
                positive_matches.append(word if intensifier == 1.0 else f"{word}(x{intensifier})")
        
        if is_negative:
            if has_negator:
                positive_count += intensifier * 0.5 
                positive_matches.append(f"NOT {word}")
            else:
                negative_count += intensifier
                negative_matches.append(word if intensifier == 1.0 else f"{word}(x{intensifier})")
    
    total = positive_count + negative_count
    if total == 0:
        score = 0.0
        label = 'neutral'
    else:
        score = (positive_count - negative_count) / total
        if score > 0.1:
            label = 'positive'
        elif score < -0.1:
            label = 'negative'
        else:
            label = 'neutral'
    
    if return_details:
        details = {
            'positive_count': positive_count,
            'negative_count': negative_count,
            'positive_matches': positive_matches,
            'negative_matches': negative_matches,
            'total_sentiment_words': total
        }
        return score, label, details
    
    return score, label


def analyze_keyword_sentiment(keyword, contexts_list):
    if not contexts_list:
        return {
            'keyword': keyword,
            'sentiment_score': 0.0,
            'sentiment_label': 'neutral',
            'positive_contexts': 0,
            'negative_contexts': 0,
            'neutral_contexts': 0,
            'sample_positive': None,
            'sample_negative': None,
        }
    
    scores = []
    labels = []
    positive_examples = []
    negative_examples = []
    
    for ctx in contexts_list:
        # combine
        full_context = f"{ctx.get('before', '')} {ctx.get('keyword', '')} {ctx.get('after', '')}"
        score, label, details = analyze_sentiment(full_context, return_details=True)
        
        scores.append(score)
        labels.append(label)
        
        if label == 'positive' and len(positive_examples) < 2:
            positive_examples.append(full_context[:100])
        elif label == 'negative' and len(negative_examples) < 2:
            negative_examples.append(full_context[:100])
    
    return {
        'keyword': keyword,
        'sentiment_score': np.mean(scores),
        'sentiment_label': 'positive' if np.mean(scores) > 0.1 else ('negative' if np.mean(scores) < -0.1 else 'neutral'),
        'positive_contexts': labels.count('positive'),
        'negative_contexts': labels.count('negative'),
        'neutral_contexts': labels.count('neutral'),
        'sample_positive': positive_examples[0] if positive_examples else None,
        'sample_negative': negative_examples[0] if negative_examples else None,
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
    
    print("TEXT INTERPRETATION")

    device = torch.device('cpu')
    model_path = model_dir / "combined_model.pth"
    saved_model = torch.load(model_path, map_location=device)
    combined_state = saved_model['combined_model_state_dict']
    
    text_model_state = {}
    for key, value in combined_state.items():
        if key.startswith('text_model.'):
            # Remove 'text_model.' prefix
            new_key = key.replace('text_model.', '')
            text_model_state[new_key] = value
    
    # Load into text model
    text_model = DistilBertTextClassifierWithAttention(num_classes=32)
    text_model.load_state_dict(text_model_state)
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
    
    print(f"Explanations saved to: {output_dir}")
    print(f"Results saved to: {results_path}")
    
    return str(output_dir), attention_results, gradient_results

def get_important_keywords(
    attention_results,
    gradient_results,
    chats_list,
    top_k=30,
    min_count=2,
    window_size=50
):
    """Extract important keywords from interpretation results"""
    
    token_data = defaultdict(lambda: {'attention': [], 'gradient': [], 'count': 0})
    
    for sample_id, token_scores in (attention_results or {}).items():
        for token, score in token_scores.items():
            token_clean = token.lower().replace('##', '')
            if token_clean not in STOPWORDS and token.lower() not in STOPWORDS and token not in STOPWORDS and len(token_clean) > 1:
                token_data[token_clean]['attention'].append(score)
                token_data[token_clean]['count'] += 1
    
    for sample_id, token_scores in (gradient_results or {}).items():
        for token, score in token_scores.items():
            token_clean = token.lower().replace('##', '')
            if token_clean in token_data:
                token_data[token_clean]['gradient'].append(score)
    
    results = []
    for keyword, data in token_data.items():
        if data['count'] < min_count:
            continue
        
        avg_attention = np.mean(data['attention']) if data['attention'] else 0
        avg_gradient = np.mean(data['gradient']) if data['gradient'] else 0
        combined_score = 0.6 * avg_attention + 0.4 * avg_gradient
        
        contexts = []
        for i, chat in enumerate(chats_list[:100]): 
            if not isinstance(chat, str):
                continue
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            for match in pattern.finditer(chat):
                start = max(0, match.start() - window_size)
                end = min(len(chat), match.end() + window_size)
                contexts.append({
                    'before': chat[start:match.start()],
                    'keyword': match.group(),
                    'after': chat[match.end():end],
                    'chat_id': i
                })
                if len(contexts) >= 10:
                    break
            if len(contexts) >= 10:
                break
        
        sentiment_result = analyze_keyword_sentiment(keyword, contexts)
        
        results.append({
            'keyword': keyword,
            'frequency': data['count'],
            'avg_attention': avg_attention,
            'avg_gradient': avg_gradient,
            'combined_score': combined_score,
            'sentiment_score': sentiment_result['sentiment_score'],
            'sentiment_label': sentiment_result['sentiment_label'],
            'positive_contexts': sentiment_result['positive_contexts'],
            'negative_contexts': sentiment_result['negative_contexts'],
        })

    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        results_df = results_df.sort_values('combined_score', ascending=False)
        results_df['rank'] = range(1, len(results_df) + 1)
    
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


def visualize_keyword_importance(results_df, save_path=None):
    import matplotlib.pyplot as plt
    
    if len(results_df) == 0:
        print("No keywords to visualize")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax1 = axes[0, 0]
    colors = ['#2ecc71' if s > 0.1 else '#e74c3c' if s < -0.1 else '#95a5a6' 
              for s in results_df['sentiment_score']]
    ax1.barh(results_df['keyword'].head(15), results_df['sentiment_score'].head(15), color=colors[:15])
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('Sentiment Score')
    ax1.set_title('Keyword Sentiment Scores (Top 15)')
    ax1.invert_yaxis()
    
    ax2 = axes[0, 1]
    sentiment_counts = results_df['sentiment_label'].value_counts()
    colors_pie = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#95a5a6'}
    ax2.pie(sentiment_counts.values, 
            labels=sentiment_counts.index, 
            colors=[colors_pie.get(l, '#95a5a6') for l in sentiment_counts.index],
            autopct='%1.1f%%',
            startangle=90)
    ax2.set_title('Sentiment Distribution')
    
    ax3 = axes[1, 0]
    scatter_colors = ['#2ecc71' if s > 0.1 else '#e74c3c' if s < -0.1 else '#95a5a6' 
                      for s in results_df['sentiment_score']]
    scatter = ax3.scatter(results_df['combined_score'], 
                         results_df['sentiment_score'],
                         c=scatter_colors, 
                         s=results_df['frequency'] * 20,
                         alpha=0.6)
    ax3.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax3.set_xlabel('Importance Score')
    ax3.set_ylabel('Sentiment Score')
    ax3.set_title('Importance vs Sentiment')
    
    for i, row in results_df.head(10).iterrows():
        ax3.annotate(row['keyword'], 
                    (row['combined_score'], row['sentiment_score']),
                    fontsize=8, alpha=0.8)

    ax4 = axes[1, 1]
    x = np.arange(min(10, len(results_df)))
    width = 0.35
    top_keywords = results_df.head(10)
    ax4.bar(x - width/2, top_keywords['positive_contexts'], width, label='Positive', color='#2ecc71')
    ax4.bar(x + width/2, top_keywords['negative_contexts'], width, label='Negative', color='#e74c3c')
    ax4.set_xticks(x)
    ax4.set_xticklabels(top_keywords['keyword'], rotation=45, ha='right')
    ax4.set_ylabel('Context Count')
    ax4.set_title('Positive vs Negative Contexts')
    ax4.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    
    return fig

def create_keyword_summary_report(
    results_df,
    contexts_dict,
    output_path=None,
    top_n=20
):    
    print("=" * 70)
    print("GENERATING KEYWORD SUMMARY REPORT")
    print("=" * 70)
    
    if output_path is None:
        output_path = model_dir / "keyword_analysis_report.html"
    
    top_keywords = results_df.head(top_n)
    
    html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Keyword Analysis Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        h1 {
            color: #2d3748;
            border-bottom: 3px solid #667eea;
            padding-bottom: 15px;
        }
        h2 {
            color: #4a5568;
            margin-top: 30px;
        }
        h3 {
            color: #764ba2;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin: 30px 0;
        }
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-value {
            font-size: 32px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .stat-label {
            font-size: 14px;
            opacity: 0.9;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th {
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }
        td {
            padding: 10px 12px;
            border-bottom: 1px solid #eee;
        }
        tr:hover {
            background: #f5f5f5;
        }
        .keyword-badge {
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 4px 12px;
            border-radius: 15px;
            font-weight: 600;
            margin: 2px;
        }
        .context-box {
            background: #f9f9f9;
            padding: 15px;
            border-left: 4px solid #667eea;
            margin: 10px 0;
            border-radius: 4px;
        }
        .highlight {
            background: #ffeb3b;
            padding: 2px 4px;
            font-weight: bold;
        }
        .rank-1 { background: #FFD700; }
        .rank-2 { background: #C0C0C0; }
        .rank-3 { background: #CD7F32; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Comprehensive Keyword Analysis Report</h1>
"""
    
    # Summary statistics
    total_keywords = len(results_df)
    avg_frequency = results_df['frequency'].mean()
    top_keyword = results_df.iloc[0]['keyword']
    top_score = results_df.iloc[0]['combined_score']
    
    html_content += f"""
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{total_keywords}</div>
                <div class="stat-label">Total Keywords</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{avg_frequency:.1f}</div>
                <div class="stat-label">Avg Frequency</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{top_keyword}</div>
                <div class="stat-label">Top Keyword</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{top_score:.3f}</div>
                <div class="stat-label">Top Score</div>
            </div>
        </div>
        
        <h2>Top {top_n} Most Important Keywords</h2>
        <table>
            <tr>
                <th>Rank</th>
                <th>Keyword</th>
                <th>Frequency</th>
                <th>Avg Attention</th>
                <th>Avg Gradient</th>
                <th>Combined Score</th>
            </tr>
"""
    
    for idx, row in top_keywords.iterrows():
        rank_class = f"rank-{row['rank']}" if row['rank'] <= 3 else ""
        html_content += f"""
            <tr class="{rank_class}">
                <td><strong>#{row['rank']}</strong></td>
                <td><span class="keyword-badge">{row['keyword']}</span></td>
                <td>{row['frequency']}</td>
                <td>{row['avg_attention']:.4f}</td>
                <td>{row['avg_gradient']:.4f}</td>
                <td><strong>{row['combined_score']:.4f}</strong></td>
            </tr>
"""
    
    html_content += """
        </table>
        
        <h2>Keyword Contexts</h2>
"""
    
    # Add context examples
    for keyword, contexts in list(contexts_dict.items())[:10]:
        if not contexts:
            continue
        
        html_content += f"""
        <h3>Keyword: <span class="keyword-badge">{keyword}</span></h3>
"""
        
        for i, ctx in enumerate(contexts[:3], 1):
            highlighted = f"{ctx['before']} <span class='highlight'>{ctx['keyword']}</span> {ctx['after']}"
            html_content += f"""
        <div class="context-box">
            <strong>Example {i}</strong> (Chat ID: {ctx['chat_id']})<br>
            ...{highlighted}...
        </div>
"""
    
    html_content += """
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Report saved to: {output_path}")


def analyze_keyword_cooccurrence(
    chats_list,
    keywords_list,
    min_cooccurrence=2,
    tokenizer=None
):
   
    print("KEYWORD CO-OCCURRENCE ANALYSIS")

    # Count co-occurrences
    cooccurrence_counts = Counter()
    
    for chat in chats_list:
        if not isinstance(chat, str):
            continue
        
        chat_lower = chat.lower()
        
        # Find which keywords appear in this chat
        present_keywords = []
        for keyword in keywords_list:
            keyword_clean = keyword.lower().replace('##', '').strip()
            if keyword_clean in chat_lower:
                present_keywords.append(keyword)
        
        # Count all pairs of keywords that appear together
        if len(present_keywords) >= 2:
            for kw1, kw2 in combinations(sorted(present_keywords), 2):
                cooccurrence_counts[(kw1, kw2)] += 1
    
    # Build results
    results = []
    for (kw1, kw2), count in cooccurrence_counts.items():
        if count >= min_cooccurrence:
            results.append({
                'keyword_1': kw1,
                'keyword_2': kw2,
                'cooccurrence_count': count
            })
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        print("No keyword co-occurrences found")
        return pd.DataFrame()
    
    results_df = results_df.sort_values('cooccurrence_count', ascending=False)
    
    print(f"\nFound {len(results_df)} keyword pairs")
    print(f"\nTop 10 co-occurring keyword pairs:")
    print(results_df.head(10))
    print("=" * 70)
    
    return results_df


def interpret_tabular_model():
    """Interpret tabular XGBoost model using feature importance and SHAP"""

    print("XGBoost model interpretation")

    xgb_model = joblib.load(model_dir / "xgboost_model.joblib")
    artifacts = joblib.load(model_dir / "preprocess_artifacts.joblib")

    # Load test data
    test_df = pd.read_csv(data_dir / "test2.csv")

    X_test_array = apply_preprocess2(
        test_df.drop(columns=["churn"]),
        artifacts
    )

    feature_names = artifacts["all_columns"]

    print(f"\nAnalyzing {X_test_array.shape[0]} samples with {X_test_array.shape[1]} features...")

    # Feature importance
    print("\nComputing feature importance...")

    feature_importance = (
        pd.DataFrame({
            "feature": feature_names,
            "importance": xgb_model.feature_importances_
        })
        .sort_values("importance", ascending=False)
    )

    feature_importance.to_csv("../model/xgb_feature_importance.csv", index=False)

    print("Top 10 important features:")
    print(feature_importance.head(10))

    # Plot feature importance
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(20)
    plt.barh(top_features["feature"], top_features["importance"])
    plt.xlabel("Importance")
    plt.title("Top 20 Feature Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(model_dir / "xgb_feature_importance.png", dpi=150)
    plt.close()

    print("Feature importance plot saved")

    # SHAP
    print("\nComputing SHAP values...")

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test_array[:100])

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        features=X_test_array[:100],
        feature_names=feature_names,
        show=False
    )
    plt.tight_layout()
    plt.savefig(model_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("SHAP summary plot saved")

    return feature_importance, shap_values

if __name__ == "__main__":
    print("Running comprehensive interpretation pipeline...\n")
    
    # Text interpretation
    output_dir, attention_results, gradient_results = explain_text_with_attention(max_samples=10)
    
    test_df = pd.read_csv(data_dir / "test2.csv")
    chats = test_df['chat_log'].dropna().tolist()
    
    # Keyword analysis with sentiment
    results_df = get_important_keywords(
        attention_results=attention_results,
        gradient_results=gradient_results,
        chats_list=chats,
        top_k=30
    )
    
    print("\nTop 20 Keywords with Sentiment:")
    print(results_df.head(20))
    
    # Context for top keywords
    if len(results_df) > 0:
        top_keyword = results_df.iloc[0]['keyword']
        contexts = obtain_context(chats, top_keyword, max_examples=5)
    
    # Visualize with sentiment
    visualize_keyword_importance(results_df, top_n=20, save_path=model_dir / 'keyword_importance.png')
    
    contexts_dict = {}
    for keyword in results_df['keyword'].head(10):
        keyword_contexts = []
        for i, chat in enumerate(chats[:100]):
            if not isinstance(chat, str):
                continue
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            for match in pattern.finditer(chat):
                start = max(0, match.start() - 50)
                end = min(len(chat), match.end() + 50)
                keyword_contexts.append({
                    'before': chat[start:match.start()],
                    'keyword': match.group(),
                    'after': chat[match.end():end],
                    'chat_id': i
                })
                if len(keyword_contexts) >= 5:
                    break
            if len(keyword_contexts) >= 5:
                break
        contexts_dict[keyword] = keyword_contexts
    
    # Create summary report
    create_keyword_summary_report(
        results_df=results_df,
        contexts_dict=contexts_dict,
        output_path=model_dir / "keyword_report.html",
        top_n=20
    )
    
    # Tabular interpretation
    interpret_tabular_model()
    
    print("\nInterpretation complete!")
