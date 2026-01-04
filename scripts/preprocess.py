# -*- coding: utf-8 -*-
""" Module for creating the dataset by combining categorical/numerical and text csv. files and preparing the data for the churn prediction model with text. 

Run in CLI example:
    'python preprocess.py --test-size 0.33'

"""


import yaml
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from pathlib import Path
from sklearn.model_selection import train_test_split

with open("../model/params.yaml", "r") as params_file:
    params = yaml.safe_load(params_file)

model_dir = params['model_dir']
   
def create_joint_dataset(df_categorical, df_text):
    result_dfs = []
    
    for churn_val in [0, 1]:
        cat_df = df_categorical[df_categorical['churn'] == churn_val].reset_index(drop=True)
        text_df = df_text[df_text['churn'] == churn_val].reset_index(drop=True)
        
        n_cat = len(cat_df)
        n_text = len(text_df)
        min_len = min(n_cat, n_text)
        
        if min_len == 0:
            print(f"Warning: No samples for churn={churn_val} in one of the datasets. Skipping.")
            continue
        
        cat_subset = cat_df.iloc[:min_len]
        text_subset = text_df.iloc[:min_len]
        
        cat_features = cat_subset.drop(columns=['churn'], errors='ignore')
        
        combined = pd.concat([
            text_subset[['churn', 'chat_log']],
            cat_features
        ], axis=1)
        
        result_dfs.append(combined)
    
    final_df = pd.concat(result_dfs, ignore_index=True)
    return final_df.sample(frac=1, random_state=42).reset_index(drop=True)


def get_feature_names(
    df
):
    num_columns = df.select_dtypes(include=np.number).columns.tolist()
    numerical_feature_names = [i for i in num_columns if i not in ['churn']]

    cat_columns = df.select_dtypes(include='object').columns.tolist()
    categorical_feature_names = [i for i in cat_columns if i not in ['chat_log']]

    textual_feature_names = ['chat_log']
    label_name = 'churn'

    return numerical_feature_names, categorical_feature_names, textual_feature_names, label_name


class ChurnDataset(Dataset):
    def __init__(self, df, tokenizer, numerical_feature_names, categorical_feature_names, max_length=512):
        self.df = df
        self.tokenizer = tokenizer
        self.numerical_feature_names = numerical_feature_names
        self.categorical_feature_names = categorical_feature_names
        self.max_length = max_length

        #  OneHotEncoder
        self.categorical_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.categorical_encoder.fit(df[categorical_feature_names])

        self.numerical_imputer = SimpleImputer(strategy='mean')
        self.numerical_imputer.fit(df[numerical_feature_names])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        chat_log = str(row['chat_log'])
        text_encoding = self.tokenizer(
            chat_log,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        numericals = row[self.numerical_feature_names].values.astype(np.float32)
        numericals = self.numerical_imputer.transform([numericals])[0] 

        categoricals = row[self.categorical_feature_names].values.reshape(1, -1) # reshape for encoder
        categoricals = self.categorical_encoder.transform(categoricals)[0].astype(np.float32)

        label = row['churn']

        return {
            'input_ids': text_encoding['input_ids'].flatten(),
            'attention_mask': text_encoding['attention_mask'].flatten(),
            'numericals': torch.tensor(numericals, dtype=torch.float),
            'categoricals': torch.tensor(categoricals, dtype=torch.float),
            'labels': torch.tensor(label, dtype=torch.long)
        }

    def get_encoders(self):
        # to save
        return self.numerical_imputer, self.categorical_encoder

def run_preprocessing(use_existing=True):
    data_dir = Path(params['data_dir'])
    df = pd.read_csv(data_dir / "churn_dataset.csv")
    df = df.drop_duplicates(subset=['chat_log'], keep='first')

    if not use_existing:
        pass

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=123, stratify=df['churn']
    )

    train_df.to_csv(data_dir / "train2.csv", index=False)
    test_df.to_csv(data_dir / "test2.csv", index=False)
    print(f" Saved to {data_dir}")

if __name__ == "__main__":
    args = parser.parse_args()
    run_preprocessing(
        use_existing=args.use_existing,
        test_size=args.test_size,
        random_state=args.random_state
    )

