# Churn prediction using combined data

This project uses the idea of "aws-samples churn-prediction-with-text-and-interpretability" (from https://github.com/aws-samples/churn-prediction-with-text-and-interpretability/blob/main/README.md), but tries to use the DistilBERT model and XGBoost together instead of Fully Connected Neural Network.

The categorical and numerical data is from Kaggle: E-Commerce Dataset (https://www.kaggle.com/datasets/anaghapaul/e-commerce-dataset), combined with a synthetic text dataset created mainly using Gemini-2.5-flash.

Requirements: 
python 3.9
sentence_transformers==2.0.0
xgboost==1.4.2
numpy==1.24.4
spacy>=3.0.0,<4.0.0
https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.0.0/en_core_web_sm-3.0.0.tar.gz#egg=en_core_web_sm

To be fully uploaded 31 Dec)
