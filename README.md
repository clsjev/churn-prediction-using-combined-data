# Churn prediction using combined data

This project uses the idea of "aws-samples churn-prediction-with-text-and-interpretability" (https://github.com/aws-samples/churn-prediction-with-text-and-interpretability/tree/main), but rather tries to use the DistilBERT model and XGBoost together instead of a Fully CNN.

The categorical and numerical data (3941 lines) is from Kaggle: E-Commerce Dataset (https://www.kaggle.com/datasets/anaghapaul/e-commerce-dataset), combined with a synthetic text dataset (1184 lines of churn and chat_log) created mainly using Gemini-2.5-flash (but this leads to a very high AUC, so in future it is possible to use neutral or real texts and then mark their churn feature).

After preprocessing they are trained in a combined model, after which they separately predict and interpret the data. See the main notebook "test_churn_prediction" (the cell [20] is redundant).

(In ../notebook another simple XGBoost model is uploaded for comparison.)

The python scripts to prepare the data, train, evaluate, predict and interpret the model, are stored in ../scripts. 

The parameters used for training and interpreting the model are stored in ../model/params.yaml.

