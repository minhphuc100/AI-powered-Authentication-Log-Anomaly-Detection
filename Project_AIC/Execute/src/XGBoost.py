import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd, inspect

#we will use XGBoost, a powerful boosting algorithm based on decision trees, to improve accuracy and reduce overfitting compared to a single decision tree. XGBoost works by building a series of smaller decision trees that focus on correcting the errors of the previous trees, and then combines them to create a stronger model.
df = pd.read_csv('../dataset/data.csv')