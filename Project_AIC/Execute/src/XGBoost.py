
# supervised learning, classification, binary classification, imbalanced dataset
#we will use XGBoost, a powerful boosting algorithm based on decision trees, to improve accuracy and reduce overfitting compared to a single decision tree. XGBoost works by building a series of smaller decision trees that focus on correcting the errors of the previous trees, and then combines them to create a stronger model.


"""

Windows Event Viewer (4624, 4625) the exact same features as the previous dataset, but with a different focus on logon events. The features include:

Feature	Description
success/failure	
logon_type	
hour_of_day	
day_of_week	
failures_last_5min	
failures_last_hour	
logins_per_minute	
unique_users_per_ip	
unique_ips_per_user	
success_after_failures	
authentication_method	

Features like:

failures_last_5min
logins_per_minute
success_after_failures
unique_users_per_ip

need to be engineered from the raw log data, which may require additional processing steps to extract these features from the original log entries. 
Once we have these features, we can use them to train an XGBoost model for binary classification (normal vs attack) and evaluate its performance using metrics like classification report and AUC-ROC score.

Dataset Sources
Option 1 — Los Alamos Authentication Dataset
Pros
Enterprise authentication data
Massive scale
Frequently used in research
Cons
Some preprocessing required
Labels are limited


Option 2 — CERT Insider Threat Dataset
Pros
Better supervised-learning target
Contains malicious behavior
Cons
More insider-threat focused than authentication attacks

the teacher is grading based on:
Probably:

Can you collect and process logs?

Can you engineer useful features?

Can you train a classifier?

Can you explain why a login looks suspicious?

Can you evaluate Precision, Recall, F1?


HDFS is not an authentication dataset ! Because of that, we will use the Los Alamos Authentication Dataset, which contains enterprise authentication data at a massive scale and is frequently used in research. 
However, it may require some preprocessing to extract the relevant features for our classification task.

So we going to do both:

#1. Testing the xgboost on the HDFS dataset, which is not an authentication dataset but can still be used for testing the model's performance on a different type of data.

2. Using the Los Alamos Authentication Dataset or another relevant dataset to train and evaluate the XGBoost model for binary classification of normal vs attack logon events, and to engineer the necessary features from the raw log data.

"""


import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import inspect

# Load data
df = pd.read_csv("auth_logs.csv")

# Encode categorical columns
categorical_cols = [
    'logon_type',
    'day_of_week',
    'authentication_method'
]

df = pd.get_dummies(df, columns=categorical_cols)

# Features
X = df.drop(['label'], axis=1)

# Label
y = df['label']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Class imbalance
n_normal = (y_train == 0).sum()
n_attack = (y_train == 1).sum()

scale_pw = n_normal / n_attack

# Model
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pw,
    eval_metric='auc',
    random_state=42
)

# Early stopping
fit_sig = inspect.signature(model.fit)

if 'early_stopping_rounds' in fit_sig.parameters:
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=20,
        verbose=False
    )
else:
    model.set_params(early_stopping_rounds=20)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

# Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))










