import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd, inspect
# supervised learning, classification, binary classification, imbalanced dataset
#we will use XGBoost, a powerful boosting algorithm based on decision trees, to improve accuracy and reduce overfitting compared to a single decision tree. XGBoost works by building a series of smaller decision trees that focus on correcting the errors of the previous trees, and then combines them to create a stronger model.
df = pd.read_csv('../dataset/data.csv')
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




"""