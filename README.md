```
authentication-log-anomaly
│
├── data
│   ├── raw
│   └── processed
│
├── models
│   ├── xgboost_weighted.pkl
│   └── metrics
│        ├── xgboost_baseline_metrics.csv
│        ├── xgboost_smote_feature_importance.csv
│        ├── xgboost_smote_metrics.csv
│        ├── xgboost_weighted_feature_importance.csv
│        └── xgboost_weight_metric.csv
│
├── results
│   ├── figures
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
├── src
│   ├── data_engineering
│   ├── models
│   └── deployment
│
├── requirements.txt
├── README.md
└── About.txt
