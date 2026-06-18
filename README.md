project/
│
├── data/
│   ├── raw_auth_logs.csv          
│   ├── features.csv               
│   └── splits/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_isolation_forest.ipynb
│   └── 04_improved_xgboost.ipynb
│
├── models/
│   ├── isolation_forest.pkl
│   └── xgboost_model.pkl
│
├── results/
│   ├── class_distribution.png
│   ├── confusion_matrix_if.png
│   ├── confusion_matrix_xgb.png
│   └── metrics_comparison.csv
│
├── app/
│   └── streamlit_app.py
│
├── generate_data.py
├── requirements.txt
└── README.md