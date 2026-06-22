import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")


print("=" * 60)
print("ISOLATION FOREST — SYSTEM LOG ANOMALY DETECTION")
print("=" * 60)

df = pd.read_csv("data/processed/features.csv")
print(f"\n[1] Dataset loaded: {len(df)} samples, {df.shape[1]} columns")

le = LabelEncoder()
df["event_type_enc"] = le.fit_transform(df["event_type"].astype(str))

FEATURE_COLS = [
    "event_id",
    "failed_logins_5m_user",
    "unique_src_ip_1h",
    "failed_logins_5m_ip",
    "failure_rate_1h",
    "unique_users_1h_per_ip",
    "hour_of_day",
    "day_of_week",
    "is_business_hours",
    "is_network_logon",
    "event_type_enc",
]

X = df[FEATURE_COLS].fillna(0).values
y_true = df["label"].values       

n_total   = len(X)
n_anomaly = int(y_true.sum())
n_normal  = n_total - n_anomaly
contamination = max(n_anomaly / n_total, 0.01)

print(f"\n[2] Feature matrix: {X.shape}")
print(f"    Normal  : {n_normal}")
print(f"    Anomaly : {n_anomaly}")
print(f"    Contamination estimate: {contamination:.4f}")

iso = IsolationForest(
    n_estimators=300,
    contamination=contamination,
    max_samples="auto",
    random_state=42,
    n_jobs=-1,
)
iso.fit(X)

raw_pred  = iso.predict(X)
scores    = iso.decision_function(X)  

y_pred_bin = (raw_pred == -1).astype(int)  

scores_anomaly = -scores

print("\n" + "=" * 60)
print("[3] CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(
    y_true, y_pred_bin,
    target_names=["Normal (0)", "Anomaly (1)"],
    digits=4,
    zero_division=0,
))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_bin)
tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

print("=" * 60)
print("[4] CONFUSION MATRIX")
print("=" * 60)
print(f"{'':20s}  Pred Normal  Pred Anomaly")
print(f"  Actual Normal  :    {tn:6d}       {fp:6d}")
print(f"  Actual Anomaly :    {fn:6d}       {tp:6d}")

# ROC-AUC & PR-AUC
if n_anomaly > 0 and n_anomaly < n_total:
    roc_auc = roc_auc_score(y_true, scores_anomaly)
    ap      = average_precision_score(y_true, scores_anomaly)
    precision_arr, recall_arr, _ = precision_recall_curve(y_true, scores_anomaly)
    print(f"\n{'=' * 60}")
    print("[5] RANKING METRICS")
    print("=" * 60)
    print(f"  ROC-AUC Score          : {roc_auc:.4f}")
    print(f"  Average Precision (AP) : {ap:.4f}")
else:
    print("\n  (Không đủ nhãn để tính ROC-AUC)")

print(f"\n{'=' * 60}")
print("[6] ANOMALY SCORE STATISTICS (lower raw score = more anomalous)")
print("=" * 60)
print(f"  Normal samples  — mean: {scores[y_true==0].mean():.4f}  "
      f"std: {scores[y_true==0].std():.4f}")
if n_anomaly > 0:
    print(f"  Anomaly samples — mean: {scores[y_true==1].mean():.4f}  "
          f"std: {scores[y_true==1].std():.4f}")

df["anomaly_score"] = scores_anomaly
df["predicted"]     = y_pred_bin

top_anomalies = (
    df[df["predicted"] == 1]
    .sort_values("anomaly_score", ascending=False)
    .head(20)
)

print(f"\n{'=' * 60}")
print("[7] TOP 20 PREDICTED ANOMALIES (by score)")
print("=" * 60)
display_cols = ["timestamp", "username", "src_ip", "event_type",
                "failed_logins_5m_user", "failed_logins_5m_ip",
                "failure_rate_1h", "hour_of_day", "anomaly_score", "label"]
available = [c for c in display_cols if c in df.columns]
print(top_anomalies[available].to_string(index=False))

fp_rows = df[(df["predicted"] == 1) & (df["label"] == 0)]
fn_rows = df[(df["predicted"] == 0) & (df["label"] == 1)]
print(f"\n{'=' * 60}")
print("[8] ERROR ANALYSIS")
print("=" * 60)
print(f"  False Positives (normal flagged as anomaly) : {len(fp_rows)}")
print(f"  False Negatives (anomaly missed)            : {len(fn_rows)}")
if len(fn_rows) > 0:
    print("\n  Missed anomalies:")
    print(fn_rows[available].to_string(index=False))

print(f"\n{'=' * 60}")
print("DONE.")
print("=" * 60)