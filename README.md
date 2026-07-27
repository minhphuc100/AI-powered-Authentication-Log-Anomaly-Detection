# AI-Powered Authentication Log Anomaly Detection

An end-to-end machine-learning project for detecting unusual Windows-style
authentication behavior from chronological event logs.

The project parses Los Alamos National Laboratory (LANL) authentication data,
matches known red-team events, builds leakage-aware rolling features, creates
chronological train/validation/test splits, trains an XGBoost binary classifier,
and contains Streamlit and Windows terminal demonstration code.

> This is an academic security-research prototype. The current recorded results
> show very low precision and a large number of false positives, so the system
> is not ready to operate as a standalone production detector.

---

## Project Workflow

```text
LANL auth.txt + redteam.txt
            ↓
Parsing, normalization, and exact red-team matching
            ↓
Chronological rolling feature engineering
            ↓
Three consecutive day-based splits
            ↓
Optional train-only SMOTE augmentation
            ↓
XGBoost training and validation-based threshold selection
            ↓
Test evaluation and saved model bundle
            ↓
LANL replay or experimental Windows Event Log demo
```

---

## Security Task

The project uses supervised binary classification:

| Label | Meaning in this project |
|---|---|
| `0` | No exact match was found in `redteam.txt` |
| `1` | Exact match on `time + src_user + src_computer + dst_computer` |

`label=0` is not proof that an event is benign. It only means that the event
does not match the available red-team ground truth.

The parser also maps the LANL authentication result into a numeric event field:

| LANL result | `event_id` |
|---|---:|
| `Success` | `4624` |
| `Fail` | `4625` |
| Other value | `0` |

The label is copied through feature engineering and splitting. It is never used
as a model input.

---

## Project Structure

```text
.
├── data/
│   ├── raw/
│   │   ├── auth.txt
│   │   └── redteam.txt
│   └── processed/
│       ├── parsed_auth_logs.csv
│       ├── features.csv
│       └── splits/
│           ├── train.csv
│           ├── valid.csv
│           ├── test.csv
│           ├── train_smote.csv
│           └── split_report.json
├── models/
│   ├── metrics/
│   ├── xgboost_weighted.pkl
│   └── xgboost_smote.pkl
├── results/
│   ├── figures/
│   └── metrics/
├── src/
│   ├── data_engineering/
│   │   ├── parser.py
│   │   ├── feature_builder.py
│   │   ├── split_builder.py
│   │   └── smote_builder.py
│   ├── models/
│   │   ├── Xgboostbaseline.py
│   │   ├── Xgboostimproved.py
│   │   └── Xgboostevaluate.py
│   └── deployment/
│       ├── app_simple.py
│       └── terminal_demo.py
├── requirements.txt
└── run_terminal_demo.cmd
```

### Main files

| File | Purpose |
|---|---|
| `src/data_engineering/parser.py` | Parses LANL authentication events and assigns red-team match labels |
| `src/data_engineering/feature_builder.py` | Builds the `auth_anomaly_v2` rolling feature schema |
| `src/data_engineering/split_builder.py` | Creates three chronological day-based splits without shuffling |
| `src/data_engineering/smote_builder.py` | Creates an optional SMOTE-augmented training file |
| `src/models/Xgboostbaseline.py` | Trains an unweighted XGBoost baseline at threshold `0.5` |
| `src/models/Xgboostimproved.py` | Trains weighted or SMOTE-source XGBoost and selects a threshold on validation data |
| `src/models/Xgboostevaluate.py` | Selects the threshold and calculates security metrics |
| `src/deployment/app_simple.py` | Runs the Streamlit LANL replay and experimental Windows modes |
| `src/deployment/terminal_demo.py` | Implements Windows Security Log scoring for the terminal demo |

`feature_builder_legacy9.py` is retained as legacy code and is not part of the
default `auth_anomaly_v2` pipeline.

---

## Dataset

The pipeline uses the LANL
[Comprehensive, Multi-Source Cyber-Security Events](https://csr.lanl.gov/data/cyber1/)
dataset.

Required source files:

```text
data/raw/auth.txt
data/raw/redteam.txt
```

The source dataset provides compressed files named `auth.txt.gz` and
`redteam.txt.gz`. Download them from LANL, extract them, and place the resulting
text files at the paths above.

Raw and processed datasets are excluded from Git because they are large and
must be prepared locally.

### Expected authentication format

```text
time,src_user,dst_user,src_computer,dst_computer,auth_type,logon_type,auth_orientation,result
```

### Expected red-team format

```text
time,src_user,src_computer,dst_computer
```

The current `split_builder.py` expects the feature file to contain exactly three
consecutive relative days. If the complete 58-day LANL authentication file is
used, prepare the intended three-day chronological range before running the
split stage.

---

## Data Engineering

### 1. Parser

`parser.py` reads `auth.txt` in chunks, normalizes fields used downstream,
checks global timestamp ordering, maps the authentication result to `event_id`,
and left-joins red-team labels on four exact keys.

Output:

```text
data/processed/parsed_auth_logs.csv
```

Canonical columns:

```text
timestamp
src_username
username
src_host
destination_host
logon_type
event_id
label
```

### 2. Rolling feature builder

`feature_builder.py` processes events in timestamp order and preserves rolling
state across chunks. For each row, it:

1. Expires history outside the active five-minute or one-hour window.
2. Calculates features from prior events.
3. Writes the current feature row.
4. Adds the current event to the rolling state.

The current event is therefore not included in its own historical counts.

Output:

```text
data/processed/features.csv
```

### 3. Chronological split builder

`split_builder.py` validates the `auth_anomaly_v2` schema and creates:

| Relative day | Split |
|---:|---|
| First | Train |
| Second | Validation |
| Third | Test |

The split is chronological and does not shuffle events.

### 4. Optional SMOTE training data

`smote_builder.py` augments only the training data and writes:

```text
data/processed/splits/train_smote.csv
```

Validation and test data remain unmodified.

---

## Feature Engineering

The `auth_anomaly_v2` schema contains 16 model features.

### Rolling activity

- `auth_attempts_5m_per_src_user`
- `successful_auths_5m_per_src_user`
- `prior_auth_count_1h_src_user_dst_computer`

### Rolling diversity

- `unique_src_computers_1h_per_src_user`
- `unique_dst_computers_5m_per_src_user`
- `unique_dst_computers_1h_per_src_user`
- `unique_dst_users_1h_per_src_computer`
- `unique_src_users_1h_per_dst_computer`

### Novelty and recency

- `is_first_seen_src_user_src_computer`
- `is_first_seen_src_user_dst_computer`
- `is_first_seen_src_computer_dst_computer`
- `seconds_since_last_auth_by_src_user`
- `seconds_since_last_src_user_dst_computer`

The value `-1` represents a missing previous-event time.

### Current-event context

- `current_event_is_success`
- `hour_of_day`
- `is_network_logon`

---

## Model

The deployed classifier is `XGBClassifier` with:

```text
objective = binary:logistic
evaluation metric = aucpr
tree method = hist
random state = 42
```

`Xgboostimproved.py` supports two training sources:

| Option | Training file | Saved bundle |
|---|---|---|
| `original` | `train.csv` | `models/xgboost_weighted.pkl` |
| `smote` | `train_smote.csv` | `models/xgboost_smote.pkl` |

The improved training path:

1. Calculates `scale_pos_weight` as the square root of the class ratio.
2. Trains with validation-based early stopping.
3. Selects the decision threshold on validation data.
4. Targets recall of at least `0.8`, then prioritizes precision.
5. Evaluates the selected model and threshold on the test split.

Each saved bundle contains the fitted model, the ordered 16-feature contract,
the selected threshold, and training metadata.

---

## Recorded Evaluation

The following values come from the current CSV artifacts in `models/metrics/`.
The test split contains `20,105,363` events, including `25` red-team matches.

| Training source | Threshold | Precision | Recall | F1 | PR-AUC | False positives | TP / FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original + class weight | `0.0395` | `0.0000397` | `0.80` | `0.0000793` | `0.0000905` | `504,238` | `20 / 5` |
| SMOTE train + class weight | `0.0549` | `0.0000400` | `0.84` | `0.0000801` | `0.0001349` | `524,525` | `21 / 4` |

Recall is high at the selected thresholds, but precision is extremely low.
These results mean that the recorded models generate hundreds of thousands of
false-positive alerts on the test split.

Accuracy is not emphasized because only 25 of more than 20 million test events
are positive red-team matches.

---

## Installation

Python 3.10 or later is required by the current type syntax.

### Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation for the current session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run all commands below from the repository root.

---

## Run the Full Pipeline

### 1. Parse the raw authentication data

```powershell
python -m src.data_engineering.parser
```

### 2. Build rolling features

```powershell
python -m src.data_engineering.feature_builder
```

### 3. Create chronological train/validation/test splits

```powershell
python -m src.data_engineering.split_builder
```

### 4. Optionally create the SMOTE training file

```powershell
python -m src.data_engineering.smote_builder
```

### 5. Train from the original chronological training split

```powershell
python -m src.models.Xgboostimproved --train-source original
```

### 6. Train from the SMOTE-augmented training split

```powershell
python -m src.models.Xgboostimproved --train-source smote
```

The raw and processed datasets are multi-gigabyte files. Pipeline stages stream
data in chunks where implemented, but training still requires enough local
memory for the selected train, validation, and test feature matrices.

---

## Run the Demonstrations

### Streamlit application

```powershell
python -m streamlit run src/deployment/app_simple.py
```

Available modes:

- **LANL Test Replay**: scores rows from the prepared LANL test split.
- **Windows Live (Experimental)**: reads supported local Windows Security Log
  events and constructs online rolling features.

The default model path is:

```text
models/xgboost_smote.pkl
```

Because `models/*.pkl` is excluded from Git, train the model locally or place a
compatible `auth_anomaly_v2` model bundle at that path before starting the app.

### Windows terminal demo

The intended commands are:

```powershell
run_terminal_demo.cmd --once --show 10
run_terminal_demo.cmd --alerts-only
```

However, the current repository has a known package-entry issue:
`src/deployment/__init__.py` imports a removed `src/deployment/app.py` module.
That import currently prevents `run_terminal_demo.cmd` from starting. Use the
Streamlit application until the package initializer is corrected.

The terminal implementation is Windows-specific, and access to the Windows
Security Log may require an elevated terminal.

---

## Generated Artifacts

| Artifact | Location |
|---|---|
| Parsed authentication events | `data/processed/parsed_auth_logs.csv` |
| Rolling feature table | `data/processed/features.csv` |
| Split validation report | `data/processed/splits/split_report.json` |
| Chronological splits | `data/processed/splits/*.csv` |
| Optional SMOTE report | `data/processed/splits/train_smote.report.json` |
| Model bundles | `models/*.pkl` |
| Metrics | `models/metrics/*_metrics.csv` |
| Feature importance | `models/metrics/*_feature_importance.csv` |

Large data files and trained model bundles are excluded from Git and are
generated locally.

---

## Limitations

- Red-team matches are extremely rare relative to all authentication events.
- `label=0` means no known red-team match; it is not verified benign behavior.
- The recorded models have extremely low precision and very high false-positive
  counts.
- The three-day split represents one narrow chronological range of the larger
  LANL dataset.
- The parser keeps a compact subset of the original authentication fields.
- The Streamlit Windows mode and terminal demo apply a LANL-trained model to
  local Windows events, where the input distribution may differ.
- The terminal wrapper is currently blocked by the stale import in
  `src/deployment/__init__.py`.
- Online first-seen and rolling features depend on the history available during
  the current demo session.
- The project does not automatically block accounts, terminate sessions, or
  modify authentication systems.

---

## Disclaimer

This project is intended for education and academic security research.

A model alert does not prove that an authentication event is malicious, and a
non-alert does not prove that an event is safe. Production use would require
additional telemetry, validation on the target environment, alert triage,
access controls, monitoring, and human analyst review.

---

## About

Course project: **AIC211**

Topic: **AI-Powered Authentication Log Anomaly Detection with XGBoost**
