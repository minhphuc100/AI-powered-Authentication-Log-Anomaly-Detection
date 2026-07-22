# Authentication Log Anomaly Desktop App

Desktop application that reads real authentication logs and scores them with
the included trained XGBoost model.

## Included runtime components

- `run_desktop_app.py` — application entry point.
- `src/deployment/app.py` — desktop UI, Windows/macOS log collectors, rolling
  feature extraction and model inference.
- `src/models/` — model feature contract and training/evaluation code.
- `models/xgboost_improved.pkl` — trained runtime model used by the app.
- `requirements.txt` — Python dependencies.
- `.vscode/launch.json` — VS Code Python debug configuration.

The app only offers real-log sources:

- **Windows Security Log** — Windows event IDs 4624, 4625 and 4648. Run as
  Administrator for full Security Log access.
- **macOS Unified Log** — authentication-related entries from macOS `/usr/bin/log`.

## Run locally

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_desktop_app.py
```

Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_desktop_app.py
```

On macOS, XGBoost requires the OpenMP runtime once:

```bash
brew install libomp
```

## Push this app directory to Git

Run these commands **inside `authentication-log-anomaly/`**:

```bash
git init
git add run_desktop_app.py src models/xgboost_improved.pkl requirements.txt .gitignore .vscode README.md
git commit -m "Add cross-platform authentication anomaly desktop app"
git branch -M main
git remote add origin <YOUR_GIT_REPOSITORY_URL>
git push -u origin main
```

`.gitignore` excludes virtual environments, build outputs, local log exports
and large raw/processed training data, while retaining the trained runtime model.
