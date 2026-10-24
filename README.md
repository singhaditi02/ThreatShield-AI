<div align="center">

# 🛡️ ThreatShield AI

**Machine-learning network intrusion detection with an interactive dashboard**

Python · scikit-learn · Gradio · pandas · matplotlib · KDD Cup 99

</div>

---

## 📖 Overview

ThreatShield AI is an AI-based **Network Intrusion Detection System (NIDS)**. It learns from labelled network connection records (the **KDD Cup 99** dataset) and classifies every connection as normal traffic or one of four attack families. Everything is wrapped in a dark-themed, five-tab web dashboard so you can load data, train and compare models, inspect single connections and scan bulk traffic without writing any code.

### Threat classes

| Class | Meaning | Severity | Example signatures |
|-------|---------|----------|--------------------|
| `normal` | Legitimate traffic | SAFE | – |
| `dos` | Denial of Service – flooding to exhaust resources | HIGH | neptune, smurf, back, teardrop |
| `probe` | Scanning / reconnaissance | MEDIUM | nmap, portsweep, ipsweep, satan |
| `r2l` | Remote-to-Local – unauthorised remote access | HIGH | guess_passwd, warezclient, ftp_write |
| `u2r` | User-to-Root – privilege escalation | CRITICAL | buffer_overflow, rootkit, loadmodule |

### Features

- **One-click data ingestion** – auto-downloads KDD Cup 99 (10 %) or accepts your own file
- **Six ML engines** – Naive Bayes, Decision Tree, Random Forest, Logistic Regression, Gradient Boosting, SVM
- **Model Lab** – leaderboard, accuracy chart, compute-time chart, confusion matrix, feature importance
- **Threat Analyzer** – inspect a single connection, with verdict, severity, confidence and class probabilities
- **Traffic Scanner** – simulate live traffic or upload a CSV, get KPIs, a composition chart and a downloadable report
- **Knowledge Base** – built-in glossary of the threat taxonomy and the processing pipeline

---

## 🗂️ Project structure

```
ThreatShield-AI/
├── app.py                      # Entry point – launches the web app
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .gitignore                  # Keeps data, reports and caches out of Git
├── threatshield/               # Application package
│   ├── __init__.py             # Package marker + version
│   ├── config.py               # Schema, attack map, encoders, model factory, class info
│   ├── state.py                # Shared in-memory state (dataset, scaler, models, results)
│   ├── data_loader.py          # Load → clean → encode → split → scale
│   ├── visuals.py              # Dark theme + all matplotlib charts
│   ├── html_components.py      # Status pills, KPI cards, banner, verdict card
│   ├── engine.py               # Training, evaluation, inference, bulk scanning
│   └── ui.py                   # Gradio layout, CSS and event wiring
├── notebooks/
│   └── ThreatShield_AI.ipynb   # Original Colab notebook (reference)
├── data/                       # Optional: put the KDD file here (git-ignored)
└── reports/                    # Generated outputs (git-ignored)
```

### File-by-file explanation

| File | What it does |
|------|--------------|
| **`app.py`** | Parses `--share` / `--port` options and starts the Gradio server with `demo.queue().launch(...)`. |
| **`threatshield/config.py`** | Holds the 41-column KDD schema (`ALL_COLUMNS`), the 23-signature → 5-class `ATTACK_MAP`, the protocol/flag encoders, the `DROP_COLS` removed during feature selection (leaving 30 `FEATURES`), per-class descriptions and colours (`CLASS_INFO`), and `MODEL_FACTORY` that builds each classifier. |
| **`threatshield/state.py`** | Defines the single dictionary `S` that stores the dataframe, train/test splits, fitted scaler, trained models and metrics so every module shares one source of truth. |
| **`threatshield/data_loader.py`** | `download_kdd()` / `read_kdd_file()` load the data, `_clean_frame()` decodes bytes and maps labels to attack types, `prepare()` encodes, splits (67 / 33, `random_state=42`) and MinMax-scales, `transform_raw()` applies the same transform to new records, `load_data()` is the UI handler for the Command Center. |
| **`threatshield/visuals.py`** | Colour palette plus every chart: class distribution, protocol vs. threat, top signatures, flags, model accuracy, compute time, confusion matrix, feature importance, class probabilities, scan composition. |
| **`threatshield/html_components.py`** | Small HTML builders: status pills, KPI cards, "best performer" banner and the threat verdict card. |
| **`threatshield/engine.py`** | `train_models()` trains and evaluates selected models (with a Fast Mode cap for SVM and Gradient Boosting), `analyze_record()` classifies one connection, `load_sample()` pulls a real test record, `simulate_traffic()` and `scan_file()` run bulk scans and write a CSV report. |
| **`threatshield/ui.py`** | Defines the CSS, header, the five tabs and connects every button to its function. Exposes `demo`. |

---

## ⚙️ How it works (task by task)

1. **Label mapping** – 23 raw KDD attack names are grouped into 5 classes (`normal`, `dos`, `probe`, `r2l`, `u2r`).
2. **Feature selection** – `service` and several constant or highly correlated columns are dropped, leaving **30 features**.
3. **Encoding** – `protocol_type` and `flag` are converted to integers with fixed maps.
4. **Split & scale** – 67 % train / 33 % test (`random_state=42`), then MinMax scaling fitted on the training set only.
5. **Training** – the selected classifiers are fitted and timed; SVM and Gradient Boosting can be trained on a capped subset (Fast Mode).
6. **Evaluation** – accuracy, macro precision / recall / F1 and a confusion matrix per model.
7. **Inference** – a single record or a batch is scaled with the stored scaler, then classified; confidence comes from `predict_proba` (SVM shows `n/a`).

### Models

| Model | Configuration |
|-------|---------------|
| Naive Bayes | `GaussianNB()` |
| Decision Tree | `criterion="entropy"`, `max_depth=4` |
| Random Forest | `n_estimators=30`, `random_state=0` |
| Logistic Regression | `max_iter=2000` |
| Gradient Boosting | `random_state=0` |
| SVM | `SVC(gamma="scale")` |

---

## 🚀 Getting started

### Requirements

- Python **3.9 – 3.12**
- Internet access on first run (to download the dataset)

### Installation

```bash
git clone https://github.com/<your-username>/ThreatShield-AI.git
cd ThreatShield-AI

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Run

```bash
python app.py
```

Open **http://127.0.0.1:7860** in your browser.

| Option | Description |
|--------|-------------|
| `--port 8000` | Use a different local port |
| `--share` | Create a temporary public `gradio.live` link |

---

## 🖥️ Using the app

1. **📊 Command Center** – choose *Auto-download KDD Cup 99 (10 %)* (or upload `kddcup.data_10_percent_corrected`) and click **Load & Preprocess Dataset**. You get KPI cards, four charts and a data preview.
2. **🧠 Model Lab** – tick the engines, click **Train & Evaluate**. Review the leaderboard, charts, confusion matrix and feature importance. Models marked `*` were trained on a capped subset.
3. **🔍 Threat Analyzer** – pick a trained model, load a random sample of a chosen class (or edit values), then **Analyze Connection**.
4. **📡 Traffic Scanner** – click **Simulate live traffic**, or upload a CSV (41 KDD features, optional label column, header optional) and click **Scan file**. Download the report CSV.
5. **📘 Knowledge Base** – read the threat taxonomy and pipeline summary.

> Train at least one model before using the Threat Analyzer or Traffic Scanner.

### Using your own dataset file

Download `kddcup.data_10_percent_corrected` and place it in `data/`, or upload it in the app. If the auto-download fails (no internet), this is the fallback.

---

## 📓 Notebook → module mapping

| Notebook section | Module |
|------------------|--------|
| 1 · Install dependencies | `requirements.txt` |
| 2 · Core configuration & schema | `threatshield/config.py`, `threatshield/state.py` |
| 3 · Data loading & preprocessing | `threatshield/data_loader.py` |
| 4 · Visual theme & charts | `threatshield/visuals.py` |
| 5 · Training & inference engine | `threatshield/engine.py`, `threatshield/html_components.py` |
| 6 · User interface | `threatshield/ui.py` |
| 7 · Launch | `app.py` |

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: gradio` | Activate the virtual environment and run `pip install -r requirements.txt` |
| Dataset download fails | Download the file manually and use *Upload dataset file* |
| Port already in use | `python app.py --port 8001` |
| SVM / Gradient Boosting very slow | Keep **Fast mode** ticked |

---

## ⚠️ Disclaimer

KDD Cup 99 is a 1999 benchmark dataset. ThreatShield AI is built for **research and educational use** and is not a replacement for a production security product.

## 👤 Author

**Vikas Gupta**

## 📚 Acknowledgements

- KDD Cup 1999 dataset (UCI / sklearn `fetch_kddcup99`)
- [scikit-learn](https://scikit-learn.org), [Gradio](https://gradio.app), [pandas](https://pandas.pydata.org), [matplotlib](https://matplotlib.org)
