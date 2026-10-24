"""
engine.py  ·  Section 5 of the notebook
Model training / evaluation, single-connection inference and bulk traffic scanning.
"""
import os
import time
import random
import tempfile

import numpy as np
import pandas as pd
import gradio as gr
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from threatshield.config import (ALL_COLUMNS, ATTACK_MAP, CATEGORICAL, CLASSES, CLASS_INFO,
                                 FEATURES, MODEL_FACTORY, SLOW_MODELS)
from threatshield.state import S
from threatshield.data_loader import transform_raw
from threatshield.visuals import (plot_model_compare, plot_time_compare, plot_confusion,
                                  plot_importance, plot_proba, plot_scan_summary)
from threatshield.html_components import status_html, model_banner, verdict_html

SLOW_TRAIN_CAP, SLOW_TEST_CAP = 40_000, 30_000

def train_models(selected, fast_mode, progress=gr.Progress()):
    if S["X_train"] is None:
        raise gr.Error("Load a dataset first (Command Center tab).")
    if not selected:
        raise gr.Error("Select at least one model.")
    rng = np.random.RandomState(0)
    Xtr, ytr, Xte, yte = S["X_train"], S["y_train"].values, S["X_test"], S["y_test"].values
    for i, name in enumerate(selected):
        progress(i / len(selected), desc=f"Training {name}…")
        xt, yt, xe, ye = Xtr, ytr, Xte, yte
        capped = False
        if fast_mode and name in SLOW_MODELS:
            ti = rng.choice(len(Xtr), min(SLOW_TRAIN_CAP, len(Xtr)), replace=False)
            ei = rng.choice(len(Xte), min(SLOW_TEST_CAP, len(Xte)), replace=False)
            xt, yt, xe, ye = Xtr[ti], ytr[ti], Xte[ei], yte[ei]; capped = True
        model = MODEL_FACTORY[name]()
        t0 = time.time(); model.fit(xt, yt); t_train = time.time() - t0
        t0 = time.time(); p_tr = model.predict(xt); p_te = model.predict(xe); t_pred = time.time() - t0
        S["models"][name] = model
        S["results"][name] = dict(
            train_acc=accuracy_score(yt, p_tr) * 100, test_acc=accuracy_score(ye, p_te) * 100,
            precision=precision_score(ye, p_te, average="macro", zero_division=0) * 100,
            recall=recall_score(ye, p_te, average="macro", zero_division=0) * 100,
            f1=f1_score(ye, p_te, average="macro", zero_division=0) * 100,
            train_time=t_train, pred_time=t_pred, capped=capped,
            cm=confusion_matrix(ye, p_te, labels=CLASSES))
    progress(1.0, desc="Done")
    trained = list(S["results"])
    best = max(S["results"], key=lambda n: S["results"][n]["test_acc"])
    return (results_table(), plot_model_compare(S["results"]), plot_time_compare(S["results"]),
            gr.update(choices=trained, value=best), plot_confusion(best), plot_importance(best),
            gr.update(choices=trained, value=best), gr.update(choices=trained, value=best),
            model_banner(best), status_html())


def results_table():
    rows = []
    for n, r in S["results"].items():
        rows.append([n + (" *" if r["capped"] else ""), round(r["train_acc"], 2), round(r["test_acc"], 2),
                     round(r["precision"], 2), round(r["recall"], 2), round(r["f1"], 2),
                     round(r["train_time"], 2), round(r["pred_time"], 2)])
    cols = ["Model", "Train Acc %", "Test Acc %", "Precision %", "Recall %", "F1 %", "Train s", "Inference s"]
    return pd.DataFrame(rows, columns=cols).sort_values("Test Acc %", ascending=False)

def show_model_details(name):
    return plot_confusion(name), plot_importance(name)

# ── Single-record inference ───────────────────────────────────
def analyze_record(model_name, *vals):
    if model_name not in S["models"]:
        raise gr.Error("Train at least one model first (Model Lab tab).")
    row = pd.DataFrame([dict(zip(FEATURES, vals))])
    X = transform_raw(row)
    m = S["models"][model_name]
    label = str(m.predict(X)[0])
    if hasattr(m, "predict_proba"):
        pr = m.predict_proba(X)[0]; cls = list(m.classes_)
        conf = float(pr[cls.index(label)])
    else:   # SVM without probability → softmax over decision scores
        d = m.decision_function(X)[0]; e = np.exp(d - d.max()); pr = e / e.sum(); cls = list(m.classes_)
        conf = None
    return verdict_html(label, conf, model_name), plot_proba(cls, pr)

def load_sample(kind):
    if S["raw_test"] is None:
        raise gr.Error("Load a dataset first (Command Center tab).")
    idx = np.where(S["y_test"].values == kind)[0]
    if len(idx) == 0:
        raise gr.Error(f"No '{kind}' samples in the test split.")
    row = S["raw_test"].iloc[int(random.choice(idx))]
    out = []
    for f in FEATURES:
        v = row[f]
        out.append(str(v) if f in CATEGORICAL else float(v))
    return out

# ── Batch / traffic scanning ──────────────────────────────────
def _scan(raw, truth, model_name):
    m = S["models"][model_name]
    preds = m.predict(transform_raw(raw))
    counts = pd.Series(preds).value_counts().to_dict()
    n = len(preds); threats = int((preds != "normal").sum())
    acc_html = ""
    if truth is not None:
        acc = (preds == truth).mean() * 100
        acc_html = f"<div class='ts-kpi'><span>ACCURACY vs. GROUND TRUTH</span><b>{acc:.2f}%</b></div>"
    kpi = f"""<div class="ts-kpis">
      <div class='ts-kpi'><span>RECORDS SCANNED</span><b>{n:,}</b></div>
      <div class='ts-kpi'><span>THREATS FLAGGED</span><b style='color:#f87171'>{threats:,}</b></div>
      <div class='ts-kpi'><span>THREAT RATE</span><b>{threats/max(n,1)*100:.1f}%</b></div>{acc_html}</div>"""
    out = raw.copy().reset_index(drop=True)
    out.insert(0, "verdict", [CLASS_INFO[p]["sev"] for p in preds])
    out.insert(0, "prediction", preds)
    if truth is not None: out.insert(1, "actual", truth)
    path = os.path.join(tempfile.gettempdir(), "threatshield_scan_report.csv")
    out.to_csv(path, index=False)
    return kpi, plot_scan_summary(counts), out.head(500), path

def simulate_traffic(model_name, n):
    if model_name not in S["models"]:
        raise gr.Error("Train at least one model first (Model Lab tab).")
    n = int(min(n, len(S["raw_test"])))
    idx = np.random.RandomState().choice(len(S["raw_test"]), n, replace=False)
    return _scan(S["raw_test"].iloc[idx], S["y_test"].values[idx], model_name)

def scan_file(model_name, file):
    if model_name not in S["models"]:
        raise gr.Error("Train at least one model first (Model Lab tab).")
    if file is None:
        raise gr.Error("Upload a CSV of network connection records.")
    path = file if isinstance(file, str) else file.name
    df = pd.read_csv(path, header=None, low_memory=False)
    first = str(df.iloc[0, 0]).strip()
    if not first.replace(".", "", 1).isdigit():          # has a header row
        df = pd.read_csv(path, low_memory=False)
        if set(FEATURES).issubset(df.columns):
            raw = df[FEATURES]; truth = None
            for c in ("Attack Type", "target"):
                if c in df.columns:
                    t = df[c].astype(str).str.rstrip(".")
                    truth = (t.map(ATTACK_MAP).fillna(t) if c == "target" else t).values; break
            return _scan(raw, truth, model_name)
        df = pd.read_csv(path, header=None, low_memory=False)
    if df.shape[1] not in (41, 42):
        raise gr.Error(f"Expected 41 KDD features (+ optional label) — found {df.shape[1]} columns.")
    truth = None
    if df.shape[1] == 42:
        t = df.iloc[:, 41].astype(str).str.rstrip(".")
        truth = t.map(ATTACK_MAP).fillna(t).values
    df = df.iloc[:, :41]; df.columns = ALL_COLUMNS
    return _scan(df[FEATURES], truth, model_name)
