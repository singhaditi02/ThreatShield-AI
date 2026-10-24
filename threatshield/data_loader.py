"""
data_loader.py  ·  Section 3 of the notebook
Loading the KDD Cup 99 dataset, cleaning, encoding, splitting and scaling.
"""
import os
import glob

import pandas as pd
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from threatshield.config import (ALL_COLUMNS, ATTACK_MAP, PROTO_MAP, FLAG_MAP,
                                 FEATURES, CATEGORICAL)
from threatshield.state import S
from threatshield.visuals import plot_attack_dist, plot_protocol, plot_top_labels, plot_flags
from threatshield.html_components import kpi_cards, status_html

NUMERIC_FEATURES = [c for c in FEATURES if c not in CATEGORICAL]

def _clean_frame(df):
    """Coerce dtypes, derive 'Attack Type', drop unknown labels."""
    for c in ["protocol_type", "service", "flag", "target"]:
        df[c] = df[c].apply(lambda v: v.decode() if isinstance(v, bytes) else str(v)).str.strip()
    df["target"] = df["target"].str.rstrip(".")
    for c in ALL_COLUMNS:
        if c not in ("protocol_type", "service", "flag"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Attack Type"] = df["target"].map(ATTACK_MAP)
    df = df.dropna(subset=["Attack Type"]).reset_index(drop=True)
    return df

def read_kdd_file(path):
    df = pd.read_csv(path, names=ALL_COLUMNS + ["target"], header=None)
    return _clean_frame(df)

def download_kdd():
    from sklearn.datasets import fetch_kddcup99
    bunch = fetch_kddcup99(percent10=True, as_frame=False, download_if_missing=True)
    df = pd.DataFrame(bunch.data, columns=ALL_COLUMNS)
    df["target"] = bunch.target
    return _clean_frame(df)

def transform_raw(raw):
    """raw feature frame (strings for protocol/flag) -> scaled numpy matrix."""
    X = raw[FEATURES].copy()
    X["protocol_type"] = X["protocol_type"].astype(str).str.lower().map(PROTO_MAP)
    X["flag"] = X["flag"].astype(str).str.upper().map(FLAG_MAP)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    return S["scaler"].transform(X.values)

def prepare(df, source):
    raw = df[FEATURES]
    y = df["Attack Type"]
    raw_tr, raw_te, y_tr, y_te = train_test_split(raw, y, test_size=0.33, random_state=42)
    X = raw_tr.copy()
    X["protocol_type"] = X["protocol_type"].map(PROTO_MAP)
    X["flag"] = X["flag"].map(FLAG_MAP)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    S["scaler"] = MinMaxScaler().fit(X.values)
    S.update(df=df, raw_train=raw_tr, raw_test=raw_te, y_train=y_tr.reset_index(drop=True),
             y_test=y_te.reset_index(drop=True), source=source, models={}, results={})
    S["X_train"] = S["scaler"].transform(X.values)
    S["X_test"]  = transform_raw(raw_te)
    S["raw_test"] = raw_te.reset_index(drop=True)


def load_data(source, file, progress=gr.Progress()):
    progress(0.1, desc="Loading dataset…")
    if source.startswith("Upload"):
        if file is None:
            path = next(iter(glob.glob("data/kddcup.data_10_percent*") + glob.glob("kddcup.data_10_percent*")), None)
            if path is None: raise gr.Error("Upload the kddcup.data_10_percent_corrected file, or choose the auto-download option.")
        else:
            path = file if isinstance(file, str) else file.name
        df = read_kdd_file(path); src = os.path.basename(path)
    else:
        try:
            df = download_kdd(); src = "KDD Cup 99 · 10% (sklearn)"
        except Exception as e:
            raise gr.Error(f"Download failed ({e}). Upload the dataset file instead.")
    progress(0.6, desc="Preprocessing & scaling…")
    prepare(df, src)
    progress(1.0)
    return (kpi_cards(df), plot_attack_dist(df), plot_protocol(df), plot_top_labels(df), plot_flags(df),
            df.head(12).drop(columns=["target"]), status_html())
