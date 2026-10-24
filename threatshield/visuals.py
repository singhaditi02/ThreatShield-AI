"""
visuals.py  ·  Section 4 of the notebook
Dark theme colours and every matplotlib chart used in the app.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless backend (Gradio renders the figures)
import matplotlib.pyplot as plt

from threatshield.config import CLASSES, FEATURES
from threatshield.state import S

BG, PANEL, GRID, TXT, MUTED = "#0b1220", "#111a2e", "#1e2a44", "#e2e8f0", "#94a3b8"
ACCENT, ACCENT2 = "#22d3ee", "#6366f1"
CLASS_COLORS = {"normal":"#22c55e","dos":"#f97316","probe":"#eab308","r2l":"#ef4444","u2r":"#dc2626"}
PALETTE = ["#22d3ee","#6366f1","#a78bfa","#f472b6","#34d399","#fbbf24"]

def _style(ax, title=None):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values(): s.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.7); ax.set_axisbelow(True)
    if title: ax.set_title(title, color=TXT, fontsize=12, fontweight="bold", loc="left", pad=12)

def _fig(w=6.2, h=3.6):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=BG)
    return fig, ax

def _empty(msg):
    fig, ax = _fig()
    ax.set_facecolor(BG); ax.axis("off")
    ax.text(0.5, 0.5, msg, ha="center", va="center", color=MUTED, fontsize=11, transform=ax.transAxes)
    return fig

def plot_attack_dist(df):
    vc = df["Attack Type"].value_counts()
    fig, ax = _fig()
    bars = ax.bar(vc.index.str.upper(), vc.values, color=[CLASS_COLORS.get(i, ACCENT) for i in vc.index], width=0.6)
    ax.set_yscale("log"); _style(ax, "Attack-Class Distribution (log scale)")
    for b, v in zip(bars, vc.values):
        ax.text(b.get_x()+b.get_width()/2, v*1.15, f"{v:,}", ha="center", color=TXT, fontsize=9)
    fig.tight_layout(); return fig

def plot_protocol(df):
    ct = pd.crosstab(df["protocol_type"], df["Attack Type"])
    fig, ax = _fig()
    bottom = np.zeros(len(ct))
    for cls in [c for c in CLASSES if c in ct.columns]:
        ax.bar(ct.index.str.upper(), ct[cls].values, bottom=bottom, color=CLASS_COLORS[cls], label=cls.upper(), width=0.55)
        bottom += ct[cls].values
    _style(ax, "Protocol vs. Threat Class")
    leg = ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TXT, fontsize=8)
    fig.tight_layout(); return fig

def plot_top_labels(df):
    vc = df["target"].value_counts().head(10)[::-1]
    fig, ax = _fig()
    ax.barh(vc.index, vc.values, color=ACCENT2, height=0.6)
    ax.set_xscale("log"); _style(ax, "Top 10 Traffic Signatures")
    ax.grid(axis="x", color=GRID, linewidth=0.6); ax.grid(axis="y", visible=False)
    fig.tight_layout(); return fig

def plot_flags(df):
    vc = df["flag"].value_counts()
    fig, ax = _fig()
    ax.bar(vc.index, vc.values, color=ACCENT, width=0.6)
    ax.set_yscale("log"); _style(ax, "Connection Flag Distribution")
    fig.tight_layout(); return fig

def plot_model_compare(res):
    if not res: return _empty("Train models to compare accuracy")
    names = list(res); acc = [res[n]["test_acc"] for n in names]
    fig, ax = _fig(6.4, 3.8)
    bars = ax.bar(names, acc, color=PALETTE[:len(names)], width=0.6)
    lo = max(0, min(acc) - 5); ax.set_ylim(lo, 100.8)
    _style(ax, "Test Accuracy by Model (%)")
    for b, v in zip(bars, acc):
        ax.text(b.get_x()+b.get_width()/2, v+0.15, f"{v:.2f}", ha="center", color=TXT, fontsize=9, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right")
    fig.tight_layout(); return fig

def plot_time_compare(res):
    if not res: return _empty("Train models to compare speed")
    names = list(res)
    fig, ax = _fig(6.4, 3.8)
    x = np.arange(len(names)); w = 0.38
    ax.bar(x-w/2, [res[n]["train_time"] for n in names], w, color=ACCENT, label="Training")
    ax.bar(x+w/2, [res[n]["pred_time"] for n in names], w, color="#f472b6", label="Inference")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=18, ha="right")
    ax.set_yscale("log"); _style(ax, "Compute Time (seconds, log scale)")
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TXT, fontsize=8)
    fig.tight_layout(); return fig

def plot_confusion(name):
    r = S["results"].get(name)
    if r is None: return _empty("Train a model to view its confusion matrix")
    cm = r["cm"]; cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = _fig(5.6, 4.4)
    ax.imshow(cmn, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("ts", [PANEL, "#155e75", ACCENT]), vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES))); ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels([c.upper() for c in CLASSES]); ax.set_yticklabels([c.upper() for c in CLASSES])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); _style(ax, f"Confusion Matrix · {name}")
    ax.grid(False)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center", fontsize=9,
                    color="white" if cmn[i, j] < 0.75 else "#0b1220", fontweight="bold")
    fig.tight_layout(); return fig

def plot_importance(name):
    m = S["models"].get(name)
    if m is None or not hasattr(m, "feature_importances_"):
        return _empty("Feature importance is available for tree-based models\n(Decision Tree · Random Forest · Gradient Boosting)")
    imp = pd.Series(m.feature_importances_, index=FEATURES).sort_values().tail(10)
    fig, ax = _fig(5.6, 4.4)
    ax.barh(imp.index, imp.values, color=ACCENT, height=0.6)
    _style(ax, f"Top Features · {name}")
    ax.grid(axis="x", color=GRID, linewidth=0.6); ax.grid(axis="y", visible=False)
    fig.tight_layout(); return fig

def plot_proba(classes, probs):
    order = np.argsort(probs)
    fig, ax = _fig(5.2, 3.0)
    ax.barh([str(classes[i]).upper() for i in order], [probs[i]*100 for i in order],
            color=[CLASS_COLORS.get(classes[i], ACCENT) for i in order], height=0.55)
    ax.set_xlim(0, 100); _style(ax, "Class Probability (%)")
    ax.grid(axis="x", color=GRID, linewidth=0.6); ax.grid(axis="y", visible=False)
    fig.tight_layout(); return fig

def plot_scan_summary(counts):
    fig, ax = _fig(5.2, 3.4)
    keys = [c for c in CLASSES if counts.get(c, 0) > 0]
    ax.bar([k.upper() for k in keys], [counts[k] for k in keys], color=[CLASS_COLORS[k] for k in keys], width=0.55)
    _style(ax, "Detected Traffic Composition")
    fig.tight_layout(); return fig
