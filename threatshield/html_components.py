"""
html_components.py  ·  small HTML snippets used by the UI
Status pills, KPI cards, best-model banner and the threat verdict card.
"""
from threatshield.config import CLASS_INFO, FEATURES
from threatshield.state import S


def status_html():
    ds = "ok" if S["df"] is not None else "off"
    md = "ok" if S["models"] else "off"
    return f"""<div class="ts-status">
      <span class="pill {ds}">● DATASET {'LOADED · ' + S['source'] if ds=='ok' else 'NOT LOADED'}</span>
      <span class="pill {md}">● {len(S['models'])} MODEL(S) ARMED</span>
      <span class="pill ok">● ENGINE ONLINE</span></div>"""


def kpi_cards(df):
    n = len(df); atk = int((df["Attack Type"] != "normal").sum())
    return f"""<div class="ts-kpis">
      <div class='ts-kpi'><span>TOTAL RECORDS</span><b>{n:,}</b></div>
      <div class='ts-kpi'><span>ATTACK RECORDS</span><b style='color:#f87171'>{atk:,}</b></div>
      <div class='ts-kpi'><span>NORMAL RECORDS</span><b style='color:#4ade80'>{n-atk:,}</b></div>
      <div class='ts-kpi'><span>SIGNATURES</span><b>{df['target'].nunique()}</b></div>
      <div class='ts-kpi'><span>MODEL FEATURES</span><b>{len(FEATURES)}</b></div></div>"""


def model_banner(best):
    r = S["results"][best]
    return (f"<div class='ts-note'>🏆 <b>Best performer:</b> {best} — {r['test_acc']:.2f}% test accuracy, "
            f"{r['f1']:.2f}% macro-F1. &nbsp;<span style='color:#94a3b8'>* trained on a capped subset (Fast Mode).</span></div>")


def verdict_html(label, conf, model_name):
    info = CLASS_INFO[label]; safe = label == "normal"
    icon = "🛡️" if safe else "🚨"
    title = "NO THREAT DETECTED" if safe else "THREAT DETECTED"
    conf_txt = f"{conf*100:.1f}%" if conf is not None else "n/a"
    return f"""
    <div class="ts-verdict" style="border-color:{info['color']};box-shadow:0 0 28px {info['color']}33">
      <div class="ts-v-head"><span class="ts-v-icon">{icon}</span>
        <div><div class="ts-v-title" style="color:{info['color']}">{title}</div>
        <div class="ts-v-sub">{info['name']}</div></div></div>
      <div class="ts-v-grid">
        <div><span>SEVERITY</span><b style="color:{info['color']}">{info['sev']}</b></div>
        <div><span>CONFIDENCE</span><b>{conf_txt}</b></div>
        <div><span>ENGINE</span><b>{model_name}</b></div>
      </div>
      <p class="ts-v-desc">{info['desc']}</p>
    </div>"""
