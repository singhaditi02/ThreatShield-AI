"""
ui.py  ·  Section 6 of the notebook
Gradio interface: CSS theme, header, five tabs and the event wiring.
`demo` is the Blocks object that app.py launches.
"""
import gradio as gr

from threatshield.config import CLASSES, CLASS_INFO, FEATURES, CATEGORICAL, FLAG_MAP, MODEL_FACTORY
from threatshield.engine import (SLOW_TRAIN_CAP, SLOW_TEST_CAP, train_models, show_model_details,
                                 analyze_record, load_sample, simulate_traffic, scan_file)
from threatshield.data_loader import load_data
from threatshield.html_components import status_html

CSS = """
:root { --ts-bg:#0b1220; --ts-panel:#111a2e; --ts-line:#1e2a44; --ts-txt:#e2e8f0; --ts-mut:#94a3b8; --ts-acc:#22d3ee; --ts-acc2:#6366f1; }
.gradio-container { max-width: 1280px !important; margin: auto; background: var(--ts-bg) !important; font-family: 'Inter', system-ui, sans-serif !important; }
footer { display: none !important; }

.ts-hero { position:relative; overflow:hidden; display:flex; align-items:center; gap:22px; padding:28px 34px; border-radius:18px;
  background: radial-gradient(1200px 300px at 0% 0%, rgba(34,211,238,.18), transparent 60%),
              radial-gradient(900px 300px at 100% 100%, rgba(99,102,241,.22), transparent 60%), linear-gradient(135deg,#0f1a33,#0b1220);
  border:1px solid var(--ts-line); margin-bottom:12px; }
.ts-hero::after { content:""; position:absolute; inset:0; background-image:linear-gradient(rgba(148,163,184,.05) 1px,transparent 1px),linear-gradient(90deg,rgba(148,163,184,.05) 1px,transparent 1px); background-size:34px 34px; pointer-events:none; }
.ts-logo { width:64px; height:64px; flex:none; filter: drop-shadow(0 0 14px rgba(34,211,238,.55)); }
.ts-brand { font-size:34px; font-weight:800; letter-spacing:-.5px; color:#fff; line-height:1.1; }
.ts-brand span { background:linear-gradient(90deg,#22d3ee,#818cf8); -webkit-background-clip:text; background-clip:text; color:transparent; }
.ts-tag { color:var(--ts-mut); font-size:14px; margin-top:6px; letter-spacing:.3px; }
.ts-badges { margin-left:auto; display:flex; gap:8px; z-index:1; flex-wrap:wrap; justify-content:flex-end; }
.ts-badge { font-size:11px; font-weight:600; letter-spacing:.8px; padding:6px 12px; border-radius:999px; color:#a5f3fc; border:1px solid rgba(34,211,238,.35); background:rgba(34,211,238,.08); }

.ts-status { display:flex; gap:10px; flex-wrap:wrap; margin:0 0 10px 0; }
.pill { font-size:11px; font-weight:700; letter-spacing:.8px; padding:6px 12px; border-radius:999px; border:1px solid var(--ts-line); background:var(--ts-panel); color:var(--ts-mut); }
.pill.ok { color:#4ade80; border-color:rgba(74,222,128,.35); background:rgba(74,222,128,.07); }
.pill.off { color:#fbbf24; border-color:rgba(251,191,36,.35); background:rgba(251,191,36,.07); }

.ts-kpis { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin:6px 0 4px; }
.ts-kpi { background:linear-gradient(180deg,#131f38,#0f1830); border:1px solid var(--ts-line); border-radius:14px; padding:16px 18px; }
.ts-kpi span { display:block; font-size:10.5px; letter-spacing:1.1px; color:var(--ts-mut); font-weight:600; }
.ts-kpi b { display:block; font-size:28px; margin-top:6px; color:#fff; font-weight:800; }

.ts-note { background:rgba(34,211,238,.07); border:1px solid rgba(34,211,238,.25); color:var(--ts-txt); padding:12px 16px; border-radius:12px; font-size:14px; }
.ts-h { color:#fff; font-size:18px; font-weight:700; margin:4px 0 2px; }
.ts-sub { color:var(--ts-mut); font-size:13.5px; margin-bottom:6px; }

.ts-verdict { background:linear-gradient(180deg,#131f38,#0f1830); border:1.5px solid; border-radius:16px; padding:22px; }
.ts-v-head { display:flex; gap:16px; align-items:center; }
.ts-v-icon { font-size:44px; }
.ts-v-title { font-size:22px; font-weight:800; letter-spacing:1px; }
.ts-v-sub { color:var(--ts-txt); font-size:15px; margin-top:2px; }
.ts-v-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin:18px 0 12px; }
.ts-v-grid div { background:rgba(255,255,255,.03); border:1px solid var(--ts-line); border-radius:10px; padding:10px 12px; }
.ts-v-grid span { display:block; font-size:10px; letter-spacing:1px; color:var(--ts-mut); font-weight:600; }
.ts-v-grid b { font-size:16px; color:#fff; }
.ts-v-desc { color:var(--ts-mut); font-size:13.5px; line-height:1.55; margin:0; }

.ts-glossary { display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:12px; }
.ts-card { background:var(--ts-panel); border:1px solid var(--ts-line); border-left:4px solid; border-radius:12px; padding:14px 16px; color:var(--ts-txt); font-size:13.5px; line-height:1.5; }
.ts-card b { display:block; font-size:15px; margin-bottom:4px; }
.ts-steps { color:var(--ts-txt); font-size:14px; line-height:1.9; }
.ts-steps code { background:#0b1220; border:1px solid var(--ts-line); padding:1px 7px; border-radius:6px; color:#67e8f9; }

button.primary, .primary { background:linear-gradient(90deg,#06b6d4,#6366f1) !important; border:none !important; color:#fff !important; font-weight:700 !important; letter-spacing:.4px; }
button.primary:hover { filter:brightness(1.12); transform:translateY(-1px); }
.tab-nav button, button[role="tab"] { font-weight:600 !important; letter-spacing:.3px; }
.ts-foot { text-align:center; color:#64748b; font-size:12px; padding:18px 0 6px; }
"""

LOGO = """<svg class="ts-logo" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#22d3ee"/><stop offset="1" stop-color="#6366f1"/></linearGradient></defs>
<path d="M32 4 8 13v17c0 15 10 26 24 30 14-4 24-15 24-30V13L32 4z" fill="rgba(34,211,238,.08)" stroke="url(#g)" stroke-width="3" stroke-linejoin="round"/>
<path d="M18 33h8l4-9 6 17 4-8h6" fill="none" stroke="url(#g)" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round"/></svg>"""

HEADER = f"""<div class="ts-hero">{LOGO}
  <div><div class="ts-brand">Threat<span>Shield</span> AI</div>
  <div class="ts-tag">Machine-learning powered network intrusion detection · KDD Cup 99</div></div>
  <div class="ts-badges"><span class="ts-badge">6 ML ENGINES</span><span class="ts-badge">5 THREAT CLASSES</span><span class="ts-badge">REAL-TIME ANALYSIS</span></div>
</div>"""

DEFAULTS = dict(protocol_type="tcp", flag="SF", duration=0, src_bytes=181, dst_bytes=5450, logged_in=1, count=8,
                srv_count=8, same_srv_rate=1.0, dst_host_count=9, dst_host_srv_count=9, dst_host_same_src_port_rate=0.11)
ESSENTIAL = ["protocol_type","flag","duration","src_bytes","dst_bytes","logged_in","count","srv_count",
             "serror_rate","rerror_rate","same_srv_rate","dst_host_count","dst_host_srv_count"]

theme = gr.themes.Base(primary_hue=gr.themes.colors.cyan, neutral_hue=gr.themes.colors.slate,
                       font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]).set(
    body_background_fill_dark="#0b1220", background_fill_primary_dark="#111a2e", background_fill_secondary_dark="#0b1220",
    block_background_fill_dark="#111a2e", block_border_color_dark="#1e2a44", border_color_primary_dark="#1e2a44",
    input_background_fill_dark="#0b1220", block_label_background_fill_dark="#111a2e")

with gr.Blocks(theme=theme, css=CSS, title="ThreatShield AI",
               js="() => { document.body.classList.add('dark'); }") as demo:

    gr.HTML(HEADER)
    status = gr.HTML(status_html())

    with gr.Tabs():
        # ───────────── 1 · COMMAND CENTER ─────────────
        with gr.Tab("📊  Command Center"):
            gr.HTML("<div class='ts-h'>Data Ingestion</div><div class='ts-sub'>Load the KDD Cup 99 (10%) dataset — about 494K labelled network connections. Auto-download works out of the box in Colab.</div>")
            with gr.Row():
                src = gr.Radio(["Auto-download KDD Cup 99 (10%)", "Upload dataset file"], value="Auto-download KDD Cup 99 (10%)", label="Data source", scale=2)
                up = gr.File(label="kddcup.data_10_percent_corrected (optional)", scale=2)
            load_btn = gr.Button("⚡  Load & Preprocess Dataset", variant="primary", size="lg")
            kpis = gr.HTML()
            with gr.Row():
                p1 = gr.Plot(label="Threat classes"); p2 = gr.Plot(label="Protocols")
            with gr.Row():
                p3 = gr.Plot(label="Signatures"); p4 = gr.Plot(label="Flags")
            preview = gr.Dataframe(label="Data preview", interactive=False, wrap=False)

        # ───────────── 2 · MODEL LAB ─────────────
        with gr.Tab("🧠  Model Lab"):
            gr.HTML("<div class='ts-h'>Detection Engines</div><div class='ts-sub'>Train and benchmark the same six classifiers from the original notebook on a 67 / 33 split with MinMax scaling.</div>")
            with gr.Row():
                model_sel = gr.CheckboxGroup(list(MODEL_FACTORY), value=["Naive Bayes","Decision Tree","Random Forest","Logistic Regression"],
                                             label="Select engines", scale=3)
                with gr.Column(scale=2):
                    fast = gr.Checkbox(True, label="Fast mode for SVM & Gradient Boosting",
                                       info=f"Trains on {SLOW_TRAIN_CAP:,} rows and tests on {SLOW_TEST_CAP:,} (full run takes ~5+ minutes each).")
                    train_btn = gr.Button("🚀  Train & Evaluate", variant="primary", size="lg")
            banner = gr.HTML()
            table = gr.Dataframe(label="Leaderboard", interactive=False, wrap=True)
            with gr.Row():
                acc_plot = gr.Plot(label="Accuracy"); time_plot = gr.Plot(label="Compute time")
            detail_model = gr.Dropdown([], label="Inspect model")
            with gr.Row():
                cm_plot = gr.Plot(label="Confusion matrix"); imp_plot = gr.Plot(label="Feature importance")

        # ───────────── 3 · THREAT ANALYZER ─────────────
        with gr.Tab("🔍  Threat Analyzer"):
            gr.HTML("<div class='ts-h'>Single-Connection Analysis</div><div class='ts-sub'>Inspect one network connection. Load a real sample from the test set or edit the features manually.</div>")
            with gr.Row():
                an_model = gr.Dropdown([], label="Detection engine", scale=2)
                sample_kind = gr.Dropdown(CLASSES, value="dos", label="Sample class", scale=1)
                sample_btn = gr.Button("🎲  Load random sample", scale=1)
            comps = {}
            def make(f):
                if f == "protocol_type": return gr.Dropdown(["tcp","udp","icmp"], value="tcp", label=f)
                if f == "flag": return gr.Dropdown(list(FLAG_MAP), value="SF", label=f)
                return gr.Number(value=DEFAULTS.get(f, 0), label=f)
            with gr.Accordion("Connection essentials", open=True):
                for i in range(0, len(ESSENTIAL), 4):
                    with gr.Row():
                        for f in ESSENTIAL[i:i+4]: comps[f] = make(f)
            rest = [f for f in FEATURES if f not in ESSENTIAL]
            with gr.Accordion("Advanced features", open=False):
                for i in range(0, len(rest), 4):
                    with gr.Row():
                        for f in rest[i:i+4]: comps[f] = make(f)
            analyze_btn = gr.Button("🛡️  Analyze Connection", variant="primary", size="lg")
            with gr.Row():
                verdict = gr.HTML(); proba = gr.Plot(label="Class probabilities")
            ordered = [comps[f] for f in FEATURES]

        # ───────────── 4 · TRAFFIC SCANNER ─────────────
        with gr.Tab("📡  Traffic Scanner"):
            gr.HTML("<div class='ts-h'>Bulk Traffic Scan</div><div class='ts-sub'>Scan a CSV of KDD-format connection records, or simulate live traffic from the held-out test set.</div>")
            with gr.Row():
                sc_model = gr.Dropdown([], label="Detection engine", scale=2)
                n_sim = gr.Slider(100, 20000, value=2000, step=100, label="Simulated records", scale=2)
                sim_btn = gr.Button("📡  Simulate live traffic", variant="primary", scale=1)
            with gr.Row():
                csv_in = gr.File(label="Upload CSV (41 KDD features, optional label column, header optional)", scale=3)
                scan_btn = gr.Button("🔎  Scan file", variant="primary", scale=1)
            scan_kpi = gr.HTML()
            with gr.Row():
                scan_plot = gr.Plot(label="Composition"); scan_file_out = gr.File(label="Download scan report")
            scan_table = gr.Dataframe(label="Scan results (first 500 rows)", interactive=False)

        # ───────────── 5 · KNOWLEDGE BASE ─────────────
        with gr.Tab("📘  Knowledge Base"):
            cards = "".join(f"<div class='ts-card' style='border-left-color:{v['color']}'><b style='color:{v['color']}'>{v['name']} · {v['sev']}</b>{v['desc']}</div>"
                            for v in CLASS_INFO.values())
            gr.HTML(f"""<div class='ts-h'>Threat Taxonomy</div><div class='ts-sub'>The five traffic classes ThreatShield AI distinguishes.</div>
              <div class='ts-glossary'>{cards}</div>
              <div class='ts-h' style='margin-top:22px'>Processing Pipeline</div>
              <div class='ts-steps'>
              1. Map 23 attack signatures → 5 classes <code>normal · dos · probe · r2l · u2r</code><br>
              2. Encode <code>protocol_type</code> and <code>flag</code>; drop <code>service</code>, constant and highly correlated columns → <b>30 features</b><br>
              3. 67 / 33 train–test split (<code>random_state=42</code>) and MinMax scaling<br>
              4. Train Naive Bayes, Decision Tree, Random Forest, SVM, Logistic Regression, Gradient Boosting<br>
              5. Evaluate accuracy, macro precision / recall / F1 and confusion matrices</div>""")

    gr.HTML("<div class='ts-foot'>ThreatShield AI · Built on the KDD Cup 1999 dataset · For research and educational use</div>")

    # ── Wiring ──
    load_btn.click(load_data, [src, up], [kpis, p1, p2, p3, p4, preview, status])
    train_btn.click(train_models, [model_sel, fast],
                    [table, acc_plot, time_plot, detail_model, cm_plot, imp_plot, an_model, sc_model, banner, status])
    detail_model.change(show_model_details, detail_model, [cm_plot, imp_plot])
    sample_btn.click(load_sample, sample_kind, ordered)
    analyze_btn.click(analyze_record, [an_model] + ordered, [verdict, proba])
    sim_btn.click(simulate_traffic, [sc_model, n_sim], [scan_kpi, scan_plot, scan_table, scan_file_out])
    scan_btn.click(scan_file, [sc_model, csv_in], [scan_kpi, scan_plot, scan_table, scan_file_out])
