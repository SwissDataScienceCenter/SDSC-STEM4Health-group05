"""Batch Monitoring Studio: our monitoring strategy as an operator workflow.

  0 Data       is the data usable?
  1 Explore    what does it look like? linear or not? correlated or not? which signals separate?
  2 Baseline   label-free PCA with 95% limits: the reference every model must beat
  3 Model      candidate models under nested cross-validation -> recommendation with reasons
  4 Monitor    new batches: two opinions -> accept / review / reject, and why

Demo samples A and B are precomputed (python frontend/precompute.py); uploaded data is computed on the fly.
Run from the repository root:   streamlit run frontend/app.py
"""
import io
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import engine as E

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "scripts" / "results"
st.set_page_config(page_title="Batch Monitoring Studio", page_icon="🧪", layout="wide",
                   initial_sidebar_state="expanded")

# ============================================================================ design tokens
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"
GOOD, BAD, NONLIN = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3 (validated all-pairs)
ZONE = {"accept": "#0ca30c", "review": "#fab219", "reject": "#d03b3b"}  # status colours, always with icon + label
ZONE_ICON = {"accept": "✓", "review": "?", "reject": "✕"}
ZONE_TEXT = {"accept": "#ffffff", "review": "#0b0b0b", "reject": "#ffffff"}
DIVERGING = [[0, "#184f95"], [0.25, "#6da7ec"], [0.5, "#f0efec"], [0.75, "#ec8a86"], [1, "#a8302f"]]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, button, input, textarea {{ font-family: 'Inter', sans-serif; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1320px; }}
h1, h2, h3, h4 {{ letter-spacing: -0.01em; color: {INK}; }}
[data-testid="stSidebar"] {{ background: #0f1b2d; }}
[data-testid="stSidebar"] * {{ color: #e8ecf3 !important; }}
[data-testid="stSidebar"] .stRadio label p {{ font-size: 0.95rem; }}
[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.12); }}
[data-testid="stSidebar"] [data-baseweb="select"] > div {{ background: #1b2a42 !important;
    border: 1px solid rgba(255,255,255,0.18) !important; border-radius: 10px; }}
[data-testid="stSidebar"] [data-baseweb="select"] svg {{ fill: #e8ecf3 !important; }}
[data-testid="stSidebar"] [data-testid="stSelectbox"] div:has(> input) {{ background: #1b2a42 !important;
    border-color: rgba(255,255,255,0.18) !important; }}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {{ color: {INK} !important; }}
.hero {{ background: linear-gradient(120deg, #0f1b2d 0%, #183459 60%, #1f4f8a 100%); color: #fff;
        border-radius: 18px; padding: 26px 30px; margin-bottom: 18px; }}
.hero h1 {{ color: #fff; font-size: 1.85rem; margin: 0 0 4px 0; font-weight: 700; }}
.hero p {{ color: #c9d6ea; margin: 0; font-size: 1rem; }}
.hero .tag {{ display:inline-block; background: rgba(255,255,255,0.12); border-radius: 999px; padding: 3px 12px;
             font-size: 0.78rem; margin-bottom: 10px; color: #dfe8f5; letter-spacing: 0.03em; }}
.stepper {{ display:flex; gap:8px; margin: 4px 0 20px 0; flex-wrap: wrap; }}
.step {{ flex:1; min-width: 150px; background:#fff; border:1px solid {GRID}; border-radius: 12px; padding: 10px 14px; }}
.step .n {{ font-size: 0.72rem; color:{MUTED}; text-transform: uppercase; letter-spacing: .06em; font-weight:600; }}
.step .t {{ font-weight: 600; color:{INK}; font-size: 0.95rem; }}
.step .q {{ font-size: 0.8rem; color:{INK2}; }}
.step.active {{ border: 2px solid {GOOD}; background: #f2f7fe; }}
.step.done .n {{ color: {GOOD}; }}
.card {{ background:#fff; border:1px solid {GRID}; border-radius: 14px; padding: 18px 20px; margin-bottom: 14px; }}
.card h4 {{ margin: 0 0 6px 0; font-size: 1.02rem; }}
.card p, .card li {{ color: {INK2}; font-size: 0.92rem; }}
.kpis {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 16px; }}
.kpi {{ background:#fff; border:1px solid {GRID}; border-radius: 14px; padding: 14px 16px; }}
.kpi .l {{ font-size: 0.74rem; text-transform: uppercase; letter-spacing: .06em; color:{MUTED}; font-weight:600; }}
.kpi .v {{ font-size: 1.7rem; font-weight: 700; color:{INK}; line-height: 1.2; }}
.kpi .s {{ font-size: 0.8rem; color:{INK2}; }}
.check {{ display:flex; gap:12px; align-items:flex-start; padding: 10px 0; border-bottom: 1px solid #f0efec; }}
.check:last-child {{ border-bottom: none; }}
.ic {{ flex: 0 0 26px; height: 26px; border-radius: 50%; display:flex; align-items:center; justify-content:center;
       font-weight:700; font-size: 0.85rem; }}
.ic.ok {{ background:#e3f5e3; color:#006300; }} .ic.warn {{ background:#fff3d6; color:#8a5a00; }}
.ic.fail {{ background:#fbe3e3; color:#a8302f; }}
.check .t {{ font-weight:600; color:{INK}; font-size: 0.92rem; }} .check .d {{ color:{INK2}; font-size: 0.86rem; }}
.insight {{ background:#fff; border:1px solid {GRID}; border-left: 4px solid {GOOD}; border-radius: 10px;
            padding: 12px 16px; margin-bottom: 10px; }}
.insight .t {{ font-weight: 600; color:{INK}; }} .insight .d {{ color:{INK2}; font-size: 0.88rem; }}
.reco {{ background: linear-gradient(120deg, #f2f7fe, #ffffff); border:2px solid {GOOD}; border-radius: 16px;
         padding: 20px 24px; margin-bottom: 16px; }}
.reco .l {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: .08em; color:{GOOD}; font-weight:700; }}
.reco .m {{ font-size: 1.6rem; font-weight: 700; color:{INK}; margin: 2px 0 8px 0; }}
.reco li {{ color:{INK2}; margin-bottom: 4px; }}
.badge {{ border-radius: 16px; padding: 18px; text-align:center; }}
.badge .z {{ font-size: 1.9rem; font-weight: 800; letter-spacing: .04em; }}
.badge .s {{ font-size: 0.86rem; opacity: .9; }}
.meter {{ margin: 10px 0 14px 0; }}
.meter .top {{ display:flex; justify-content: space-between; font-size: 0.86rem; color:{INK2}; }}
.meter .bar {{ position: relative; height: 10px; background:#f0efec; border-radius: 999px; margin-top: 6px; }}
.meter .fill {{ position:absolute; left:0; top:0; bottom:0; border-radius: 999px; }}
.meter .cut {{ position:absolute; top:-4px; bottom:-4px; width:2px; background:{INK}; }}
.why {{ padding: 10px 0; border-bottom: 1px solid #f0efec; }}
.why:last-child {{ border-bottom: none; }}
.why .s {{ font-weight: 600; color:{INK}; }} .why .d {{ color:{INK2}; font-size: 0.9rem; }}
.pill {{ display:inline-block; border-radius: 999px; padding: 2px 10px; font-size: 0.76rem; font-weight:600;
         background:#f0efec; color:{INK2}; margin-left: 6px; }}
.muted {{ color:{MUTED}; font-size: 0.85rem; }}
div[data-testid="stTabs"] button p {{ font-weight: 600; }}
</style>
""", unsafe_allow_html=True)


# ============================================================================ small UI helpers
def html(s):
    st.markdown(s, unsafe_allow_html=True)


def kpis(items):
    html('<div class="kpis">' + "".join(
        f'<div class="kpi"><div class="l">{l}</div><div class="v">{v}</div><div class="s">{s}</div></div>'
        for l, v, s in items) + "</div>")


def card(title, body):
    html(f'<div class="card"><h4>{title}</h4><p>{body}</p></div>')


def md_bold(s):
    """**x** -> <b>x</b>, `x` -> <code>x</code> for use inside HTML blocks."""
    import re
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", s)


def zone_pill(z):
    return (f'<span style="background:{ZONE[z]};color:{ZONE_TEXT[z]};border-radius:999px;padding:2px 10px;'
            f'font-weight:700;font-size:.8rem">{ZONE_ICON[z]} {z.upper()}</span>')


def style(fig, h=340, legend=True, title=None):
    fig.update_layout(
        height=h, margin=dict(l=8, r=8, t=(34 if title else 6) + (26 if legend else 8), b=8),
        title=dict(text=title, x=0, xanchor="left", y=1, yref="container", yanchor="top", pad=dict(t=8),
                   font=dict(size=14, color=INK)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=SURFACE, font=dict(family="Inter, sans-serif", color=INK2, size=12),
        showlegend=legend, legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                                       bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#fff", bordercolor=GRID, font=dict(color=INK, family="Inter")))
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, ticks="")
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, ticks="")
    return fig


def show(fig, key=None):
    st.plotly_chart(fig, width="stretch", key=key, config={"displayModeBar": False})


STEPS = [("0", "Data", "Is the data usable?"), ("1", "Explore", "What does it look like?"),
         ("2", "Baseline", "How far does a simple model get?"), ("3", "Model", "Which model, and why?"),
         ("4", "Monitor", "Accept, review or reject?")]


def stepper(active):
    html('<div class="stepper">' + "".join(
        f'<div class="step {"active" if i == active else "done" if i < active else ""}">'
        f'<div class="n">Step {n}</div><div class="t">{t}</div><div class="q">{q}</div></div>'
        for i, (n, t, q) in enumerate(STEPS)) + "</div>")


# ============================================================================ data access
@st.cache_resource(show_spinner=False)
def _load_demo(p, mtime):
    with open(HERE / "demo" / f"{p}.pkl", "rb") as f:
        return pickle.load(f)


def load_demo(p):
    """Precomputed sample results; the file time is part of the cache key, so re-running precompute is picked up."""
    path = HERE / "demo" / f"{p}.pkl"
    return dict(_load_demo(p, path.stat().st_mtime)) if path.exists() else None


@st.cache_data(show_spinner=False)
def demo_raw(p):
    d = E.load_sample(p)
    return d["cal"], d["labels"], d["new"]


@st.cache_data(show_spinner=False)
def parse_upload(cal_b, lab_b, new_b):
    cal = pd.read_csv(io.BytesIO(cal_b))
    labels = pd.read_csv(io.BytesIO(lab_b)).set_index("batch_id")["label"]
    labels = labels[labels.index.isin(cal.batch_id.unique())]
    new = pd.read_csv(io.BytesIO(new_b)) if new_b else None
    return cal, labels, new


@st.cache_data(show_spinner=False)
def features_of(cal, labels, new):
    sensors = E.active_sensors(cal)
    F_cal = E.FN.features_all(cal[["batch_id"] + sensors]).loc[labels.index]
    F_new = E.FN.features_all(new[["batch_id"] + sensors])[F_cal.columns] if new is not None else None
    return F_cal, F_new


@st.cache_data(show_spinner="Exploring the data (linearity test with random forests, ~1 min)...")
def upload_explore(cal, labels, new):
    return E.explore(cal, labels, new, linearity_trees=100)


@st.cache_data(show_spinner="Running the label-free baseline...")
def upload_baseline(F_cal, y):
    return E.baseline(F_cal, y)


@st.cache_data(show_spinner="Scoring batches with the recommended model...")
def upload_monitor(F_cal, y, labels, F_new, model):
    return E.monitor(F_cal, y, labels, F_new, model)


# ============================================================================ sidebar
with st.sidebar:
    html('<div style="font-size:1.25rem;font-weight:700;margin:4px 0 2px">🧪 Batch Monitoring Studio</div>'
         '<div style="font-size:.8rem;opacity:.7;margin-bottom:14px">SDSC STEM4Health · Group 05</div>')
    source = st.selectbox("Dataset", ["Sample A · industrial process", "Sample B · penicillin fermentation",
                                      "Upload your own data"])
    st.markdown("---")
    step = st.radio("Strategy step", range(5), format_func=lambda i: f"{STEPS[i][0]} · {STEPS[i][1]}")
    st.markdown("---")
    if source.startswith("Upload"):
        up_cal = st.file_uploader("Calibration data (CSV)", type="csv")
        up_lab = st.file_uploader("Calibration labels (CSV)", type="csv")
        up_new = st.file_uploader("New batches to monitor (CSV, optional)", type="csv")
    html('<div style="font-size:.75rem;opacity:.6;margin-top:10px">Validated results come from nested, '
         'repeated, stratified 5-fold cross-validation. The three zones and the live view are prototypes.</div>')

DEMO = not source.startswith("Upload")
if DEMO:
    P = source.split()[1]
    R = load_demo(P)
    if R is None:
        st.error(f"Demo results for sample {P} not found. Run `python frontend/precompute.py {P}` first.")
        st.stop()
    cal, labels, new = demo_raw(P)
    TITLE = {"A": "Process A · industrial batches, 11 sensors",
             "B": "Process B · simulated penicillin fermentation, 20 variables"}[P]
else:
    if not (up_cal and up_lab):
        html('<div class="hero"><span class="tag">UPLOAD</span><h1>Bring your own process</h1>'
             '<p>Upload calibration data and labels in the sidebar to run the same strategy on a new process.</p></div>')
        card("Expected format", "<b>Calibration data</b>: one row per time step, a <code>batch_id</code> column and one "
             "numeric column per sensor, rows in time order.<br><b>Calibration labels</b>: <code>batch_id</code>, "
             "<code>label</code> with values <code>accepted</code> / <code>rejected</code>.<br><b>New batches</b> "
             "(optional): same columns as the calibration data, no labels. These are monitored in step 4.")
        st.stop()
    cal, labels, new = parse_upload(up_cal.getvalue(), up_lab.getvalue(), up_new.getvalue() if up_new else None)
    R = dict(name="upload", quality=E.quality(cal, labels, new))
    TITLE = f"Uploaded process · {cal.batch_id.nunique()} batches"

y = E.y_of(labels)
F_cal, F_new = features_of(cal, labels, new)


def get(key):
    """Precomputed result for samples, computed (and cached) for uploads."""
    if key in R:
        return R[key]
    if key == "explore":
        R[key] = upload_explore(cal, labels, new)
    elif key == "baseline":
        R[key] = upload_baseline(F_cal, y)
    elif key in ("benchmark", "recommendation"):
        b = st.session_state.get(f"bench_{hash(up_cal.getvalue())}")
        if b is None:
            return None
        R["benchmark"] = b
        R["recommendation"] = E.recommend(b, get("explore"))
    elif key == "monitor":
        rec = get("recommendation")
        if rec is None:
            return None
        R[key] = upload_monitor(F_cal, y, labels, F_new, rec["model"])
    return R.get(key)


html(f'<div class="hero"><span class="tag">{"DEMO SAMPLE" if DEMO else "UPLOADED DATA"}</span>'
     f'<h1>{STEPS[step][1]}</h1><p>{TITLE}</p></div>')
stepper(step)


# ============================================================================ step 0: data
def page_data():
    n_new = new.batch_id.nunique() if new is not None else 0
    lengths = cal.groupby("batch_id").size()
    kpis([("Calibration batches", len(labels), "labelled history"),
          ("Rejected", int(y.sum()), f"{y.mean():.0%} of the history"),
          ("Sensors", len(E.sensors_of(cal)), f"{len(E.active_sensors(cal))} vary over time"),
          ("Batch length", int(lengths.median()), f"time steps (range {lengths.min()}–{lengths.max()})"),
          ("New batches", n_new, "to monitor in step 4")])
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("#### Data quality checklist")
        html('<div class="card">' + "".join(
            f'<div class="check"><div class="ic {c["status"]}">{dict(ok="✓", warn="!", fail="✕")[c["status"]]}</div>'
            f'<div><div class="t">{c["title"]}</div><div class="d">{md_bold(c["detail"])}</div></div></div>'
            for c in R["quality"]) + "</div>")
    with right:
        st.markdown("#### The strategy in one picture")
        html('<div class="card">'
             '<p><b>1 · Explore.</b> Look before modelling: trajectories, correlations, and a linearity test '
             'that decides which model family fits.</p>'
             '<p><b>2 · Baseline.</b> PCA learned on good batches only, with 95% control limits. '
             'It needs no labels and is the bar every model must beat.</p>'
             '<p><b>3 · Model.</b> Candidates compete under nested cross-validation; among those tied with the best, '
             'the one that catches most rejects wins.</p>'
             '<p><b>4 · Monitor.</b> Every new batch gets two opinions: <i>“unlike the good batches?”</i> (PCA) and '
             '<i>“like a known reject?”</i> (supervised). Agreement decides; disagreement goes to a person.</p></div>')
    with st.expander("Preview the raw data"):
        st.dataframe(cal.head(200), width="stretch", height=260)


# ============================================================================ step 1: explore
def page_explore():
    ex = get("explore")
    L = ex["linearity"]
    kpis([("Linearity verdict", L["verdict"].capitalize(), f"linear R² {L['linear']:.2f} · forest R² {L['nonlinear']:.2f}"),
          ("Mean |correlation|", f"{ex['mean_abs_corr']:.2f}", "between sensor levels, good batches"),
          ("PCA components for 80%", f"{ex['k80']} / {ex['n_basic']}", "lower = more compressible"),
          ("Best single feature", f"AUC {ex['separation'].auc.iloc[0]:.2f}", f"<code>{ex['separation'].feature.iloc[0]}</code>")])
    st.markdown("#### What this means for the model")
    cols = st.columns(2)
    for i, (t, d) in enumerate(E.implications(ex)):
        with cols[i % 2]:
            html(f'<div class="insight"><div class="t">{md_bold(t)}</div><div class="d">{md_bold(d)}</div></div>')

    t1, t2, t3, t4, t5 = st.tabs(["Trajectories", "Correlation", "Linearity test", "Separating signals", "Batch map"])
    sensors = ex["sensors"]
    good_ids = list(labels.index[labels == "accepted"])
    bad_ids = list(labels.index[labels == "rejected"])
    t = np.linspace(0, 100, E.N_POINTS)

    with t1:
        c1, c2 = st.columns([2, 1])
        s = c1.selectbox("Sensor", sensors, index=sensors.index("aeration_rate") if "aeration_rate" in sensors else 0)
        rel = c2.toggle("Relative to the batch's own level", value=False,
                        help="Divides each batch by its own level. Hidden dips that are smaller than the "
                             "batch-to-batch differences become visible.")
        j = sensors.index(s)
        band = ex["band_rel"] if rel else ex["band_raw"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=band[2][:, j], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=t, y=band[0][:, j], fill="tonexty", fillcolor="rgba(42,120,214,0.16)",
                                 line=dict(width=0), name="good batches, 5–95%", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=t, y=band[1][:, j], line=dict(color=GOOD, width=2), name="good median"))
        for k, b in enumerate(bad_ids[:25]):
            tr = ex["trajs"][b]
            v = E.relative(tr)[:, j] if rel else tr[:, j]
            fig.add_trace(go.Scatter(x=t, y=v, line=dict(color=BAD, width=1.2), opacity=0.55, name="rejected batches",
                                     legendgroup="bad", showlegend=k == 0, hovertemplate=f"{b}<br>%{{y:.3g}}<extra></extra>"))
        fig.update_xaxes(title="batch progress (%)")
        show(style(fig, 420, title=f"{s}: rejected batches against the good-batch band"))
        st.caption("Batches have different lengths, so time is shown as % of the batch. The blue band holds 90% of "
                   "the good batches at every moment.")

    with t2:
        c1, c2 = st.columns(2)
        for col, M, ttl, cap in [
            (c1, ex["corr"], "Between batches (what PCA uses)",
             "Correlation of each sensor's average level across good batches: when one batch runs high on X, "
             "does it also run high on Y?"),
            (c2, ex["within_corr"], "Within a batch (over time)",
             "Correlation over time inside each batch, averaged over good batches: do X and Y move together during a run?")]:
            fig = go.Figure(go.Heatmap(z=M.values, x=M.columns, y=M.index, zmin=-1, zmax=1, colorscale=DIVERGING,
                                       colorbar=dict(thickness=10, len=0.8),
                                       hovertemplate="%{y} × %{x}<br>r = %{z:.2f}<extra></extra>"))
            fig.update_yaxes(autorange="reversed")
            with col:
                show(style(fig, 470, legend=False, title=ttl))
                st.caption(cap)

    with t3:
        P_ = L["per_feature"]
        c1, c2 = st.columns([3, 2], gap="large")
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[-0.5, 1], y=[-0.5, 1], mode="lines", line=dict(color=AXIS, dash="dash", width=1),
                                     showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=P_.linear, y=P_.nonlinear, mode="markers", text=P_.feature, name="feature",
                                     marker=dict(size=9, color=NONLIN, line=dict(color="#fff", width=1.5)),
                                     hovertemplate="%{text}<br>linear R² %{x:.2f}<br>forest R² %{y:.2f}<extra></extra>"))
            fig.update_xaxes(title="linear model R²", range=[-0.3, 1.02])
            fig.update_yaxes(title="random forest R²", range=[-0.3, 1.02])
            fig.add_annotation(x=0.05, y=0.95, text="above the line: non-linear structure", showarrow=False,
                               font=dict(color=INK2, size=11), xanchor="left")
            fig.add_annotation(x=0.95, y=0.05, text="below the line: linear is enough", showarrow=False,
                               font=dict(color=INK2, size=11), xanchor="right")
            show(style(fig, 440, legend=False, title="Can each feature be predicted from the others?"))
        with c2:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=["Linear (PCA + ridge)", "Non-linear (random forest)"], y=[L["linear"], L["nonlinear"]],
                                 marker_color=[GOOD, NONLIN], text=[f"{L['linear']:.2f}", f"{L['nonlinear']:.2f}"],
                                 textposition="outside", width=0.5,
                                 hovertemplate="%{x}<br>median R² %{y:.2f}<extra></extra>"))
            fig.update_yaxes(range=[0, 1.1], title="median cross-validated R²")
            show(style(fig, 300, legend=False, title="Summary"))
            card("How to read it", "For every per-batch feature (median, percentiles, noise of each sensor), we hide it "
                 "and predict it from the others <b>across good batches</b>, once with a linear model and once with a "
                 "random forest. If the forest wins clearly, the batch-to-batch relationships are non-linear and PCA "
                 "alone (a linear method) will miss them.")

    with t4:
        S = ex["separation"].head(12)[::-1]
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            fig = go.Figure(go.Bar(x=S.auc, y=S.feature, orientation="h", marker_color=GOOD,
                                   text=[f"{a:.2f}" for a in S.auc], textposition="outside",
                                   hovertemplate="%{y}<br>AUC %{x:.2f}<extra></extra>"))
            fig.update_xaxes(range=[0.5, 1.08], title="AUC on its own (0.5 = useless, 1 = perfect)")
            show(style(fig, 460, legend=False, title="Features that separate rejected from good batches"))
        with c2:
            f = st.selectbox("Feature", list(ex["separation"].feature), index=0)
            fig = go.Figure()
            for lab, colr, ids in [("accepted", GOOD, good_ids), ("rejected", BAD, bad_ids)]:
                fig.add_trace(go.Box(y=F_cal.loc[ids, f], name=lab, marker_color=colr, boxpoints="all", jitter=0.4,
                                     pointpos=0, marker=dict(size=6, opacity=0.7), line=dict(width=1.5),
                                     fillcolor="rgba(0,0,0,0)", text=ids,
                                     hovertemplate="%{text}<br>%{y:.4g}<extra></extra>"))
            show(style(fig, 400, legend=False, title=f))
        st.caption("Feature names: sensor + statistic (mean, std, median, p05, p95, noise, relstep = largest sudden "
                   "change relative to the sensor's range). This is a screening view; models combine all features.")

    with t5:
        c1, c2 = st.columns([3, 2], gap="large")
        xy = ex["pca_xy"]
        with c1:
            fig = go.Figure()
            for lab, colr in [("accepted", GOOD), ("rejected", BAD)]:
                d = xy[xy.label == lab]
                fig.add_trace(go.Scatter(x=d.PC1, y=d.PC2, mode="markers", name=lab, text=d.index,
                                         marker=dict(size=9, color=colr, line=dict(color="#fff", width=1.5)),
                                         hovertemplate="%{text}<extra></extra>"))
            fig.update_xaxes(title="PC1")
            fig.update_yaxes(title="PC2")
            show(style(fig, 440, title="Every batch on the first two PCA axes (learned from good batches)"))
        with c2:
            cum = np.cumsum(ex["explained"])
            fig = go.Figure(go.Scatter(x=np.arange(1, len(cum) + 1), y=cum, mode="lines+markers",
                                       line=dict(color=GOOD, width=2), marker=dict(size=7),
                                       hovertemplate="%{x} components<br>%{y:.0%} of variance<extra></extra>"))
            fig.add_hline(y=0.8, line=dict(color=AXIS, dash="dash"))
            fig.update_yaxes(tickformat=".0%", range=[0, 1.02], title="variance explained")
            fig.update_xaxes(title="number of components")
            show(style(fig, 300, legend=False, title="How compressible is the data?"))
            st.caption("If rejected batches already sit apart on this map, a label-free PCA will catch them. "
                       "If they mix with good ones, labels and non-linear models are needed.")


# ============================================================================ step 2: baseline
def page_baseline():
    B = get("baseline")
    m, pb = B["metrics"], B["per_batch"]
    kpis([("F1 (accepted = positive)", f"{m['f1']:.3f}", "leaderboard metric"),
          ("Rejects caught", f"{m['rejects_caught']:.0%}", f"{int(((pb.pred == 1) & (pb.truth == 1)).sum())} of {int(pb.truth.sum())}"),
          ("False alarms", f"{m['false_alarms']:.0%}", "good batches flagged"),
          ("Precision", f"{m['precision']:.3f}", "of batches accepted, truly good"),
          ("Recall", f"{m['recall']:.3f}", "of good batches, accepted")])
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        fig = go.Figure()
        lim = max(pb[["T2", "Q"]].max().max() * 1.3, 3)
        fig.add_shape(type="rect", x0=0.01, y0=0.01, x1=1, y1=1, fillcolor="rgba(12,163,12,0.07)", line_width=0, layer="below")
        for lab, colr, tv in [("accepted", GOOD, 0), ("rejected", BAD, 1)]:
            d = pb[pb.truth == tv]
            fig.add_trace(go.Scatter(x=d.T2.clip(0.01), y=d.Q.clip(0.01), mode="markers", name=f"truly {lab}", text=d.index,
                                     marker=dict(size=9, color=colr, line=dict(color="#fff", width=1.5)),
                                     hovertemplate="%{text}<br>T² %{x:.2f} × limit<br>Q %{y:.2f} × limit<extra></extra>"))
        fig.add_vline(x=1, line=dict(color=INK, dash="dash", width=1))
        fig.add_hline(y=1, line=dict(color=INK, dash="dash", width=1))
        fig.update_xaxes(type="log", title="T² / limit  (unusual combination of known patterns)", range=[-1.3, np.log10(lim)])
        fig.update_yaxes(type="log", title="Q / limit  (pattern never seen in good batches)", range=[-1.3, np.log10(lim)])
        show(style(fig, 470, title="PCA control space: every calibration batch, judged out-of-fold"))
        st.caption("Green area = inside both limits → accepted. Dashed lines = 95% limits.")
    with c2:
        card("How the baseline works",
             "1. Summarise each batch by the mean and spread of every sensor.<br>"
             "2. Learn PCA on <b>good batches only</b>, no labels needed.<br>"
             "3. <b>T²</b> measures how extreme a batch is along normal patterns; <b>Q</b> measures what PCA "
             "cannot explain.<br>4. Limits = 95th percentile on good batches the PCA has not seen.<br>"
             "5. Beyond either limit → rejected.")
        cm = B["confusion"]
        fig = go.Figure(go.Heatmap(z=cm, x=["predicted accepted", "predicted rejected"], y=["truly accepted", "truly rejected"],
                                   colorscale=[[0, "#f2f7fe"], [1, "#1c5cab"]], showscale=False,
                                   text=cm, texttemplate="<b>%{text}</b>", textfont=dict(size=18),
                                   hovertemplate="%{y}, %{x}: %{z}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        show(style(fig, 230, legend=False, title="Confusion matrix"))
    missed = pb[(pb.truth == 1) & (pb.pred == 0)].index
    fa = pb[(pb.truth == 0) & (pb.pred == 1)].index
    st.markdown(f"**Missed rejects ({len(missed)}):** {', '.join(missed) or '–'}  \n"
                f"**False alarms ({len(fa)}):** {', '.join(fa) or '–'}")
    st.caption("Out-of-fold: each batch is scored by a PCA fitted without it (stratified 5-fold), so these numbers "
               "estimate performance on new batches.")


# ============================================================================ step 3: model
def run_upload_benchmark():
    key = f"bench_{hash(up_cal.getvalue())}"
    repeats = st.radio("Validation depth", [1, 3], horizontal=True,
                       format_func=lambda r: "Quick (1 × 5-fold, ~2–5 min)" if r == 1 else "Full (3 × 5-fold, ~10–20 min)")
    if st.button("Run the benchmark", type="primary"):
        bar = st.progress(0.0, text="Starting...")
        b = E.benchmark(F_cal, y, repeats, progress=lambda f, n: bar.progress(f, text=f"{n} done"))
        b.attrs["repeats"] = repeats
        st.session_state[key] = b
        R.pop("benchmark", None)
        st.rerun()


def page_model():
    bench = get("benchmark")
    if bench is None:
        card("Benchmark not run yet", "Every candidate model is trained and tested with nested, repeated, stratified "
             "5-fold cross-validation. Cut-offs are tuned on inner folds only, so the scores are honest.")
        run_upload_benchmark()
        return
    rec = get("recommendation")
    html(f'<div class="reco"><div class="l">Recommended model</div><div class="m">{rec["model"]}</div><ul>'
         + "".join(f"<li>{md_bold(r)}</li>" for r in rec["reasons"]) + "</ul></div>")

    order = bench.sort_values("f1")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        colors = [BAD if n == rec["model"] else (MUTED if n == "PCA baseline" else GOOD) for n in order.index]
        fig = go.Figure(go.Scatter(x=order.f1, y=order.index, mode="markers", marker=dict(size=13, color=colors,
                                   line=dict(color="#fff", width=2)),
                                   error_x=dict(type="data", array=order.f1_sd, color=AXIS, thickness=2, width=4),
                                   hovertemplate="%{y}<br>F1 %{x:.3f}<extra></extra>"))
        fig.update_xaxes(title="validated F1 (± 1 sd)")
        show(style(fig, 360, legend=False, title="Which model scores best?"))
        st.caption("Orange = recommended, grey = label-free baseline.")
    with c2:
        from plotly.subplots import make_subplots
        fig = make_subplots(1, 2, shared_yaxes=True, horizontal_spacing=0.06,
                            subplot_titles=("Rejects caught ↑", "False alarms ↓"))
        fig.add_trace(go.Bar(x=order.rejects_caught, y=order.index, orientation="h", marker_color=GOOD,
                             text=[f"{v:.0%}" for v in order.rejects_caught], textposition="outside",
                             hovertemplate="%{y}<br>caught %{x:.0%}<extra></extra>"), 1, 1)
        fig.add_trace(go.Bar(x=order.false_alarms, y=order.index, orientation="h", marker_color=BAD,
                             text=[f"{v:.0%}" for v in order.false_alarms], textposition="outside",
                             hovertemplate="%{y}<br>false alarms %{x:.0%}<extra></extra>"), 1, 2)
        fig.update_xaxes(tickformat=".0%", range=[0, 1.25], col=1)
        fig.update_xaxes(tickformat=".0%", range=[0, max(order.false_alarms.max() * 1.5, 0.05)], col=2)
        fig.update_annotations(font=dict(size=12, color=INK2))
        show(style(fig, 360, legend=False, title="The trade-off behind the F1"))

    tbl = bench[["labels", "f1", "f1_sd", "recall", "precision", "rejects_caught", "false_alarms", "description"]].copy()
    tbl["labels"] = np.where(tbl["labels"], "yes", "no")
    st.dataframe(tbl.sort_values("f1", ascending=False), width="stretch",
                 column_config={"labels": "needs labels", "f1": st.column_config.NumberColumn("F1", format="%.3f"),
                                "f1_sd": st.column_config.NumberColumn("± sd", format="%.3f"),
                                "recall": st.column_config.NumberColumn(format="%.3f"),
                                "precision": st.column_config.NumberColumn(format="%.3f"),
                                "rejects_caught": st.column_config.ProgressColumn("caught", format="%.2f", min_value=0, max_value=1),
                                "false_alarms": st.column_config.ProgressColumn("false alarms", format="%.2f", min_value=0, max_value=1)})
    st.caption(f"Validation: {R.get('repeats', bench.attrs.get('repeats', ''))} × stratified 5-fold, nested (cut-offs tuned inside the training folds). "
               "Selection rule: among models within one standard deviation of the best F1, the one catching most rejects wins; "
               "ties go to the simpler model.")

    if DEMO:
        with st.expander("Also tested during our research (not re-run here)"):
            hist = RESULTS / "submission_history.csv"
            if hist.exists():
                h = pd.read_csv(hist)
                st.markdown("**Leaderboard history** (score = mean F1 of A and B on the hidden test labels)")
                st.dataframe(h[["timestamp", "a_model", "b_model", "validated_score", "leaderboard_score"]],
                             width="stretch", hide_index=True)
            if P == "A":
                for f, ttl in [("compare_trees_A.csv", "Tree models and XGBoost (process A)"),
                               ("compare_recall_A.csv", "Stricter limits and a third ensemble opinion (process A)")]:
                    if (RESULTS / f).exists():
                        st.markdown(f"**{ttl}**")
                        st.dataframe(pd.read_csv(RESULTS / f), width="stretch", hide_index=True)
            if (RESULTS / "compare_autoencoder.csv").exists():
                st.markdown("**Autoencoders vs PCA (label-free)**")
                st.dataframe(pd.read_csv(RESULTS / "compare_autoencoder.csv").query("process == @P"),
                             width="stretch", hide_index=True)


# ============================================================================ step 4: monitor
def meter(label, v, colr, t=1.0):
    pos = min(v, 3) / 3 * 100
    html(f'<div class="meter"><div class="top"><span>{label}</span><span><b>{v:.2f}</b> × cut-off</span></div>'
         f'<div class="bar"><div class="fill" style="width:{pos:.0f}%;background:{colr}"></div>'
         f'<div class="cut" style="left:{min(t, 3) / 3 * 100:.1f}%"></div></div></div>')


def apply_threshold(S, t):
    """Re-decide zones and the model decision with every cut-off multiplied by t (t = 1: as fitted)."""
    S = S.copy()
    n = (S.pca_risk >= t).astype(int) + (S.sup_risk >= t).astype(int)
    S["zone"] = np.array(E.ZONES)[n.to_numpy()]
    if "decision_risk" in S:
        S["decision"] = np.where(S.decision_risk >= t, "rejected", "accepted")
    return S


def page_monitor():
    M = get("monitor")
    if M is None:
        card("Choose a model first", "Run the benchmark in step 3; the recommended model is then used here.")
        return
    ex = get("explore")
    sensors = ex["sensors"]
    options = (["New batches (unlabelled)"] if "new" in M else []) + ["History (true label known)"]
    c1, c2 = st.columns([2, 3])
    view = c1.radio("Batches", options, horizontal=True,
                    help="History batches are scored by models trained without them, so the true outcome can be compared.")
    S = M["new"] if view.startswith("New") else M["cal"]
    t = c2.slider("Rejection threshold (× cut-off)", 0.25, 4.0, 1.0, 0.05, key=f"thr_{source}",
                  help="Multiplies every cut-off. 1.0 = the model as validated. Lower = more sensitive: more rejects "
                       "caught, more false alarms. Higher = fewer false alarms, more missed rejects.")
    if t != 1.0 and "decision_risk" not in S:
        st.warning("These results predate the threshold option: only the zones follow it. "
                   "Re-run step 4 (python frontend/refresh_monitor.py --force) to update the model decision too.")
    S = apply_threshold(S, t)
    if "truth" in S:
        bad, rej = S.truth == "rejected", S.decision == "rejected"
        c2.caption(f"At {t:.2f}: catches **{int((bad & rej).sum())} of {int(bad.sum())}** known rejects, "
                   f"**{int((~bad & rej).sum())}** false alarms of {int((~bad).sum())} good batches."
                   + ("" if t == 1.0 else " Only 1.0 is validated."))
    elif t != 1.0:
        c2.caption("Only 1.0 is validated. Switch to History to see what this threshold costs.")
    counts = S.zone.value_counts()
    n_rej = int((S.decision == "rejected").sum())
    kpis([("Batches", len(S), view.split(" (")[0].lower()),
          (f"{ZONE_ICON['accept']} Accept", int(counts.get("accept", 0)), "both opinions accept"),
          (f"{ZONE_ICON['review']} Review", int(counts.get("review", 0)), "opinions disagree → a person decides"),
          (f"{ZONE_ICON['reject']} Reject", int(counts.get("reject", 0)), "both opinions reject"),
          ("Model decision", f"{n_rej} rejected", M["model"])])

    left, right = st.columns([3, 2], gap="large")
    with left:
        fig = go.Figure()
        hi = max(S.pca_risk.max(), S.sup_risk.max(), 3) * 1.5
        lo = float(np.clip(np.nanpercentile(np.r_[S.pca_risk, S.sup_risk], 2) / 1.5, 1e-4, 0.05))
        for (x0, x1, y0, y1, z) in [(lo, t, lo, t, "accept"), (t, hi, t, hi, "reject"),
                                    (t, hi, lo, t, "review"), (lo, t, t, hi, "review")]:
            fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, line_width=0, layer="below",
                          fillcolor={"accept": "rgba(12,163,12,0.08)", "review": "rgba(250,178,25,0.12)",
                                     "reject": "rgba(208,59,59,0.09)"}[z])
        for z in E.ZONES:
            d = S[S.zone == z]
            fig.add_trace(go.Scatter(x=d.pca_risk.clip(lo), y=d.sup_risk.clip(lo), mode="markers",
                                     name=f"{ZONE_ICON[z]} {z}", text=d.index,
                                     marker=dict(size=10, color=ZONE[z], line=dict(color="#fff", width=1.5),
                                                 symbol={"accept": "circle", "review": "diamond", "reject": "x"}[z]),
                                     hovertemplate="%{text}<br>PCA %{x:.2f} · supervised %{y:.2f}<extra></extra>"))
        a, b = np.sqrt(lo), np.sqrt(hi)
        for x, yy, txt in [(a, lo * 1.8, "ACCEPT"), (b, hi / 1.8, "REJECT"), (b, lo * 1.8, "REVIEW"), (a, hi / 1.8, "REVIEW")]:
            fig.add_annotation(x=np.log10(x), y=np.log10(yy), text=txt, showarrow=False,
                               font=dict(size=11, color=INK2, family="Inter"), opacity=0.8)
        fig.add_vline(x=t, line=dict(color=INK, dash="dash", width=1))
        fig.add_hline(y=t, line=dict(color=INK, dash="dash", width=1))
        fig.update_xaxes(type="log", range=[np.log10(lo), np.log10(hi)], title="PCA: unlike the good batches? (× limit)")
        fig.update_yaxes(type="log", range=[np.log10(lo), np.log10(hi)],
                         title=f"{M['partner']}: like a known reject? (× cut-off)")
        show(style(fig, 480, title="Two opinions per batch"), key="opinion_map")
    with right:
        rank = S.assign(r=np.maximum(S.pca_risk, S.sup_risk)).sort_values("r", ascending=False)
        default = next((b for b in rank.index if S.loc[b, "zone"] == "review"), rank.index[0])
        batch = st.selectbox("Inspect a batch (sorted by risk)", list(rank.index), index=list(rank.index).index(default),
                             format_func=lambda b: f"{b}  ·  {S.loc[b, 'zone']}")
        r = S.loc[batch]
        z = r.zone
        html(f'<div class="badge" style="background:{ZONE[z]};color:{ZONE_TEXT[z]}">'
             f'<div class="z">{ZONE_ICON[z]} {z.upper()}</div>'
             f'<div class="s">{"Both opinions agree" if z != "review" else "Opinions disagree: send to a person"}</div></div>')
        meter("PCA · unlike good batches", float(r.pca_risk), GOOD, t)
        meter(f"{M['partner']} · like a known reject", float(r.sup_risk), BAD, t)
        html(f'<div class="muted">Recommended model (<b>{M["model"]}</b>) decision: <b>{r.decision}</b>'
             + (f' · true outcome: <b>{r.truth}</b>' if "truth" in r else "") + "</div>")

    # ------------------------------------------------------------------ why
    st.markdown(f"#### Why is {batch} flagged?")
    z_row = r[M["z_cols"]].astype(float)
    traj = ex["trajs"].get(batch)
    contrib, reasons = E.why(z_row, traj, ex["band_raw"], ex["band_rel"], sensors)
    rz = M["rejected_z"].drop(index=batch, errors="ignore")
    near, dist = E.nearest_reject(z_row, rz)
    c1, c2 = st.columns([2, 3], gap="large")
    with c1:
        if reasons:
            items = "".join(
                f'<div class="why"><div class="s">{x["sensor"]} <span class="pill">{x["z"]:.1f} spreads from normal</span></div>'
                f'<div class="d">{x["text"]}'
                + ((" · throughout the batch" if x["window"][1] - x["window"][0] >= 90 else
                    f' · mostly between <b>{x["window"][0]}%</b> and <b>{x["window"][1]}%</b> of the batch') if x["window"] else "")
                + "</div></div>" for x in reasons)
        else:
            items = '<div class="why"><div class="d">No sensor is more than 2 typical spreads from the good batches.</div></div>'
        sim = (f'<p class="muted" style="margin-top:10px">Most similar known reject: <b>{near}</b> '
               f'(distance {dist:.1f}; below ~2 is very similar)</p>' if near else "")
        html(f'<div class="card"><h4>In plain words</h4>{items}{sim}</div>')
        top = contrib.head(8)[::-1]
        fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h", marker_color=GOOD,
                               hovertemplate="%{y}: %{x:.1f}<extra></extra>"))
        fig.add_vline(x=2, line=dict(color=AXIS, dash="dash"))
        fig.update_xaxes(title="spreads from normal")
        show(style(fig, 280, legend=False, title="Sensor contributions"))
    with c2:
        default_s = [x["sensor"] for x in reasons if x["sensor"] in sensors][:2] or [s for s in contrib.index if s in sensors][:2]
        shown = st.multiselect("Sensors", sensors, default=default_s)
        rel = st.toggle("Relative to the batch's own level", value=bool(reasons) and reasons[0]["feature"].endswith("relstep"))
        k = st.slider("Replay the batch", 10, E.N_POINTS, E.N_POINTS, 10, format="%d / 200 points",
                      help="Shows how the batch looked while it was running.")
        band = ex["band_rel"] if rel else ex["band_raw"]
        tt = np.linspace(0, 100, E.N_POINTS)
        for s in shown:
            j = sensors.index(s)
            v = E.relative(traj)[:, j] if rel else traj[:, j]
            out = (v < band[0][:, j]) | (v > band[2][:, j])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=tt, y=band[2][:, j], line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=tt, y=band[0][:, j], fill="tonexty", fillcolor="rgba(42,120,214,0.14)",
                                     line=dict(width=0), name="good batches 5–95%", hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=tt, y=band[1][:, j], line=dict(color=GOOD, width=1.2, dash="dot"), name="good median"))
            fig.add_trace(go.Scatter(x=tt[:k], y=v[:k], line=dict(color=INK, width=2), name=batch,
                                     hovertemplate="%{x:.0f}%: %{y:.4g}<extra></extra>"))
            o = np.where(out[:k])[0]
            fig.add_trace(go.Scatter(x=tt[o], y=v[o], mode="markers", marker=dict(color=BAD, size=6),
                                     name="outside the band", hoverinfo="skip"))
            fig.update_xaxes(range=[0, 100], title="batch progress (%)")
            show(style(fig, 260, title=s), key=f"tr_{batch}_{s}_{rel}")

    # ------------------------------------------------------------------ review queue
    st.markdown("#### Review queue")
    q = S[S.zone == "review"].assign(risk=lambda d: np.maximum(d.pca_risk, d.sup_risk)).sort_values("risk", ascending=False)
    key = f"decisions_{source}"
    st.session_state.setdefault(key, {})
    dec = st.session_state[key]
    queue = pd.DataFrame({"batch": q.index, "PCA": q.pca_risk.round(2).values, "supervised": q.sup_risk.round(2).values,
                          "model decision": q.decision.values,
                          "main reason": [E.why(q.loc[b, M["z_cols"]].astype(float))[0].index[0] for b in q.index],
                          "operator decision": [dec.get(b, "") for b in q.index]})
    edited = st.data_editor(queue, hide_index=True, width="stretch",
                            disabled=["batch", "PCA", "supervised", "model decision", "main reason"],
                            column_config={"operator decision": st.column_config.SelectboxColumn(
                                options=["", "accepted", "rejected"])}, key=f"q_{source}_{view}_{t}")
    dec.update({b: d for b, d in zip(edited.batch, edited["operator decision"]) if d})
    done = sum(1 for b in q.index if dec.get(b))
    st.progress(done / max(len(q), 1), text=f"{done} of {len(q)} reviewed")
    st.caption("Designed loop: operator decisions become new labelled batches, and the model is retrained "
               "periodically. A drift guard should stop a slowly drifting process from being learned as normal.")


[page_data, page_explore, page_baseline, page_model, page_monitor][step]()
