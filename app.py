import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from io import BytesIO
import warnings
import os
warnings.filterwarnings("ignore")

# ── PALETTE ───────────────────────────────────────────────────────────────────

PALETTE = ["#636EFA", "#EF553B", "#00CC96", "#FFB703", "#AB63FA",
           "#FFA15A", "#19D3F3", "#FF6692", "#B6E880", "#FF97FF"]
PALETTE_LIGHT = ["#2B4EFF", "#CC2200", "#009966", "#CC8800", "#7733CC",
                 "#CC5500", "#0099BB", "#CC3366", "#669900", "#CC44CC"]

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="SurvivaLab", page_icon="📈", layout="wide")

# ── LIGHT MODE TOGGLE ─────────────────────────────────────────────────────────

light_mode = st.sidebar.toggle("☀️ Light mode", value=False)

BG_PAGE   = "#ffffff" if light_mode else "#0f0f1a"
BG_FIG    = "#ffffff" if light_mode else "#0f0f1a"
BG_AX     = "#f8f8f8" if light_mode else "#0f0f1a"
BG_BOX    = "#f0f0f0" if light_mode else "#1a1a2e"
COL_SPINE = "#cccccc" if light_mode else "#333333"
COL_TICK  = "#333333" if light_mode else "#aaaaaa"
COL_GRID  = "#dddddd" if light_mode else "#1e1e2e"
COL_TEXT  = "#111111" if light_mode else "#eeeeee"
COL_SUBTEXT = "#444444" if light_mode else "#aaaaaa"
COL_WM    = "#cccccc" if light_mode else "#333333"
COL_LEG_FC= "#f0f0f0" if light_mode else "#1a1a2e"
COL_LEG_EC= "#cccccc" if light_mode else "#333333"
COL_LEG_L = "#111111" if light_mode else "#cccccc"
CURR_PALETTE = PALETTE_LIGHT if light_mode else PALETTE

hero_color  = "#2B4EFF, #009966" if light_mode else "#636EFA, #00CC96"
desc_color  = "#555" if light_mode else "#aaa"
check_ok    = "#007744" if light_mode else "#00CC96"
check_warn  = "#AA6600" if light_mode else "#FFB703"
check_err   = "#CC2200" if light_mode else "#EF553B"
footer_col  = "#888" if light_mode else "#555"
gh_col      = "#444" if light_mode else "#aaa"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800;900&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
.hero-title {{
    font-size: 2.8rem; font-weight: 900; text-align: center;
    background: linear-gradient(135deg, {hero_color});
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 0.3rem;
}}
.hero-desc {{ text-align: center; color: {desc_color}; font-size: 1rem; margin-bottom: 1.5rem; }}
.check-ok   {{ color: {check_ok};   font-weight: 600; }}
.check-warn {{ color: {check_warn}; font-weight: 600; }}
.check-err  {{ color: {check_err};  font-weight: 600; }}
.block-container {{ max-width: 1100px !important; margin: 0 auto !important; padding: 2rem 3rem !important; }}
.github-link {{ color:{gh_col}; text-decoration:none; display:inline-flex; align-items:center; gap:6px; font-weight:600; transition:all 0.2s ease; }}
.github-link:hover {{ color:#ffd700; }}
.github-link svg {{ transition: transform 0.6s ease; }}
.github-link:hover svg {{ transform: rotate(360deg); }}
</style>
""", unsafe_allow_html=True)

# Hero
st.markdown('<div class="hero-title">SurvivaLab</div>', unsafe_allow_html=True)
st.markdown(f'<div class="hero-desc">Carica un CSV · Scegli le colonne · Ottieni curve di sopravvivenza pronte da pubblicare.</div>', unsafe_allow_html=True)
st.markdown("<h2 style='text-align:center;margin-top:-1rem;'>📈</h2>", unsafe_allow_html=True)

# ── UPLOAD ────────────────────────────────────────────────────────────────────

uploaded = st.file_uploader("Carica il tuo file (CSV o Excel)", type=["csv", "xlsx", "xls"])

# ── SAMPLE CSV BADGE ──────────────────────────────────────────────────────────
# Badge con download del file di esempio, subito sotto il file uploader.
# L'utente scarica il file e lo trascina nel box qui sopra.

_SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "sample_data.csv")

if os.path.exists(_SAMPLE_PATH):
    with open(_SAMPLE_PATH, "rb") as _f:
        _sample_bytes = _f.read()

    _badge_bg     = "#f0f0f0" if light_mode else "#1a1a2e"
    _badge_border = "#cccccc" if light_mode else "#2e2e4e"
    _badge_text   = "#333333" if light_mode else "#cccccc"
    _badge_hint   = "#888888" if light_mode else "#666688"

if os.path.exists(_SAMPLE_PATH):
    with open(_SAMPLE_PATH, "rb") as _f:
        _sample_bytes = _f.read()

    import base64
    _b64 = base64.b64encode(_sample_bytes).decode()

    _badge_bg     = "#f0f0f0" if light_mode else "#1a1a2e"
    _badge_border = "#cccccc" if light_mode else "#2e2e4e"
    _badge_text   = "#333333" if light_mode else "#cccccc"
    _badge_hint   = "#888888" if light_mode else "#666688"

    st.markdown(
        f"""
        <style>
        .sample-badge-wrap {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 6px 0 12px 0;
        }}
        .sample-badge {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: {_badge_bg};
            border: 1.5px dashed {_badge_border};
            border-radius: 8px;
            padding: 7px 14px;
            font-size: 0.85rem;
            color: {_badge_text};
        }}
        .sample-badge .icon {{ font-size: 1.1rem; }}
        .dl-link {{
            display: inline-flex;
            align-items: center;
            gap: 5px;
            background: {_badge_bg};
            border: 1.5px solid {_badge_border};
            border-radius: 6px;
            padding: 6px 12px;
            font-size: 0.82rem;
            color: {_badge_text};
            text-decoration: none;
            font-weight: 600;
            transition: border-color 0.2s;
        }}
        .dl-link:hover {{ border-color: #636EFA; color: #636EFA; }}
        .sample-hint {{
            font-size: 0.78rem;
            color: {_badge_hint};
            font-style: italic;
        }}
        </style>
        <div class="sample-badge-wrap">
            <span class="sample-badge">
                <span class="icon">📊</span>
                <strong>sample_data.csv</strong>
            </span>
            <a class="dl-link" href="data:text/csv;base64,{_b64}" download="sample_data.csv">
                ⬇️ Scarica
            </a>
            <span class="sample-hint">← scarica e trascinalo nel box qui sopra per testare</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # st.download_button(
    #     label="⬇️ Scarica sample_data.csv",
    #     data=_sample_bytes,
    #     file_name="sample_data.csv",
    #     mime="text/csv",
    #     key="download_sample",
    # )

# ── STOP SE NESSUN FILE ───────────────────────────────────────────────────────

if not uploaded:
    st.info("⬆️ Carica un CSV o Excel per iniziare. Oppure scarica il file `sample_data.csv` qui sopra e trascinalo nel box.")
    st.stop()

@st.cache_data
def load_file(f, sheet=None):
    name = f.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(f), None
    else:
        xl = pd.ExcelFile(f)
        sheets = xl.sheet_names
        chosen = sheet if sheet else sheets[0]
        return xl.parse(chosen), sheets

is_excel = uploaded.name.lower().endswith((".xlsx", ".xls"))
sheet_choice = None
if is_excel:
    import openpyxl
    raw_bytes = uploaded.read()
    uploaded.seek(0)
    from io import BytesIO as _BIO
    wb = openpyxl.load_workbook(_BIO(raw_bytes), read_only=True, data_only=True)
    sheet_names = wb.sheetnames
    wb.close()
    if len(sheet_names) > 1:
        sheet_choice = st.selectbox("📑 Seleziona foglio Excel", sheet_names)
    else:
        sheet_choice = sheet_names[0]
        st.caption(f"📑 Foglio: **{sheet_choice}**")

df, _ = load_file(uploaded, sheet=sheet_choice)

st.markdown("---")
st.markdown("### 📋 Anteprima dati")
st.dataframe(df.head(10), use_container_width=True)
st.caption(f"{df.shape[0]:,} righe · {df.shape[1]} colonne")

# ── DATA QUALITY ──────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### 🔍 Data Quality Check")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
all_cols     = df.columns.tolist()

checks = []
checks.append(("Righe nel dataset", f"{df.shape[0]:,}",
               "ok" if df.shape[0] >= 30 else "warn",
               f"< 30 righe: risultati poco affidabili"))

dupes = df.duplicated().sum()
checks.append(("Righe duplicate", str(dupes),
               "ok" if dupes == 0 else "warn",
               f"{dupes} righe duplicate trovate"))

total_miss = df.isnull().sum().sum()
pct_miss   = total_miss / df.size * 100
checks.append(("Valori mancanti", f"{total_miss} ({pct_miss:.1f}%)",
               "ok" if pct_miss == 0 else ("warn" if pct_miss < 10 else "err"),
               f"{pct_miss:.1f}% valori mancanti"))

checks.append(("Colonne numeriche", str(len(numeric_cols)),
               "ok" if len(numeric_cols) >= 2 else "err",
               "Servono almeno 2 colonne numeriche"))

icon_map = {"ok": "✅", "warn": "⚠️", "err": "❌"}
cls_map  = {"ok": "check-ok", "warn": "check-warn", "err": "check-err"}

c1, c2, c3, c4 = st.columns(4)
for col_ui, (label, value, status, detail) in zip([c1,c2,c3,c4], checks):
    with col_ui:
        st.markdown(f'<div class="{cls_map[status]}">{icon_map[status]} {label}</div>', unsafe_allow_html=True)
        st.markdown(f"**{value}**")
        if status != "ok":
            st.caption(detail)

if total_miss > 0:
    with st.expander("📊 Dettaglio valori mancanti per colonna"):
        miss_df = df.isnull().sum().reset_index()
        miss_df.columns = ["Colonna", "Mancanti"]
        miss_df["% su totale"] = (miss_df["Mancanti"] / len(df) * 100).round(2)
        miss_df = miss_df[miss_df["Mancanti"] > 0].sort_values("Mancanti", ascending=False)
        st.dataframe(miss_df, use_container_width=True, hide_index=True)

# ── CONFIGURAZIONE ────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### ⚙️ Configurazione")

col_cfg1, col_cfg2 = st.columns(2)
with col_cfg1:
    time_col  = st.selectbox("⏱️ Colonna TIME", numeric_cols)
    event_col = st.selectbox("🎯 Colonna EVENT (0/1)", numeric_cols, index=min(1, len(numeric_cols)-1))
with col_cfg2:
    strat_cols = st.multiselect("🎨 Stratificazione (opzionale)",
                                [c for c in all_cols if c not in [time_col, event_col]])

with st.expander("🎨 Opzioni grafico"):
    col_o1, col_o2, col_o3 = st.columns(3)
    with col_o1:
        show_ci      = st.toggle("Banda CI 95%", value=True)
        show_censors = st.toggle("Mostra censure ( | )", value=True)
    with col_o2:
        show_at_risk = st.toggle("Tabella at-risk", value=False)
        show_median  = st.toggle("Linea mediana", value=True)
    with col_o3:
        title_input  = st.text_input("Titolo grafico", value="Curva di Sopravvivenza")
        xlabel_input = st.text_input("Label asse X", value="Tempo")

# ── VALIDAZIONE ───────────────────────────────────────────────────────────────

errors = []
if time_col == event_col:
    errors.append("TIME e EVENT non possono essere la stessa colonna.")
if time_col in df.columns and (df[time_col] < 0).sum() > 0:
    errors.append(f"Colonna TIME ha {(df[time_col]<0).sum()} valori negativi.")
if event_col in df.columns:
    bad_ev = [v for v in df[event_col].dropna().unique() if v not in [0,1]]
    if bad_ev:
        errors.append(f"Colonna EVENT contiene valori non 0/1: {bad_ev[:5]}")

for e in errors:
    st.error(f"❌ {e}")
if errors:
    st.stop()

# ── HELPERS ───────────────────────────────────────────────────────────────────

def compute_hr(df_c, time_col, event_col, strat_col):
    """HR via CoxPH per variabile binaria (2 gruppi)."""
    try:
        df_cox = df_c[[time_col, event_col, strat_col]].copy()
        df_cox[strat_col] = pd.Categorical(df_cox[strat_col]).codes
        cph = CoxPHFitter()
        cph.fit(df_cox, duration_col=time_col, event_col=event_col)
        hr    = np.exp(cph.params_[strat_col])
        ci_low = np.exp(cph.confidence_intervals_.loc[strat_col, "95% lower-bound"])
        ci_hi  = np.exp(cph.confidence_intervals_.loc[strat_col, "95% upper-bound"])
        return hr, ci_low, ci_hi
    except Exception:
        return None, None, None

def style_axes(ax, light):
    ax.set_facecolor(BG_AX)
    for spine in ax.spines.values():
        spine.set_edgecolor(COL_SPINE)
    ax.tick_params(colors=COL_TICK, labelsize=11)
    ax.xaxis.label.set_color(COL_TEXT)
    ax.yaxis.label.set_color(COL_TEXT)
    ax.grid(True, color=COL_GRID, linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

# ── PLOT ──────────────────────────────────────────────────────────────────────

if st.button("📈 Genera curva di sopravvivenza", type="primary", use_container_width=True):

    df_clean = df[[time_col, event_col] + strat_cols].dropna()
    n_dropped = len(df) - len(df_clean)
    if n_dropped > 0:
        st.warning(f"⚠️ {n_dropped} righe rimosse per valori mancanti.")

    T = df_clean[time_col]
    E = df_clean[event_col]

    if show_at_risk:
        fig, (ax, ax_risk) = plt.subplots(
            2, 1, figsize=(12, 10),
            facecolor=BG_FIG,
            gridspec_kw={"height_ratios": [4, 1], "hspace": 0.05}
        )
        ax_risk.set_facecolor(BG_FIG)
        ax_risk.axis("off")
    else:
        fig, ax = plt.subplots(figsize=(12, 7), facecolor=BG_FIG)

    style_axes(ax, light_mode)

    legend_elements  = []
    logrank_results  = None
    kmf_list         = []

    if not strat_cols:
        # ── CURVA SINGOLA ─────────────────────────────────────────────────────
        kmf = KaplanMeierFitter()
        kmf.fit(T, E, label="Popolazione")
        kmf.plot_survival_function(
            ax=ax, ci_show=show_ci, show_censors=show_censors,
            color=CURR_PALETTE[0], linewidth=2.5, ci_alpha=0.12,
            censor_styles={"ms": 7, "marker": "|"},
        )
        kmf_list.append((kmf, "Popolazione", CURR_PALETTE[0]))

        if show_median:
            med = kmf.median_survival_time_
            if not np.isinf(med) and not np.isnan(med):
                ax.axvline(med, color=CURR_PALETTE[0], linestyle=":", linewidth=1.8, alpha=0.8)
                ax.axhline(0.5, color=CURR_PALETTE[0], linestyle=":", linewidth=1.2, alpha=0.4,
                           xmin=0, xmax=med / ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 0.5)
                ax.text(med + (T.max() * 0.01), 0.53,
                        f"Mediana = {med:.1f}",
                        color=CURR_PALETTE[0], fontsize=9, va="bottom",
                        bbox=dict(facecolor=BG_BOX, edgecolor="none", alpha=0.7, pad=2))

        n_total  = len(T)
        n_events = int(E.sum())
        legend_elements.append(
            Line2D([0],[0], color=CURR_PALETTE[0], linewidth=2.5,
                   label=f"Popolazione n={n_total:,} eventi={n_events:,}")
        )

    else:
        # ── CURVE STRATIFICATE ────────────────────────────────────────────────
        df_clean["_strat_"] = df_clean[strat_cols].astype(str).agg(" | ".join, axis=1)
        groups = sorted(df_clean["_strat_"].unique())

        for idx, grp in enumerate(groups):
            mask  = df_clean["_strat_"] == grp
            T_g   = df_clean.loc[mask, time_col]
            E_g   = df_clean.loc[mask, event_col]
            color = CURR_PALETTE[idx % len(CURR_PALETTE)]

            kmf = KaplanMeierFitter()
            kmf.fit(T_g, E_g, label=grp)
            kmf.plot_survival_function(
                ax=ax, ci_show=show_ci, show_censors=show_censors,
                color=color, linewidth=2.5, ci_alpha=0.10,
                censor_styles={"ms": 7, "marker": "|"},
            )
            kmf_list.append((kmf, grp, color))

            if show_median:
                med = kmf.median_survival_time_
                if not np.isinf(med) and not np.isnan(med):
                    ax.axvline(med, color=color, linestyle=":", linewidth=1.5, alpha=0.7)
                    ax.axhline(0.5, color=color, linestyle=":", linewidth=1.0, alpha=0.35)
                    ax.text(med + (T.max() * 0.01), 0.5 + 0.03 * idx,
                            f"Med {grp} = {med:.1f}",
                            color=color, fontsize=8, va="bottom",
                            bbox=dict(facecolor=BG_BOX, edgecolor="none", alpha=0.7, pad=2))

            n_g = len(T_g)
            e_g = int(E_g.sum())
            legend_elements.append(
                Line2D([0],[0], color=color, linewidth=2.5,
                       label=f"{grp} n={n_g:,} eventi={e_g:,}")
            )

        if len(groups) == 2:
            g0, g1 = groups
            res = logrank_test(
                df_clean.loc[df_clean["_strat_"]==g0, time_col],
                df_clean.loc[df_clean["_strat_"]==g1, time_col],
                df_clean.loc[df_clean["_strat_"]==g0, event_col],
                df_clean.loc[df_clean["_strat_"]==g1, event_col],
            )
            logrank_results = res
            pval     = res.p_value
            pval_str = f"p = {pval:.4f}" if pval >= 0.0001 else "p < 0.0001"
            sig_str  = "★ Significativo (p<0.05)" if pval < 0.05 else "Non significativo"

            strat_single = strat_cols[0] if len(strat_cols) == 1 else "_strat_"
            df_for_cox   = df_clean.copy()
            if strat_single == "_strat_":
                df_for_cox["_strat_cox_"] = df_for_cox["_strat_"]
                strat_single = "_strat_cox_"
            hr, ci_lo, ci_hi = compute_hr(df_for_cox, time_col, event_col, strat_single)
            hr_str = f"HR = {hr:.2f} (95% CI: {ci_lo:.2f}–{ci_hi:.2f})" if hr else ""

            stat_text = f"Log-rank test\n{pval_str}\n{sig_str}"
            if hr_str:
                stat_text += f"\n{hr_str}"
            ax.text(0.97, 0.97, stat_text,
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=9.5, color=COL_TEXT,
                    bbox=dict(facecolor=BG_BOX, edgecolor=COL_SPINE, boxstyle="round,pad=0.5", alpha=0.9))

        elif len(groups) > 2:
            res = multivariate_logrank_test(T, df_clean["_strat_"], E)
            logrank_results = res
            pval     = res.p_value
            pval_str = f"p = {pval:.4f}" if pval >= 0.0001 else "p < 0.0001"
            sig_str  = "★ Significativo (p<0.05)" if pval < 0.05 else "Non significativo"
            ax.text(0.97, 0.97, f"Log-rank test (multivariato)\n{pval_str}\n{sig_str}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=9.5, color=COL_TEXT,
                    bbox=dict(facecolor=BG_BOX, edgecolor=COL_SPINE, boxstyle="round,pad=0.5", alpha=0.9))

    if show_censors:
        legend_elements.append(
            Line2D([0],[0], marker="|", color=COL_SUBTEXT, linewidth=0,
                   markersize=9, markeredgewidth=1.5,
                   label="| = Osservazione censurata")
        )

    ax.set_xlabel(xlabel_input, fontsize=13, labelpad=10, color=COL_TEXT)
    ax.set_ylabel("Probabilità di sopravvivenza", fontsize=13, labelpad=10, color=COL_TEXT)
    ax.set_title(title_input, fontsize=16, fontweight="bold", color=COL_TEXT, pad=15)
    ax.title.set_color(COL_TEXT)

    legend = ax.legend(
        handles=legend_elements,
        loc="upper right" if not strat_cols else "lower left",
        framealpha=0.9 if light_mode else 0.15,
        edgecolor=COL_LEG_EC,
        labelcolor=COL_LEG_L,
        fontsize=10,
        facecolor=COL_LEG_FC,
    )

    if show_at_risk:
        n_groups   = max(len(kmf_list), 1)
        timepoints = np.linspace(T.min(), T.max(), 7).astype(int)
        xlim       = ax.get_xlim()

        ax_risk.set_xlim(xlim)
        ax_risk.set_ylim(-n_groups - 0.5, 0.5)
        ax_risk.set_facecolor(BG_FIG)
        ax_risk.axis("off")

        ax_risk.text(-0.01, 0.3, "Tempo →", transform=ax_risk.transAxes,
                     ha="right", va="center", fontsize=8,
                     color=COL_SUBTEXT, style="italic")

        for tp in timepoints:
            x_norm = (tp - xlim[0]) / (xlim[1] - xlim[0])
            ax_risk.text(x_norm, 0.95, str(int(tp)),
                         transform=ax_risk.transAxes,
                         ha="center", va="top", fontsize=8, color=COL_SUBTEXT)

        for row_idx, (kmf_obj, grp_lbl, color) in enumerate(kmf_list):
            y_norm = 1.0 - (row_idx + 1) * (1.0 / (n_groups + 1))
            ax_risk.text(-0.01, y_norm, grp_lbl[:14],
                         transform=ax_risk.transAxes,
                         ha="right", va="center", fontsize=8,
                         color=color, fontweight="600")
            for tp in timepoints:
                n_at_risk = int((kmf_obj.durations >= tp).sum())
                x_norm    = (tp - xlim[0]) / (xlim[1] - xlim[0])
                ax_risk.text(x_norm, y_norm, str(n_at_risk),
                             transform=ax_risk.transAxes,
                             ha="center", va="center", fontsize=9,
                             color=color, fontweight="700")

    ax.text(0.01, 0.01, "SurvivaLab", transform=ax.transAxes,
            fontsize=8, color=COL_WM, va="bottom")

    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)

    st.markdown("#### 📊 Statistiche descrittive")
    stats_rows = [{
        "Gruppo": "Totale",
        "N": len(T),
        "Eventi": int(E.sum()),
        "% censurati": f"{(1-E.mean())*100:.1f}%",
        "T min": f"{T.min():.1f}",
        "T max": f"{T.max():.1f}",
        "T mediano": f"{T.median():.1f}",
    }]
    if strat_cols:
        for kmf_obj, grp_lbl, _ in kmf_list:
            mask = df_clean["_strat_"] == grp_lbl
            T_g  = df_clean.loc[mask, time_col]
            E_g  = df_clean.loc[mask, event_col]
            stats_rows.append({
                "Gruppo": grp_lbl,
                "N": len(T_g),
                "Eventi": int(E_g.sum()),
                "% censurati": f"{(1-E_g.mean())*100:.1f}%",
                "T min": f"{T_g.min():.1f}",
                "T max": f"{T_g.max():.1f}",
                "T mediano": f"{T_g.median():.1f}",
            })
    st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight", facecolor=BG_FIG)
    buf.seek(0)
    st.download_button(
        label="⬇️ Scarica PNG (alta risoluzione)",
        data=buf, file_name="survival_curve.png", mime="image/png",
        use_container_width=True, type="primary",
    )
    plt.close(fig)

# ── FOOTER ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    f'<p style="text-align:center;color:{footer_col};font-size:0.9rem;">'
    f'SurvivaLab · Powered by <a href="https://lifelines.readthedocs.io" target="_blank" style="color:{gh_col};">lifelines</a> · '
    '<a href="https://github.com/ricciarello" target="_blank" rel="noopener" class="github-link">'
    '<svg height="16" width="16" viewBox="0 0 16 16" fill="currentColor">'
    '<path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>'
    '</svg> ricciarello</a></p>',
    unsafe_allow_html=True
)
