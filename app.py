import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

# ── PALETTE ───────────────────────────────────────────────────────────────────
PALETTE = ["#636EFA", "#EF553B", "#00CC96", "#FFB703", "#AB63FA",
           "#FFA15A", "#19D3F3", "#FF6692", "#B6E880", "#FF97FF"]

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="SurvivaLab", page_icon="📈", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.hero-title {
    font-size: 2.8rem; font-weight: 900; text-align: center;
    background: linear-gradient(135deg, #636EFA, #00CC96);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 0.3rem;
}
.hero-desc { text-align: center; color: #aaa; font-size: 1rem; margin-bottom: 1.5rem; }
.check-ok  { color: #00CC96; font-weight: 600; }
.check-warn{ color: #FFB703; font-weight: 600; }
.check-err { color: #EF553B; font-weight: 600; }
.block-container { max-width: 1100px !important; margin: 0 auto !important; padding: 2rem 3rem !important; }
.github-link { color:#aaa; text-decoration:none; display:inline-flex; align-items:center; gap:6px; font-weight:600; transition:all 0.2s ease; }
.github-link:hover { color:#ffd700; }
.github-link svg { transition: transform 0.6s ease; }
.github-link:hover svg { transform: rotate(360deg); }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero-title">📈 SurvivaLab</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-desc">Carica un CSV · Scegli le colonne · Ottieni curve di sopravvivenza pronte da pubblicare.</div>', unsafe_allow_html=True)

# ── UPLOAD ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Carica il tuo CSV", type=["csv"])

if not uploaded:
    st.info("⬆️ Carica un CSV per iniziare.")
    st.stop()

# ── LOAD ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_csv(f):
    return pd.read_csv(f)

df = load_csv(uploaded)

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

# Dimensione
checks.append(("Righe nel dataset", f"{df.shape[0]:,}", "ok" if df.shape[0] >= 30 else "warn",
               "ok" if df.shape[0] >= 30 else "< 30 righe: risultati poco affidabili"))

# Duplicati
dupes = df.duplicated().sum()
checks.append(("Righe duplicate", str(dupes), "ok" if dupes == 0 else "warn",
               "ok" if dupes == 0 else f"{dupes} righe duplicate trovate"))

# Valori mancanti
total_miss = df.isnull().sum().sum()
pct_miss   = total_miss / df.size * 100
checks.append(("Valori mancanti", f"{total_miss} ({pct_miss:.1f}%)",
               "ok" if pct_miss == 0 else ("warn" if pct_miss < 10 else "err"),
               "ok" if pct_miss == 0 else f"{pct_miss:.1f}% valori mancanti"))

# Colonne numeriche
checks.append(("Colonne numeriche disponibili", str(len(numeric_cols)),
               "ok" if len(numeric_cols) >= 2 else "err",
               "ok" if len(numeric_cols) >= 2 else "Servono almeno 2 colonne numeriche (TIME, EVENT)"))

icon_map = {"ok": "✅", "warn": "⚠️", "err": "❌"}
cls_map  = {"ok": "check-ok", "warn": "check-warn", "err": "check-err"}

c1, c2, c3, c4 = st.columns(4)
for col_ui, (label, value, status, detail) in zip([c1,c2,c3,c4], checks):
    with col_ui:
        st.markdown(f'<div class="{cls_map[status]}">{icon_map[status]} {label}</div>', unsafe_allow_html=True)
        st.markdown(f"**{value}**")
        if status != "ok":
            st.caption(detail)

# Tabella mancanti per colonna
if total_miss > 0:
    with st.expander("📊 Dettaglio valori mancanti per colonna"):
        miss_df = df.isnull().sum().reset_index()
        miss_df.columns = ["Colonna", "Mancanti"]
        miss_df["% su totale"] = (miss_df["Mancanti"] / len(df) * 100).round(2)
        miss_df = miss_df[miss_df["Mancanti"] > 0].sort_values("Mancanti", ascending=False)
        st.dataframe(miss_df, use_container_width=True, hide_index=True)

# ── CONFIGURAZIONE KM ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### ⚙️ Configurazione")

col_cfg1, col_cfg2 = st.columns(2)

with col_cfg1:
    time_col = st.selectbox("⏱️ Colonna TIME (durata)", numeric_cols,
                             help="Colonna con la durata/tempo di osservazione")
    event_col = st.selectbox("🎯 Colonna EVENT (0/1)", numeric_cols,
                              index=min(1, len(numeric_cols)-1),
                              help="Colonna con l'evento: 1=avvenuto, 0=censurato")

with col_cfg2:
    strat_cols = st.multiselect("🎨 Variabili di stratificazione (opzionale)",
                                 [c for c in all_cols if c not in [time_col, event_col]],
                                 help="Seleziona 1+ variabili per plottare curve separate per gruppo")

# Opzioni grafiche
with st.expander("🎨 Opzioni grafico"):
    col_o1, col_o2, col_o3 = st.columns(3)
    with col_o1:
        show_ci      = st.toggle("Banda di confidenza (95%)", value=True)
        show_censors = st.toggle("Mostra censure (tick)", value=True)
    with col_o2:
        show_at_risk = st.toggle("Tabella at-risk sotto grafico", value=False)
        show_median  = st.toggle("Linea mediana", value=True)
    with col_o3:
        title_input  = st.text_input("Titolo grafico", value="Curva di Sopravvivenza")
        xlabel_input = st.text_input("Label asse X", value="Tempo")

# ── VALIDAZIONE ───────────────────────────────────────────────────────────────
errors = []

if time_col == event_col:
    errors.append("TIME e EVENT non possono essere la stessa colonna.")

if time_col in df.columns:
    neg_time = (df[time_col] < 0).sum()
    if neg_time > 0:
        errors.append(f"Colonna TIME ha {neg_time} valori negativi.")

if event_col in df.columns:
    unique_ev = df[event_col].dropna().unique()
    bad_ev = [v for v in unique_ev if v not in [0, 1]]
    if bad_ev:
        errors.append(f"Colonna EVENT contiene valori non 0/1: {bad_ev[:5]}")

if errors:
    for e in errors:
        st.error(f"❌ {e}")
    st.stop()

# ── PLOT ──────────────────────────────────────────────────────────────────────
if st.button("📈 Genera curva di sopravvivenza", type="primary", use_container_width=True):

    df_clean = df[[time_col, event_col] + strat_cols].dropna()
    n_dropped = len(df) - len(df_clean)

    if n_dropped > 0:
        st.warning(f"⚠️ {n_dropped} righe rimosse per valori mancanti nelle colonne selezionate.")

    T = df_clean[time_col]
    E = df_clean[event_col]

    # ── FIGURE SETUP ──────────────────────────────────────────────────────────
    fig_height = 7 if not show_at_risk else 9
    fig, ax = plt.subplots(figsize=(12, fig_height), facecolor="#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    ax.tick_params(colors="#aaa", labelsize=11)
    ax.xaxis.label.set_color("#ccc")
    ax.yaxis.label.set_color("#ccc")
    ax.grid(True, color="#1e1e2e", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

    legend_elements = []
    logrank_results = None

    if not strat_cols:
        # ── CURVA SINGOLA ─────────────────────────────────────────────────────
        kmf = KaplanMeierFitter()
        kmf.fit(T, E, label="Popolazione")
        kmf.plot_survival_function(
            ax=ax, ci_show=show_ci, show_censors=show_censors,
            color=PALETTE[0], linewidth=2.5,
            ci_alpha=0.12,
            censor_styles={"ms": 6, "marker": "|"},
        )
        if show_median:
            med = kmf.median_survival_time_
            if not np.isinf(med):
                ax.axvline(med, color=PALETTE[0], linestyle=":", linewidth=1.5, alpha=0.7)
                ax.text(med, 0.52, f" mediana\n {med:.1f}", color=PALETTE[0],
                        fontsize=9, va="bottom")

        n_total   = len(T)
        n_events  = int(E.sum())
        legend_elements.append(
            Line2D([0],[0], color=PALETTE[0], linewidth=2.5,
                   label=f"Popolazione  n={n_total:,}  eventi={n_events:,}")
        )

    else:
        # ── CURVE STRATIFICATE ────────────────────────────────────────────────
        strat_label = " × ".join(strat_cols)
        df_clean["_strat_"] = df_clean[strat_cols].astype(str).agg(" | ".join, axis=1)
        groups = sorted(df_clean["_strat_"].unique())

        for idx, grp in enumerate(groups):
            mask = df_clean["_strat_"] == grp
            T_g  = df_clean.loc[mask, time_col]
            E_g  = df_clean.loc[mask, event_col]
            color = PALETTE[idx % len(PALETTE)]

            kmf = KaplanMeierFitter()
            kmf.fit(T_g, E_g, label=grp)
            kmf.plot_survival_function(
                ax=ax, ci_show=show_ci, show_censors=show_censors,
                color=color, linewidth=2.5, ci_alpha=0.10,
                censor_styles={"ms": 6, "marker": "|"},
            )
            if show_median:
                med = kmf.median_survival_time_
                if not np.isinf(med):
                    ax.axvline(med, color=color, linestyle=":", linewidth=1.2, alpha=0.5)

            n_g = len(T_g)
            e_g = int(E_g.sum())
            legend_elements.append(
                Line2D([0],[0], color=color, linewidth=2.5,
                       label=f"{grp}  n={n_g:,}  eventi={e_g:,}")
            )

        # Log-rank test
        if len(groups) == 2:
            g0, g1 = groups
            res = logrank_test(
                df_clean.loc[df_clean["_strat_"]==g0, time_col],
                df_clean.loc[df_clean["_strat_"]==g1, time_col],
                df_clean.loc[df_clean["_strat_"]==g0, event_col],
                df_clean.loc[df_clean["_strat_"]==g1, event_col],
            )
            logrank_results = res
        elif len(groups) > 2:
            res = multivariate_logrank_test(T, df_clean["_strat_"], E)
            logrank_results = res

        if logrank_results:
            pval = logrank_results.p_value
            pval_str = f"p = {pval:.4f}" if pval >= 0.0001 else "p < 0.0001"
            sig_str  = "★ Differenza significativa" if pval < 0.05 else "Differenza non significativa"
            ax.text(0.97, 0.97, f"Log-rank test\n{pval_str}\n{sig_str}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=10, color="#ccc",
                    bbox=dict(facecolor="#1a1a2e", edgecolor="#333", boxstyle="round,pad=0.5"))

    # ── LEGENDA E LABELS ──────────────────────────────────────────────────────
    ax.set_xlabel(xlabel_input, fontsize=13, labelpad=10)
    ax.set_ylabel("Probabilità di sopravvivenza", fontsize=13, labelpad=10)
    ax.set_title(title_input, fontsize=16, fontweight="bold", color="#eee", pad=15)

    legend = ax.legend(
        handles=legend_elements,
        loc="upper right" if not strat_cols else "lower left",
        framealpha=0.15, edgecolor="#333",
        labelcolor="#ccc", fontsize=10,
        facecolor="#1a1a2e",
    )

    # Watermark
    ax.text(0.01, 0.01, "SurvivaLab", transform=ax.transAxes,
            fontsize=8, color="#333", va="bottom")

    plt.tight_layout()

    # ── MOSTRA E DOWNLOAD ─────────────────────────────────────────────────────
    st.pyplot(fig, use_container_width=True)

    # Statistiche sotto
    st.markdown("#### 📊 Statistiche descrittive")
    stats_data = {
        "N totale": [len(T)],
        "N eventi": [int(E.sum())],
        "% censurati": [f"{(1 - E.mean())*100:.1f}%"],
        "Tempo min": [f"{T.min():.2f}"],
        "Tempo max": [f"{T.max():.2f}"],
        "Tempo mediano": [f"{T.median():.2f}"],
    }
    st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

    # Download PNG
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight", facecolor="#0f0f1a")
    buf.seek(0)
    st.download_button(
        label="⬇️ Scarica PNG (alta risoluzione)",
        data=buf,
        file_name="survival_curve.png",
        mime="image/png",
        use_container_width=True,
        type="primary",
    )
    plt.close(fig)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#555;font-size:0.9rem;">'
    'SurvivaLab · Powered by <a href="https://lifelines.readthedocs.io" target="_blank" style="color:#aaa;">lifelines</a> · '
    '<a href="https://github.com/ricciarello" target="_blank" rel="noopener" class="github-link">'
    '<svg height="16" width="16" viewBox="0 0 16 16" fill="currentColor">'
    '<path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>'
    '</svg> ricciarello</a></p>',
    unsafe_allow_html=True
)
