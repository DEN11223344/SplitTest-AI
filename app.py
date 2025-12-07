import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

import plotly.express as px
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
from scipy.stats import chi2_contingency, norm

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


# ================== Utility functions ================== #

REQUIRED_COLS = ["variant", "clicked_cta", "purchased"]


def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    if "timestamp" in df.columns:
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        except Exception:
            pass
    return df


def check_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    return missing


def compute_ab_stats(df: pd.DataFrame, metric_col: str):
    """Compute A/B stats for a binary metric (clicked_cta / purchased)."""
    df_ab = df[df["variant"].isin(["A", "B"])].copy()

    A = df_ab[df_ab["variant"] == "A"][metric_col]
    B = df_ab[df_ab["variant"] == "B"][metric_col]

    conv_A = A.mean()
    conv_B = B.mean()
    n_A = A.count()
    n_B = B.count()
    count_A = A.sum()
    count_B = B.sum()

    stat, p_val = proportions_ztest([count_B, count_A], [n_B, n_A], alternative="larger")

    ci_A = proportion_confint(count_A, n_A, alpha=0.05, method="normal")
    ci_B = proportion_confint(count_B, n_B, alpha=0.05, method="normal")

    abs_lift = conv_B - conv_A
    rel_lift = abs_lift / conv_A * 100 if conv_A > 0 else np.nan

    return {
        "conv_A": conv_A,
        "conv_B": conv_B,
        "n_A": n_A,
        "n_B": n_B,
        "count_A": count_A,
        "count_B": count_B,
        "z_stat": stat,
        "p_value": p_val,
        "ci_A": ci_A,
        "ci_B": ci_B,
        "abs_lift": abs_lift,
        "rel_lift": rel_lift,
    }


def randomization_checks(df: pd.DataFrame, cols_to_check=None):
    """
    For each categorical column, run chi-square test between variant and that column.
    """
    if cols_to_check is None:
        cols_to_check = ["device", "country", "traffic_source"]

    results = []
    for col in cols_to_check:
        if col not in df.columns:
            continue
        ct = pd.crosstab(df["variant"], df[col])
        if ct.shape[1] < 2:
            continue
        chi2, p, dof, exp = chi2_contingency(ct)
        results.append({
            "column": col,
            "p_value": p,
            "status": "OK" if p > 0.05 else "Imbalance?",
            "detail": "Randomization looks balanced ✅" if p > 0.05 else "Potential allocation bias – review this segment ⚠️"
        })
    return results


def sample_size_two_prop(baseline_rate, rel_lift_percent, alpha=0.05, power=0.8):
    """
    Basic sample size calculator for two-proportion A/B test.
    baseline_rate: baseline conversion (0–1)
    rel_lift_percent: desired relative lift in % (e.g., 10 for +10%)
    Returns: (n_per_variant, total_n)
    """
    p1 = baseline_rate
    p2 = p1 * (1 + rel_lift_percent / 100.0)
    delta = p2 - p1

    if delta <= 0 or p1 <= 0 or p1 >= 1:
        return None, None

    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)

    pooled_var = p1 * (1 - p1) + p2 * (1 - p2)
    n_per_group = (z_alpha * np.sqrt(2 * p1 * (1 - p1)) + z_beta * np.sqrt(pooled_var))**2 / (delta**2)
    n_per_group = int(np.ceil(n_per_group))
    return n_per_group, n_per_group * 2


def compute_segment_lifts(df: pd.DataFrame, seg_col: str, metric_col: str = "clicked_cta"):
    """
    For a segment column (e.g., device), compute CTR lift (B - A) per segment.
    Returns dataframe with segment, CTR_A, CTR_B, Abs_Lift, Rel_Lift, etc.
    """
    if seg_col not in df.columns:
        return None

    tmp = (
        df[df["variant"].isin(["A", "B"])]
        .groupby([seg_col, "variant"])[metric_col]
        .agg(["mean", "count"])
        .reset_index()
    )

    pivot = tmp.pivot(index=seg_col, columns="variant", values="mean")
    counts = tmp.pivot(index=seg_col, columns="variant", values="count")

    if "A" not in pivot.columns or "B" not in pivot.columns:
        return None

    out = pd.DataFrame({
        seg_col: pivot.index,
        "CTR_A": pivot["A"],
        "CTR_B": pivot["B"],
        "Users_A": counts["A"],
        "Users_B": counts["B"],
    })
    out["Abs_Lift"] = out["CTR_B"] - out["CTR_A"]
    out["Rel_Lift_%"] = np.where(
        out["CTR_A"] > 0,
        out["Abs_Lift"] / out["CTR_A"] * 100,
        np.nan
    )
    out["Total_Users"] = out["Users_A"] + out["Users_B"]
    out = out.sort_values("Abs_Lift", ascending=False)
    return out


def generate_ai_summary(ctr_stats: dict, conv_stats: dict | None = None, audience: str = "Product Manager"):
    """
    Generate an AI-style natural language summary.
    Audience can be: "Product Manager", "Executive", "Data Scientist".
    """
    ctr_A = ctr_stats["conv_A"] * 100
    ctr_B = ctr_stats["conv_B"] * 100
    ctr_lift_abs = ctr_stats["abs_lift"] * 100
    ctr_lift_rel = ctr_stats["rel_lift"]
    p_ctr = ctr_stats["p_value"]

    if p_ctr < 0.001:
        ctr_sig_text = "highly statistically significant (p < 0.001)"
    elif p_ctr < 0.05:
        ctr_sig_text = "statistically significant (p < 0.05)"
    else:
        ctr_sig_text = "not statistically significant (p ≥ 0.05)"

    lines = []

    # Common core
    lines.append(
        f"• Click-Through Rate (CTR) for Variant A is **{ctr_A:.2f}%**, while Variant B reaches **{ctr_B:.2f}%**."
    )
    lines.append(
        f"• This corresponds to an absolute uplift of **{ctr_lift_abs:.2f} percentage points** "
        f"and a relative improvement of **{ctr_lift_rel:.2f}%** in CTR."
    )
    lines.append(
        f"• The CTR difference is {ctr_sig_text}, suggesting that Variant B "
        f"{'clearly outperforms' if p_ctr < 0.05 else 'does not reliably outperform'} Variant A."
    )

    # Purchase part (if available)
    if conv_stats is not None:
        conv_A = conv_stats["conv_A"] * 100
        conv_B = conv_stats["conv_B"] * 100
        conv_lift_abs = conv_stats["abs_lift"] * 100
        conv_lift_rel = conv_stats["rel_lift"]
        p_conv = conv_stats["p_value"]

        if p_conv < 0.001:
            conv_sig_text = "highly statistically significant (p < 0.001)"
        elif p_conv < 0.05:
            conv_sig_text = "statistically significant (p < 0.05)"
        else:
            conv_sig_text = "not statistically significant (p ≥ 0.05)"

        lines.append("")
        lines.append(
            f"• Purchase conversion for Variant A is **{conv_A:.3f}%**, and Variant B is **{conv_B:.3f}%**."
        )
        lines.append(
            f"• This is an absolute uplift of **{conv_lift_abs:.3f} percentage points** "
            f"({conv_lift_rel:.2f}% relative)."
        )
        lines.append(
            f"• The impact on purchases is {conv_sig_text}, meaning the evidence for downstream "
            f"conversion improvement is {'strong' if p_conv < 0.05 else 'weak or inconclusive'}."
        )

    # Recommendation logic
    if p_ctr < 0.05:
        rec_core = "Variant B is recommended as the primary experience based on CTR."
        if conv_stats and conv_stats["p_value"] >= 0.05:
            conv_note = (
                " However, since purchase uplift is not clearly significant, you may want to "
                "focus on checkout or pricing improvements to translate extra clicks into revenue."
            )
        else:
            conv_note = ""
    else:
        rec_core = (
            "The experiment does not provide strong enough evidence to favor Variant B. "
            "You can keep Variant A, collect more data, or test a new variation."
        )
        conv_note = ""

    # Audience-specific framing
    lines.append("")
    if audience == "Executive":
        rec = (
            f"From a business standpoint, {rec_core} "
            f"{conv_note} Overall, the experiment provides a clear directional signal for decision-making."
        )
    elif audience == "Data Scientist":
        rec = (
            f"{rec_core} {conv_note} "
            "Consider running follow-up experiments or richer instrumentation to better understand "
            "which user segments drive the observed effect."
        )
    else:  # Product Manager (default)
        rec = (
            f"{rec_core} {conv_note} "
            "This gives a solid basis for roadmap decisions while highlighting where UX or funnel optimization "
            "can further improve impact."
        )

    lines.append("**Recommendation:** " + rec)

    return "\n".join(lines)


def generate_pdf_report(ctr_stats: dict, conv_stats: dict | None, ai_text: str) -> bytes:
    """Generate a simple PDF report."""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, height - 72, "A/B Test Experiment Report")

    y = height - 110
    c.setFont("Helvetica", 10)

    def draw_line(line, y_pos):
        max_chars = 90
        while len(line) > max_chars:
            c.drawString(72, y_pos, line[:max_chars])
            line = line[max_chars:]
            y_pos -= 14
        c.drawString(72, y_pos, line)
        return y_pos - 14

    lines = [
        f"CTR Variant A: {ctr_stats['conv_A']*100:.2f}% "
        f"(n={ctr_stats['n_A']}, clicks={ctr_stats['count_A']})",
        f"CTR Variant B: {ctr_stats['conv_B']*100:.2f}% "
        f"(n={ctr_stats['n_B']}, clicks={ctr_stats['count_B']})",
        f"CTR absolute uplift: {ctr_stats['abs_lift']*100:.2f} percentage points",
        f"CTR relative uplift: {ctr_stats['rel_lift']:.2f}%",
        f"CTR p-value: {ctr_stats['p_value']:.6f}",
        "",
    ]

    if conv_stats is not None:
        lines.extend(
            [
                f"Purchase rate Variant A: {conv_stats['conv_A']*100:.3f}% "
                f"(n={conv_stats['n_A']}, purchases={conv_stats['count_A']})",
                f"Purchase rate Variant B: {conv_stats['conv_B']*100:.3f}% "
                f"(n={conv_stats['n_B']}, purchases={conv_stats['count_B']})",
                f"Purchase absolute uplift: {conv_stats['abs_lift']*100:.3f} percentage points",
                f"Purchase relative uplift: {conv_stats['rel_lift']:.2f}%",
                f"Purchase p-value: {conv_stats['p_value']:.6f}",
                "",
            ]
        )

    lines.append("AI-style Summary and Recommendation:")
    lines.append("")

    for part in ai_text.split("\n"):
        lines.append(part)

    for line in lines:
        if y < 72:
            c.showPage()
            c.setFont("Helvetica", 10)
            y = height - 72
        y = draw_line(line, y)

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()


# ================== Streamlit App ================== #

st.set_page_config(
    page_title="SplitTest AI - A/B Experiment Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Custom CSS ---- #
st.markdown(
    """
    <style>
        .main {
            background: radial-gradient(circle at top, #0f172a 0, #020617 45%, #020617 100%);
            color: #e5e7eb;
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        h1, h2, h3, h4 {
            color: #e5e7eb !important;
        }
        .metric-card {
            background: rgba(15,23,42,0.9);
            border-radius: 16px;
            padding: 16px 18px;
            border: 1px solid rgba(148,163,184,0.35);
            box-shadow: 0 18px 35px rgba(15,23,42,0.65);
        }
        .metric-label {
            font-size: 0.85rem;
            color: #9ca3af;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .metric-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: #e5e7eb;
        }
        .metric-sub {
            font-size: 0.85rem;
            color: #9ca3af;
        }
        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(90deg,#38bdf8,#a855f7,#ec4899);
            -webkit-background-clip: text;
            color: transparent;
        }
        .hero-subtitle {
            font-size: 0.98rem;
            color: #9ca3af;
            max-width: 650px;
        }
        .tag-pill {
            display: inline-flex;
            align-items: center;
            padding: 4px 10px;
            border-radius: 999px;
            background: rgba(15,23,42,0.9);
            border: 1px solid rgba(148,163,184,0.4);
            font-size: 0.8rem;
            color: #e5e7eb;
            margin-right: 6px;
        }
        .stMetric > div {
            padding: 10px !important;
            background: rgba(15,23,42,0.9) !important;
            border-radius: 12px !important;
            border: 1px solid rgba(148,163,184,0.5) !important;
        }
        .stDownloadButton button, .stButton button {
            border-radius: 999px !important;
            padding: 8px 18px !important;
            border: none;
            background: linear-gradient(135deg,#6366f1,#22c55e) !important;
            color: white !important;
            font-weight: 600 !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            background: rgba(15,23,42,0.85);
            border-radius: 999px;
            padding-top: 4px;
            padding-bottom: 4px;
        }
        footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# ---- Sidebar ---- #
with st.sidebar:
    st.markdown("### ⚙️ Experiment Configuration")
    uploaded = st.file_uploader("Upload A/B experiment CSV", type=["csv"])
    use_sample = st.checkbox("Just show schema (no data)", value=False)

    st.markdown("---")
    st.markdown("**ℹ️ Required Columns**")
    st.caption("`variant`, `clicked_cta`, `purchased`")
    st.markdown("**✨ Optional Columns**")
    st.caption("`timestamp`, `device`, `country`, `traffic_source`, `session_duration_sec`, `pages_viewed`, `scroll_depth_pct`")

# ---- Hero Section ---- #
st.markdown(
    '<div class="tag-pill">📊 Experiment Intelligence • A/B Testing • AI Summary</div>',
    unsafe_allow_html=True
)
st.markdown('<div class="hero-title">SplitTest AI – A/B Experiment Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">Upload an A/B test dataset, validate experiment health, run proper statistical tests, '
    'visualize results, and auto-generate a business-ready summary and PDF report – in a single tool.</p>',
    unsafe_allow_html=True
)

st.markdown("")

if not uploaded and not use_sample:
    st.info("⬅️ Upload a CSV in the sidebar to get started.")
    st.stop()

if use_sample:
    st.markdown("### 🔍 Sample Schema (No Data Loaded)")
    st.write(
        """
        Your CSV should contain at least:
        - `variant`: A / B  
        - `clicked_cta`: 0/1 (user clicked main CTA)  
        - `purchased`: 0/1 (user completed purchase)  

        Optional columns unlock richer insights:
        - `timestamp` for time trends  
        - `device`, `country`, `traffic_source` for segmentation  
        - `session_duration_sec`, `pages_viewed`, `scroll_depth_pct` for engagement analysis  
        """
    )
    st.stop()

df = load_data(uploaded)
missing = check_columns(df)

if missing:
    st.error(f"❌ Missing required columns: {missing}. Please check your CSV.")
    st.stop()

st.success("✅ Data loaded successfully.")

# -------------- Dataset Overview -------------- #
ctr_overall = df["clicked_cta"].mean() * 100 if "clicked_cta" in df.columns else 0.0
total_rows = len(df)
variants = ", ".join(sorted(df["variant"].dropna().unique().astype(str)))

c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Total Users</div>
            <div class="metric-value">{total_rows}</div>
            <div class="metric-sub">Rows in uploaded dataset</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Variants</div>
            <div class="metric-value">{variants}</div>
            <div class="metric-sub">Detected experiment groups</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Overall CTR</div>
            <div class="metric-value">{ctr_overall:.2f}%</div>
            <div class="metric-sub">Baseline click-through rate</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# -------------- Experiment Health Check -------------- #
st.subheader("🧪 Experiment Health Check")

health_results = randomization_checks(df)
if not health_results:
    st.info("No segment columns (`device`, `country`, `traffic_source`) found for randomization check.")
else:
    h1, h2 = st.columns([1.5, 1.5])
    with h1:
        for res in health_results:
            icon = "✅" if res["status"] == "OK" else "⚠️"
            st.markdown(
                f"**{icon} {res['column']}**  \n"
                f"- p-value: `{res['p_value']:.3f}`  \n"
                f"- {res['detail']}"
            )
    with h2:
        st.caption(
            "We run chi-square tests between `variant` and key categorical columns. "
            "p-value > 0.05 indicates no evidence of imbalance; lower values may hint at biased allocation."
        )

st.markdown("---")

# -------------- A/B Statistical Analysis -------------- #
ctr_stats = compute_ab_stats(df, "clicked_cta")
conv_stats = compute_ab_stats(df, "purchased")

st.subheader("📐 A/B Statistical Summary")

col_a, col_b, col_c = st.columns([1.1, 1.1, 1.2])

with col_a:
    st.markdown("##### Control – Variant A")
    st.metric("Users", ctr_stats["n_A"])
    st.metric("CTR", f"{ctr_stats['conv_A']*100:.2f}%")
    st.metric("Purchase", f"{conv_stats['conv_A']*100:.3f}%")

with col_b:
    st.markdown("##### Treatment – Variant B")
    st.metric("Users", ctr_stats["n_B"])
    st.metric("CTR", f"{ctr_stats['conv_B']*100:.2f}%")
    st.metric("Purchase", f"{conv_stats['conv_B']*100:.3f}%")

with col_c:
    st.markdown("##### Lift & Significance")
    st.metric("CTR Lift (pp)", f"{ctr_stats['abs_lift']*100:.2f}")
    st.metric("CTR Relative Lift", f"{ctr_stats['rel_lift']:.2f}%")
    st.metric("CTR p-value", f"{ctr_stats['p_value']:.3e}")
    st.markdown("---")
    st.metric("Purchase Lift (pp)", f"{conv_stats['abs_lift']*100:.3f}")
    st.metric("Purchase Rel. Lift", f"{conv_stats['rel_lift']:.2f}%")
    st.metric("Purchase p-value", f"{conv_stats['p_value']:.3e}")

st.markdown("---")

# -------------- Visual & Design Tools -------------- #
st.subheader("📊 Visual & Design Tools")

tab1, tab2, tab3, tab4 = st.tabs(
    ["CTR by Variant", "Purchase by Variant", "Segment View", "Design & Sample Size"]
)

with tab1:
    st.markdown("##### CTR by Variant")
    ctr_df = df.groupby("variant")["clicked_cta"].mean().reset_index()
    ctr_df["CTR"] = ctr_df["clicked_cta"] * 100
    fig_ctr = px.bar(
        ctr_df,
        x="variant",
        y="CTR",
        text="CTR",
        labels={"CTR": "CTR (%)"},
        color="variant",
        color_discrete_sequence=["#38bdf8", "#a855f7"],
    )
    fig_ctr.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig_ctr.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_ctr, use_container_width=True)

with tab2:
    st.markdown("##### Purchase Conversion by Variant")
    conv_df = df.groupby("variant")["purchased"].mean().reset_index()
    conv_df["PurchaseRate"] = conv_df["purchased"] * 100
    fig_conv = px.bar(
        conv_df,
        x="variant",
        y="PurchaseRate",
        text="PurchaseRate",
        labels={"PurchaseRate": "Purchase Rate (%)"},
        color="variant",
        color_discrete_sequence=["#22c55e", "#a855f7"],
    )
    fig_conv.update_traces(texttemplate="%{text:.3f}%", textposition="outside")
    fig_conv.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_conv, use_container_width=True)

with tab3:
    st.markdown("##### Segment-wise CTR")
    selectable_cols = [c for c in ["device", "country", "traffic_source"] if c in df.columns]
    if selectable_cols:
        seg_col = st.selectbox("Segment by column", options=selectable_cols)
        seg_df = df.groupby([seg_col, "variant"])["clicked_cta"].mean().reset_index()
        seg_df["CTR"] = seg_df["clicked_cta"] * 100
        fig_seg = px.bar(
            seg_df,
            x=seg_col,
            y="CTR",
            color="variant",
            barmode="group",
            labels={"CTR": "CTR (%)"},
            color_discrete_sequence=["#38bdf8", "#a855f7"],
        )
        fig_seg.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_seg, use_container_width=True)

        st.markdown("###### Top segments where Variant B outperforms A")
        seg_lifts = compute_segment_lifts(df, seg_col, metric_col="clicked_cta")
        if seg_lifts is not None:
            show_cols = [seg_col, "CTR_A", "CTR_B", "Abs_Lift", "Rel_Lift_%", "Total_Users"]
            st.dataframe(
                seg_lifts[show_cols]
                .head(5)
                .style.format({
                    "CTR_A": "{:.3f}",
                    "CTR_B": "{:.3f}",
                    "Abs_Lift": "{:.3f}",
                    "Rel_Lift_%": "{:.2f}",
                    "Total_Users": "{:,.0f}"
                }),
                use_container_width=True
            )
            st.caption("Top 5 segments ranked by absolute CTR uplift (B vs A).")
        else:
            st.info("Not enough variation in this segment to compute lifts.")
    else:
        st.info("No segment columns (`device`, `country`, `traffic_source`) found in this dataset.")

with tab4:
    st.markdown("##### Sample Size & Experiment Design Helper")

    baseline_guess = df[df["variant"] == "A"]["clicked_cta"].mean() if "clicked_cta" in df.columns else 0.05
    if np.isnan(baseline_guess) or baseline_guess == 0:
        baseline_guess = 0.05

    col_design1, col_design2 = st.columns(2)

    with col_design1:
        base_ctr_input = st.number_input(
            "Baseline CTR for Variant A (%)",
            min_value=0.01,
            max_value=100.0,
            value=float(round(baseline_guess * 100, 2)),
            step=0.1,
            help="Use your current CTR or the observed CTR of Variant A."
        )
        desired_lift = st.number_input(
            "Desired relative uplift for Variant B (%)",
            min_value=1.0,
            max_value=200.0,
            value=10.0,
            step=1.0,
            help="For example, 10% means going from 5% to 5.5% CTR."
        )

    with col_design2:
        alpha = st.number_input(
            "Significance level (alpha)",
            min_value=0.001,
            max_value=0.2,
            value=0.05,
            step=0.005
        )
        power = st.number_input(
            "Statistical power",
            min_value=0.50,
            max_value=0.99,
            value=0.80,
            step=0.05
        )

    if st.button("Calculate required sample size"):
        n_per_group, total_n = sample_size_two_prop(
            baseline_rate=base_ctr_input / 100.0,
            rel_lift_percent=desired_lift,
            alpha=alpha,
            power=power
        )
        if n_per_group is None:
            st.error("Invalid input combination. Make sure baseline rate and uplift are positive.")
        else:
            st.success(
                f"You need **~{n_per_group:,} users per variant** "
                f"(~{total_n:,} total) to detect a {desired_lift:.1f}% uplift "
                f"with {power*100:.0f}% power at α = {alpha}."
            )
            st.caption(
                "If your current experiment has fewer users than this, your test may be underpowered, "
                "and non-significant results could just be due to insufficient sample size."
            )

st.markdown("---")

# -------------- AI-style Summary -------------- #
st.subheader("🤖 AI-style Summary & Recommendation")

audience = st.selectbox(
    "Tailor summary for audience",
    options=["Product Manager", "Executive", "Data Scientist"],
    index=0,
    help="Choose how the explanation should be framed."
)

ai_text = generate_ai_summary(ctr_stats, conv_stats, audience=audience)
st.markdown(ai_text)

st.markdown("---")

# -------------- PDF Report Download -------------- #
st.subheader("📄 Export A/B Test Report")

if st.button("Generate PDF Report"):
    pdf_bytes = generate_pdf_report(ctr_stats, conv_stats, ai_text)
    st.download_button(
        label="⬇️ Download PDF Report",
        data=pdf_bytes,
        file_name="ab_test_report.pdf",
        mime="application/pdf"
    )
