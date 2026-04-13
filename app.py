import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="The Economics of Intelligence", page_icon=":bar_chart:", layout="wide")

DATA_PATH = Path("data/models_2026-03-29.json")
ALIASES = {
    "model_creator.name": "provider",
    "evaluations.artificial_analysis_intelligence_index": "intelligence",
    "evaluations.artificial_analysis_coding_index": "coding_index",
    "evaluations.artificial_analysis_math_index": "math_index",
    "evaluations.gpqa": "gpqa",
    "evaluations.hle": "hle",
    "evaluations.lcr": "lcr",
    "evaluations.tau2": "tau2",
    "evaluations.terminalbench_hard": "terminalbench_hard",
    "evaluations.mmlu_pro": "mmlu_pro",
    "evaluations.livecodebench": "livecodebench",
    "evaluations.ifbench": "ifbench",
    "evaluations.scicode": "scicode",
    "pricing.price_1m_blended_3_to_1": "blended_price",
    "pricing.price_1m_input_tokens": "input_price",
    "pricing.price_1m_output_tokens": "output_price",
    "median_output_tokens_per_second": "output_tps",
    "median_time_to_first_token_seconds": "ttft",
    "median_time_to_first_answer_token": "ttfat",
}
BENCHMARKS = ["intelligence", "coding_index", "math_index", "gpqa", "hle", "lcr", "tau2", "terminalbench_hard", "mmlu_pro", "livecodebench", "ifbench", "scicode"]
CORE = ["intelligence", "gpqa", "lcr", "tau2", "terminalbench_hard"]
DERIVED = ["cai", "sai", "ttv", "bes", "agent_readiness_index", "cost_to_agent_score", "reasoning_action_gap", "throughput_efficiency", "energy_proxy", "cost_per_intelligence"]
LABELS = {
    "name": "Model", "provider": "Provider", "release_date": "Release date", "release_month": "Release month",
    "intelligence": "Intelligence", "coding_index": "Coding", "math_index": "Math", "gpqa": "GPQA", "hle": "HLE",
    "lcr": "LCR", "tau2": "tau2", "terminalbench_hard": "TerminalBench Hard", "mmlu_pro": "MMLU Pro",
    "livecodebench": "LiveCodeBench", "ifbench": "IFBench", "scicode": "SciCode", "blended_price": "Blended price",
    "input_price": "Input price", "output_price": "Output price", "output_tps": "Output tok/s", "ttft": "TTFT", "ttfat": "TTFAT",
    "cai": "Cost-adjusted intelligence", "sai": "Speed-adjusted intelligence", "ttv": "Time-to-value", "bes": "BES",
    "agent_readiness_index": "Agent readiness", "cost_to_agent_score": "Cost to agent score", "reasoning_action_gap": "Reasoning-action gap",
    "throughput_efficiency": "Throughput efficiency", "energy_proxy": "Energy proxy", "cost_per_intelligence": "Cost per intelligence",
    "is_frontier": "Pareto frontier", "benchmark_coverage_count": "Benchmark coverage count",
}


def safe_divide(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    if np.isscalar(b):
        return a / (np.nan if pd.isna(b) or b == 0 else b)
    return a / b.replace(0, np.nan)


def ensure_columns(df, cols):
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan
    return df


def infer_reasoning_type(name):
    text = str(name).lower()
    if "non-reasoning" in text or "non reasoning" in text:
        return "non-reasoning"
    for level in ["xhigh", "medium", "high", "low"]:
        if level in text:
            return level
    return "reasoning" if "reasoning" in text or "thinking" in text else "unspecified"


def compute_metrics(df):
    df = ensure_columns(df, [*BENCHMARKS, "blended_price", "output_tps", "ttft"])
    df["cai"] = safe_divide(df["intelligence"], df["blended_price"])
    df["sai"] = pd.to_numeric(df["intelligence"], errors="coerce") * pd.to_numeric(df["output_tps"], errors="coerce")
    df["ttv"] = safe_divide(df["intelligence"], df["ttft"])
    df["bes"] = safe_divide(df["sai"], pd.to_numeric(df["blended_price"], errors="coerce") * pd.to_numeric(df["ttft"], errors="coerce"))
    df["agent_readiness_index"] = 0.4 * df["lcr"] + 0.3 * df["terminalbench_hard"] + 0.3 * df["tau2"]
    df["cost_to_agent_score"] = safe_divide(df["agent_readiness_index"], df["blended_price"])
    df["reasoning_action_gap"] = df["gpqa"] - df["lcr"]
    df["throughput_efficiency"] = safe_divide(df["output_tps"], df["blended_price"])
    df["energy_proxy"] = safe_divide(df["blended_price"], df["output_tps"])
    df["cost_per_intelligence"] = safe_divide(df["blended_price"], df["intelligence"])
    for src, out in {"gpqa": "gpqa_dominance", "lcr": "lcr_dominance", "tau2": "tau2_dominance", "terminalbench_hard": "terminalbench_dominance"}.items():
        max_score = df[src].max(skipna=True)
        df[out] = safe_divide(df[src], max_score) if pd.notna(max_score) and max_score != 0 else np.nan
    df["benchmark_coverage_count"] = df[BENCHMARKS].notna().sum(axis=1)
    return df


@st.cache_data(show_spinner=False)
def load_data(path=DATA_PATH):
    if not path.exists():
        st.error("Missing data file: data/models_2026-03-29.json. Add the JSON file to the repo before deploying.")
        st.stop()
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    records = payload.get("data", []) if isinstance(payload, dict) else payload
    df = pd.json_normalize(records).rename(columns=ALIASES)
    df = ensure_columns(df, ["name", "slug", "release_date", "provider", *ALIASES.values()])
    df["name"] = df["name"].fillna(df["slug"]).fillna("Unknown model")
    df["provider"] = df["provider"].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date"].dt.year.astype("Int64")
    df["release_month"] = df["release_date"].dt.to_period("M").dt.to_timestamp()
    df["reasoning_type"] = df["name"].map(infer_reasoning_type)
    for col in [*BENCHMARKS, "blended_price", "input_price", "output_price", "output_tps", "ttft", "ttfat"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return compute_metrics(df)


def display_df(df, cols):
    cols = [c for c in dict.fromkeys(cols) if c in df.columns]
    out = df[cols].copy()
    for col in out.select_dtypes(include=np.number).columns:
        out[col] = out[col].round(3)
    if "release_date" in out:
        out["release_date"] = out["release_date"].dt.date
    if "release_month" in out:
        out["release_month"] = out["release_month"].dt.strftime("%Y-%m")
    return out.rename(columns={c: LABELS.get(c, c) for c in out.columns})


def build_frontier(df, x="blended_price", y="intelligence"):
    work = df.dropna(subset=[x, y]).copy()
    work = work[(work[x] > 0) & np.isfinite(work[x]) & np.isfinite(work[y])].sort_values([x, y], ascending=[True, False])
    best, flags = -np.inf, []
    for value in work[y]:
        flag = value > best
        flags.append(flag)
        if flag:
            best = value
    work["is_frontier"] = flags
    return work


def add_frontier_flag(df):
    out = df.copy()
    out["is_frontier"] = False
    frontier = build_frontier(out)
    if not frontier.empty:
        out.loc[frontier.index, "is_frontier"] = frontier["is_frontier"]
    return out


def monthly_summary(df):
    work = df.dropna(subset=["release_month"])
    return work.groupby("release_month", as_index=False).agg(
        model_count=("name", "count"), max_intelligence=("intelligence", "max"), median_intelligence=("intelligence", "median"),
        max_agent_readiness=("agent_readiness_index", "max"), median_agent_readiness=("agent_readiness_index", "median"),
        min_cost_per_intelligence=("cost_per_intelligence", "min"), median_cost_per_intelligence=("cost_per_intelligence", "median"),
    ).sort_values("release_month")


def provider_monthly_summary(df):
    work = df.dropna(subset=["release_month"])
    return work.groupby(["release_month", "provider"], as_index=False).agg(
        model_count=("name", "count"), best_intelligence=("intelligence", "max"), best_agent_readiness=("agent_readiness_index", "max"),
        median_cost_per_intelligence=("cost_per_intelligence", "median"),
    )


def pct_change(a, b):
    return np.nan if pd.isna(a) or pd.isna(b) or a == 0 else ((b - a) / abs(a)) * 100


def generate_release_trend_insights(df):
    insights, monthly = [], monthly_summary(df)
    if monthly.empty:
        return ["No dated model releases are available after filtering."]
    valid = monthly.dropna(subset=["max_intelligence"])
    if not valid.empty:
        first, last = valid.iloc[0], valid.iloc[-1]
        insights.append(f"Frontier release-cohort intelligence moved from {first.max_intelligence:.1f} in {first.release_month:%Y-%m} to {last.max_intelligence:.1f} in {last.release_month:%Y-%m}, a {pct_change(first.max_intelligence, last.max_intelligence):.1f}% change.")
    valid = monthly.dropna(subset=["min_cost_per_intelligence"])
    if not valid.empty:
        first, last = valid.iloc[0], valid.iloc[-1]
        change = pct_change(first.min_cost_per_intelligence, last.min_cost_per_intelligence)
        direction = "fell" if pd.notna(change) and change < 0 else "rose"
        insights.append(f"Best observed cost per intelligence {direction} from {first.min_cost_per_intelligence:.4f} to {last.min_cost_per_intelligence:.4f} across release cohorts ({change:.1f}%).")
    frontier = build_frontier(df)
    f = frontier[frontier["is_frontier"]] if not frontier.empty else pd.DataFrame()
    if not f.empty:
        insights.append(f"On the price-intelligence frontier, {f.sort_values('blended_price').iloc[0].provider} anchors the lowest-cost point while {f.sort_values('intelligence', ascending=False).iloc[0].provider} anchors the highest-capability point.")
        lat = f[["intelligence", "ttft"]].dropna()
        if len(lat) >= 3:
            insights.append(f"Among frontier models, latency and intelligence have correlation {lat.intelligence.corr(lat.ttft):.2f}.")
    leaders = df.dropna(subset=["intelligence"]).sort_values("intelligence", ascending=False).head(20)
    if not leaders.empty:
        insights.append(f"{leaders.provider.value_counts().idxmax()} appears most often among the top 20 releases by intelligence in the filtered dataset.")
    return insights[:6]


def filter_data(df):
    st.sidebar.title("Model Filters")
    providers = sorted(df.provider.dropna().unique())
    selected = st.sidebar.multiselect("Provider", providers, default=providers)
    min_date, max_date = df.release_date.min(), df.release_date.max()
    date_range = st.sidebar.date_input("Release date range", value=(min_date.date(), max_date.date()), min_value=min_date.date(), max_value=max_date.date())
    max_price = st.sidebar.slider("Max blended price / 1M tokens", 0.0, float(df.blended_price.max(skipna=True)), float(df.blended_price.quantile(0.95)), step=0.01)
    min_intel = st.sidebar.slider("Minimum intelligence", float(df.intelligence.min(skipna=True)), float(df.intelligence.max(skipna=True)), float(df.intelligence.min(skipna=True)), step=0.1)
    require_core = st.sidebar.checkbox("Require non-null core benchmarks", value=False)
    search = st.sidebar.text_input("Search model name", "")
    out = df[df.provider.isin(selected)].copy()
    if len(date_range) == 2:
        out = out[out.release_date.between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])) | out.release_date.isna()]
    out = out[(out.blended_price.isna()) | (out.blended_price <= max_price)]
    out = out[(out.intelligence.isna()) | (out.intelligence >= min_intel)]
    if require_core:
        out = out.dropna(subset=CORE)
    if search.strip():
        out = out[out.name.str.contains(search.strip(), case=False, na=False)]
    st.sidebar.download_button("Download filtered CSV", out.to_csv(index=False).encode("utf-8"), "filtered_ai_models.csv", "text/csv", width="stretch")
    st.sidebar.caption(f"{len(out):,} of {len(df):,} models visible")
    return out


def scatter(df, x, y, title, color="provider", size="blended_price"):
    plot = df.dropna(subset=[x, y])
    if plot.empty:
        st.info(f"No complete data for {title}.")
        return
    fig = px.scatter(plot, x=x, y=y, color=color, size=size if size in plot and plot[size].notna().any() else None, hover_name="name", hover_data={"release_date": "|%Y-%m-%d", "intelligence": ":.2f", "blended_price": ":.3f", "bes": ":.2f", "agent_readiness_index": ":.3f"}, labels=LABELS, title=title, template="plotly_white")
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(fig, width="stretch", key=f"{title}-{x}-{y}")


def metric_row(df):
    cols = st.columns(6)
    vals = [("Models", len(df)), ("Providers", df.provider.nunique()), ("Median intelligence", df.intelligence.median()), ("Median price", df.blended_price.median()), ("Median speed", df.output_tps.median()), ("Median latency", df.ttft.median())]
    for col, (label, value) in zip(cols, vals):
        col.metric(label, "NA" if pd.isna(value) else f"{value:,.2f}" if isinstance(value, float) else f"{value:,}")


def main():
    st.title("The Economics of Intelligence")
    st.caption("Measuring Progress in Frontier AI Models through release-date trends, benchmark performance, pricing, latency, throughput, and efficiency.")
    df = load_data()
    filtered = filter_data(df)
    tabs = st.tabs(["Overview", "Research Trends", "Rankings", "Benchmark Explorer", "Efficiency Frontier", "Model Profile", "Benchmark Coverage"])

    with tabs[0]:
        metric_row(filtered)
        c1, c2, c3 = st.columns(3)
        for col, metric, title in [(c1, "intelligence", "Top Intelligence"), (c2, "bes", "Top BES"), (c3, "cai", "Top CAI")]:
            col.subheader(title)
            col.dataframe(display_df(filtered.dropna(subset=[metric]).sort_values(metric, ascending=False).head(10), ["name", "provider", "release_date", "intelligence", "blended_price", metric]), width="stretch", hide_index=True)

    with tabs[1]:
        st.subheader("Release-Date Research Trends")
        st.caption("These are release-cohort trends from one JSON file, not repeated snapshot comparisons.")
        for insight in generate_release_trend_insights(filtered):
            st.markdown(f"- {insight}")
        monthly, provider_monthly = monthly_summary(filtered), provider_monthly_summary(filtered)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.line(monthly, x="release_month", y=["max_intelligence", "median_intelligence"], markers=True, title="Capability Progress by Release Month", template="plotly_white")
            st.plotly_chart(fig, width="stretch")
        with c2:
            fig = px.line(monthly, x="release_month", y=["max_agent_readiness", "median_agent_readiness"], markers=True, title="Agent Readiness by Release Month", template="plotly_white")
            st.plotly_chart(fig, width="stretch")
        fig = px.line(monthly, x="release_month", y=["min_cost_per_intelligence", "median_cost_per_intelligence"], markers=True, title="Cost of Intelligence Over Time", template="plotly_white")
        st.plotly_chart(fig, width="stretch")
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(monthly, x="release_month", y="model_count", title="Release Velocity", labels=LABELS, template="plotly_white")
            st.plotly_chart(fig, width="stretch")
        with c2:
            fig = px.bar(provider_monthly, x="release_month", y="model_count", color="provider", title="Release Count by Provider", template="plotly_white")
            st.plotly_chart(fig, width="stretch")
        scatter(filtered, "release_date", "intelligence", "Intelligence vs Release Date")
        scatter(filtered, "release_date", "cai", "CAI vs Release Date")
        scatter(filtered, "release_date", "agent_readiness_index", "ARI vs Release Date")

    with tabs[2]:
        metric = st.selectbox("Ranking metric", ["intelligence", "cai", "sai", "ttv", "bes", "agent_readiness_index", "cost_to_agent_score", "throughput_efficiency"], format_func=lambda x: LABELS.get(x, x))
        st.dataframe(display_df(filtered.dropna(subset=[metric]).sort_values(metric, ascending=False), ["name", "provider", "release_date", "intelligence", "blended_price", "output_tps", "ttft", metric, "agent_readiness_index"]), width="stretch", hide_index=True)

    with tabs[3]:
        for x, y, title in [("blended_price", "intelligence", "Intelligence vs Price"), ("output_tps", "intelligence", "Intelligence vs Speed"), ("ttft", "intelligence", "Latency vs Intelligence"), ("gpqa", "lcr", "GPQA vs LCR"), ("lcr", "terminalbench_hard", "LCR vs TerminalBench")]:
            scatter(filtered, x, y, title)

    with tabs[4]:
        frontier = build_frontier(filtered)
        if frontier.empty:
            st.info("No complete price-intelligence data available.")
        else:
            fig = px.scatter(frontier, x="blended_price", y="intelligence", color="is_frontier", symbol="is_frontier", hover_name="name", hover_data={"provider": True, "release_date": "|%Y-%m-%d", "bes": ":.2f", "agent_readiness_index": ":.3f"}, labels=LABELS, title="Price-Intelligence Pareto Frontier", template="plotly_white")
            line = frontier[frontier.is_frontier].sort_values("blended_price")
            fig.add_trace(go.Scatter(x=line.blended_price, y=line.intelligence, mode="lines", name="Frontier line", line=dict(color="#111827", width=3), hoverinfo="skip"))
            st.plotly_chart(fig, width="stretch")
            st.dataframe(display_df(line.sort_values("release_date"), ["name", "provider", "release_date", "intelligence", "blended_price", "cai", "bes"]), width="stretch", hide_index=True)

    with tabs[5]:
        if filtered.empty:
            st.info("No models match current filters.")
        else:
            profiled = add_frontier_flag(filtered)
            name = st.selectbox("Select a model", profiled.sort_values(["provider", "name"]).name.tolist())
            row = profiled[profiled.name == name].iloc[0]
            st.subheader(row["name"])
            st.caption(f"{row.provider} | {row.release_date.date() if pd.notna(row.release_date) else 'Unknown release date'} | Pareto frontier: {'Yes' if row.is_frontier else 'No'}")
            values = pd.DataFrame({"Metric": [LABELS.get(c, c) for c in BENCHMARKS if c in row.index], "Score": [row[c] for c in BENCHMARKS if c in row.index]}).dropna()
            fig = px.bar(values, x="Score", y="Metric", orientation="h", title="Benchmark Profile", template="plotly_white")
            st.plotly_chart(fig, width="stretch")
            c1, c2 = st.columns(2)
            c1.dataframe(pd.DataFrame({"Metric": [LABELS.get(c, c) for c in DERIVED], "Value": [row[c] for c in DERIVED]}).round(4), width="stretch", hide_index=True)
            med = pd.DataFrame({"Metric": [LABELS.get(c, c) for c in ["intelligence", "blended_price", "output_tps", "ttft", "cai", "bes", "agent_readiness_index"]], "Model": [row[c] for c in ["intelligence", "blended_price", "output_tps", "ttft", "cai", "bes", "agent_readiness_index"]], "Dataset median": [filtered[c].median() for c in ["intelligence", "blended_price", "output_tps", "ttft", "cai", "bes", "agent_readiness_index"]]})
            c2.dataframe(med.round(4), width="stretch", hide_index=True)

    with tabs[6]:
        coverage = pd.DataFrame({"Benchmark": [LABELS.get(c, c) for c in BENCHMARKS], "Non-null models": [filtered[c].notna().sum() for c in BENCHMARKS], "Coverage %": [filtered[c].notna().mean() * 100 for c in BENCHMARKS]})
        st.dataframe(coverage.round(1), width="stretch", hide_index=True)
        provider_cov = filtered.groupby("provider", as_index=False).agg(models=("name", "count"), avg_coverage=("benchmark_coverage_count", "mean"))
        st.dataframe(provider_cov.round(2), width="stretch", hide_index=True)
        year_cov = filtered.dropna(subset=["release_year"]).groupby("release_year", as_index=False).agg(models=("name", "count"), avg_coverage=("benchmark_coverage_count", "mean"))
        st.dataframe(year_cov.round(2), width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
