"""
Streamlit dashboard for the PySpark e-commerce segmentation project.

The heavy lifting (cleaning, RFM, K-Means, FP-Growth) is done by pipeline.py with PySpark.
This app only reads the small aggregated CSVs next to it, so it deploys on
Streamlit Community Cloud without needing Java or Spark.

Run locally:  streamlit run app.py
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

OUT = Path(__file__).parent

SEGMENT_COLORS = {
    "Champions": "#2E7D32",
    "Loyal Customers": "#1E88E5",
    "Potential Loyalists": "#8E24AA",
    "At-Risk": "#FB8C00",
    "Lost / Hibernating": "#C62828",
}

SEGMENT_ACTIONS = {
    "Champions": "Reward them: early access, loyalty perks, ask for reviews and referrals.",
    "Loyal Customers": "Upsell higher-value products, offer bundles and membership programs.",
    "Potential Loyalists": "Nudge a repeat purchase with personalised recommendations.",
    "At-Risk": "Win-back emails, time-limited discounts, 'we miss you' campaigns.",
    "Lost / Hibernating": "Low-cost reactivation only; survey them to learn why they left.",
}

st.set_page_config(page_title="E-commerce Customer Segmentation", page_icon="🛒", layout="wide")


@st.cache_data
def load(name):
    path = OUT / name
    return pd.read_csv(path) if path.exists() else None


def money(x):
    return f"£{x:,.0f}"


kpis = load("kpis.csv")
monthly = load("monthly_revenue.csv")
products = load("top_products.csv")
countries = load("country_revenue.csv")
hourly = load("hourly_sales.csv")
rfm = load("rfm_segments.csv")
profile = load("cluster_profile.csv")
k_sel = load("k_selection.csv")
rules = load("association_rules.csv")

if kpis is None or rfm is None:
    st.error("No result CSV files found next to app.py. Run `python pipeline.py` first to generate them.")
    st.stop()

st.title("🛒 E-commerce Sales & Customer Segmentation")
st.caption("Big data project built with Apache Spark (PySpark) · Spark SQL · MLlib K-Means · FP-Growth")

with st.sidebar:
    st.header("About")
    st.write(
        "Transactions were cleaned and aggregated with **PySpark**. Customers were scored on "
        "**RFM** (Recency, Frequency, Monetary value) and grouped with **K-Means** from Spark MLlib."
    )
    st.markdown("**Pipeline:** Ingest → Clean → Spark SQL → RFM → K-Means → FP-Growth → Dashboard")
    k = kpis.iloc[0]
    st.divider()
    st.metric("Raw rows processed", f"{int(k.raw_rows):,}")
    st.metric("Rows after cleaning", f"{int(k.clean_rows):,}")

tab_overview, tab_sales, tab_seg, tab_lookup, tab_basket = st.tabs(
    ["📊 Overview", "🛍️ Sales Explorer", "👥 Customer Segments", "🔎 Customer Lookup", "🧺 Market Basket"]
)

# ------------------------------------------------------------------ Overview
with tab_overview:
    k = kpis.iloc[0]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total revenue", money(k.total_revenue))
    c2.metric("Orders", f"{int(k.total_orders):,}")
    c3.metric("Customers", f"{int(k.total_customers):,}")
    c4.metric("Products", f"{int(k.total_products):,}")
    c5.metric("Avg order value", f"£{k.avg_order_value:,.2f}")

    st.subheader("Monthly revenue")
    fig = px.bar(monthly, x="month", y="revenue", labels={"month": "Month", "revenue": "Revenue (£)"})
    fig.update_layout(height=380, margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Orders and active customers per month")
        m = monthly.melt(id_vars="month", value_vars=["orders", "customers"])
        fig = px.line(m, x="month", y="value", color="variable", markers=True,
                      labels={"value": "Count", "month": "Month", "variable": ""})
        fig.update_layout(height=340, margin=dict(t=20))
        st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader("Revenue by segment")
        fig = px.pie(profile, names="segment", values="total_revenue", hole=0.45,
                     color="segment", color_discrete_map=SEGMENT_COLORS)
        fig.update_layout(height=340, margin=dict(t=20))
        st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ Sales explorer
with tab_sales:
    left, right = st.columns(2)
    with left:
        st.subheader("Top products by revenue")
        n = st.slider("Number of products", 5, 30, 10)
        top = products.head(n).sort_values("revenue")
        fig = px.bar(top, x="revenue", y="product", orientation="h",
                     labels={"revenue": "Revenue (£)", "product": ""})
        fig.update_layout(height=max(320, 28 * n), margin=dict(t=20))
        st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader("Revenue by country")
        view = st.radio("View", ["All countries", "Excluding United Kingdom"], horizontal=True)
        c = countries if view == "All countries" else countries[countries.country != "United Kingdom"]
        fig = px.bar(c.sort_values("revenue"), x="revenue", y="country", orientation="h",
                     labels={"revenue": "Revenue (£)", "country": ""})
        fig.update_layout(height=420, margin=dict(t=20))
        st.plotly_chart(fig, width="stretch")

    st.subheader("When do customers buy?")
    day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    metric = st.radio("Metric", ["revenue", "orders"], horizontal=True)
    heat = hourly.pivot_table(index="weekday", columns="hour", values=metric, aggfunc="sum").fillna(0)
    heat = heat.reindex([d for d in day_order if d in heat.index])
    fig = px.imshow(heat, aspect="auto", color_continuous_scale="Blues",
                    labels=dict(x="Hour of day", y="Weekday", color=metric.title()))
    fig.update_layout(height=360, margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ Segments
with tab_seg:
    st.subheader("Segment profiles")
    show = profile[["segment", "customers", "avg_recency", "avg_frequency", "avg_monetary",
                    "revenue_share_pct"]].rename(columns={
        "segment": "Segment", "customers": "Customers", "avg_recency": "Avg recency (days)",
        "avg_frequency": "Avg orders", "avg_monetary": "Avg spend (£)",
        "revenue_share_pct": "Revenue share (%)"})
    st.dataframe(show, hide_index=True, width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Customers per segment")
        fig = px.bar(profile.sort_values("customers"), x="customers", y="segment", orientation="h",
                     color="segment", color_discrete_map=SEGMENT_COLORS,
                     labels={"customers": "Customers", "segment": ""})
        fig.update_layout(showlegend=False, height=320, margin=dict(t=20))
        st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader("Revenue share per segment")
        fig = px.bar(profile.sort_values("revenue_share_pct"), x="revenue_share_pct", y="segment",
                     orientation="h", color="segment", color_discrete_map=SEGMENT_COLORS,
                     labels={"revenue_share_pct": "% of revenue", "segment": ""})
        fig.update_layout(showlegend=False, height=320, margin=dict(t=20))
        st.plotly_chart(fig, width="stretch")

    st.subheader("RFM scatter")
    chosen = st.multiselect("Segments to show", sorted(rfm.segment.unique()),
                            default=sorted(rfm.segment.unique()))
    sub = rfm[rfm.segment.isin(chosen)]
    mode = st.radio("Plot", ["2D (Frequency vs Monetary)", "3D (Recency, Frequency, Monetary)"], horizontal=True)
    if mode.startswith("2D"):
        fig = px.scatter(sub, x="Frequency", y="Monetary", color="segment", opacity=0.6,
                         color_discrete_map=SEGMENT_COLORS, log_x=True, log_y=True,
                         hover_data=["CustomerID", "Recency"])
    else:
        fig = px.scatter_3d(sub, x="Recency", y="Frequency", z="Monetary", color="segment",
                            opacity=0.7, color_discrete_map=SEGMENT_COLORS, log_y=True, log_z=True,
                            hover_data=["CustomerID"])
    fig.update_layout(height=520, margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Recommended actions")
    for seg in profile.segment:
        st.markdown(f"**{seg}** — {SEGMENT_ACTIONS.get(seg, 'Tailor campaigns to this group.')}")

    if k_sel is not None:
        with st.expander("How was the number of clusters chosen?"):
            a, b = st.columns(2)
            with a:
                fig = px.line(k_sel, x="k", y="silhouette", markers=True, title="Silhouette score (higher is better)")
                fig.update_layout(height=300)
                st.plotly_chart(fig, width="stretch")
            with b:
                fig = px.line(k_sel, x="k", y="wssse", markers=True, title="Elbow curve (WSSSE)")
                fig.update_layout(height=300)
                st.plotly_chart(fig, width="stretch")
            st.caption("k = 4 gives a good balance between a high silhouette score and segments "
                       "that are distinct and actionable for marketing.")

# ------------------------------------------------------------------ Lookup
with tab_lookup:
    st.subheader("Find a customer")
    cid = st.number_input("Customer ID", min_value=int(rfm.CustomerID.min()),
                          max_value=int(rfm.CustomerID.max()), value=int(rfm.CustomerID.min()), step=1)
    row = rfm[rfm.CustomerID == cid]
    if row.empty:
        st.warning("Customer not found (they may have been removed during cleaning).")
    else:
        r = row.iloc[0]
        a, b, c, d = st.columns(4)
        a.metric("Segment", r.segment)
        b.metric("Recency", f"{int(r.Recency)} days")
        c.metric("Orders", int(r.Frequency))
        d.metric("Total spend", f"£{r.Monetary:,.2f}")
        st.info(SEGMENT_ACTIONS.get(r.segment, ""))

    st.subheader("Download segmented customers")
    seg_filter = st.multiselect("Filter by segment", sorted(rfm.segment.unique()),
                                default=sorted(rfm.segment.unique()), key="dl")
    dl = rfm[rfm.segment.isin(seg_filter)].drop(columns=["cluster"])
    st.dataframe(dl, hide_index=True, width="stretch", height=300)
    st.download_button("⬇️ Download CSV", dl.to_csv(index=False).encode(), "customer_segments.csv", "text/csv")

# ------------------------------------------------------------------ Basket
with tab_basket:
    st.subheader("Frequently bought together (FP-Growth)")
    if rules is None or rules.empty:
        st.info("No association rules available.")
    else:
        min_lift = st.slider("Minimum lift", 1.0, float(max(2.0, rules.lift.max())), 1.5, 0.1)
        min_conf = st.slider("Minimum confidence", 0.0, 1.0, 0.3, 0.05)
        r = rules[(rules.lift >= min_lift) & (rules.confidence >= min_conf)]
        st.caption("Lift > 1 means the items are bought together more often than chance would predict.")
        st.dataframe(r.rename(columns={"antecedent": "If customer buys", "consequent": "They also buy"}),
                     hide_index=True, width="stretch")
        if not r.empty:
            fig = px.scatter(r, x="support", y="confidence", size="lift", color="lift",
                             hover_data=["antecedent", "consequent"], color_continuous_scale="Viridis")
            fig.update_layout(height=380)
            st.plotly_chart(fig, width="stretch")
