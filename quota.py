import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Quota Purchases - Leadership Dashboard", layout="wide")

# ============================================================
# Pricing Rules (Provided by you)
# ============================================================
LOCAL_SEO_PRICE_PER_1000 = 10.0  # USD per 1000 quota

# Keyword Rank Tracker tier prices
# quota -> price
KRT_TIER_MAP = {
    50: 4,
    100: 7,
    200: 10,
    500: 22,
    1000: 33,
    2000: 61,
    3000: 92,
    4000: 123,
    5000: 143,
}
KRT_BEYOND_5000_PRICE_PER_QUOTA = 0.033  # USD per quota beyond 5000


# ============================================================
# Expected pricing calculators
# ============================================================
def expected_local_seo_price(quota: float) -> float:
    """
    Local SEO: 10 USD per 1000 quota
    """
    if pd.isna(quota) or quota <= 0:
        return 0.0
    return (quota / 1000.0) * LOCAL_SEO_PRICE_PER_1000


def expected_krt_price(quota: float) -> float:
    """
    Keyword Rank Tracker:
    - If quota <= 5000: must match exact tier quantity
    - If quota > 5000: $143 + (quota-5000)*0.033
    """
    if pd.isna(quota) or quota <= 0:
        return 0.0

    if quota > 5000:
        return float(KRT_TIER_MAP[5000] + (quota - 5000) * KRT_BEYOND_5000_PRICE_PER_QUOTA)

    # exact tier only
    if int(quota) in KRT_TIER_MAP:
        return float(KRT_TIER_MAP[int(quota)])

    # invalid tier quantity
    return np.nan


def safe_div(n, d):
    if d is None or d == 0 or pd.isna(d):
        return np.nan
    return n / d


# ============================================================
# Load + enrich dataset
# ============================================================
@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)

    # Parse dates
    df["first_purchase_date"] = pd.to_datetime(df["first_purchase_date"], format="%d/%m/%y", errors="coerce")
    df["last_purchase_date"] = pd.to_datetime(df["last_purchase_date"], format="%d/%m/%y", errors="coerce")

    # Purchase span
    df["purchase_span_days"] = (df["last_purchase_date"] - df["first_purchase_date"]).dt.days.fillna(0).astype(int)

    # Segment
    df["product_segment"] = np.select(
        [
            (df["local_seo_quantity_bought"] > 0) & (df["keyword_rank_quantity_bought"] == 0),
            (df["keyword_rank_quantity_bought"] > 0) & (df["local_seo_quantity_bought"] == 0),
            (df["local_seo_quantity_bought"] > 0) & (df["keyword_rank_quantity_bought"] > 0),
        ],
        ["Local SEO Only", "Keyword Rank Only", "Both"],
        default="None"
    )

    # Expected amounts
    df["expected_local_seo_amount"] = df["local_seo_quantity_bought"].apply(expected_local_seo_price)
    df["expected_keyword_rank_amount"] = df["keyword_rank_quantity_bought"].apply(expected_krt_price)

    df["expected_total_amount"] = df["expected_local_seo_amount"].fillna(0) + df["expected_keyword_rank_amount"].fillna(0)

    # Deltas (paid - expected)
    df["local_seo_delta"] = df["local_seo_amount_paid"] - df["expected_local_seo_amount"]
    df["keyword_rank_delta"] = df["keyword_rank_amount_paid"] - df["expected_keyword_rank_amount"]
    df["total_delta"] = df["total_amount_paid"] - df["expected_total_amount"]

    # Flags
    df["keyword_invalid_tier"] = (
        (df["keyword_rank_quantity_bought"] > 0) &
        (df["keyword_rank_quantity_bought"] <= 5000) &
        (df["expected_keyword_rank_amount"].isna())
    )

    # Unit economics (actual)
    df["local_seo_actual_price_per_quota"] = np.where(
        df["local_seo_quantity_bought"] > 0,
        df["local_seo_amount_paid"] / df["local_seo_quantity_bought"],
        np.nan
    )
    df["keyword_rank_actual_price_per_quota"] = np.where(
        df["keyword_rank_quantity_bought"] > 0,
        df["keyword_rank_amount_paid"] / df["keyword_rank_quantity_bought"],
        np.nan
    )
    df["total_actual_price_per_quota"] = np.where(
        df["total_quantity_bought"] > 0,
        df["total_amount_paid"] / df["total_quantity_bought"],
        np.nan
    )

    # Unit economics (expected)
    df["local_seo_expected_price_per_quota"] = np.where(
        df["local_seo_quantity_bought"] > 0,
        df["expected_local_seo_amount"] / df["local_seo_quantity_bought"],
        np.nan
    )
    df["keyword_rank_expected_price_per_quota"] = np.where(
        df["keyword_rank_quantity_bought"] > 0,
        df["expected_keyword_rank_amount"] / df["keyword_rank_quantity_bought"],
        np.nan
    )

    # Whale detection (high volume)
    df["is_whale_quantity"] = df["total_quantity_bought"] >= 1_000_000
    df["is_whale_revenue"] = df["total_amount_paid"] >= 200  # adjustable heuristic

    return df


# ============================================================
# UI Header
# ============================================================
st.title("📊 Quota Purchases — Leadership Dashboard")
st.caption("Local SEO Points + Keyword Rank Tracker | Pricing Mismatch & Revenue Risk Analysis")

uploaded = st.file_uploader("Upload your CSV file", type=["csv"])
if not uploaded:
    st.info("Upload the CSV file to begin.")
    st.stop()

df = load_data(uploaded)

# ============================================================
# Sidebar controls
# ============================================================
st.sidebar.header("Filters & Controls")

segments = sorted(df["product_segment"].unique())
segment_filter = st.sidebar.multiselect("Product Segment", segments, default=segments)

# Leadership-friendly mismatch tolerance
tolerance = st.sidebar.number_input(
    "Mismatch tolerance (USD)",
    min_value=0.0,
    value=0.50,
    step=0.10,
    help="If absolute difference between expected and paid exceeds this, it's flagged."
)

show_only_mismatches = st.sidebar.checkbox("Show only mismatches", value=False)
show_whales_only = st.sidebar.checkbox("Show whales only (>= 1M quota)", value=False)

# Apply filters
filtered = df[df["product_segment"].isin(segment_filter)].copy()

# Flags based on tolerance
filtered["local_seo_mismatch"] = filtered["local_seo_delta"].abs() > tolerance
filtered["keyword_rank_mismatch"] = filtered["keyword_rank_delta"].abs() > tolerance
filtered["total_mismatch"] = filtered["total_delta"].abs() > tolerance

if show_whales_only:
    filtered = filtered[filtered["is_whale_quantity"]].copy()

if show_only_mismatches:
    filtered = filtered[
        filtered["local_seo_mismatch"] |
        filtered["keyword_rank_mismatch"] |
        filtered["total_mismatch"] |
        filtered["keyword_invalid_tier"]
    ].copy()

# ============================================================
# Leadership KPIs
# ============================================================
total_customers = filtered["customer_id"].nunique()
actual_revenue = filtered["total_amount_paid"].sum()
expected_revenue = filtered["expected_total_amount"].sum()
revenue_delta = actual_revenue - expected_revenue

mismatch_rows = filtered[
    filtered["local_seo_mismatch"] |
    filtered["keyword_rank_mismatch"] |
    filtered["total_mismatch"] |
    filtered["keyword_invalid_tier"]
].copy()

mismatch_count = len(mismatch_rows)
mismatch_rate = (mismatch_count / len(filtered) * 100) if len(filtered) else 0

# Risk view:
# Underpayment (expected > paid) => negative delta
# Overpayment (paid > expected) => positive delta
underpayment_total = mismatch_rows[mismatch_rows["total_delta"] < 0]["total_delta"].sum()
overpayment_total = mismatch_rows[mismatch_rows["total_delta"] > 0]["total_delta"].sum()

# Convert underpayment to positive "leakage amount"
revenue_leakage = abs(underpayment_total)
overcharge_risk = overpayment_total

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Customers", f"{total_customers:,}")
k2.metric("Actual Revenue", f"${actual_revenue:,.2f}")
k3.metric("Expected Revenue", f"${expected_revenue:,.2f}")
k4.metric("Revenue Delta", f"${revenue_delta:,.2f}")
k5.metric("Mismatch Rate", f"{mismatch_rate:.1f}%")
k6.metric("Mismatch Rows", f"{mismatch_count:,}")

k7, k8, k9 = st.columns(3)
k7.metric("Revenue Leakage (Underpayment)", f"${revenue_leakage:,.2f}")
k8.metric("Overcharge Risk (Customer Impact)", f"${overcharge_risk:,.2f}")
k9.metric("Keyword Invalid Tier Rows", f"{int(filtered['keyword_invalid_tier'].sum()):,}")

st.divider()

# ============================================================
# Executive Summary (Auto narrative)
# ============================================================
st.subheader("🧠 Executive Summary (Auto-Generated)")

local_actual = filtered["local_seo_amount_paid"].sum()
keyword_actual = filtered["keyword_rank_amount_paid"].sum()

local_expected = filtered["expected_local_seo_amount"].sum()
keyword_expected = filtered["expected_keyword_rank_amount"].fillna(0).sum()

seg_mix = filtered["product_segment"].value_counts(normalize=True) * 100
cross_sell_rate = seg_mix.get("Both", 0)

one_time_rate = (filtered["purchase_span_days"] == 0).mean() * 100 if len(filtered) else 0

summary_left, summary_right = st.columns([1.2, 1])

with summary_left:
    st.markdown(f"""
### Key takeaways
- **Revenue (Actual):** **${actual_revenue:,.2f}** vs **Expected:** **${expected_revenue:,.2f}**
- **Revenue Delta:** **${revenue_delta:,.2f}**  
  - **Leakage risk (underpayment):** **${revenue_leakage:,.2f}**
  - **Overcharge risk:** **${overcharge_risk:,.2f}**
- **Mismatch rate:** **{mismatch_rate:.1f}%** (tolerance = **${tolerance:.2f}**)
- **Product mix:**
  - Local SEO actual revenue: **${local_actual:,.2f}** (expected **${local_expected:,.2f}**)
  - Keyword Rank actual revenue: **${keyword_actual:,.2f}** (expected **${keyword_expected:,.2f}**)
- **Cross-sell rate (Both products):** **{cross_sell_rate:.1f}%**
- **One-time purchase behavior:** **{one_time_rate:.1f}%** of customers purchased only on a single day in this window
""")

with summary_right:
    st.markdown("### Leadership actions this enables")
    st.markdown("""
1. **Identify revenue leakage** from quota grants not matching payments  
2. **Protect customer experience** by catching overcharges early  
3. **Validate Unified Quota rollout readiness** by isolating enterprise/whale accounts  
4. **Quantify pricing tier compliance** for Keyword Rank Tracker  
""")

st.divider()

# ============================================================
# Charts
# ============================================================
st.subheader("📈 Revenue & Customer Breakdown")

cA, cB = st.columns(2)

with cA:
    seg_counts = filtered["product_segment"].value_counts().reset_index()
    seg_counts.columns = ["segment", "customers"]
    fig_seg = px.pie(seg_counts, names="segment", values="customers", title="Customer Mix by Product Segment")
    st.plotly_chart(fig_seg, use_container_width=True)

with cB:
    rev_split = pd.DataFrame({
        "product": ["Local SEO", "Keyword Rank"],
        "actual_revenue": [local_actual, keyword_actual],
        "expected_revenue": [local_expected, keyword_expected],
    })
    fig_rev = px.bar(
        rev_split.melt(id_vars="product", value_vars=["actual_revenue", "expected_revenue"]),
        x="product",
        y="value",
        color="variable",
        barmode="group",
        title="Actual vs Expected Revenue by Product"
    )
    st.plotly_chart(fig_rev, use_container_width=True)

st.divider()

# ============================================================
# Timeline
# ============================================================
st.subheader("📅 Timeline: Revenue Trend (Actual vs Expected)")

basis = st.radio("Group by:", ["first_purchase_date", "last_purchase_date"], horizontal=True)
tmp = filtered.dropna(subset=[basis]).copy()
tmp["month"] = tmp[basis].dt.to_period("M").astype(str)

monthly = tmp.groupby("month", as_index=False).agg(
    actual_revenue=("total_amount_paid", "sum"),
    expected_revenue=("expected_total_amount", "sum"),
    customers=("customer_id", "nunique"),
    mismatches=("total_mismatch", "sum")
)

fig_month = px.line(
    monthly.melt(id_vars="month", value_vars=["actual_revenue", "expected_revenue"]),
    x="month",
    y="value",
    color="variable",
    markers=True,
    title="Actual vs Expected Revenue by Month"
)
st.plotly_chart(fig_month, use_container_width=True)

st.divider()

# ============================================================
# Mismatch Analysis
# ============================================================
st.subheader("🚨 Mismatch Analysis")

m1, m2 = st.columns(2)

with m1:
    # Delta distribution
    fig_delta = px.histogram(
        filtered,
        x="total_delta",
        nbins=40,
        title="Distribution: Total Delta (Paid - Expected)"
    )
    st.plotly_chart(fig_delta, use_container_width=True)

with m2:
    mismatch_breakdown = pd.DataFrame({
        "Mismatch Type": ["Local SEO mismatch", "Keyword Rank mismatch", "Total mismatch", "Keyword invalid tier"],
        "Count": [
            int(filtered["local_seo_mismatch"].sum()),
            int(filtered["keyword_rank_mismatch"].sum()),
            int(filtered["total_mismatch"].sum()),
            int(filtered["keyword_invalid_tier"].sum()),
        ]
    })
    fig_break = px.bar(mismatch_breakdown, x="Mismatch Type", y="Count", title="Mismatch Breakdown")
    st.plotly_chart(fig_break, use_container_width=True)

st.divider()

# ============================================================
# Top Customers (Revenue + Quantity)
# ============================================================
st.subheader("🏆 Top Customers")

top_left, top_right = st.columns(2)

with top_left:
    st.markdown("### Top by Revenue")
    top_rev = filtered.sort_values("total_amount_paid", ascending=False).head(20)
    st.dataframe(
        top_rev[[
            "customer_email",
            "product_segment",
            "total_amount_paid",
            "expected_total_amount",
            "total_delta",
            "total_quantity_bought",
            "purchase_span_days",
        ]],
        use_container_width=True
    )

with top_right:
    st.markdown("### Top by Quota Quantity")
    top_qty = filtered.sort_values("total_quantity_bought", ascending=False).head(20)
    st.dataframe(
        top_qty[[
            "customer_email",
            "product_segment",
            "total_quantity_bought",
            "total_amount_paid",
            "expected_total_amount",
            "total_delta",
            "is_whale_quantity",
        ]],
        use_container_width=True
    )

st.divider()

# ============================================================
# Full mismatch table + downloads
# ============================================================
st.subheader("📋 Detailed Mismatch Table (Exportable)")

display_cols = [
    "customer_email",
    "product_segment",

    "local_seo_quantity_bought",
    "local_seo_amount_paid",
    "expected_local_seo_amount",
    "local_seo_delta",
    "local_seo_mismatch",

    "keyword_rank_quantity_bought",
    "keyword_rank_amount_paid",
    "expected_keyword_rank_amount",
    "keyword_rank_delta",
    "keyword_rank_mismatch",
    "keyword_invalid_tier",

    "total_amount_paid",
    "expected_total_amount",
    "total_delta",
    "total_mismatch",

    "is_whale_quantity",
    "purchase_span_days",
    "first_purchase_date",
    "last_purchase_date",
]

mismatch_sorted = mismatch_rows.sort_values("total_delta").copy()

st.dataframe(mismatch_sorted[display_cols], use_container_width=True)

# Download mismatch report
mismatch_csv = mismatch_sorted[display_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download mismatch report (CSV)",
    data=mismatch_csv,
    file_name="quota_payment_mismatch_report.csv",
    mime="text/csv"
)

# Download full enriched dataset
enriched_csv = filtered[display_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download filtered enriched dataset (CSV)",
    data=enriched_csv,
    file_name="quota_dashboard_enriched_dataset.csv",
    mime="text/csv"
)

st.divider()

# ============================================================
# Explain pricing logic (for leadership + finance)
# ============================================================
with st.expander("ℹ️ Pricing Rules Used (for audit / finance validation)"):
    st.markdown("""
### Local SEO pricing
- **$10 per 1000 quota**
- Expected = `(local_seo_quantity_bought / 1000) × 10`

### Keyword Rank Tracker pricing
Tier-based pricing:
- 50 quota = $4  
- 100 quota = $7  
- 200 quota = $10  
- 500 quota = $22  
- 1000 quota = $33  
- 2000 quota = $61  
- 3000 quota = $92  
- 4000 quota = $123  
- 5000 quota = $143  

Beyond 5000 quota:
- Expected = `$143 + (quota - 5000) × 0.033`

### Mismatch logic
A row is flagged if:
- `abs(paid - expected) > tolerance`

Also flagged if:
- Keyword quota <= 5000 is **not exactly one of the tier values** (invalid tier)
""")
