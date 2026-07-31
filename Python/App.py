import streamlit as st
import pandas as pd
import plotly.express as px

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Buyer Segmentation Dashboard",
    page_icon="🏠",
    layout="wide"
)

# --------------------------------------------------
# Dashboard Title
# --------------------------------------------------
st.title("🏠 Machine Learning Based Buyer Segmentation and Investment Profiling")

st.markdown("""
This dashboard provides interactive insights into customer demographics,
investment behaviour, property transactions, and machine learning-based buyer segmentation.
""")

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("../Dataset/final_clustered_data.xlsx")
    return df

df = load_data()
# --------------------------------------------------
# Sidebar Filters
# --------------------------------------------------
st.sidebar.header("🔍 Filters")
st.sidebar.markdown("""
### 📌 Project Details

**Algorithm Used:** K-Means Clustering

**No. of Clusters:** 3

**Silhouette Score:** 0.302

**Dashboard Type:** Interactive Real Estate Analytics

**Machine Learning:** Customer Segmentation
""")

country = st.sidebar.multiselect(
    "Country",
    options=sorted(df["country"].unique()),
    default=sorted(df["country"].unique())
)

region = st.sidebar.multiselect(
    "Region",
    options=sorted(df["region"].unique()),
    default=sorted(df["region"].unique())
)

client_type = st.sidebar.multiselect(
    "Client Type",
    options=sorted(df["client_type"].unique()),
    default=sorted(df["client_type"].unique())
)

cluster = st.sidebar.multiselect(
    "Cluster",
    options=sorted(df["Cluster"].unique()),
    default=sorted(df["Cluster"].unique())
)

purpose = st.sidebar.multiselect(
    "Acquisition Purpose",
    options=sorted(df["acquisition_purpose"].unique()),
    default=sorted(df["acquisition_purpose"].unique())
)

loan = st.sidebar.multiselect(
    "Loan Applied",
    options=sorted(df["loan_applied"].unique()),
    default=sorted(df["loan_applied"].unique())
)
filtered = df[
    (df["country"].isin(country)) &
    (df["region"].isin(region)) &
    (df["client_type"].isin(client_type)) &
    (df["Cluster"].isin(cluster)) &
    (df["acquisition_purpose"].isin(purpose)) &
    (df["loan_applied"].isin(loan))
]
# --------------------------------------------------
# KPI Cards
# --------------------------------------------------
st.markdown("---")
st.header("📊 Dashboard KPIs")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Buyers", len(filtered))

with col2:
    st.metric(
        "Average Sale Price",
        f"${filtered['sale_price'].mean():,.0f}"
    )

with col3:
    st.metric(
        "Average Satisfaction",
        round(filtered["satisfaction_score"].mean(), 2)
    )

with col4:
    st.metric(
        "Number of Clusters",
        filtered["Cluster"].nunique()
    )
  # ==================================================
# Row 1
# ==================================================

col1, col2 = st.columns(2)

with col1:

    st.subheader("📊 Cluster Distribution")

    fig = px.histogram(
        filtered,
        x="Cluster",
        color="Cluster",
        text_auto=True,
        title="Customer Distribution Across Clusters"
    )

    fig.update_layout(
        xaxis_title="Cluster",
        yaxis_title="Number of Customers"
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:

    st.subheader("🌍 Country-wise Buyers")

    country_df = (
        filtered.groupby("country")
        .size()
        .reset_index(name="Customers")
        .sort_values("Customers", ascending=False)
    )

    fig = px.bar(
        country_df,
        x="country",
        y="Customers",
        color="Customers",
        text_auto=True
    )

    fig.update_layout(
        xaxis_title="Country",
        yaxis_title="Total Buyers"
    )

    st.plotly_chart(fig, use_container_width=True)
col1, col2 = st.columns(2)

with col1:

    st.subheader("👥 Client Type Distribution")

    fig = px.pie(
        filtered,
        names="client_type",
        hole=0.45
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:

    st.subheader("🏡 Acquisition Purpose")

    fig = px.pie(
        filtered,
        names="acquisition_purpose",
        hole=0.45
    )

    st.plotly_chart(fig, use_container_width=True)
    # ==================================================
# Row 3
# Loan Applied + Referral Channel
# ==================================================

col1, col2 = st.columns(2)

# ---------------- Left ----------------
with col1:

    st.subheader("💳 Loan Applied Analysis")

    fig = px.pie(
        filtered,
        names="loan_applied",
        hole=0.45
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------- Right ----------------
with col2:

    st.subheader("📢 Referral Channel")

    ref = (
        filtered.groupby("referral_channel")
        .size()
        .reset_index(name="Customers")
    )

    fig = px.bar(
        ref,
        x="referral_channel",
        y="Customers",
        color="Customers",
        text_auto=True
    )

    fig.update_layout(
        xaxis_title="Referral Channel",
        yaxis_title="Customers"
    )

    st.plotly_chart(fig, use_container_width=True)
# ==================================================
# Row 4
# Sale Price + Satisfaction
# ==================================================

col1, col2 = st.columns(2)

# ---------------- Left ----------------
with col1:

    st.subheader("💰 Sale Price by Cluster")

    fig = px.box(
        filtered,
        x="Cluster",
        y="sale_price",
        color="Cluster"
    )

    fig.update_layout(
        xaxis_title="Cluster",
        yaxis_title="Sale Price"
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------- Right ----------------
with col2:

    st.subheader("😊 Average Satisfaction by Cluster")

    sat = (
        filtered.groupby("Cluster")["satisfaction_score"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        sat,
        x="Cluster",
        y="satisfaction_score",
        color="Cluster",
        text_auto=".2f"
    )

    fig.update_layout(
        xaxis_title="Cluster",
        yaxis_title="Average Satisfaction"
    )

    st.plotly_chart(fig, use_container_width=True)
# ==================================================
# Row 5
# Property Size (Full Width)
# ==================================================

st.markdown("---")

st.subheader("📐 Average Property Size by Cluster")

area = (
    filtered.groupby("Cluster")["floor_area_sqft"]
    .mean()
    .reset_index()
)

fig = px.bar(
    area,
    x="Cluster",
    y="floor_area_sqft",
    color="Cluster",
    text_auto=".0f"
)

fig.update_layout(
    xaxis_title="Cluster",
    yaxis_title="Average Area (Sq.ft)"
)

st.plotly_chart(fig, use_container_width=True)
# ==================================================
# Cluster Summary
# ==================================================

st.markdown("---")

st.subheader("📋 Cluster Summary")

summary = (
    filtered.groupby("Cluster")
    .agg(
        Total_Buyers=("client_id", "count"),
        Average_Sale_Price=("sale_price", "mean"),
        Average_Satisfaction=("satisfaction_score", "mean"),
        Average_Property_Size=("floor_area_sqft", "mean")
    )
    .round(2)
)

st.dataframe(summary, use_container_width=True)
# ==================================================
# Business Insights
# ==================================================

st.markdown("---")
st.subheader("💡 Business Insights")

st.success(f"""
## 📈 Executive Summary

👥 Total Buyers : **{len(filtered):,}**

💰 Average Sale Price : **${filtered['sale_price'].mean():,.0f}**

😊 Average Satisfaction : **{filtered['satisfaction_score'].mean():.2f}**

🏠 Machine Learning Segments : **{filtered['Cluster'].nunique()}**

""")

st.info("""

## 📌 Strategic Recommendations

• Increase marketing efforts for high-value buyer segments.

• Introduce personalized financing schemes for buyers applying for loans.

• Focus premium marketing campaigns on countries with the highest property purchases.

• Improve customer satisfaction through post-sale engagement.

• Utilize machine learning clusters for targeted marketing and personalized recommendations.

• Expand investment opportunities for customers interested in investment properties.

""")
st.divider()

st.subheader("Customer Dataset")

st.dataframe(
    filtered,
    use_container_width=True
)
# ==================================================
# Complete Dataset
# ==================================================

st.markdown("---")

st.subheader("📄 Filtered Customer Dataset")

st.dataframe(
    filtered,
    use_container_width=True,
    height=500
)
# ==================================================
# Download
# ==================================================

st.markdown("---")

csv = filtered.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Filtered Dataset",
    data=csv,
    file_name="Filtered_Buyer_Data.csv",
    mime="text/csv"
)
st.markdown("---")

st.caption("""
Developed by **Satyam Rakeshpratap Singh**

Unified Mentor Internship Project

Machine Learning Based Buyer Segmentation and Investment Profiling for Real Estate Market Intelligence

Technology Used:
• Python
• Streamlit
• Plotly
• Pandas
• Scikit-learn
""")