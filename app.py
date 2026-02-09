import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background-color: #f5f7fa;
}
.main {
    padding: 1.5rem;
}
h1, h2, h3 {
    color: #1f2937;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("📊 Customer Segmentation & Personality Analysis")
st.write("Upload customer data and explore segmentation results")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    kmeans = pickle.load(open("kmeans_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return kmeans, scaler

kmeans, scaler = load_model()

# ---------------- FILE UPLOAD ----------------
st.sidebar.header("📁 Upload Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"]
)

# ---------------- MAIN LOGIC ----------------
if uploaded_file is not None:

    # ---- Read file safely ----
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, encoding="utf-8")
        else:
            df = pd.read_excel(uploaded_file)
    except UnicodeDecodeError:
        df = pd.read_csv(uploaded_file, encoding="latin1")

    st.success("File uploaded successfully!")

    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    # ---------------- FEATURE ENGINEERING ----------------
    df['Total_Spending'] = (
        df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] +
        df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds']
    )

    df['Total_Purchases'] = (
        df['NumWebPurchases'] +
        df['NumStorePurchases'] +
        df['NumCatalogPurchases']
    )

    df['Deal_purchase_ratio'] = df['NumDealsPurchases'] / df['Total_Purchases']
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    features = [
        'Income',
        'Recency',
        'Total_Spending',
        'Total_Purchases',
        'Deal_purchase_ratio'
    ]

    # ---------------- SCALING ----------------
    X = df[features]
    X_scaled = scaler.transform(X)

    # ---------------- CLUSTERING ----------------
    df['Cluster'] = kmeans.predict(X_scaled)

    # ---------------- CLUSTER DISTRIBUTION ----------------
    st.subheader("📌 Cluster Distribution")
    col1, col2 = st.columns(2)

    with col1:
        st.dataframe(
            df['Cluster'].value_counts().rename("Customers"),
            use_container_width=True
        )

    with col2:
        fig, ax = plt.subplots()
        sns.countplot(x='Cluster', data=df, ax=ax)
        ax.set_title("Customers per Cluster")
        st.pyplot(fig)

    # ---------------- CLUSTER PROFILE ----------------
    st.subheader("🧠 Cluster Profiling")
    profile = df.groupby('Cluster')[features].mean().round(2)
    st.dataframe(profile, use_container_width=True)

    # ---------------- SCATTER PLOT ----------------
    st.subheader("📈 Income vs Total Spending")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(
        x='Income',
        y='Total_Spending',
        hue='Cluster',
        palette='Set2',
        data=df,
        ax=ax2
    )
    ax2.set_title("Customer Segments")
    st.pyplot(fig2)

    # ---------------- CLUSTER INTERPRETATION ----------------
    st.subheader("📝 Cluster Interpretation")
    st.markdown("""
    - **Cluster 0:** High-income, high-spending premium customers  
    - **Cluster 1:** Price-sensitive customers with high deal purchases  
    - **Cluster 2:** Low engagement, low spending customers  
    - **Cluster 3:** Moderate income customers with balanced behavior  
    """)

else:
    st.info("⬅️ Upload a CSV or Excel file to start analysis")
