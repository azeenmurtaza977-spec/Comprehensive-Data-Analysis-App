import streamlit as st
import pandas as pd

# Set page config
st.set_page_config(page_title="📊 Comprehensive Data Analysis", layout="wide")

# Title
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>📊 Comprehensive Data Analysis App</h1>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Dataset preview
    st.markdown("### 🔎 Dataset Preview")
    st.dataframe(df.head())

    # Dataset info
    st.markdown("### 📋 Dataset Overview")
    st.write(f"**Shape of dataset:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.write("**Column names:**", list(df.columns))

    # Summary
    st.markdown("### 📈 Summary Statistics")
    st.dataframe(df.describe(include="all"))

    # Quick Insights
    numeric_cols = df.select_dtypes(include="number").columns
    categorical_cols = df.select_dtypes(exclude="number").columns

    if len(numeric_cols) > 0:
        st.markdown("### ⚡ Quick Insights")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Minimum Values**")
            st.dataframe(df[numeric_cols].min())
        with col2:
            st.write("**Maximum Values**")
            st.dataframe(df[numeric_cols].max())

        # Correlation heatmap alternative
        st.markdown("### 🔗 Correlation Matrix (numeric columns)")
        corr = df[numeric_cols].corr()
        st.dataframe(corr.style.background_gradient(cmap="Blues"))

        # Line chart of numeric columns
        st.markdown("### 📉 Trend of Numeric Columns")
        st.line_chart(df[numeric_cols])

    # Top performer
    if len(numeric_cols) > 0:
        st.markdown("### 🏆 Top Performer")
        top_col = st.selectbox("Select a column to find top performer:", numeric_cols)
        top_row = df.loc[df[top_col].idxmax()]
        st.success(f"Highest {top_col}: {top_row[top_col]}")
        st.write("**Details of top performer:**")
        st.dataframe(top_row.to_frame().T)

    # Column-wise analysis
    st.markdown("### 🔍 Column-wise Analysis")
    col = st.selectbox("Pick a column for analysis", df.columns)

    if pd.api.types.is_numeric_dtype(df[col]):
        st.write("**Summary:**")
        st.dataframe(df[col].describe().to_frame())

        st.markdown("**Histogram (approx via bar chart)**")
        st.bar_chart(df[col].value_counts().sort_index())
        
        st.markdown("**Line Chart**")
        st.line_chart(df[col])

        st.markdown("**Area Chart**")
        st.area_chart(df[col])
    else:
        st.write("**Value counts:**")
        st.dataframe(df[col].value_counts().head(10).to_frame())

        st.markdown("**Top Categories (Bar Chart)**")
        st.bar_chart(df[col].value_counts().head(10))

    # Compare two columns
    st.markdown("### 📊 Compare Two Columns")
    col1 = st.selectbox("Select first column", df.columns, key="c1")
    col2 = st.selectbox("Select second column", df.columns, key="c2")

    if col1 != col2:
        if pd.api.types.is_numeric_dtype(df[col2]) and not pd.api.types.is_numeric_dtype(df[col1]):
            st.markdown(f"**Average {col2} by {col1}**")
            grouped = df.groupby(col1)[col2].mean().sort_values()
            st.bar_chart(grouped)
        elif pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
            st.markdown(f"**Scatter Plot: {col1} vs {col2}**")
            st.scatter_chart(df[[col1, col2]])

    # Overall insights
    st.markdown("### 🧾 Overall Insights")
    st.write(f"- Dataset has **{df.shape[0]} rows** and **{df.shape[1]} columns**.")
    st.write("- Quick descriptive statistics generated above.")
    if len(numeric_cols) > 0:
        st.write("- Strong correlations shown in the correlation matrix.")
    if len(categorical_cols) > 0:
        st.write("- Categorical distributions shown in bar charts.")

