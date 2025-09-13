import streamlit as st
import pandas as pd

# Title
st.title("📊 Simple Data Analysis App")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Preview
    st.subheader("🔎 Dataset Preview")
    st.dataframe(df.head())

    # Summary
    st.subheader("📈 Summary Statistics")
    st.write(df.describe(include="all"))

    # Quick Insights
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) > 0:
        st.subheader("⚡ Quick Insights")
        st.write("**Minimum Values**")
        st.write(df[numeric_cols].min())
        st.write("**Maximum Values**")
        st.write(df[numeric_cols].max())

        # Top performer
        top_col = st.selectbox("🏆 Select column to find top performer:", numeric_cols)
        top_row = df.loc[df[top_col].idxmax()]
        st.success(f"Highest {top_col}: {top_row[top_col]}")
        st.write("Details of top performer:", top_row)

    # Column Analysis
    st.subheader("🔍 Column Analysis")
    col = st.selectbox("Pick a column", df.columns)

    if pd.api.types.is_numeric_dtype(df[col]):
        st.write(df[col].describe())
        st.bar_chart(df[col])
    else:
        st.write(df[col].value_counts().head(10))
        st.bar_chart(df[col].value_counts().head(10))

    # Compare two variables
    st.subheader("📊 Compare Two Columns")
    col1 = st.selectbox("Select first column", df.columns, key="c1")
    col2 = st.selectbox("Select second column", df.columns, key="c2")

    if col1 != col2:
        if pd.api.types.is_numeric_dtype(df[col2]) and not pd.api.types.is_numeric_dtype(df[col1]):
            st.bar_chart(df.groupby(col1)[col2].mean())
        elif pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
            st.scatter_chart(df[[col1, col2]])

                
