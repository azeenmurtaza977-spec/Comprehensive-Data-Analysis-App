import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# App title
st.title("📊 Simple Data Analysis App")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

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
        fig, ax = plt.subplots()
        df[col].hist(ax=ax, bins=20)
        ax.set_title(f"Histogram of {col}")
        st.pyplot(fig)
    else:
        st.write(df[col].value_counts())
        fig, ax = plt.subplots()
        df[col].value_counts().head(10).plot.pie(autopct='%1.1f%%', ax=ax)
        ax.set_ylabel("")
        ax.set_title(f"Top 10 {col} distribution")
        st.pyplot(fig)

    # Compare two variables
    st.subheader("📊 Compare Two Columns")
    col1 = st.selectbox("Select first column", df.columns, key="c1")
    col2 = st.selectbox("Select second column", df.columns, key="c2")

    if col1 != col2:
        if pd.api.types.is_numeric_dtype(df[col2]) and not pd.api.types.is_numeric_dtype(df[col1]):
            fig, ax = plt.subplots()
            df.groupby(col1)[col2].mean().plot(kind="bar", ax=ax)
            ax.set_title(f"Average {col2} by {col1}")
            st.pyplot(fig)

        elif pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
            fig, ax = plt.subplots()
            ax.scatter(df[col1], df[col2], alpha=0.6)
            ax.set_xlabel(col1)
            ax.set_ylabel(col2)
            ax.set_title(f"{col1} vs {col2}")
            st.pyplot(fig)
                
