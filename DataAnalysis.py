import streamlit as st
import pandas as pd
import plotly.express as px

# Set page config
st.set_page_config(page_title="📊 Exploratory Data Analysis", layout="wide")

# Title
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>📊 Exploratory Data Analysis App</h1>", unsafe_allow_html=True)

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
            st.dataframe(df[numeric_cols].min().to_frame("Min"))
        with col2:
            st.write("**Maximum Values**")
            st.dataframe(df[numeric_cols].max().to_frame("Max"))

        # Histogram of numeric columns
        st.markdown("### 📊 Distribution of Numeric Columns (Histogram)")
        for col in numeric_cols:
            fig = px.histogram(df, x=col, nbins=30,
                               title=f"Distribution of {col}",
                               color_discrete_sequence=px.colors.qualitative.Safe)
            fig.update_layout(bargap=0.1, plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

    # Top performer (numeric)
    if len(numeric_cols) > 0:
        st.markdown("### 🏆 Top Performer (Numeric)")
        top_col = st.selectbox("Select a numeric column to find top performer:", numeric_cols)
        top_row = df.loc[df[top_col].idxmax()]
        st.success(f"Highest {top_col}: {top_row[top_col]}")
        st.write("**Details of top performer:**")
        st.dataframe(top_row.to_frame().T)

    # Highest occurrence (categorical)
    if len(categorical_cols) > 0:
        st.markdown("### 🥇 Most Frequent (Categorical)")
        cat_col = st.selectbox("Select a categorical column to find most frequent:", categorical_cols)
        most_common_value = df[cat_col].value_counts().idxmax()
        most_common_count = df[cat_col].value_counts().max()
        st.success(f"Most frequent {cat_col}: {most_common_value} (appears {most_common_count} times)")
        st.write("**Top value counts (Pie Chart):**")
        fig = px.pie(df, names=cat_col, title=f"Distribution of {cat_col}",
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True)

    # Column-wise analysis
    st.markdown("### 🔍 Column-wise Analysis")
    col = st.selectbox("Pick a column for analysis", df.columns)

    if pd.api.types.is_numeric_dtype(df[col]):
        st.write("**Summary:**")
        st.dataframe(df[col].describe().to_frame())

        st.markdown("**Histogram (Bar Chart)**")
        fig = px.histogram(df, x=col, nbins=30,
                           title=f"Histogram of {col}",
                           color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_layout(bargap=0.1, plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("**Value counts:**")
        st.dataframe(df[col].value_counts().head(10).to_frame())

        st.markdown("**Top Categories (Bar Chart)**")
        fig = px.bar(df[col].value_counts().head(10),
                     x=df[col].value_counts().head(10).index,
                     y=df[col].value_counts().head(10).values,
                     color=df[col].value_counts().head(10).index,
                     title=f"Top Categories in {col}",
                     color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig, use_container_width=True)

    # Compare two columns
    st.markdown("### 📊 Compare Two Columns")
    col1 = st.selectbox("Select first column", df.columns, key="c1")
    col2 = st.selectbox("Select second column", df.columns, key="c2")

    if col1 != col2:
        if pd.api.types.is_numeric_dtype(df[col2]) and not pd.api.types.is_numeric_dtype(df[col1]):
            st.markdown(f"**Average {col2} by {col1} (Bar Chart)**")
            grouped = df.groupby(col1)[col2].mean().sort_values()
            fig = px.bar(grouped, x=grouped.index, y=grouped.values,
                         color=grouped.index, title=f"Average {col2} by {col1}",
                         color_discrete_sequence=px.colors.qualitative.Vivid)
            st.plotly_chart(fig, use_container_width=True)

        elif pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
            st.markdown(f"**Comparison of {col1} and {col2} (Bar Chart)**")
            avg_df = df.groupby(col1)[col2].mean().reset_index()
            fig = px.bar(avg_df, x=col1, y=col2,
                         title=f"Average {col2} by {col1}",
                         color=col1, color_discrete_sequence=px.colors.qualitative.Bold)
            st.plotly_chart(fig, use_container_width=True)

        elif not pd.api.types.is_numeric_dtype(df[col1]) and not pd.api.types.is_numeric_dtype(df[col2]):
            st.markdown("**Cross-tabulation (Bar Chart)**")
            cross_tab = df.groupby([col1, col2]).size().unstack(fill_value=0)
            fig = px.bar(cross_tab, barmode="group",
                         title=f"Cross-tabulation of {col1} and {col2}",
                         color_discrete_sequence=px.colors.qualitative.Safe)
            st.plotly_chart(fig, use_container_width=True)

    # Extra EDA Features
    st.markdown("### 🔬 Extra EDA Features")
    if len(numeric_cols) > 1:
        st.write("**Correlation Heatmap**")
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                        title="Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)

    st.write("**Missing Values:**")
    missing = df.isnull().sum()
    st.dataframe(missing[missing > 0].to_frame("Missing Values"))

    # ✅ NEW FEATURE: Unique value counts
    st.markdown("### 📌 Unique Value Counts per Column")
    unique_counts = df.nunique().to_frame("Unique Values")
    st.dataframe(unique_counts)

    # 📥 Download dataset
    st.markdown("### 📥 Download Cleaned Dataset")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "cleaned_dataset.csv", "text/csv")

    # 📝 Export insights summary
    st.markdown("### 📝 Export Insights")
    insights = f"""
    Dataset has {df.shape[0]} rows and {df.shape[1]} columns.
    Numeric Columns: {list(numeric_cols)}
    Categorical Columns: {list(categorical_cols)}
    Missing Values: {missing.sum()}
    """
    st.download_button("Download Insights Summary", insights, "insights.txt")

