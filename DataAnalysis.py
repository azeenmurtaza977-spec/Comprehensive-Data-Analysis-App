import streamlit as st
import pandas as pd

# Page config
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

    # Summary stats
    st.markdown("### 📈 Summary Statistics")
    st.dataframe(df.describe(include="all"))

    # Auto analysis defaults
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    # ✅ Default analysis
    st.markdown("### 📊 Default Analysis")
    if categorical_cols and numeric_cols:
        default_cat = categorical_cols[0]
        default_num = numeric_cols[0]
        st.write(f"**Average {default_num} by {default_cat} (default view):**")
        grouped = df.groupby(default_cat)[default_num].mean().sort_values()
        st.bar_chart(grouped)

    # 🔄 Column comparison
    st.markdown("### 🔍 Compare Any Two Columns")
    col1 = st.selectbox("Select first column", df.columns, key="col1")
    col2 = st.selectbox("Select second column", df.columns, key="col2")

    if col1 != col2:
        if col1 in categorical_cols and col2 in numeric_cols:
            st.write(f"**Average {col2} by {col1}**")
            grouped = df.groupby(col1)[col2].mean().sort_values()
            st.bar_chart(grouped)

            st.write(f"**Distribution of {col2} by {col1} (Pie Chart)**")
            pie_data = df.groupby(col1)[col2].sum()
            st.dataframe(pie_data.to_frame(name="Total"))
            st.pyplot(pie_data.plot.pie(autopct="%1.1f%%", figsize=(4,4), legend=False).get_figure())

        elif col1 in numeric_cols and col2 in categorical_cols:
            st.write(f"**Average {col1} by {col2}**")
            grouped = df.groupby(col2)[col1].mean().sort_values()
            st.bar_chart(grouped)

            st.write(f"**Distribution of {col1} by {col2} (Pie Chart)**")
            pie_data = df.groupby(col2)[col1].sum()
            st.pyplot(pie_data.plot.pie(autopct="%1.1f%%", figsize=(4,4), legend=False).get_figure())

        elif col1 in numeric_cols and col2 in numeric_cols:
            st.write(f"**Scatter Plot: {col1} vs {col2}**")
            st.scatter_chart(df[[col1, col2]])

        elif col1 in categorical_cols and col2 in categorical_cols:
            st.write(f"**Cross-tab of {col1} and {col2}**")
            cross = pd.crosstab(df[col1], df[col2])
            st.dataframe(cross)
            st.bar_chart(cross)


