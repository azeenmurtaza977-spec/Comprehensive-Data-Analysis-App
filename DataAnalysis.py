import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Title
st.title("Comprehensive Data Analysis App")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Show dataset preview
    st.subheader("Dataset Preview")
    st.write(df.head())
    
    # Show basic info
    st.subheader("Summary Statistics")
    st.write(df.describe())
    
    # NEW: Quick insights - Min/Max values
    st.subheader("Quick Insights")
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    if len(numeric_cols) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Minimum Values:**")
            for col in numeric_cols:
                min_idx = df[col].idxmin()
                st.write(f"• {col}: {df[col].min():.2f} (Row {min_idx})")
        
        with col2:
            st.write("**Maximum Values:**")
            for col in numeric_cols:
                max_idx = df[col].idxmax()
                st.write(f"• {col}: {df[col].max():.2f} (Row {max_idx})")
    
    # NEW: Top performer (highest value in selected column)
    st.subheader("Top Performer")
    if len(numeric_cols) > 0:
        top_col = st.selectbox("Select column to find top performer:", numeric_cols)
        max_idx = df[top_col].idxmax()
        max_value = df[top_col].max()
        
        st.success(f"🏆 Highest {top_col}: {max_value:.2f}")
        st.write("**Top performer details:**")
        st.write(df.iloc[max_idx])
    
    # Column selection for analysis
    st.subheader("Column-wise Analysis")
    column = st.selectbox("Select a column for analysis", df.columns)
    
    if pd.api.types.is_numeric_dtype(df[column]):
        st.write(f"Summary of {column}:")
        st.write(df[column].describe())
        
        # Histogram
        fig, ax = plt.subplots()
        df[column].hist(ax=ax, bins=20)
        ax.set_title(f"Histogram of {column}")
        st.pyplot(fig)
    else:
        st.write(f"Value counts of {column}:")
        st.write(df[column].value_counts())
        
        # NEW: Pie chart for categorical data
        fig, ax = plt.subplots()
        df[column].value_counts().head(10).plot(kind='pie', ax=ax, autopct='%1.1f%%')
        ax.set_title(f"Distribution of {column}")
        st.pyplot(fig)
    
    # NEW: Compare two variables with bar chart
    st.subheader("Compare Two Variables")
    col1_select = st.selectbox("Select first variable:", df.columns, key="var1")
    col2_select = st.selectbox("Select second variable:", df.columns, key="var2")
    
    if col1_select != col2_select:
        if pd.api.types.is_numeric_dtype(df[col2_select]) and not pd.api.types.is_numeric_dtype(df[col1_select]):
            # Categorical vs Numeric
            fig, ax = plt.subplots(figsize=(10, 6))
            df.groupby(col1_select)[col2_select].mean().plot(kind='bar', ax=ax)
            ax.set_title(f"Average {col2_select} by {col1_select}")
            ax.set_ylabel(f"Average {col2_select}")
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
        elif pd.api.types.is_numeric_dtype(df[col1_select]) and pd.api.types.is_numeric_dtype(df[col2_select]):
            # Both numeric - scatter plot
            fig, ax = plt.subplots()
            ax.scatter(df[col1_select], df[col2_select], alpha=0.6)
            ax.set_xlabel(col1_select)
            ax.set_ylabel(col2_select)
            ax.set_title(f"{col1_select} vs {col2_select}")
            st.pyplot(fig)                 
