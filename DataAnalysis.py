import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(page_title="Advanced Data Analysis App", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🔍 Advanced Data Analysis App</h1>', unsafe_allow_html=True)
st.markdown("Upload your CSV file to get comprehensive insights and analysis")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load data
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Successfully loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()
    
    # Sidebar for navigation
    st.sidebar.title("Analysis Navigation")
    analysis_type = st.sidebar.selectbox(
        "Choose Analysis Type:",
        ["📊 Dataset Overview", "🔢 Descriptive Statistics", "📈 Distribution Analysis", 
         "🔗 Correlation Analysis", "❓ Missing Data Analysis", "🎯 Outlier Detection",
         "📋 Categorical Analysis", "🔍 Advanced Analytics", "📝 Data Quality Report"]
    )
    
    # Helper functions
    def detect_column_types(df):
        """Detect and categorize column types"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = []
        
        # Try to detect datetime columns
        for col in categorical_cols:
            try:
                pd.to_datetime(df[col].dropna().iloc[:100])
                datetime_cols.append(col)
            except:
                continue
                
        categorical_cols = [col for col in categorical_cols if col not in datetime_cols]
        
        return numeric_cols, categorical_cols, datetime_cols
    
    def calculate_data_quality_score(df):
        """Calculate overall data quality score"""
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        completeness = (1 - missing_cells / total_cells) * 100
        
        # Check for duplicates
        duplicate_rows = df.duplicated().sum()
        uniqueness = (1 - duplicate_rows / len(df)) * 100
        
        # Overall quality score
        quality_score = (completeness + uniqueness) / 2
        return quality_score, completeness, uniqueness
    
    # Get column types
    numeric_cols, categorical_cols, datetime_cols = detect_column_types(df)
    
    # Analysis sections
    if analysis_type == "📊 Dataset Overview":
        st.markdown('<h2 class="section-header">Dataset Overview</h2>', unsafe_allow_html=True)
        
        # Basic info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", f"{df.shape[0]:,}")
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            st.metric("Numeric Columns", len(numeric_cols))
        with col4:
            st.metric("Categorical Columns", len(categorical_cols))
        
        # Data quality score
        quality_score, completeness, uniqueness = calculate_data_quality_score(df)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Data Quality Score", f"{quality_score:.1f}%")
        with col2:
            st.metric("Data Completeness", f"{completeness:.1f}%")
        with col3:
            st.metric("Data Uniqueness", f"{uniqueness:.1f}%")
        
        # Dataset preview
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Column information
        st.subheader("Column Information")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.astype(str),
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Null Percentage': (df.isnull().sum() / len(df) * 100).round(2),
            'Unique Values': df.nunique()
        })
        st.dataframe(col_info, use_container_width=True)
    
    elif analysis_type == "🔢 Descriptive Statistics":
        st.markdown('<h2 class="section-header">Descriptive Statistics</h2>', unsafe_allow_html=True)
        
        if numeric_cols:
            st.subheader("Numeric Columns Summary")
            desc_stats = df[numeric_cols].describe()
            
            # Add additional statistics
            additional_stats = pd.DataFrame({
                'variance': df[numeric_cols].var(),
                'skewness': df[numeric_cols].skew(),
                'kurtosis': df[numeric_cols].kurtosis()
            }).T
            
            full_stats = pd.concat([desc_stats, additional_stats])
            st.dataframe(full_stats.round(3), use_container_width=True)
            
            # Individual column analysis
            st.subheader("Individual Column Analysis")
            selected_col = st.selectbox("Select a numeric column:", numeric_cols)
            
            if selected_col:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Statistics for {selected_col}:**")
                    col_stats = df[selected_col].describe()
                    for stat, value in col_stats.items():
                        st.write(f"• {stat.title()}: {value:.3f}")
                
                with col2:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    df[selected_col].hist(bins=30, alpha=0.7, ax=ax)
                    ax.axvline(df[selected_col].mean(), color='red', linestyle='--', label='Mean')
                    ax.axvline(df[selected_col].median(), color='green', linestyle='--', label='Median')
                    ax.set_title(f'Distribution of {selected_col}')
                    ax.legend()
                    st.pyplot(fig)
        
        if categorical_cols:
            st.subheader("Categorical Columns Summary")
            cat_summary = []
            for col in categorical_cols:
                cat_summary.append({
                    'Column': col,
                    'Unique Values': df[col].nunique(),
                    'Most Frequent': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'N/A',
                    'Most Frequent Count': df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0
                })
            
            cat_df = pd.DataFrame(cat_summary)
            st.dataframe(cat_df, use_container_width=True)
    
    elif analysis_type == "📈 Distribution Analysis":
        st.markdown('<h2 class="section-header">Distribution Analysis</h2>', unsafe_allow_html=True)
        
        if numeric_cols:
            # Distribution plots for all numeric columns
            st.subheader("Distribution Plots")
            
            # Select columns to plot
            selected_cols = st.multiselect("Select columns to analyze:", numeric_cols, default=numeric_cols[:3])
            
            if selected_cols:
                # Create subplots
                fig, axes = plt.subplots(len(selected_cols), 2, figsize=(15, 5*len(selected_cols)))
                if len(selected_cols) == 1:
                    axes = axes.reshape(1, -1)
                
                for i, col in enumerate(selected_cols):
                    # Histogram
                    axes[i, 0].hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
                    axes[i, 0].set_title(f'Histogram: {col}')
                    axes[i, 0].set_ylabel('Frequency')
                    
                    # Box plot
                    axes[i, 1].boxplot(df[col].dropna())
                    axes[i, 1].set_title(f'Box Plot: {col}')
                    axes[i, 1].set_ylabel('Values')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Normality tests
                st.subheader("Normality Tests")
                normality_results = []
                for col in selected_cols:
                    data = df[col].dropna()
                    if len(data) >= 8:  # Minimum sample size for Shapiro-Wilk
                        statistic, p_value = stats.shapiro(data[:5000])  # Limit for performance
                        is_normal = p_value > 0.05
                        normality_results.append({
                            'Column': col,
                            'Shapiro-Wilk Statistic': f"{statistic:.4f}",
                            'P-value': f"{p_value:.4f}",
                            'Is Normal? (α=0.05)': 'Yes' if is_normal else 'No'
                        })
                
                if normality_results:
                    norm_df = pd.DataFrame(normality_results)
                    st.dataframe(norm_df, use_container_width=True)
    
    elif analysis_type == "🔗 Correlation Analysis":
        st.markdown('<h2 class="section-header">Correlation Analysis</h2>', unsafe_allow_html=True)
        
        if len(numeric_cols) >= 2:
            # Correlation matrix
            corr_matrix = df[numeric_cols].corr()
            
            # Heatmap
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, linewidths=0.5, ax=ax)
            ax.set_title('Correlation Heatmap')
            st.pyplot(fig)
            
            # Strong correlations
            st.subheader("Strong Correlations (|r| > 0.5)")
            strong_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.5:
                        strong_corr.append({
                            'Variable 1': corr_matrix.columns[i],
                            'Variable 2': corr_matrix.columns[j],
                            'Correlation': f"{corr_val:.3f}",
                            'Strength': 'Strong' if abs(corr_val) > 0.7 else 'Moderate'
                        })
            
            if strong_corr:
                corr_df = pd.DataFrame(strong_corr)
                st.dataframe(corr_df, use_container_width=True)
            else:
                st.info("No strong correlations found (|r| > 0.5)")
            
            # Scatter plot matrix for selected variables
            if len(numeric_cols) <= 6:
                st.subheader("Scatter Plot Matrix")
                fig = px.scatter_matrix(df[numeric_cols], 
                                      title="Scatter Plot Matrix",
                                      height=600)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least 2 numeric columns for correlation analysis")
    
    elif analysis_type == "❓ Missing Data Analysis":
        st.markdown('<h2 class="section-header">Missing Data Analysis</h2>', unsafe_allow_html=True)
        
        # Missing data summary
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        
        missing_summary = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing Percentage': missing_percent.values
        })
        missing_summary = missing_summary[missing_summary['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        
        if len(missing_summary) > 0:
            st.subheader("Missing Data Summary")
            st.dataframe(missing_summary, use_container_width=True)
            
            # Missing data visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Bar plot
            missing_summary.set_index('Column')['Missing Percentage'].plot(kind='bar', ax=ax1)
            ax1.set_title('Missing Data Percentage by Column')
            ax1.set_ylabel('Percentage Missing')
            ax1.tick_params(axis='x', rotation=45)
            
            # Heatmap of missing data pattern
            sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis', ax=ax2)
            ax2.set_title('Missing Data Pattern')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Missing data patterns
            st.subheader("Missing Data Insights")
            total_missing = missing_data.sum()
            st.write(f"• Total missing values: {total_missing:,}")
            st.write(f"• Percentage of dataset missing: {(total_missing / (len(df) * len(df.columns))) * 100:.2f}%")
            st.write(f"• Columns with missing data: {len(missing_summary)}")
        else:
            st.success("🎉 No missing data found in the dataset!")
    
    elif analysis_type == "🎯 Outlier Detection":
        st.markdown('<h2 class="section-header">Outlier Detection</h2>', unsafe_allow_html=True)
        
        if numeric_cols:
            selected_col = st.selectbox("Select column for outlier analysis:", numeric_cols)
            
            if selected_col:
                data = df[selected_col].dropna()
                
                # Calculate outliers using IQR method
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = data[(data < lower_bound) | (data > upper_bound)]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Outliers", len(outliers))
                with col2:
                    st.metric("Outlier Percentage", f"{(len(outliers)/len(data)*100):.2f}%")
                with col3:
                    st.metric("Clean Data Points", len(data) - len(outliers))
                
                # Visualization
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Box plot
                ax1.boxplot(data, vert=True)
                ax1.set_title(f'Box Plot: {selected_col}')
                ax1.set_ylabel('Values')
                
                # Scatter plot with outliers highlighted
                ax2.scatter(range(len(data)), data, alpha=0.6, label='Normal')
                outlier_indices = data[(data < lower_bound) | (data > upper_bound)].index
                ax2.scatter(outlier_indices, data[outlier_indices], color='red', alpha=0.8, label='Outliers')
                ax2.axhline(y=lower_bound, color='r', linestyle='--', alpha=0.7, label='IQR Bounds')
                ax2.axhline(y=upper_bound, color='r', linestyle='--', alpha=0.7)
                ax2.set_title(f'Outlier Detection: {selected_col}')
                ax2.set_xlabel('Data Point Index')
                ax2.set_ylabel('Values')
                ax2.legend()
                
                plt.tight_layout()
                st.pyplot(fig)
                
                if len(outliers) > 0:
                    st.subheader("Outlier Statistics")
                    st.write(f"• Lower bound (Q1 - 1.5*IQR): {lower_bound:.3f}")
                    st.write(f"• Upper bound (Q3 + 1.5*IQR): {upper_bound:.3f}")
                    st.write(f"• Minimum outlier value: {outliers.min():.3f}")
                    st.write(f"• Maximum outlier value: {outliers.max():.3f}")
        else:
            st.warning("No numeric columns available for outlier detection")
    
    elif analysis_type == "📋 Categorical Analysis":
        st.markdown('<h2 class="section-header">Categorical Analysis</h2>', unsafe_allow_html=True)
        
        if categorical_cols:
            selected_cat_col = st.selectbox("Select categorical column:", categorical_cols)
            
            if selected_cat_col:
                value_counts = df[selected_cat_col].value_counts()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Value Counts")
                    st.dataframe(value_counts.to_frame().reset_index(), use_container_width=True)
                
                with col2:
                    # Pie chart
                    fig, ax = plt.subplots(figsize=(8, 8))
                    value_counts.head(10).plot(kind='pie', ax=ax, autopct='%1.1f%%')
                    ax.set_title(f'Distribution of {selected_cat_col}')
                    ax.set_ylabel('')
                    st.pyplot(fig)
                
                # Bar chart for all categories
                st.subheader("Bar Chart")
                fig, ax = plt.subplots(figsize=(12, 6))
                value_counts.plot(kind='bar', ax=ax)
                ax.set_title(f'Distribution of {selected_cat_col}')
                ax.set_xlabel(selected_cat_col)
                ax.set_ylabel('Count')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Cross-tabulation with numeric columns
                if numeric_cols:
                    st.subheader("Categorical vs Numeric Analysis")
                    selected_num_col = st.selectbox("Select numeric column for comparison:", numeric_cols)
                    
                    if selected_num_col:
                        # Group statistics
                        group_stats = df.groupby(selected_cat_col)[selected_num_col].agg([
                            'count', 'mean', 'median', 'std', 'min', 'max'
                        ]).round(3)
                        st.dataframe(group_stats, use_container_width=True)
                        
                        # Box plot by category
                        fig, ax = plt.subplots(figsize=(12, 6))
                        df.boxplot(column=selected_num_col, by=selected_cat_col, ax=ax)
                        ax.set_title(f'{selected_num_col} by {selected_cat_col}')
                        plt.suptitle('')  # Remove default title
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
        else:
            st.warning("No categorical columns available for analysis")
    
    elif analysis_type == "🔍 Advanced Analytics":
        st.markdown('<h2 class="section-header">Advanced Analytics</h2>', unsafe_allow_html=True)
        
        # Principal Component Analysis
        if len(numeric_cols) >= 2:
            st.subheader("Principal Component Analysis (PCA)")
            
            # Prepare data for PCA
            pca_data = df[numeric_cols].dropna()
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(pca_data)
            
            # Perform PCA
            pca = PCA()
            pca_result = pca.fit_transform(scaled_data)
            
            # Explained variance
            explained_var_ratio = pca.explained_variance_ratio_
            cumulative_var_ratio = np.cumsum(explained_var_ratio)
            
            # Plot explained variance
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Individual explained variance
            ax1.bar(range(1, len(explained_var_ratio) + 1), explained_var_ratio)
            ax1.set_xlabel('Principal Component')
            ax1.set_ylabel('Explained Variance Ratio')
            ax1.set_title('Explained Variance by Component')
            
            # Cumulative explained variance
            ax2.plot(range(1, len(cumulative_var_ratio) + 1), cumulative_var_ratio, 'bo-')
            ax2.axhline(y=0.8, color='r', linestyle='--', label='80% Variance')
            ax2.axhline(y=0.95, color='g', linestyle='--', label='95% Variance')
            ax2.set_xlabel('Number of Components')
            ax2.set_ylabel('Cumulative Explained Variance Ratio')
            ax2.set_title('Cumulative Explained Variance')
            ax2.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # PCA insights
            st.write("**PCA Insights:**")
            components_80 = np.argmax(cumulative_var_ratio >= 0.8) + 1
            components_95 = np.argmax(cumulative_var_ratio >= 0.95) + 1
            st.write(f"• Components needed for 80% variance: {components_80}")
            st.write(f"• Components needed for 95% variance: {components_95}")
            st.write(f"• Total variance explained by first 2 components: {cumulative_var_ratio[1]:.1%}")
        
        # Statistical tests
        if len(numeric_cols) >= 2:
            st.subheader("Statistical Tests")
            
            col1_test = st.selectbox("Select first variable:", numeric_cols, key="test1")
            col2_test = st.selectbox("Select second variable:", numeric_cols, key="test2")
            
            if col1_test != col2_test:
                data1 = df[col1_test].dropna()
                data2 = df[col2_test].dropna()
                
                # Correlation test
                corr_coef, corr_p = stats.pearsonr(data1, data2)
                
                # T-test (assuming we want to test if means are different)
                t_stat, t_p = stats.ttest_ind(data1, data2)
                
                test_results = pd.DataFrame({
                    'Test': ['Pearson Correlation', 'Independent T-test'],
                    'Statistic': [f"{corr_coef:.4f}", f"{t_stat:.4f}"],
                    'P-value': [f"{corr_p:.4f}", f"{t_p:.4f}"],
                    'Significant (α=0.05)': ['Yes' if corr_p < 0.05 else 'No', 
                                           'Yes' if t_p < 0.05 else 'No']
                })
                
                st.dataframe(test_results, use_container_width=True)
    
    elif analysis_type == "📝 Data Quality Report":
        st.markdown('<h2 class="section-header">Data Quality Report</h2>', unsafe_allow_html=True)
        
        # Overall quality score
        quality_score, completeness, uniqueness = calculate_data_quality_score(df)
        
        # Quality metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            if quality_score >= 80:
                st.success(f"Overall Quality Score: {quality_score:.1f}%")
            elif quality_score >= 60:
                st.warning(f"Overall Quality Score: {quality_score:.1f}%")
            else:
                st.error(f"Overall Quality Score: {quality_score:.1f}%")
        
        with col2:
            st.metric("Data Completeness", f"{completeness:.1f}%")
        with col3:
            st.metric("Data Uniqueness", f"{uniqueness:.1f}%")
        
        # Detailed quality checks
        st.subheader("Detailed Quality Checks")
        
        quality_issues = []
        
        # Check for missing data
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            quality_issues.append({
                'Issue Type': 'Missing Data',
                'Affected Columns': ', '.join(missing_cols),
                'Severity': 'High' if len(missing_cols) > len(df.columns) * 0.3 else 'Medium',
                'Description': f'{len(missing_cols)} columns have missing values'
            })
        
        # Check for duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            quality_issues.append({
                'Issue Type': 'Duplicate Rows',
                'Affected Columns': 'All',
                'Severity': 'High' if duplicates > len(df) * 0.1 else 'Medium',
                'Description': f'{duplicates} duplicate rows found'
            })
        
        # Check for constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            quality_issues.append({
                'Issue Type': 'Constant Values',
                'Affected Columns': ', '.join(constant_cols),
                'Severity': 'Low',
                'Description': f'{len(constant_cols)} columns have constant values'
            })
        
        # Check for high cardinality categorical columns
        high_card_cols = [col for col in categorical_cols if df[col].nunique() > len(df) * 0.5]
        if high_card_cols:
            quality_issues.append({
                'Issue Type': 'High Cardinality',
                'Affected Columns': ', '.join(high_card_cols),
                'Severity': 'Medium',
                'Description': f'{len(high_card_cols)} categorical columns have very high cardinality'
            })
        
        if quality_issues:
            issues_df = pd.DataFrame(quality_issues)
            st.dataframe(issues_df, use_container_width=True)
        else:
            st.success("🎉 No major data quality issues detected!")
        
        # Recommendations
        st.subheader("Recommendations")
        recommendations = []
        
        if missing_cols:
            recommendations.append("• Consider imputation strategies for missing data or remove columns with high missing rates")
        
        if duplicates > 0:
            recommendations.append("• Remove duplicate rows to improve data quality")
        
        if constant_cols:
            recommendations.append("• Consider removing constant columns as they provide no information")
        
        if len(numeric_cols) > 10:
            recommendations.append("• Consider dimensionality reduction techniques (PCA) for high-dimensional data")
        
        if not recommendations:
            recommendations.append("• Your data appears to be in good quality!")
        
        for rec in recommendations:
            st.write(rec)

else:
    st.info("👆 Please upload a CSV file to begin analysis")
    
    # Sample data info
    st.markdown("---")
    st.subheader("What this app analyzes:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📊 Basic Analysis:**
        - Dataset overview and structure
        - Descriptive statistics
        - Data quality assessment
        - Missing data analysis
        """)
        
        st.markdown("""
        **📈 Statistical Analysis:**
        - Distribution analysis
        - Normality tests
        - Correlation analysis
        - Outlier detection
        """)
    
    with col2:
        st.markdown("""
        **🔍 Advanced Analytics:**
        - Principal Component Analysis (PCA)
        - Statistical hypothesis tests
        - Categorical data analysis
        - Cross-tabulation analysis
        """)
        
        st.markdown("""
        **📝 Reporting:**
        - Comprehensive data quality report
        - Actionable recommendations
        - Interactive visualizations
        - Export-ready insights
        """)

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit • Upload your CSV and discover insights!")
