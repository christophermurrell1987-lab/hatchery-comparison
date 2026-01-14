# Hatchery Batch Comparison Tool - Web Application
# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Hatchery Batch Comparison",
    page_icon="🥚",
    layout="wide"
)

st.title("🥚 Hatchery Batch Comparison Tool")
st.markdown("Compare hatchery performance on eggs from the same flock & collection date")

# ============================================
# CACHED DATA LOADING
# ============================================
@st.cache_data
def load_default_data():
    """Load and cache the default hatchery data"""
    df = pd.read_excel('hatchdata.xlsx')
    
    # ----------------------------------------
    # DATA CLEANING
    # ----------------------------------------
    # Standardize Hatchery names
    df['Hatchery'] = df['Hatchery'].str.strip().str.title()
    
    # Standardize Type names (Fix PHns vs PH discrepancy)
    if 'Type' in df.columns:
        df['Type'] = df['Type'].astype(str).str.strip()
        df['Type'] = df['Type'].str.replace('PHns', 'PH', regex=False)
        
    return df

# ============================================
# DATA SOURCE
# ============================================
st.sidebar.header("📁 Data Source")

use_default = st.sidebar.checkbox("Use central hatchery data", value=True)

if use_default:
    try:
        df = load_default_data()
        st.sidebar.success("✓ Loaded central data")
    except FileNotFoundError:
        st.error("hatchdata.xlsx not found. Please upload a file instead.")
        use_default = False

if not use_default:
    uploaded_file = st.file_uploader("Upload your own data", type=['xlsx', 'xls'])
    if uploaded_file is None:
        st.info("👆 Upload an Excel file or check 'Use central hatchery data'")
        st.stop()
    df = pd.read_excel(uploaded_file)
    
    # Apply cleaning to uploaded data
    df['Hatchery'] = df['Hatchery'].str.strip().str.title()
    if 'Type' in df.columns:
        df['Type'] = df['Type'].astype(str).str.strip()
        df['Type'] = df['Type'].str.replace('PHns', 'PH', regex=False)

# ============================================
# SIDEBAR FILTERS
# ============================================
st.sidebar.header("🔧 Filters")

# Date range
min_date = df['Egg Collect Date'].min().date()
max_date = df['Egg Collect Date'].max().date()

st.sidebar.subheader("Date Range")
start_date = st.sidebar.date_input("Start date", min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End date", max_date, min_value=min_date, max_value=max_date)

# Hatchery filter
st.sidebar.subheader("Hatcheries")
all_hatcheries = df['Hatchery'].unique().tolist()
selected_hatcheries = st.sidebar.multiselect(
    "Select hatcheries to include",
    all_hatcheries,
    default=all_hatcheries
)

# Apply filters
df = df[(df['Egg Collect Date'].dt.date >= start_date) & 
        (df['Egg Collect Date'].dt.date <= end_date) &
        (df['Hatchery'].isin(selected_hatcheries))]

# Show data summary
st.sidebar.markdown("---")
st.sidebar.metric("Total Records", f"{len(df):,}")
st.sidebar.metric("Unique Flocks", f"{df['flock_number'].nunique():,}")

# ============================================
# BATCH MATCHING
# ============================================
df['Batch_Key'] = df['Egg Collect Date'].astype(str) + '_' + df['flock_number'].astype(str)

# Find batches at multiple hatcheries
batch_summary = df.groupby('Batch_Key').agg({
    'Hatchery': 'nunique',
    'Egg Collect Date': 'first',
    'flock_number': 'first',
    'Set': 'sum'
}).reset_index()
batch_summary.columns = ['Batch_Key', 'Num_Hatcheries', 'Egg Collect Date', 'Flock', 'Total_Set']

multi_hatchery_batches = batch_summary[batch_summary['Num_Hatcheries'] > 1]['Batch_Key'].tolist()
df_compare = df[df['Batch_Key'].isin(multi_hatchery_batches)].copy()

# Key metrics
metrics = ['Hatch of Set', 'Hatch Of Fertile', 'Fertility %', 'Act Dead %', 'Banger %']

# Ensure "Egg age at Set" is available
if 'Egg age at Set' not in df_compare.columns:
    df_compare['Egg age at Set'] = df_compare['Egg Age']
else:
    df_compare['Egg age at Set'] = df_compare['Egg age at Set'].fillna(df_compare['Egg Age'])

# Aggregate by Batch + Hatchery
comparison = df_compare.groupby(['Batch_Key', 'Egg Collect Date', 'flock_number', 'Hatchery']).agg({
    'Farm': 'first',
    'Type': 'first',
    'Flock Age': 'mean',
    'Egg Age': 'mean',
    'Egg age at Set': 'mean',
    'Set': 'sum',
    'Hatched': 'sum',
    'Infertile': 'sum',
    'Bangers': 'sum',
    'Hatch of Set': 'mean',
    'Hatch Of Fertile': 'mean',
    'Fertility %': 'mean',
    'Act Dead %': 'mean',
    'Banger %': 'mean'
}).reset_index()

# Clean up Flock Age
comparison['Flock Age'] = comparison['Flock Age'].round(0).astype(int)

# ============================================
# MAIN DISPLAY
# ============================================
st.success(f"✓ Found **{len(multi_hatchery_batches)}** batches (flock + date) sent to multiple hatcheries")

# Tabs for different views (Removed Summary, Renamed Pairwise & Pivot)
tab1, tab2, tab3, tab4 = st.tabs([
    "🔄 Hatchery Comparison", 
    "⚠️ High Variance Batches",
    "📋 Individual Flocks",
    "📈 Charts"
])

# ============================================
# TAB 1: HATCHERY COMPARISON (Formerly Pairwise)
# ============================================
with tab1:
    st.subheader("Head-to-Head Hatchery Comparison")
    
    hatcheries = comparison['Hatchery'].unique()
    pairwise_results = []
    
    for i, h1 in enumerate(hatcheries):
        for h2 in hatcheries[i+1:]:
            h1_batches = set(comparison[comparison['Hatchery'] == h1]['Batch_Key'])
            h2_batches = set(comparison[comparison['Hatchery'] == h2]['Batch_Key'])
            shared = h1_batches & h2_batches
            
            if len(shared) > 0:
                h1_data = comparison[(comparison['Hatchery'] == h1) & (comparison['Batch_Key'].isin(shared))]
                h2_data = comparison[(comparison['Hatchery'] == h2) & (comparison['Batch_Key'].isin(shared))]
                
                diff = (h1_data['Hatch of Set'].mean() - h2_data['Hatch of Set'].mean()) * 100
                
                pairwise_results.append({
                    'Hatchery 1': h1,
                    'Hatchery 2': h2,
                    'Shared Batches': len(shared),
                    'H1 Avg HoS %': round(h1_data['Hatch of Set'].mean() * 100, 2),
                    'H2 Avg HoS %': round(h2_data['Hatch of Set'].mean() * 100, 2),
                    'Difference %': round(diff, 2),
                    'H1 Avg HoF %': round(h1_data['Hatch Of Fertile'].mean() * 100, 2),
                    'H2 Avg HoF %': round(h2_data['Hatch Of Fertile'].mean() * 100, 2)
                })
    
    if pairwise_results:
        pairwise_df = pd.DataFrame(pairwise_results)
        
        def highlight_diff(val):
            if val > 2:
                return 'background-color: #90EE90'  # Green - H1 better
            elif val < -2:
                return 'background-color: #FFB6C1'  # Red - H2 better
            return ''
        
        styled_df = pairwise_df.style.applymap(highlight_diff, subset=['Difference %'])
        
        st.dataframe(
            styled_df.format({
                'H1 Avg HoS %': '{:.2f}',
                'H2 Avg HoS %': '{:.2f}',
                'Difference %': '{:.2f}',
                'H1 Avg HoF %': '{:.2f}',
                'H2 Avg HoF %': '{:.2f}'
            }), 
            use_container_width=True
        )
        
        st.caption("🟢 Green = Hatchery 1 better by >2% | 🔴 Red = Hatchery 2 better by >2%")
    else:
        st.warning("No shared batches found between hatcheries")

# ============================================
# TAB 2: HIGH VARIANCE BATCHES
# ============================================
with tab2:
    st.subheader("Batches with Highest Variance Between Hatcheries")
    
    variance_analysis = comparison.groupby(['Batch_Key', 'Egg Collect Date', 'flock_number', 'Type']).agg({
        'Flock Age': 'mean',
        'Egg Age': 'mean',
        'Hatch of Set': ['mean', 'min', 'max', 'count'],
        'Hatch Of Fertile': ['mean', 'min', 'max'],
    }).reset_index()
    
    variance_analysis.columns = ['Batch_Key', 'Egg Collect Date', 'Flock', 'Type', 'Flock Age', 'Avg Egg Age',
                                 'Avg HoS %', 'Min HoS %', 'Max HoS %', 'Num Hatcheries',
                                 'Avg HoF %', 'Min HoF %', 'Max HoF %']
    
    variance_analysis['HoS Range %'] = (variance_analysis['Max HoS %'] - variance_analysis['Min HoS %']) * 100
    
    variance_analysis['Egg Collect Date'] = variance_analysis['Egg Collect Date'].dt.date
    
    for col in ['Avg HoS %', 'Min HoS %', 'Max HoS %', 'Avg HoF %', 'Min HoF %', 'Max HoF %']:
        variance_analysis[col] = (variance_analysis[col] * 100).round(2)
    
    variance_analysis['HoS Range %'] = variance_analysis['HoS Range %'].round(2)
    variance_analysis['Avg Egg Age'] = variance_analysis['Avg Egg Age'].round(1)
    variance_analysis['Flock Age'] = variance_analysis['Flock Age'].astype(int)
    
    threshold = st.slider("Minimum variance threshold (%)", 0, 30, 10)
    high_var = variance_analysis[variance_analysis['HoS Range %'] > threshold].sort_values('HoS Range %', ascending=False)
    
    st.metric("Batches above threshold", len(high_var))
    
    display_cols = ['Egg Collect Date', 'Flock', 'Type', 'Flock Age', 'Avg Egg Age', 'Avg HoS %', 'Min HoS %', 'Max HoS %', 'HoS Range %', 'Num Hatcheries']
    st.dataframe(high_var[display_cols].head(50), use_container_width=True)

# ============================================
# TAB 3: INDIVIDUAL FLOCKS (Formerly Pivot)
# ============================================
with tab3:
    st.subheader("Detailed Flock Comparison")
    st.caption("Comparing Performance vs Egg Age at Set")
    
    metric_choice = st.selectbox("Select metric", ['Hatch of Set', 'Hatch Of Fertile', 'Fertility %', 'Act Dead %'])
    
    # Pivot both the Metric AND Egg Age at Set
    pivot = comparison.pivot_table(
        index=['Egg Collect Date', 'flock_number', 'Type', 'Flock Age'],
        columns='Hatchery',
        values=[metric_choice, 'Egg age at Set'],
        aggfunc='mean'
    )
    
    # Handle Metric Percentage
    pivot[metric_choice] = pivot[metric_choice] * 100
    pivot = pivot.round(2)
    
    # Swap levels to group by Hatchery
    pivot = pivot.swaplevel(0, 1, axis=1).sort_index(axis=1)
    
    # Reset index for display
    pivot_display = pivot.reset_index()
    pivot_display['Egg Collect Date'] = pivot_display['Egg Collect Date'].dt.date

    # ---------------------------------------------------------
    # HIGHLIGHTING LOGIC
    # ---------------------------------------------------------
    def highlight_valid_data(val):
        """Highlights cells that are not null (i.e., have data)"""
        # Checks if value is not NaN (pandas uses isnull() or isna())
        if pd.notnull(val) and val != '':
            return 'background-color: #e6f3ff; color: black' # Light blue background
        return ''

    # Apply style to the dataframe
    # We apply this to the data columns only (subset)
    # The columns in pivot_display are now: Egg Collect Date, flock, type, age ... [Hatchery A - Metric], [Hatchery A - Egg Age]...
    
    st.dataframe(
        pivot_display.style.applymap(highlight_valid_data),
        use_container_width=True, 
        height=600
    )

# ============================================
# TAB 4: CHARTS
# ============================================
with tab4:
    st.subheader("Visual Comparisons")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Hatch of Set Distribution by Hatchery**")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        df_compare_pct = df_compare.copy()
        df_compare_pct['Hatch of Set'] = df_compare_pct['Hatch of Set'] * 100
        df_compare_pct.boxplot(column='Hatch of Set', by='Hatchery', ax=ax1)
        ax1.set_ylabel('Hatch of Set (%)')
        ax1.set_xlabel('Hatchery')
        ax1.set_title('')
        plt.suptitle('')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig1)
    
    with col2:
        st.markdown("**Hatch of Fertile Distribution by Hatchery**")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        df_compare_pct['Hatch Of Fertile'] = df_compare['Hatch Of Fertile'] * 100
        df_compare_pct.boxplot(column='Hatch Of Fertile', by='Hatchery', ax=ax2)
        ax2.set_ylabel('Hatch of Fertile (%)')
        ax2.set_xlabel('Hatchery')
        ax2.set_title('')
        plt.suptitle('')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig2)
    
    st.markdown("**Performance Heatmap**")
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    heatmap_data = df_compare.groupby('Hatchery')[metrics].mean() * 100
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax3)
    ax3.set_title('Average Performance Metrics by Hatchery (%)')
    plt.tight_layout()
    st.pyplot(fig3)

# ============================================
# EXPORT
# ============================================
st.sidebar.markdown("---")
st.sidebar.subheader("📥 Export Results")

comparison_export = comparison.copy()
for col in ['Hatch of Set', 'Hatch Of Fertile', 'Fertility %', 'Act Dead %', 'Banger %']:
    comparison_export[col] = (comparison_export[col] * 100).round(2)

comparison_export['Egg Collect Date'] = comparison_export['Egg Collect Date'].dt.date

output = BytesIO()
with pd.ExcelWriter(output, engine='openpyxl') as writer:
    comparison_export.to_excel(writer, sheet_name='Batch_by_Hatchery', index=False)
    if pairwise_results:
        pd.DataFrame(pairwise_results).to_excel(writer, sheet_name='Hatchery_Comparison', index=False)
    
    variance_export = variance_analysis.copy()
    variance_export.to_excel(writer, sheet_name='Variance_Analysis', index=False)
    
    pivot_display.to_excel(writer, sheet_name='Individual_Flocks')

st.sidebar.download_button(
    label="Download Excel Report",
    data=output.getvalue(),
    file_name="batch_comparison_results.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)