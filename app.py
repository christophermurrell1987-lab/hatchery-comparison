# Hatchery Hatch Analysis - Web Application
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
    page_title="Hatchery Hatch Analysis",
    page_icon="🥚",
    layout="wide"
)

st.title("🥚 Hatchery Hatch Analysis")
st.markdown("Compare hatchery performance on eggs from the same flock & collection date (or week)")

# ============================================
# CACHED DATA LOADING
# ============================================
@st.cache_data
def load_hatchery_data_v6():
    """Load and cache the default hatchery data with AGGRESSIVE cleaning"""
    df = pd.read_excel('hatchdata.xlsx')
    
    # ----------------------------------------
    # DATA CLEANING - HATCHERY
    # ----------------------------------------
    df['Hatchery'] = df['Hatchery'].str.strip().str.title()
    
    # ----------------------------------------
    # DATA CLEANING - TYPE / LINE
    # ----------------------------------------
    if 'Type' in df.columns:
        df['Type'] = df['Type'].astype(str)
        df['Type'] = df['Type'].str.upper()
        df['Type'] = df['Type'].str.strip()
        df['Type'] = df['Type'].str.replace(' ', '', regex=False)
        df['Type'] = df['Type'].str.replace('PHNS', 'PH', regex=False)
    
    # ----------------------------------------
    # DATA CLEANING - DATES
    # ----------------------------------------
    if 'Egg Collect Date' in df.columns:
        df['Egg Collect Date'] = pd.to_datetime(df['Egg Collect Date'], errors='coerce')
        # Remove 1900 errors
        df = df[df['Egg Collect Date'].dt.year >= 2000]
        # Remove time component
        df['Egg Collect Date'] = df['Egg Collect Date'].dt.normalize()
        
    return df

# ============================================
# SIDEBAR - DATA CLEANING
# ============================================
st.sidebar.header("🚫 Data Cleaning")
with st.sidebar.container():
    st.sidebar.info("📉 **Duplicate Data Filter**")
    filter_suspicious = st.sidebar.checkbox(
        "Hide suspicious duplicate data", 
        value=True,
        help="Check this to automatically hide batches where different hatcheries have the EXACT same 'Set' and 'Hatched' numbers."
    )
st.sidebar.markdown("---")

# ============================================
# DATA SOURCE
# ============================================
st.sidebar.header("📁 Data Source")

use_default = st.sidebar.checkbox("Use central hatchery data", value=True)

if use_default:
    try:
        df = load_hatchery_data_v6()
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
    
    # Apply SAME cleaning to uploaded data
    df['Hatchery'] = df['Hatchery'].str.strip().str.title()
    
    if 'Type' in df.columns:
        df['Type'] = df['Type'].astype(str).str.upper().str.strip()
        df['Type'] = df['Type'].str.replace(' ', '', regex=False)
        df['Type'] = df['Type'].str.replace('PHNS', 'PH', regex=False)
    
    if 'Egg Collect Date' in df.columns:
        df['Egg Collect Date'] = pd.to_datetime(df['Egg Collect Date'], errors='coerce')
        df = df[df['Egg Collect Date'].dt.year >= 2000]
        df['Egg Collect Date'] = df['Egg Collect Date'].dt.normalize()

# ============================================
# SIDEBAR FILTERS
# ============================================
st.sidebar.header("🔧 Filters")

# Calculate min/max dates
if not df.empty:
    min_date = df['Egg Collect Date'].min().date()
    max_date = df['Egg Collect Date'].max().date()
    if min_date.year < 2000:
        min_date = pd.to_datetime("2000-01-01").date()
else:
    min_date = pd.to_datetime('today').date()
    max_date = pd.to_datetime('today').date()

st.sidebar.subheader("Date Range")
start_date = st.sidebar.date_input(
    "Start date", 
    value=min_date, 
    min_value=min_date, 
    max_value=max_date,
    key=f"start_{min_date}_{max_date}",
    format="DD/MM/YYYY" 
)
end_date = st.sidebar.date_input(
    "End date", 
    value=max_date, 
    min_value=min_date, 
    max_value=max_date,
    key=f"end_{min_date}_{max_date}",
    format="DD/MM/YYYY" 
)

# Hatchery filter
st.sidebar.subheader("Select Hatcheries")
all_hatcheries = sorted(df['Hatchery'].unique().tolist()) if not df.empty else []
selected_hatcheries = st.sidebar.multiselect(
    "Hatcheries",
    all_hatcheries,
    default=all_hatcheries
)

# Line / Type Filter
st.sidebar.subheader("Select Line / Type")
all_types = sorted(df['Type'].unique().tolist()) if not df.empty and 'Type' in df.columns else []
selected_types = st.sidebar.multiselect(
    "Line / Breed",
    all_types,
    default=all_types
)

# ============================================
# FILTERING LOGIC
# ============================================
if not df.empty:
    # Filter Date
    df = df[(df['Egg Collect Date'].dt.date >= start_date) & 
            (df['Egg Collect Date'].dt.date <= end_date)]
    
    # Filter Hatchery
    df = df[df['Hatchery'].isin(selected_hatcheries)]
    
    # Filter Line/Type
    if 'Type' in df.columns:
        df = df[df['Type'].isin(selected_types)]
            
    if df.empty:
        st.warning("No data found for the selected filters.")
        st.stop()
else:
    st.warning("No data available.")
    st.stop()

# Show data summary
st.sidebar.markdown("---")
st.sidebar.metric("Total Records", f"{len(df):,}")
st.sidebar.metric("Unique Flocks", f"{df['flock_number'].nunique():,}")

# ============================================
# DATE GROUPING SELECTION
# ============================================
col_group_mode, col_spacer = st.columns([1, 3])
with col_group_mode:
    date_mode = st.radio(
        "📅 **Group Collection Dates:**",
        options=["Daily (Exact Date)", "Weekly (Combine Mon/Thu)"],
        index=0,
        help="Daily matches exact dates. Weekly groups all collections in the same week (starting Monday) to compare data from different collection days."
    )

# Apply Grouping Logic
if date_mode == "Weekly (Combine Mon/Thu)":
    # Convert date to the Monday of that week
    df['Comparison_Date'] = df['Egg Collect Date'].dt.to_period('W-MON').apply(lambda r: r.start_time)
else:
    # Use exact date
    df['Comparison_Date'] = df['Egg Collect Date']

# ============================================
# BATCH MATCHING
# ============================================
# Create Batch Key using Comparison_Date
df['Batch_Key'] = df['Comparison_Date'].dt.strftime('%Y-%m-%d') + '_' + df['flock_number'].astype(str)

# Find batches that exist in MORE THAN ONE of the *selected* hatcheries
batch_summary = df.groupby('Batch_Key').agg({
    'Hatchery': 'nunique',
    'Comparison_Date': 'first',
    'flock_number': 'first',
    'Set': 'sum'
}).reset_index()
batch_summary.columns = ['Batch_Key', 'Num_Hatcheries', 'Date', 'Flock', 'Total_Set']

multi_hatchery_batches = batch_summary[batch_summary['Num_Hatcheries'] > 1]['Batch_Key'].tolist()
df_compare = df[df['Batch_Key'].isin(multi_hatchery_batches)].copy()

# --------------------------------------------
# DUPLICATE DETECTION LOGIC
# --------------------------------------------
if filter_suspicious and not df_compare.empty:
    suspicious_batches = []
    
    for batch in multi_hatchery_batches:
        batch_data = df_compare[df_compare['Batch_Key'] == batch]
        unique_sets = batch_data['Set'].nunique()
        unique_hatched = batch_data['Hatched'].nunique()
        
        if unique_sets == 1 and unique_hatched == 1:
            suspicious_batches.append(batch)
            
    if suspicious_batches:
        df_compare = df_compare[~df_compare['Batch_Key'].isin(suspicious_batches)]
        st.sidebar.warning(f"Filtered out {len(suspicious_batches)} suspicious duplicate batches.")

# Key metrics
metrics = ['Hatch of Set', 'Hatch Of Fertile', 'Fertility %', 'Act Dead %', 'Banger %']

# Ensure "Egg age at Set" is available
if 'Egg age at Set' not in df_compare.columns:
    df_compare['Egg age at Set'] = df_compare['Egg Age']
else:
    df_compare['Egg age at Set'] = df_compare['Egg age at Set'].fillna(df_compare['Egg Age'])

# Aggregate by Batch + Hatchery
if not df_compare.empty:
    comparison = df_compare.groupby(['Batch_Key', 'Comparison_Date', 'flock_number', 'Hatchery']).agg({
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
    
    comparison['Flock Age'] = comparison['Flock Age'].round(0).astype(int)
else:
    comparison = pd.DataFrame()

# ============================================
# MAIN DISPLAY
# ============================================
num_batches = comparison['Batch_Key'].nunique() if not comparison.empty else 0
st.success(f"✓ Found **{num_batches}** valid shared batches between the selected hatcheries")

tab1, tab2, tab3 = st.tabs([
    "🔄 Hatchery Comparison", 
    "📋 Individual Flocks",
    "📈 Charts"
])

# ============================================
# TAB 1: HATCHERY COMPARISON
# ============================================
with tab1:
    st.subheader("Head-to-Head Hatchery Comparison")
    
    if not comparison.empty:
        hatcheries = sorted(comparison['Hatchery'].unique())
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
                    return 'background-color: #90EE90'
                elif val < -2:
                    return 'background-color: #FFB6C1'
                return ''
            
            styled_df = pairwise_df.style.map(highlight_diff, subset=['Difference %'])
            styled_df.hide(axis="index") 
            
            st.dataframe(
                styled_df.format({
                    'H1 Avg HoS %': '{:.2f}',
                    'H2 Avg HoS %': '{:.2f}',
                    'Difference %': '{:.2f}',
                    'H1 Avg HoF %': '{:.2f}',
                    'H2 Avg HoF %': '{:.2f}'
                }), 
                use_container_width=True,
                hide_index=True 
            )
            
            st.caption("🟢 Green = Hatchery 1 better by >2% | 🔴 Red = Hatchery 2 better by >2%")
        else:
            st.warning("No shared batches found between the selected hatcheries.")
    else:
        st.info("Select at least two hatcheries with overlapping batches to see comparisons.")

# ============================================
# TAB 2: INDIVIDUAL FLOCKS
# ============================================
with tab2:
    st.subheader("Detailed Flock Comparison")
    
    col_metric, col_slider = st.columns([1, 2])
    
    with col_metric:
        metric_choice = st.selectbox("Select metric", ['Hatch of Set', 'Hatch Of Fertile', 'Fertility %', 'Act Dead %'])
        
    with col_slider:
        threshold = st.slider("Show only batches with variance > X % (0 = show all)", 0, 30, 0)
    
    if not comparison.empty:
        df_display = comparison.copy()
        
        if threshold > 0:
            var_calc = df_display.groupby(['Batch_Key'])[metric_choice].agg(['min', 'max'])
            var_calc['diff'] = (var_calc['max'] - var_calc['min']) * 100
            
            high_var_keys = var_calc[var_calc['diff'] > threshold].index.tolist()
            df_display = df_display[df_display['Batch_Key'].isin(high_var_keys)]
            
            st.caption(f"Showing {len(high_var_keys)} batches with >{threshold}% variance in {metric_choice}")
        
        if not df_display.empty:
            
            # --------------------------------------------
            # CRITICAL FIX: STANDARDIZE FLOCK AGE
            # --------------------------------------------
            # Calculate the average age for the entire batch (group) and overwrite the individual rows
            # This ensures that if Hatchery A says "30" and Hatchery B says "31", they both get "31" (grouped)
            df_display['Flock Age'] = df_display.groupby(['Batch_Key'])['Flock Age'].transform('mean').round().astype(int)
            
            # Now proceed with Pivot
            pivot_data = df_display.copy()
            pivot_data[metric_choice] = pivot_data[metric_choice] * 100
            
            pivot = pivot_data.pivot_table(
                index=['Comparison_Date', 'Farm', 'flock_number', 'Type', 'Flock Age'],
                columns='Hatchery',
                values=[metric_choice, 'Egg age at Set'],
                aggfunc='mean'
            )
            
            pivot = pivot.swaplevel(0, 1, axis=1).sort_index(axis=1)
            pivot_display = pivot.reset_index()
            pivot_display['Comparison_Date'] = pd.to_datetime(pivot_display['Comparison_Date'])

            def highlight_metric_only(val):
                if pd.notnull(val) and val != '':
                    return 'background-color: #e6f3ff; color: black'
                return ''

            subset_metric = [c for c in pivot_display.columns if c[1] == metric_choice]
            subset_egg_age = [c for c in pivot_display.columns if c[1] == 'Egg age at Set']
            
            date_col_label = "Week Commencing" if date_mode == "Weekly (Combine Mon/Thu)" else "Collection Date"
            
            st.dataframe(
                pivot_display.style
                .map(highlight_metric_only, subset=subset_metric)
                .format("{:.2f}", subset=subset_metric)
                .format("{:.0f}", subset=subset_egg_age)
                .hide(axis="index"), 
                use_container_width=True, 
                height=600,
                hide_index=True,
                column_config={
                    "Comparison_Date": st.column_config.DateColumn(
                        date_col_label,
                        format="DD/MM/YY"
                    )
                }
            )
        else:
            st.warning(f"No batches found with a variance greater than {threshold}% in {metric_choice}.")
    else:
        st.info("No overlapping data to display.")

# ============================================
# TAB 3: CHARTS
# ============================================
with tab3:
    st.subheader("Visual Comparisons")
    
    if not df_compare.empty:
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
    else:
        st.warning("No shared data found to visualize.")

# ============================================
# EXPORT
# ============================================
st.sidebar.markdown("---")
st.sidebar.subheader("📥 Export Results")

if not comparison.empty:
    comparison_export = comparison.copy()
    for col in ['Hatch of Set', 'Hatch Of Fertile', 'Fertility %', 'Act Dead %', 'Banger %']:
        comparison_export[col] = (comparison_export[col] * 100).round(2)

    comparison_export['Egg age at Set'] = comparison_export['Egg age at Set'].round(0)
    
    comparison_export['Comparison_Date'] = comparison_export['Comparison_Date'].apply(
        lambda x: x.strftime('%d/%m/%y') if pd.notnull(x) else ''
    )

    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        comparison_export.to_excel(writer, sheet_name='Batch_by_Hatchery', index=False)
        if 'pairwise_results' in locals() and pairwise_results:
            pd.DataFrame(pairwise_results).to_excel(writer, sheet_name='Hatchery_Comparison', index=False)
        
        if 'pivot_display' in locals():
            pivot_export = pivot_display.copy()
            for c in subset_metric:
                pivot_export[c] = pivot_export[c].round(2)
            for c in subset_egg_age:
                pivot_export[c] = pivot_export[c].round(0)
            
            pivot_export['Comparison_Date'] = pivot_export['Comparison_Date'].apply(
                lambda x: x.strftime('%d/%m/%y') if pd.notnull(x) else ''
            )
            pivot_export.to_excel(writer, sheet_name='Individual_Flocks')

    st.sidebar.download_button(
        label="Download Excel Report",
        data=output.getvalue(),
        file_name="hatchery_hatch_analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )