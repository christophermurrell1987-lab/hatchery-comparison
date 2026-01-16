# Hatchery Hatch Analysis - Web Application
# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
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
def load_hatchery_data_v11():
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
        df['Type'] = df['Type'].astype(str).str.upper().str.strip()
        df['Type'] = df['Type'].str.replace(' ', '', regex=False)
        df['Type'] = df['Type'].str.replace('PHNS', 'PH', regex=False)
    
    # ----------------------------------------
    # DATA CLEANING - DATES
    # ----------------------------------------
    if 'Egg Collect Date' in df.columns:
        df['Egg Collect Date'] = pd.to_datetime(df['Egg Collect Date'], errors='coerce')
        df = df[df['Egg Collect Date'].dt.year >= 2000]
        df['Egg Collect Date'] = df['Egg Collect Date'].dt.normalize()
    
    # ----------------------------------------
    # DATA CLEANING - FLOCK AGE (Standardize)
    # ----------------------------------------
    if 'Flock Age' in df.columns:
        df['Flock Age'] = pd.to_numeric(df['Flock Age'], errors='coerce').fillna(0).round(0).astype(int)
        
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
        df_raw = load_hatchery_data_v11()
        st.sidebar.success("✓ Loaded central data")
    except FileNotFoundError:
        st.error("hatchdata.xlsx not found. Please upload a file instead.")
        use_default = False

if not use_default:
    uploaded_file = st.file_uploader("Upload your own data", type=['xlsx', 'xls'])
    if uploaded_file is None:
        st.info("👆 Upload an Excel file or check 'Use central hatchery data'")
        st.stop()
    df_raw = pd.read_excel(uploaded_file)
    
    # Apply cleaning to uploaded data
    df_raw['Hatchery'] = df_raw['Hatchery'].str.strip().str.title()
    if 'Type' in df_raw.columns:
        df_raw['Type'] = df_raw['Type'].astype(str).str.upper().str.strip()
        df_raw['Type'] = df_raw['Type'].str.replace(' ', '', regex=False)
        df_raw['Type'] = df_raw['Type'].str.replace('PHNS', 'PH', regex=False)
    if 'Egg Collect Date' in df_raw.columns:
        df_raw['Egg Collect Date'] = pd.to_datetime(df_raw['Egg Collect Date'], errors='coerce')
        df_raw = df_raw[df_raw['Egg Collect Date'].dt.year >= 2000]
        df_raw['Egg Collect Date'] = df_raw['Egg Collect Date'].dt.normalize()
    if 'Flock Age' in df_raw.columns:
        df_raw['Flock Age'] = pd.to_numeric(df_raw['Flock Age'], errors='coerce').fillna(0).round(0).astype(int)

# Initialize working dataframe
df = df_raw.copy()

# ============================================
# SIDEBAR FILTERS
# ============================================
st.sidebar.header("🔧 Filters")

if not df.empty:
    min_date = df['Egg Collect Date'].min().date()
    max_date = df['Egg Collect Date'].max().date()
    if min_date.year < 2000:
        min_date = pd.to_datetime("2000-01-01").date()
else:
    min_date = pd.to_datetime('today').date()
    max_date = pd.to_datetime('today').date()

st.sidebar.subheader("Date Range (Global)")
start_date = st.sidebar.date_input("Start date", value=min_date, min_value=min_date, max_value=max_date, key=f"start_{min_date}_{max_date}", format="DD/MM/YYYY")
end_date = st.sidebar.date_input("End date", value=max_date, min_value=min_date, max_value=max_date, key=f"end_{min_date}_{max_date}", format="DD/MM/YYYY")

st.sidebar.subheader("Select Hatcheries")
all_hatcheries = sorted(df['Hatchery'].unique().tolist()) if not df.empty else []
selected_hatcheries = st.sidebar.multiselect("Hatcheries", all_hatcheries, default=all_hatcheries)

st.sidebar.subheader("Select Line / Type")
all_types = sorted(df['Type'].unique().tolist()) if not df.empty and 'Type' in df.columns else []
selected_types = st.sidebar.multiselect("Line / Breed", all_types, default=all_types)

# ============================================
# GLOBAL FILTER APPLICATION
# ============================================
if not df.empty:
    df = df[(df['Egg Collect Date'].dt.date >= start_date) & (df['Egg Collect Date'].dt.date <= end_date)]
    df = df[df['Hatchery'].isin(selected_hatcheries)]
    if 'Type' in df.columns:
        df = df[df['Type'].isin(selected_types)]
    if df.empty:
        st.warning("No data found for the selected filters.")
        st.stop()
else:
    st.warning("No data available.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.metric("Total Records", f"{len(df):,}")
st.sidebar.metric("Unique Flocks", f"{df['flock_number'].nunique():,}")

# ============================================
# DATE GROUPING
# ============================================
col_group_mode, col_spacer = st.columns([1, 3])
with col_group_mode:
    date_mode = st.radio("📅 **Group Collection Dates:**", options=["Daily (Exact Date)", "Weekly (Combine Mon/Thu)"], index=0)

if date_mode == "Weekly (Combine Mon/Thu)":
    df['Comparison_Date'] = df['Egg Collect Date'].dt.to_period('W-MON').apply(lambda r: r.start_time)
else:
    df['Comparison_Date'] = df['Egg Collect Date']

df['Batch_Key'] = df['Comparison_Date'].dt.strftime('%Y-%m-%d') + '_' + df['flock_number'].astype(str)

# ============================================
# BATCH MATCHING
# ============================================
batch_summary = df.groupby('Batch_Key').agg({
    'Hatchery': 'nunique',
    'Comparison_Date': 'first',
    'flock_number': 'first',
    'Set': 'sum'
}).reset_index()
batch_summary.columns = ['Batch_Key', 'Num_Hatcheries', 'Date', 'Flock', 'Total_Set']

multi_hatchery_batches = batch_summary[batch_summary['Num_Hatcheries'] > 1]['Batch_Key'].tolist()
df_compare = df[df['Batch_Key'].isin(multi_hatchery_batches)].copy()

if filter_suspicious and not df_compare.empty:
    suspicious_batches = []
    for batch in multi_hatchery_batches:
        batch_data = df_compare[df_compare['Batch_Key'] == batch]
        if batch_data['Set'].nunique() == 1 and batch_data['Hatched'].nunique() == 1:
            suspicious_batches.append(batch)
    if suspicious_batches:
        df_compare = df_compare[~df_compare['Batch_Key'].isin(suspicious_batches)]
        st.sidebar.warning(f"Filtered out {len(suspicious_batches)} suspicious duplicate batches.")

if 'Egg age at Set' not in df_compare.columns:
    df_compare['Egg age at Set'] = df_compare['Egg Age']
else:
    df_compare['Egg age at Set'] = df_compare['Egg age at Set'].fillna(df_compare['Egg Age'])

if not df_compare.empty:
    comparison = df_compare.groupby(['Batch_Key', 'Comparison_Date', 'flock_number', 'Hatchery']).agg({
        'Farm': 'first', 'Type': 'first', 'Flock Age': 'mean', 'Egg Age': 'mean', 'Egg age at Set': 'mean',
        'Set': 'sum', 'Hatched': 'sum', 'Infertile': 'sum', 'Bangers': 'sum',
        'Hatch of Set': 'mean', 'Hatch Of Fertile': 'mean', 'Fertility %': 'mean',
        'Act Dead %': 'mean', 'Banger %': 'mean'
    }).reset_index()
    comparison['Flock Age'] = comparison['Flock Age'].round(0).astype(int)
else:
    comparison = pd.DataFrame()

# ============================================
# TABS
# ============================================
st.success(f"✓ Found **{comparison['Batch_Key'].nunique() if not comparison.empty else 0}** valid shared batches")

tab1, tab2, tab3, tab4 = st.tabs(["🔄 Hatchery Comparison", "📋 Individual Flocks", "🐣 Hatch Curve", "📈 Comparison Charts"])

# ============================================
# TAB 1
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
                        'Hatchery 1': h1, 'Hatchery 2': h2, 'Shared Batches': len(shared),
                        'H1 Avg HoS %': round(h1_data['Hatch of Set'].mean() * 100, 2),
                        'H2 Avg HoS %': round(h2_data['Hatch of Set'].mean() * 100, 2),
                        'Difference %': round(diff, 2),
                        'H1 Avg HoF %': round(h1_data['Hatch Of Fertile'].mean() * 100, 2),
                        'H2 Avg HoF %': round(h2_data['Hatch Of Fertile'].mean() * 100, 2)
                    })
        if pairwise_results:
            st.dataframe(pd.DataFrame(pairwise_results).style.map(
                lambda v: 'background-color: #90EE90' if v > 2 else ('background-color: #FFB6C1' if v < -2 else ''), 
                subset=['Difference %']
            ).hide(axis="index").format("{:.2f}", subset=['H1 Avg HoS %', 'H2 Avg HoS %', 'Difference %', 'H1 Avg HoF %', 'H2 Avg HoF %']), use_container_width=True, hide_index=True)
            st.caption("🟢 Green = Hatchery 1 better by >2% | 🔴 Red = Hatchery 2 better by >2%")
        else:
            st.warning("No overlapping batches found.")
    else:
        st.info("No overlapping data.")

# ============================================
# TAB 2
# ============================================
with tab2:
    st.subheader("Detailed Flock Comparison")
    col_metric, col_slider = st.columns([1, 2])
    with col_metric:
        metric_choice = st.selectbox("Select metric", ['Hatch of Set', 'Hatch Of Fertile', 'Fertility %', 'Act Dead %'])
    with col_slider:
        threshold = st.slider("Show only batches with variance > X %", 0, 30, 0)
    
    if not comparison.empty:
        df_display = comparison.copy()
        if threshold > 0:
            var_calc = df_display.groupby(['Batch_Key'])[metric_choice].agg(['min', 'max'])
            high_var_keys = var_calc[(var_calc['max'] - var_calc['min']) * 100 > threshold].index.tolist()
            df_display = df_display[df_display['Batch_Key'].isin(high_var_keys)]
            st.caption(f"Showing {len(high_var_keys)} batches with >{threshold}% variance")
        
        if not df_display.empty:
            df_display['Flock Age'] = df_display.groupby(['Batch_Key'])['Flock Age'].transform('mean').round().astype(int)
            pivot_data = df_display.copy()
            pivot_data[metric_choice] = pivot_data[metric_choice] * 100
            
            pivot = pivot_data.pivot_table(
                index=['Comparison_Date', 'Farm', 'flock_number', 'Type', 'Flock Age'],
                columns='Hatchery',
                values=[metric_choice, 'Egg age at Set'],
                aggfunc='mean'
            ).swaplevel(0, 1, axis=1).sort_index(axis=1)
            
            pivot_display = pivot.reset_index()
            pivot_display['Comparison_Date'] = pd.to_datetime(pivot_display['Comparison_Date'])
            
            subset_metric = [c for c in pivot_display.columns if c[1] == metric_choice]
            subset_egg_age = [c for c in pivot_display.columns if c[1] == 'Egg age at Set']
            
            date_col_label = "Week Commencing" if date_mode == "Weekly (Combine Mon/Thu)" else "Collection Date"
            st.dataframe(
                pivot_display.style.map(
                    lambda v: 'background-color: #e6f3ff; color: black' if pd.notnull(v) and v != '' else '', 
                    subset=subset_metric
                ).format("{:.2f}", subset=subset_metric).format("{:.0f}", subset=subset_egg_age).hide(axis="index"),
                use_container_width=True, height=600, hide_index=True,
                column_config={"Comparison_Date": st.column_config.DateColumn(date_col_label, format="DD/MM/YY")}
            )
        else:
            st.warning(f"No batches found with >{threshold}% variance.")
    else:
        st.info("No data.")

# ============================================
# TAB 3: HATCH CURVE (HOVER FIX)
# ============================================
with tab3:
    st.subheader("🐣 Flock Hatch Curve")
    if selected_types:
        df_curve_source = df_raw[df_raw['Type'].isin(selected_types)]
    else:
        df_curve_source = df_raw
    
    available_flocks = sorted(df_curve_source['flock_number'].unique().tolist())
    
    col_sel_flock, col_sel_metric = st.columns(2)
    with col_sel_flock:
        selected_flock_curve = st.selectbox("Select Flock to Graph", available_flocks)
    with col_sel_metric:
        curve_metric = st.selectbox("Y-Axis Metric", ['Hatch of Set', 'Hatch Of Fertile', 'Fertility %'], index=0)

    if selected_flock_curve:
        curve_data = df_raw[df_raw['flock_number'] == selected_flock_curve].copy()
        
        # Ensure Egg Age is populated
        if 'Egg Age' not in curve_data.columns and 'Egg age at Set' in curve_data.columns:
             curve_data['Egg Age'] = curve_data['Egg age at Set']
        elif 'Egg Age' in curve_data.columns:
             if 'Egg age at Set' in curve_data.columns:
                  curve_data['Egg Age'] = curve_data['Egg Age'].fillna(curve_data['Egg age at Set'])

        curve_data[curve_metric] = curve_data[curve_metric] * 100
        
        # Rename for tooltip clarity
        curve_data = curve_data.rename(columns={
            'Egg Age': 'Egg Age (Days)',
            'Egg Collect Date': 'Collect Date',
            'Set': 'Eggs Set'
        })
        
        curve_data['Date Str'] = curve_data['Collect Date'].dt.strftime('%d/%m/%y')
        curve_data = curve_data.sort_values('Flock Age')
        
        # PLOTLY CHART (Mode changed to 'closest')
        fig = px.line(
            curve_data, 
            x="Flock Age", 
            y=curve_metric, 
            color="Hatchery",
            markers=True,
            title=f"Hatch Curve: Flock {selected_flock_curve}",
            hover_data={
                "Flock Age": True,
                "Hatchery": True,
                "Date Str": True,
                "Egg Age (Days)": ":.1f", 
                "Eggs Set": True,
                curve_metric: ":.2f"
            },
            labels={"Date Str": "Date"} 
        )
        
        fig.update_layout(
            hovermode="closest", # CRITICAL FIX: Allows hovering over individual dots
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis_title=f"{curve_metric} (%)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("Show Raw Data for Curve"):
            display_cols = ['Collect Date', 'Hatchery', 'Flock Age', 'Egg Age (Days)', 'Eggs Set', curve_metric]
            curve_data_table = curve_data.copy()
            curve_data_table['Collect Date'] = curve_data_table['Collect Date'].dt.strftime('%d/%m/%y')
            curve_data_table[curve_metric] = curve_data_table[curve_metric].round(2)
            
            st.dataframe(
                curve_data_table[display_cols].sort_values('Flock Age'), 
                use_container_width=True, 
                hide_index=True
            )

# ============================================
# TAB 4
# ============================================
with tab4:
    st.subheader("Hatchery Distributions")
    if not df_compare.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Hatch of Set**")
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            df_compare_pct = df_compare.copy()
            df_compare_pct['Hatch of Set'] = df_compare_pct['Hatch of Set'] * 100
            df_compare_pct.boxplot(column='Hatch of Set', by='Hatchery', ax=ax1)
            ax1.set_title('')
            st.pyplot(fig1)
        with col2:
            st.markdown("**Hatch of Fertile**")
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            df_compare_pct['Hatch Of Fertile'] = df_compare['Hatch Of Fertile'] * 100
            df_compare_pct.boxplot(column='Hatch Of Fertile', by='Hatchery', ax=ax2)
            ax2.set_title('')
            st.pyplot(fig2)
    else:
        st.info("No comparison data.")

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
    comparison_export['Comparison_Date'] = comparison_export['Comparison_Date'].apply(lambda x: x.strftime('%d/%m/%y') if pd.notnull(x) else '')

    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        comparison_export.to_excel(writer, sheet_name='Batch_by_Hatchery', index=False)
        if 'pairwise_results' in locals() and pairwise_results:
            pd.DataFrame(pairwise_results).to_excel(writer, sheet_name='Hatchery_Comparison', index=False)
        if 'pivot_display' in locals():
            pivot_export = pivot_display.copy()
            pivot_export['Comparison_Date'] = pivot_export['Comparison_Date'].apply(lambda x: x.strftime('%d/%m/%y') if pd.notnull(x) else '')
            pivot_export.to_excel(writer, sheet_name='Individual_Flocks')

    st.sidebar.download_button(label="Download Excel Report", data=output.getvalue(), file_name="hatchery_hatch_analysis.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")