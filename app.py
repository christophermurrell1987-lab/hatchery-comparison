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
# FILE UPLOAD
# ============================================
uploaded_file = st.file_uploader("Upload hatchery data (Excel file)", type=['xlsx', 'xls'])

if uploaded_file is not None:
    # Load data
    df = pd.read_excel(uploaded_file)
    df['Hatchery'] = df['Hatchery'].str.strip().str.title()
    
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
    
    # Aggregate by Batch + Hatchery
    comparison = df_compare.groupby(['Batch_Key', 'Egg Collect Date', 'flock_number', 'Hatchery']).agg({
        'Farm': 'first',
        'Type': 'first',
        'Flock Age': 'mean',
        'Egg Age': 'mean',
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
    
    # ============================================
    # MAIN DISPLAY
    # ============================================
    st.success(f"✓ Found **{len(multi_hatchery_batches)}** batches (flock + date) sent to multiple hatcheries")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Hatchery Summary", 
        "🔄 Pairwise Comparison", 
        "⚠️ High Variance Batches",
        "📋 Pivot Table",
        "📈 Charts"
    ])
    
    # ============================================
    # TAB 1: HATCHERY SUMMARY
    # ============================================
    with tab1:
        st.subheader("Overall Hatchery Performance (on shared batches only)")
        
        hatchery_summary = df_compare.groupby('Hatchery').agg({
            'Set': 'sum',
            'Hatched': 'sum',
            'Hatch of Set': 'mean',
            'Hatch Of Fertile': 'mean',
            'Fertility %': 'mean',
            'Act Dead %': 'mean',
            'Batch_Key': 'nunique'
        })
        hatchery_summary.columns = ['Total Set', 'Total Hatched', 'Avg HoS %', 'Avg HoF %', 
                                    'Avg Fertility %', 'Avg Dead %', 'Num Batches']
        
        # Convert to percentages
        for col in ['Avg HoS %', 'Avg HoF %', 'Avg Fertility %', 'Avg Dead %']:
            hatchery_summary[col] = (hatchery_summary[col] * 100).round(1)
        
        hatchery_summary['Total Set'] = hatchery_summary['Total Set'].astype(int)
        hatchery_summary['Total Hatched'] = hatchery_summary['Total Hatched'].astype(int)
        
        st.dataframe(hatchery_summary, use_container_width=True)
        
        # Quick comparison metrics
        st.subheader("Quick Comparison")
        cols = st.columns(len(hatchery_summary))
        for i, (hatchery, row) in enumerate(hatchery_summary.iterrows()):
            with cols[i]:
                st.metric(
                    label=hatchery,
                    value=f"{row['Avg HoS %']:.1f}%",
                    delta=f"{row['Num Batches']} batches",
                    delta_color="off"
                )
    
    # ============================================
    # TAB 2: PAIRWISE COMPARISON
    # ============================================
    with tab2:
        st.subheader("Head-to-Head Hatchery Comparison (on same batches)")
        
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
                        'H1 Avg HoS %': round(h1_data['Hatch of Set'].mean() * 100, 1),
                        'H2 Avg HoS %': round(h2_data['Hatch of Set'].mean() * 100, 1),
                        'Difference %': round(diff, 1),
                        'H1 Avg HoF %': round(h1_data['Hatch Of Fertile'].mean() * 100, 1),
                        'H2 Avg HoF %': round(h2_data['Hatch Of Fertile'].mean() * 100, 1)
                    })
        
        if pairwise_results:
            pairwise_df = pd.DataFrame(pairwise_results)
            
            # Color code the difference
            def highlight_diff(val):
                if val > 2:
                    return 'background-color: #90EE90'  # Green - H1 better
                elif val < -2:
                    return 'background-color: #FFB6C1'  # Red - H2 better
                return ''
            
            styled_df = pairwise_df.style.applymap(highlight_diff, subset=['Difference %'])
            st.dataframe(styled_df, use_container_width=True)
            
            st.caption("🟢 Green = Hatchery 1 better by >2% | 🔴 Red = Hatchery 2 better by >2%")
        else:
            st.warning("No shared batches found between hatcheries")
    
    # ============================================
    # TAB 3: HIGH VARIANCE BATCHES
    # ============================================
    with tab3:
        st.subheader("Batches with Highest Variance Between Hatcheries")
        st.markdown("*Same flock, same collection date - different results*")
        
        # Calculate variance
        variance_analysis = comparison.groupby(['Batch_Key', 'Egg Collect Date', 'flock_number']).agg({
            'Hatch of Set': ['mean', 'min', 'max', 'count'],
            'Hatch Of Fertile': ['mean', 'min', 'max'],
        }).reset_index()
        
        variance_analysis.columns = ['Batch_Key', 'Egg Collect Date', 'Flock', 
                                     'Avg HoS %', 'Min HoS %', 'Max HoS %', 'Num Hatcheries',
                                     'Avg HoF %', 'Min HoF %', 'Max HoF %']
        
        variance_analysis['HoS Range %'] = (variance_analysis['Max HoS %'] - variance_analysis['Min HoS %']) * 100
        
        # Convert to percentages
        for col in ['Avg HoS %', 'Min HoS %', 'Max HoS %', 'Avg HoF %', 'Min HoF %', 'Max HoF %']:
            variance_analysis[col] = (variance_analysis[col] * 100).round(1)
        
        variance_analysis['HoS Range %'] = variance_analysis['HoS Range %'].round(1)
        
        # Threshold slider
        threshold = st.slider("Minimum variance threshold (%)", 0, 30, 10)
        high_var = variance_analysis[variance_analysis['HoS Range %'] > threshold].sort_values('HoS Range %', ascending=False)
        
        st.metric("Batches above threshold", len(high_var))
        
        display_cols = ['Egg Collect Date', 'Flock', 'Avg HoS %', 'Min HoS %', 'Max HoS %', 'HoS Range %', 'Num Hatcheries']
        st.dataframe(high_var[display_cols].head(50), use_container_width=True)
    
    # ============================================
    # TAB 4: PIVOT TABLE
    # ============================================
    with tab4:
        st.subheader("Side-by-Side Hatchery Comparison")
        
        metric_choice = st.selectbox("Select metric", ['Hatch of Set', 'Hatch Of Fertile', 'Fertility %', 'Act Dead %'])
        
        pivot = comparison.pivot_table(
            index=['Egg Collect Date', 'flock_number'],
            columns='Hatchery',
            values=metric_choice,
            aggfunc='mean'
        ) * 100
        
        pivot = pivot.round(1)
        
        st.dataframe(pivot, use_container_width=True, height=500)
    
    # ============================================
    # TAB 5: CHARTS
    # ============================================
    with tab5:
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
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax3)
        ax3.set_title('Average Performance Metrics by Hatchery (%)')
        plt.tight_layout()
        st.pyplot(fig3)
    
    # ============================================
    # EXPORT
    # ============================================
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 Export Results")
    
    # Prepare export data
    comparison_export = comparison.copy()
    for col in ['Hatch of Set', 'Hatch Of Fertile', 'Fertility %', 'Act Dead %', 'Banger %']:
        comparison_export[col] = (comparison_export[col] * 100).round(1)
    
    # Create Excel file in memory
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        comparison_export.to_excel(writer, sheet_name='Batch_by_Hatchery', index=False)
        hatchery_summary.to_excel(writer, sheet_name='Hatchery_Summary')
        if pairwise_results:
            pd.DataFrame(pairwise_results).to_excel(writer, sheet_name='Pairwise_Comparison', index=False)
        variance_analysis.to_excel(writer, sheet_name='Variance_Analysis', index=False)
        pivot.to_excel(writer, sheet_name='Pivot_Table')
    
    st.sidebar.download_button(
        label="Download Excel Report",
        data=output.getvalue(),
        file_name="batch_comparison_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    # No file uploaded - show instructions
    st.info("👆 Upload your hatchery Excel file to get started")
    
    st.markdown("""
    ### Required columns:
    - `Egg Collect Date` - Date eggs were collected
    - `flock_number` - Flock identifier
    - `Hatchery` - Hatchery name
    - `Hatch of Set` - Hatch rate (as decimal)
    - `Hatch Of Fertile` - Hatch of fertile rate
    - `Fertility %` - Fertility rate
    - `Act Dead %` - Dead in shell rate
    
    ### What this tool does:
    1. Finds batches (same flock + collection date) sent to multiple hatcheries
    2. Compares performance across hatcheries on the same eggs
    3. Identifies high-variance batches where hatcheries performed differently
    4. Provides head-to-head pairwise comparisons
    """)
