import streamlit as st
import pandas as pd
import numpy as np
import openai
import json
import re

# ==========================================
# 1. CONFIGURATION & UI SETUP
# ==========================================
st.set_page_config(
    page_title="AdRank AI | Behavioral Analysis",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main { background-color: #f9f9f9; }
    h1 { color: #1E3A8A; }
    div[data-testid="stExpander"] { border: 1px solid #e0e0e0; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("OpenAI API Key", type="password", help="Required for AI analysis.")
    st.info("Key is not stored. Used only for this session.")
    st.markdown("---")
    st.markdown("""
    **How to get the data:**
    1. Go to **Google Ads > Assets**.
    2. Click **Columns** icon.
    3. Add: *'Performance Label'*, *'Conv. value / cost'*.
    4. Download as **CSV**.
    """)

# ==========================================
# 2. SMART DATA LOADER & CLEANER
# ==========================================
def load_csv_smartly(uploaded_file):
    """
    Scans the file to find the row where the actual data headers start.
    Ignores Google's 'Report Name' or 'Date Range' metadata at the top.
    """
    try:
        # Read first 20 lines to find the header
        preview = pd.read_csv(uploaded_file, header=None, nrows=20, on_bad_lines='skip')
        
        header_row_idx = None
        for i, row in preview.iterrows():
            # Convert row to string and look for key columns
            row_str = row.astype(str).str.cat(sep=' ').lower()
            # We look for 'asset' AND 'impressions' to confirm it's the header
            if ('asset' in row_str or 'item' in row_str) and 'impressions' in row_str:
                header_row_idx = i
                break
        
        # Reset file pointer and read from the found row
        uploaded_file.seek(0)
        
        if header_row_idx is not None:
            # Read from the specific header row
            return pd.read_csv(uploaded_file, skiprows=header_row_idx)
        else:
            # Fallback: Try reading normally, then try skipping 2 rows (standard Google format)
            try:
                return pd.read_csv(uploaded_file)
            except:
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file, skiprows=2)

    except Exception as e:
        st.error(f"Could not read CSV structure: {e}")
        return None

def clean_and_aggregate_ads(df):
    """
    Standardizes column names, removes garbage rows, and aggregates stats by Text.
    """
    try:
        # 1. Normalize Columns (Strip whitespace, lowercase)
        df.columns = df.columns.astype(str).str.strip().str.lower()
        
        # 2. Map Google's variable names to our standard names
        col_map = {
            'asset': 'Cleaned_Text',
            'item': 'Cleaned_Text',      # Sometimes Google exports as 'Item'
            'headline': 'Cleaned_Text',  # Sometimes 'Headline'
            'impressions': 'Impressions',
            'clicks': 'Clicks',
            'cost': 'Cost',
            'conv. value': 'Conv_Value',
            'conversions': 'Conversions',
            'ctr': 'Raw_CTR',
            'conv. value / cost': 'Raw_ROAS'
        }
        
        # Rename columns that exist in the map
        df = df.rename(columns={c: col_map[c] for c in df.columns if c in col_map})

        # Debugging: Show user what we found if things go wrong
        if 'Cleaned_Text' not in df.columns:
            st.error("❌ Could not find an 'Asset' or 'Headline' column.")
            with st.expander("See detected columns"):
                st.write(df.columns.tolist())
            return None

        # 3. Clean Metric Values (Remove %, commas, currencies)
        def clean_metric(val):
            if pd.isna(val): return 0.0
            s = str(val).replace('%', '').replace(',', '')
            s = re.sub(r'[^\d.-]', '', s) # Remove currency symbols
            try:
                return float(s)
            except:
                return 0.0

        target_metrics = ['Impressions', 'Clicks', 'Cost', 'Conv_Value', 'Conversions']
        for m in target_metrics:
            if m in df.columns:
                df[m] = df[m].apply(clean_metric)
            else:
                df[m] = 0.0

        # 4. Filter Rows (Remove empty text, IDs, or weird artifacts)
        df['Cleaned_Text'] = df['Cleaned_Text'].astype(str).str.strip('"').str.strip("'").str.strip()
        df = df[df['Cleaned_Text'].str.len() > 3] # Remove text shorter than 3 chars
        df = df[~df['Cleaned_Text'].str.match(r'^[\d\-]+$')] # Remove pure numbers or hyphens
        
        # 5. Aggregate by Text (Merge duplicates)
        grouped = df.groupby('Cleaned_Text').agg({
            'Impressions': 'sum',
            'Clicks': 'sum',
            'Cost': 'sum',
            'Conv_Value': 'sum',
            'Conversions': 'sum'
        }).reset_index()

        # 6. Recalculate KPIs on Aggregated Data
        grouped['CTR'] = np.where(grouped['Impressions']>0, (grouped['Clicks']/grouped['Impressions'])*100, 0)
        grouped['ROAS'] = np.where(grouped['Cost']>0, grouped['Conv_Value']/grouped['Cost'], 0)
        grouped['CPA'] = np.where(grouped['Conversions']>0, grouped['Cost']/grouped['Conversions'], 0)

        # Rounding
        grouped = grouped.round(2)

        # Filter out low-volume noise
        grouped = grouped[grouped['Impressions'] > 50] 
        
        return grouped

    except Exception as e:
        st.error(f"Data Cleaning Error: {e}")
        return None

# ==========================================
# 3. AI ANALYSIS ENGINE
# ==========================================
def run_ai_analysis(df, key):
    client = openai.OpenAI(api_key=key)
    
    # Sort and slice data for context
    top_ads = df.sort_values(by='ROAS', ascending=False).head(10).to_dict(orient='records')
    click_magnets = df.sort_values(by='CTR', ascending=False).head(8).to_dict(orient='records')
    wasters = df[(df['Cost'] > df['Cost'].median()) & (df['ROAS'] < df['ROAS'].mean())].sort_values(by='Cost', ascending=False).head(8).to_dict(orient='records')
    
    system_prompt = """
    You are AdRank AI, a Behavioral Science Marketing Expert.
    
    ## YOUR TASK
    Analyze the provided ad performance data. Identify correlations between *specific words/tones* and *performance metrics*.
    
    ## THE CODEBOOK (Tags to use)
    1. **Psychology:** Scarcity, Social Proof, Authority, Reciprocity, Loss Aversion, Curiosity.
    2. **Tone:** Urgent, Corporate, Playful, Empathetic, Direct.
    3. **Format:** Question, List, "How-to", Statement.

    ## OUTPUT FORMAT
    1. **EXECUTIVE DIAGNOSIS:** One bold sentence on the account's biggest win or failure.
    2. **🏆 THE WINNING FORMULA:** What defines the high ROAS ads? (e.g., "Short questions + Scarcity").
    3. **🚩 THE CLICKBAIT TRAP:** Analyze ads with High CTR but Low ROAS. Why are they failing to convert?
    4. **📉 THE MONEY PIT:** Analyze the 'Wasters'. What phrases are costing money with no return?
    5. **🚀 3 HYPOTHESIS TESTS:** Write 3 NEW headlines to test based on the Winning Formula.
    """

    user_data = f"""
    TOP PERFORMERS (High ROAS - Model these):
    {json.dumps(top_ads)}

    CLICK MAGNETS (High Interest, Check Conversion Quality):
    {json.dumps(click_magnets)}

    WASTERS (High Spend, Poor Results - Avoid these):
    {json.dumps(wasters)}
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_data}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content

# ==========================================
# 4. MAIN APPLICATION
# ==========================================
st.title("🧠 AdRank AI: Behavioral Copy Analyzer")
st.markdown("Upload your Google Ads **Assets Report** to reveal the psychology behind your performance.")

uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

if uploaded_file is not None:
    # 1. Load
    raw_df = load_csv_smartly(uploaded_file)
    
    if raw_df is not None:
        # 2. Clean
        with st.spinner("Sanitizing data & aggregating duplicates..."):
            clean_df = clean_and_aggregate_ads(raw_df)
        
        if clean_df is not None and not clean_df.empty:
            # 3. Show Dashboard
            c1, c2, c3 = st.columns(3)
            c1.metric("Unique Assets", len(clean_df))
            c2.metric("Total Spend Analyzed", f"${clean_df['Cost'].sum():,.0f}")
            c3.metric("Avg ROAS", f"{clean_df['ROAS'].mean():.2f}")
            
            with st.expander("🔎 View Processed Data (Top 10 by Spend)"):
                st.dataframe(clean_df.sort_values(by='Cost', ascending=False).head(10))

            # 4. Run AI
            st.markdown("### 🤖 Behavioral Analysis")
            if st.button("Generate Insights"):
                if not api_key:
                    st.warning("⚠️ Please enter your OpenAI API Key in the sidebar.")
                else:
                    with st.spinner("Analyzing semantic patterns..."):
                        try:
                            insights = run_ai_analysis(clean_df, api_key)
                            st.success("Analysis Complete!")
                            st.markdown("---")
                            st.markdown(insights)
                        except Exception as e:
                            st.error(f"OpenAI Error: {e}")
        else:
            st.warning("⚠️ No valid data found after cleaning. Ensure your CSV contains text assets (Headlines/Descriptions) and performance metrics.")
