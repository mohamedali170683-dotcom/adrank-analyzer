import streamlit as st
import pandas as pd
import numpy as np
import openai
import json
import re

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AdRank AI | Universal Loader",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
    <style>
    .main { background-color: #f9f9f9; }
    h1 { color: #1E3A8A; }
    div[data-testid="stExpander"] { border: 1px solid #ddd; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("OpenAI API Key", type="password")
    st.info("Key not stored. Used for this session only.")

# ==========================================
# 2. THE UNIVERSAL MAPPER (The "Brain")
# ==========================================
def find_header_row(df_preview):
    """
    Scans the first 20 rows to find the one that looks like a header.
    Criteria: Must contain 'Impr' AND ('Cost' OR 'Clicks').
    """
    best_row_idx = None
    
    for idx, row in df_preview.iterrows():
        # Convert row to a single lowercase string
        row_str = " ".join(row.astype(str)).lower()
        
        # Check for strong signals of being a header
        has_metrics = 'impr' in row_str and ('cost' in row_str or 'click' in row_str)
        has_text = 'asset' in row_str or 'headline' in row_str or 'item' in row_str or 'ad' in row_str
        
        if has_metrics:
            return idx # Found it!
            
    return 0 # Default to first row if nothing found

def normalize_columns(df):
    """
    Renames columns based on their content, not just their exact name.
    """
    df.columns = df.columns.astype(str).str.strip().str.lower()
    
    # Define our dictionary of "Concepts" -> "Keywords"
    # We map specific keywords to our internal standard names
    mapping_logic = {
        'Cleaned_Text': ['asset', 'headline', 'item', 'ad text', 'description'],
        'Impressions': ['impr', 'views'],
        'Clicks': ['click'],
        'Cost': ['cost'], # We will need to be careful not to grab "Cost / conv"
        'Conv_Value': ['conv. value', 'total conv. value', 'conversion value'],
        'Conversions': ['conversions']
    }
    
    new_columns = {}
    assigned_cols = []

    # Iterate through our standard names
    for standard_name, keywords in mapping_logic.items():
        # Find the best matching column in the CSV
        for col in df.columns:
            if col in assigned_cols: continue # Don't map same column twice
            
            # Special Logic for Cost to avoid "Cost / Conversion"
            if standard_name == 'Cost' and 'cost /' in col:
                continue
            
            # Check if any keyword exists in this column name
            if any(k in col for k in keywords):
                new_columns[col] = standard_name
                assigned_cols.append(col)
                break # Stop looking for this standard name once found
    
    # Rename the columns
    return df.rename(columns=new_columns)

def process_data(uploaded_file):
    try:
        # 1. READ RAW (No header assumption yet)
        # Read first 20 rows to hunt for header
        preview = pd.read_csv(uploaded_file, header=None, nrows=20, on_bad_lines='skip')
        header_idx = find_header_row(preview)
        
        # Reload with correct header
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, skiprows=header_idx)
        
        # 2. RENAME COLUMNS INTELLIGENTLY
        df = normalize_columns(df)
        
        # DEBUG: Save detected columns to session state for user visibility
        st.session_state['detected_cols'] = df.columns.tolist()

        # 3. VERIFY REQUIRED COLUMNS
        required = ['Cleaned_Text', 'Impressions', 'Cost']
        missing = [req for req in required if req not in df.columns]
        
        if missing:
            return None, f"Could not detect columns for: {', '.join(missing)}"

        # 4. CLEAN DATA TYPES
        def clean_num(x):
            try:
                if pd.isna(x): return 0.0
                return float(str(x).replace('%','').replace(',','').replace('--','0'))
            except: return 0.0

        for col in ['Impressions', 'Clicks', 'Cost', 'Conv_Value', 'Conversions']:
            if col in df.columns:
                df[col] = df[col].apply(clean_num)
            else:
                df[col] = 0.0 # Fill missing optional metrics with 0

        # 5. AGGREGATE
        # Clean text
        df['Cleaned_Text'] = df['Cleaned_Text'].astype(str).str.strip('"').str.strip()
        df = df[df['Cleaned_Text'].str.len() > 2]
        df = df[~df['Cleaned_Text'].str.match(r'^[\d\-]+$')] # No ID numbers

        # Group
        grouped = df.groupby('Cleaned_Text').agg({
            'Impressions': 'sum', 'Clicks': 'sum', 'Cost': 'sum', 
            'Conv_Value': 'sum', 'Conversions': 'sum'
        }).reset_index()

        # KPIs
        grouped['CTR'] = np.where(grouped['Impressions']>0, (grouped['Clicks']/grouped['Impressions'])*100, 0)
        grouped['ROAS'] = np.where(grouped['Cost']>0, grouped['Conv_Value']/grouped['Cost'], 0)
        
        # Filter Noise
        grouped = grouped[grouped['Impressions'] > 50]
        
        return grouped, None

    except Exception as e:
        return None, str(e)

# ==========================================
# 3. AI ENGINE
# ==========================================
def run_gpt_analysis(df, key):
    client = openai.OpenAI(api_key=key)
    
    top_performers = df.sort_values(by='ROAS', ascending=False).head(10).to_dict(orient='records')
    click_bait = df.sort_values(by='CTR', ascending=False).head(8).to_dict(orient='records')
    
    prompt = f"""
    You are AdRank AI. Analyze this Google Ads Data.
    
    TOP ADS (High ROAS): {json.dumps(top_performers)}
    HIGH CTR ADS: {json.dumps(click_bait)}
    
    OUTPUT FORMAT:
    1. Executive Summary.
    2. Winning Psychological Patterns (Tone, Words).
    3. Losing Patterns.
    4. 3 New Headlines to Test.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# ==========================================
# 4. UI MAIN
# ==========================================
st.title("🧠 AdRank AI: Universal Loader")
st.write("Upload your Google Ads CSV (Assets or Ads report). We will auto-detect the columns.")

uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

if uploaded_file is not None:
    # RUN THE PROCESSOR
    with st.spinner("Scanning file structure..."):
        clean_df, error_msg = process_data(uploaded_file)
    
    # SUCCESS STATE
    if clean_df is not None and not clean_df.empty:
        st.success("✅ File Successfully Mapped!")
        
        col1, col2 = st.columns(2)
        col1.metric("Ads Analyzed", len(clean_df))
        col2.metric("Total Spend", f"${clean_df['Cost'].sum():,.0f}")
        
        with st.expander("👀 View Processed Data"):
            st.dataframe(clean_df.head())
            
        if st.button("Run AI Analysis"):
            if not api_key:
                st.error("Please enter API Key.")
            else:
                with st.spinner("Thinking..."):
                    res = run_gpt_analysis(clean_df, api_key)
                    st.markdown("---")
                    st.markdown(res)

    # FAILURE STATE - DEBUGGER
    else:
        st.error("⚠️ Data Mapping Failed")
        if error_msg:
            st.write(f"**Reason:** {error_msg}")
        
        st.markdown("### 🕵️‍♂️ Debug Info")
        st.write("We detected these columns in your file. Did we miss the 'Asset' or 'Headline' column?")
        if 'detected_cols' in st.session_state:
            st.write(st.session_state['detected_cols'])
        
        st.info("Tip: Ensure your CSV has a column with your Ad Text (named 'Asset', 'Headline', or 'Item').")
