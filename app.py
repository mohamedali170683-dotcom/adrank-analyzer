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
    page_title="AdRank AI | RSA Analyzer",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
    <style>
    .main { background-color: #f9f9f9; }
    div[data-testid="stExpander"] { border: 1px solid #ddd; border-radius: 8px; }
    h1 { color: #1E3A8A; }
    .stAlert { border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("OpenAI API Key", type="password")
    st.info("Key is not stored. Used for this session only.")
    st.markdown("---")
    st.markdown("""
    ### 📊 Data Guide
    **For Real Analysis:**
    Go to Google Ads > Ads > Columns.
    Ensure you check:
    - **Clicks**
    - **Cost**
    - **Impressions**
    
    *If these are missing, the tool will run in Demo Mode.*
    """)

# ==========================================
# 2. DATA PROCESSING ENGINE
# ==========================================
def find_header_row(uploaded_file):
    """
    Scans the file line-by-line to find the header row.
    Ignores Google's 'Untitled Report' metadata.
    """
    uploaded_file.seek(0)
    for i in range(20):
        # Read raw bytes to avoid decoding errors on metadata
        line = uploaded_file.readline().decode('utf-8', errors='ignore').lower()
        
        # We look for a line that looks like a header (has 'ad state' or 'asset')
        if ('ad state' in line) or ('asset' in line and 'status' in line) or ('headline 1' in line):
            uploaded_file.seek(0)
            return i
    uploaded_file.seek(0)
    return 0

def clean_metric(val):
    if pd.isna(val) or str(val).strip() == '--': return 0.0
    s = str(val).replace('%', '').replace(',', '')
    s = re.sub(r'[^\d.-]', '', s)
    try:
        return float(s)
    except:
        return 0.0

def process_rsa_file(df):
    # 1. Normalize Columns
    df.columns = df.columns.astype(str).str.strip().str.lower()
    
    # 2. Map Columns (Universal Mapper)
    col_map = {
        'impr. (abs. top) %': 'Abs_Top',
        'ctr': 'CTR',
        'conv. rate': 'Conv_Rate',
        'cost / conv.': 'CPA',
        'avg. cpc': 'CPC',
        'clicks': 'Clicks',
        'cost': 'Cost',
        'impressions': 'Impressions',
        'conversions': 'Conversions',
        'conv. value': 'Conv_Value'
    }
    df = df.rename(columns=col_map)
    
    # 3. DEMO MODE BYPASS (The Fix for your current file)
    # Check if we are missing volume data
    missing_vol = []
    if 'Clicks' not in df.columns: missing_vol.append('Clicks')
    if 'Cost' not in df.columns: missing_vol.append('Cost')
    
    if len(missing_vol) >= 1:
        st.warning(f"⚠️ **Missing Volume Data ({', '.join(missing_vol)}):** Running in **DEMO MODE**. \n\nWe are simulating 100 clicks & $50 spend per ad so you can test the AI analysis. \n\n**To get real insights:** Please re-download your CSV from Google Ads with 'Clicks', 'Cost', and 'Impressions' columns enabled.")
        
        # Inject Dummy Data so the math works
        df['Clicks'] = 100.0
        df['Impressions'] = 1000.0
        df['Cost'] = 50.0 
    
    # 4. Fill other missing columns with 0.0
    for m in ['Cost', 'Impressions', 'CTR', 'Conversions', 'Conv_Value', 'Clicks']:
        if m not in df.columns:
            df[m] = 0.0
        else:
            df[m] = df[m].apply(clean_metric)

    # 5. MELT (Wide to Long)
    # Grab all Headline/Description columns (Headline 1...15)
    text_cols = [c for c in df.columns if ('headline' in c or 'description' in c) and 'position' not in c and 'image' not in c]
    
    # Keep metrics associated with the text
    id_vars = ['Cost', 'Impressions', 'Conversions', 'Conv_Value', 'Clicks', 'CTR', 'CPC']
    id_vars = [c for c in id_vars if c in df.columns]
    
    melted = df.melt(id_vars=id_vars, value_vars=text_cols, value_name='Cleaned_Text')
    
    # 6. Clean Text
    melted['Cleaned_Text'] = melted['Cleaned_Text'].astype(str).str.strip('"').str.strip()
    melted = melted[melted['Cleaned_Text'].str.len() > 2] 
    melted = melted[~melted['Cleaned_Text'].str.contains('--', na=False)] 
    
    return melted, None

def aggregate_data(df):
    grouped = df.groupby('Cleaned_Text').agg({
        'Impressions': 'sum',
        'Cost': 'sum',
        'Conversions': 'sum',
        'Conv_Value': 'sum',
        'Clicks': 'sum'
    }).reset_index()
    
    # KPI Calculation
    grouped['ROAS'] = np.where(grouped['Cost']>0, grouped['Conv_Value']/grouped['Cost'], 0)
    grouped['CPA'] = np.where(grouped['Conversions']>0, grouped['Cost']/grouped['Conversions'], 0)
    grouped['CTR'] = np.where(grouped['Impressions']>0, (grouped['Clicks']/grouped['Impressions'])*100, 0)
    
    # If no impressions exist (even after demo mode), use Clicks
    sort_metric = 'Impressions' if grouped['Impressions'].sum() > 0 else 'Clicks'
    grouped = grouped[grouped[sort_metric] > 0] 
    
    return grouped

# ==========================================
# 3. AI ENGINE
# ==========================================
def run_ai_analysis(df, key):
    client = openai.OpenAI(api_key=key)
    
    # Prepare Context (Sort by implied performance)
    # Since we might be in demo mode (equal cost), we also look at CTR if available
    top_ads = df.sort_values(by='ROAS', ascending=False).head(10).to_dict(orient='records')
    traffic_drivers = df.sort_values(by='Clicks', ascending=False).head(10).to_dict(orient='records')
    
    system_prompt = """
    You are AdRank AI, a Behavioral Science Marketing Strategist.
    
    ### THE CODEBOOK
    1. **Psychology:** Scarcity, Social Proof, Authority, Reciprocity, Loss Aversion.
    2. **Tone:** Urgent, Corporate, Playful, Empathetic, Direct.

    ### OUTPUT FORMAT
    1. **EXECUTIVE SUMMARY:** Diagnose the account health.
    2. **🏆 WINNING PATTERNS:** What words/tones correlate with high performance?
    3. **🚩 WASTED SPEND:** What high-traffic ads are failing to convert?
    4. **🚀 3 TEST IDEAS:** Write 3 new headlines based on the winning data.
    """
    
    user_data = f"""
    TOP PERFORMERS (ROAS): {json.dumps(top_ads)}
    HIGH TRAFFIC (Clicks): {json.dumps(traffic_drivers)}
    
    Note: If metrics look identical (e.g. all 100 clicks), the user is in DEMO MODE. 
    Focus your analysis purely on the TEXT SEMANTICS and best practices, ignoring the fake numbers.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_data}],
        temperature=0.7
    )
    return response.choices[0].message.content

# ==========================================
# 4. MAIN UI
# ==========================================
st.title("🧠 AdRank AI: RSA Analyzer")
st.write("Upload your **Ad Report** or **Asset Report**.")

uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

if uploaded_file is not None:
    # 1. LOAD
    header_idx = find_header_row(uploaded_file)
    try:
        # Use python engine for robustness
        raw_df = pd.read_csv(uploaded_file, skiprows=header_idx, engine='python')
    except:
        st.error("Failed to read file.")
        st.stop()

    # 2. PROCESS
    with st.spinner("Analyzing columns..."):
        clean_df, error = process_rsa_file(raw_df)
        
        if error:
            st.error(error)
        
        elif clean_df is not None:
            final_df = aggregate_data(clean_df)
            
            if final_df.empty:
                st.warning("⚠️ Data processed, but no valid text assets found.")
            else:
                st.success("✅ Analysis Ready!")
                
                # Metrics
                c1, c2, c3 = st.columns(3)
                c1.metric("Assets Analyzed", len(final_df))
                c2.metric("Total Spend", f"${final_df['Cost'].sum():,.0f}")
                c3.metric("Max ROAS", f"{final_df['ROAS'].max():.2f}")
                
                with st.expander("🔎 View Top Copy"):
                    st.dataframe(final_df.sort_values(by='Clicks', ascending=False).head(10))
                
                # AI
                st.markdown("### 🤖 Behavioral Analysis")
                if st.button("Generate Insights"):
                    if not api_key:
                        st.error("Please enter OpenAI API Key in the sidebar.")
                    else:
                        with st.spinner("Thinking..."):
                            try:
                                res = run_ai_analysis(final_df, api_key)
                                st.markdown("---")
                                st.markdown(res)
                            except Exception as e:
                                st.error(f"AI Error: {e}")
