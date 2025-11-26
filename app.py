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
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("OpenAI API Key", type="password")
    st.info("Key is not stored. Used for this session only.")
    st.markdown("---")
    st.warning("⚠️ **Requirement:** Your CSV must contain **'Impressions'** and **'Cost'**. If missing, the tool cannot calculate performance.")

# ==========================================
# 2. ROBUST FILE LOADER (The Fix)
# ==========================================
def find_header_row(uploaded_file):
    """
    Reads the file line-by-line (as raw text) to find the header.
    This prevents Pandas from crashing on column count mismatches.
    """
    uploaded_file.seek(0)
    
    current_pos = 0
    for i in range(20): # Scan first 20 lines
        line_bytes = uploaded_file.readline()
        
        # Decode bytes to string
        try:
            line_str = line_bytes.decode('utf-8').lower()
        except:
            try:
                line_str = line_bytes.decode('latin-1').lower() # Fallback encoding
            except:
                continue # Skip unreadable lines
        
        # Check for header signatures
        if 'ad state' in line_str and 'headline' in line_str:
            uploaded_file.seek(0)
            return i
        if 'asset' in line_str and 'status' in line_str:
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
    """
    Logic for RSA "Wide" Files (Headline 1, Headline 2...)
    Melts them into "Long" format so we can analyze specific text.
    """
    # 1. Map user's columns to standard metrics
    df.columns = df.columns.astype(str).str.strip().str.lower()
    
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
    
    # 2. Handle Missing Metrics (Graceful Fallback)
    if 'Cost' not in df.columns:
        if 'Clicks' in df.columns and 'CPC' in df.columns:
            df['Cost'] = df['Clicks'] * df['CPC'] # Infer cost
        else:
            df['Cost'] = 0.0
            
    if 'Impressions' not in df.columns:
        if 'Clicks' in df.columns and 'CTR' in df.columns:
             # Reverse engineer impressions: Clicks / (CTR/100)
             # Avoid div by zero
             df['Impressions'] = df.apply(lambda x: x['Clicks'] / (x['CTR']/100) if x['CTR'] > 0 else 0, axis=1)
        else:
             df['Impressions'] = df['Clicks'] if 'Clicks' in df.columns else 0.0

    # Clean Metrics
    for m in ['Cost', 'Impressions', 'CTR', 'Conversions', 'Conv_Value', 'Clicks']:
        if m in df.columns:
            df[m] = df[m].apply(clean_metric)
        else:
            df[m] = 0.0

    # 3. MELT (The Magic Step)
    text_cols = [c for c in df.columns if ('headline' in c or 'description' in c) and 'position' not in c and 'image' not in c]
    
    id_vars = ['Cost', 'Impressions', 'Conversions', 'Conv_Value', 'Clicks']
    id_vars = [c for c in id_vars if c in df.columns]
    
    melted = df.melt(id_vars=id_vars, value_vars=text_cols, value_name='Cleaned_Text')
    
    # 4. Clean Text
    melted['Cleaned_Text'] = melted['Cleaned_Text'].astype(str).str.strip('"').str.strip()
    melted = melted[melted['Cleaned_Text'].str.len() > 2] # Remove empty
    melted = melted[~melted['Cleaned_Text'].str.contains('--', na=False)] 
    
    return melted

def aggregate_data(df):
    grouped = df.groupby('Cleaned_Text').agg({
        'Impressions': 'sum',
        'Cost': 'sum',
        'Conversions': 'sum',
        'Conv_Value': 'sum',
        'Clicks': 'sum'
    }).reset_index()
    
    grouped['ROAS'] = np.where(grouped['Cost']>0, grouped['Conv_Value']/grouped['Cost'], 0)
    grouped['CPA'] = np.where(grouped['Conversions']>0, grouped['Cost']/grouped['Conversions'], 0)
    grouped['CTR'] = np.where(grouped['Impressions']>0, (grouped['Clicks']/grouped['Impressions'])*100, 0)
    
    return grouped

# ==========================================
# 3. AI ANALYSIS ENGINE
# ==========================================
def run_ai_analysis(df, key):
    client = openai.OpenAI(api_key=key)
    
    top_ads = df.sort_values(by='ROAS', ascending=False).head(10).to_dict(orient='records')
    click_magnets = df.sort_values(by='Clicks', ascending=False).head(10).to_dict(orient='records')
    
    system_prompt = """
    You are AdRank AI, a Behavioral Science Marketing Strategist.
    
    ### THE CODEBOOK
    1. **Psychology:** Scarcity, Social Proof, Authority, Reciprocity, Loss Aversion.
    2. **Tone:** Urgent, Corporate, Playful, Empathetic, Direct.

    ### OUTPUT FORMAT
    1. **EXECUTIVE DIAGNOSIS:** One bold sentence on the account's biggest opportunity.
    2. **🏆 THE WINNING FORMULA:** Analyze 'Top Performers'. What specific tone/words correlate with high ROAS?
    3. **🚩 THE CLICKBAIT TRAP:** Analyze 'Click Magnets'. Why do these get traffic but less efficiency?
    4. **🚀 3 HYPOTHESIS TESTS:** Write 3 NEW headlines to test based on the Winning Formula.
    """

    user_data = f"""
    TOP PERFORMERS (High Efficiency):
    {json.dumps(top_ads)}

    HIGH TRAFFIC ASSETS (Volume Drivers):
    {json.dumps(click_magnets)}
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
# 4. MAIN APP FLOW
# ==========================================
st.title("🧠 AdRank AI: RSA & Asset Analyzer")
st.write("Upload your Google Ads **Ad Report** or **Asset Report**.")

uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

if uploaded_file is not None:
    # 1. ROBUST LOAD
    header_idx = find_header_row(uploaded_file)
    
    # Use python engine which is slower but much more robust to "Expected 1 fields saw 171" errors
    try:
        raw_df = pd.read_csv(uploaded_file, skiprows=header_idx, engine='python')
    except Exception as e:
        st.error(f"Failed to read CSV. Try opening it in Excel and saving as a fresh CSV. Error: {e}")
        st.stop()

    # 2. Detect & Transform
    with st.spinner("Processing file structure..."):
        try:
            clean_df = process_rsa_file(raw_df)
            
            if clean_df is not None and not clean_df.empty:
                final_df = aggregate_data(clean_df)
                
                # Show Dashboard
                st.success("✅ File Processed Successfully!")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Unique Copy Assets", len(final_df))
                c2.metric("Total Spend", f"${final_df['Cost'].sum():,.0f}")
                c3.metric("Top Asset ROAS", f"{final_df['ROAS'].max():.2f}")
                
                with st.expander("🔎 View Top Performing Copy"):
                    st.dataframe(final_df.sort_values(by='Cost', ascending=False).head(10))
                
                st.markdown("### 🤖 Behavioral Analysis")
                if st.button("Generate Insights"):
                    if not api_key:
                        st.error("Please enter OpenAI API Key.")
                    else:
                        with st.spinner("Analyzing semantic patterns..."):
                            try:
                                insights = run_ai_analysis(final_df, api_key)
                                st.success("Analysis Complete!")
                                st.markdown("---")
                                st.markdown(insights)
                            except Exception as e:
                                st.error(f"OpenAI Error: {e}")
            else:
                st.error("Could not find valid text assets. Please check your CSV.")
                
        except Exception as e:
             st.error(f"Processing Error: {e}")
             with st.expander("Debug Info"):
                 st.write(raw_df.head())
