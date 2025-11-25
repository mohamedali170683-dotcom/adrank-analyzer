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
    .main {
        background-color: #f9f9f9;
    }
    h1 {
        color: #1E3A8A;
    }
    .stAlert {
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar for API Key
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("OpenAI API Key", type="password", help="Required to generate the analysis.")
    st.info("Your key is not stored. It is used only for this session.")
    st.markdown("---")
    st.markdown("**Instructions:**\n1. Go to Google Ads > Assets\n2. Columns: Add 'Conv. Value/Cost' & 'Performance Label'\n3. Download as CSV")

# ==========================================
# 2. THE JANITOR (Data Cleaning Engine)
# ==========================================
def clean_and_aggregate_ads(df):
    """
    Cleans Google Ads CSVs, removes artifacts, and aggregates by Text.
    """
    try:
        # Standardize headers
        df.columns = df.columns.str.strip()
        
        # Check requirements
        if 'Asset' not in df.columns:
            st.error("❌ Error: Column 'Asset' not found. Please check your CSV.")
            return None

        # 1. Metric Cleaning Helper
        def clean_metric(val):
            if pd.isna(val): return 0.0
            s = str(val)
            # Remove %, commas, currency symbols
            s = re.sub(r'[^\d.-]', '', s)
            try:
                return float(s)
            except ValueError:
                return 0.0

        # 2. Apply cleaning to specific columns
        metric_cols = {
            'Impressions': 'Impressions',
            'Clicks': 'Clicks',
            'Cost': 'Cost',
            'Conv. value': 'Conv_Value',
            'Conversions': 'Conversions'
        }
        
        for raw_col, new_col in metric_cols.items():
            found_col = next((c for c in df.columns if raw_col.lower() in c.lower()), None)
            if found_col:
                df[new_col] = df[found_col].apply(clean_metric)
            else:
                df[new_col] = 0.0

        # 3. Filter for TEXT assets only (Headlines/Descriptions)
        # We assume 'Asset' column contains the text. 
        # Remove rows that are just numbers or empty
        df['Cleaned_Text'] = df['Asset'].astype(str).str.strip().str.strip('"').str.strip("'")
        df = df[df['Cleaned_Text'].str.len() > 3] # Remove tiny artifacts
        df = df[~df['Cleaned_Text'].str.match(r'^[\d\-]+$')] # Remove pure numbers/hyphens

        # 4. Global Aggregation (The "Simpson's Paradox" Fix)
        # Group by the actual text to see global performance
        grouped = df.groupby(['Cleaned_Text']).agg({
            'Impressions': 'sum',
            'Clicks': 'sum',
            'Cost': 'sum',
            'Conv_Value': 'sum',
            'Conversions': 'sum'
        }).reset_index()

        # 5. Calculate KPIs
        grouped['CTR'] = np.where(grouped['Impressions']>0, (grouped['Clicks'] / grouped['Impressions']) * 100, 0)
        grouped['ROAS'] = np.where(grouped['Cost']>0, grouped['Conv_Value'] / grouped['Cost'], 0)
        grouped['CPA'] = np.where(grouped['Conversions']>0, grouped['Cost'] / grouped['Conversions'], 0)
        
        # Rounding
        grouped = grouped.round(2)
        
        # Filter insignificant data (optional threshold)
        grouped = grouped[grouped['Impressions'] > 50]

        return grouped

    except Exception as e:
        st.error(f"Data Processing Error: {e}")
        return None

# ==========================================
# 3. THE BRAIN (LLM Analysis)
# ==========================================
def run_ai_analysis(df, key):
    client = openai.OpenAI(api_key=key)
    
    # Segment Data for the LLM
    top_performers = df.sort_values(by='ROAS', ascending=False).head(15).to_dict(orient='records')
    click_magnets = df.sort_values(by='CTR', ascending=False).head(8).to_dict(orient='records')
    wasters = df[df['Cost'] > df['Cost'].mean()].sort_values(by='ROAS', ascending=True).head(8).to_dict(orient='records')

    # Calculate Account Health for Context
    total_spend = df['Cost'].sum()
    total_conv = df['Conversions'].sum()
    data_reliability = "HIGH" if total_conv > 50 else "LOW (Caution: Low Sample Size)"

    # The Prompt Construction
    system_prompt = """
    You are AdRank AI, a PhD-level Behavioral Scientist and Direct Response Copywriter.
    
    ### THE CODEBOOK (Use these tags for analysis)
    1. **Psychological Levers:** Scarcity, Social Proof, Authority, Loss Aversion, Reciprocity, Curiosity Gap.
    2. **Tone:** Urgent, Professional, Playful, Direct, Empathetic.
    3. **Structure:** Question-based, Listicle, "How-to", Negative Framing (Stop doing X).

    ### YOUR OUTPUT STRUCTURE
    1. **Executive Summary:** One bold sentence diagnosing the account's biggest opportunity.
    2. **🏆 The Winning Formula:** Analyze the 'Top Performers'. What specific tone/words correlate with high ROAS?
    3. **🚩 The Clickbait Trap:** Analyze 'Click Magnets'. Why do these get clicks but no sales? (e.g. "Misleading 'Free' offer").
    4. **📉 The Money Pit:** Analyze 'Wasters'. Identify patterns in expensive/failing ads.
    5. **🚀 3 Actionable Tests:** Write 3 NEW headlines to test based on the Winning Formula. Explain the psychology behind each.

    **Tone:** Professional, direct, analytical. No fluff.
    """

    user_content = f"""
    DATA CONTEXT:
    - Data Reliability: {data_reliability}
    - Total Spend Analyzed: ${total_spend}
    
    TOP PERFORMERS (High ROAS):
    {json.dumps(top_performers)}

    CLICK MAGNETS (High CTR, check for low ROAS):
    {json.dumps(click_magnets)}

    WASTERS (High Spend, Low ROAS):
    {json.dumps(wasters)}
    """

    response = client.chat.completions.create(
        model="gpt-4o", # Recommended model
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        temperature=0.7
    )
    
    return response.choices[0].message.content

# ==========================================
# 4. MAIN APP FLOW
# ==========================================
st.title("🧠 AdRank AI: Behavioral Copy Analyzer")
st.markdown("Upload your Google Ads export to uncover the *psychology* behind your performance.")

uploaded_file = st.file_uploader("Upload CSV (Ads & Assets Report)", type=['csv'])

if uploaded_file is not None:
    # 1. Load Data (Robust Method)
    try:
        # Google Ads CSVs often have 2 lines of metadata at top. Try skipping them.
        raw_df = pd.read_csv(uploaded_file, skiprows=2)
        
        # Check if 'Asset' column exists. If not, maybe the user deleted the top rows already.
        # So we try reading normally without skipping.
        if 'Asset' not in raw_df.columns:
            uploaded_file.seek(0) # Reset file pointer
            raw_df = pd.read_csv(uploaded_file)
            
    except pd.errors.ParserError:
        # If standard parsing fails, try a different engine (useful for some encodings)
        uploaded_file.seek(0)
        raw_df = pd.read_csv(uploaded_file, skiprows=2, engine='python')
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()
    
    # 2. Process Data
    with st.spinner("Cleaning data and aggregating duplicate assets..."):
        clean_df = clean_and_aggregate_ads(raw_df)
    
    if clean_df is not None and not clean_df.empty:
        # Show Stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Unique Assets Analyzed", len(clean_df))
        col2.metric("Total Spend", f"${clean_df['Cost'].sum():,.2f}")
        col3.metric("Avg ROAS", f"{clean_df['ROAS'].mean():.2f}")

        with st.expander("🔎 View Cleaned Data (Top 10 by Spend)"):
            st.dataframe(clean_df.sort_values(by='Cost', ascending=False).head(10))

        # 3. Analyze Button
        st.markdown("### 🤖 Generate Behavioral Report")
        if st.button("Run AI Analysis"):
            if not api_key:
                st.warning("⚠️ Please enter your OpenAI API Key in the sidebar.")
            else:
                with st.spinner("Analyzing semantic patterns & psychological triggers..."):
                    try:
                        analysis_result = run_ai_analysis(clean_df, api_key)
                        st.success("Analysis Complete!")
                        st.markdown("---")
                        st.markdown(analysis_result)
                    except Exception as e:
                        st.error(f"AI Error: {e}")
    else:

        st.warning("No valid text assets found. Ensure your CSV has 'Asset', 'Impressions', and 'Cost' columns.")
