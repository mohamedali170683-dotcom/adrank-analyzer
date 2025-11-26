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
    
    # 3. DEMO MODE BYPASS (The Fix)
    # Check if we are missing volume data
    missing_vol = []
    if 'Clicks' not in df.columns: missing_vol.append('Clicks')
    if 'Cost' not in df.columns: missing_vol.append('Cost')
    
    if len(missing_vol) >= 2:
        st.warning(f"⚠️ **Missing Volume Data ({', '.join(missing_vol)}):** Running in **DEMO MODE**. \n\nWe are simulating 100 clicks per ad so you can see the AI analysis. For real results, please re-download your CSV with 'Clicks' & 'Cost' columns added.")
        
        # Inject Dummy Data so the math works
        df['Clicks'] = 100.0
        df['Impressions'] = 1000.0
        df['Cost'] = 50.0 # Arbitrary $50 spend per ad
    
    # 4. Fill other missing columns with 0.0
    for m in ['Cost', 'Impressions', 'CTR', 'Conversions', 'Conv_Value', 'Clicks']:
        if m not in df.columns:
            df[m] = 0.0
        else:
            df[m] = df[m].apply(clean_metric)

    # 5. MELT (Wide to Long)
    text_cols = [c for c in df.columns if ('headline' in c or 'description' in c) and 'position' not in c and 'image' not in c]
    
    id_vars = ['Cost', 'Impressions', 'Conversions', 'Conv_Value', 'Clicks', 'CTR', 'CPC']
    id_vars = [c for c in id_vars if c in df.columns]
    
    melted = df.melt(id_vars=id_vars, value_vars=text_cols, value_name='Cleaned_Text')
    
    # 6. Clean Text
    melted['Cleaned_Text'] = melted['Cleaned_Text'].astype(str).str.strip('"').str.strip()
    melted = melted[melted['Cleaned_Text'].str.len() > 2] 
    melted = melted[~melted['Cleaned_Text'].str.contains('--', na=False)] 
    
    return melted, None
