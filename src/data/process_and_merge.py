import pandas as pd
import numpy as np
import os
from datetime import timedelta

def clean_and_engineer_features(
    base_df: pd.DataFrame, 
    trend_df: pd.DataFrame, 
    twitch_df: pd.DataFrame, 
    update_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Process, merge, clean, and engineer features for a specific game's time series analysis.
    """
    
    # ==========================================
    # 1. Base Data Setup
    # ==========================================
    # Ensure the base dataframe has a daily Date index
    base_df['Date'] = pd.to_datetime(base_df['Date'])
    base_df.set_index('Date', inplace=True)
    # Sort index just in case the raw data is out of order
    base_df.sort_index(inplace=True)
    
    df = base_df.copy()

    # ==========================================
    # 2. Google Trends (Monthly to Daily via FFill)
    # ==========================================
    # Standardize trend date column name (often 'Time', 'Month' or 'Date')
    if 'Date' in trend_df.columns:
        trend_date_col = 'Date'
    elif 'Month' in trend_df.columns:
        trend_date_col = 'Month'
    elif 'Time' in trend_df.columns:
        trend_date_col = 'Time'
    else:
        trend_date_col = trend_df.columns[0]
        
    trend_df['Date'] = pd.to_datetime(trend_df[trend_date_col])
    trend_df.set_index('Date', inplace=True)
    
    # Identify the trend value column (assume 'Trend_Value' or similar)
    # If not explicitly named, take the first numeric column
    if 'Trend_Value' not in trend_df.columns:
        num_cols = trend_df.select_dtypes(include='number').columns
        if len(num_cols) > 0:
            trend_df.rename(columns={num_cols[0]: 'Trend_Value'}, inplace=True)
            
    # Merge the monthly Trend data onto the daily base data using a left join
    df = df.merge(trend_df[['Trend_Value']], left_index=True, right_index=True, how='left')
    
    # Apply forward fill (ffill()) to populate daily rows with the monthly trend value
    df['Trend_Value'] = df['Trend_Value'].ffill()
    # Backfill in case the beginning days don't have preceding trend values
    df['Trend_Value'] = df['Trend_Value'].bfill()
    
    # Normalize: Create Trend_Index
    df['Trend_Index'] = df['Trend_Value'] / 100.0

    # ==========================================
    # 3. Twitch Data Processing
    # ==========================================
    # Standardize twitch date column and index
    if 'Date' in twitch_df.columns:
        twitch_df['Date'] = pd.to_datetime(twitch_df['Date'])
        twitch_df.set_index('Date', inplace=True)
        
    # Merge Twitch data
    df = df.merge(
        twitch_df[['Twitch_Avg_Viewers', 'Twitch_Peak_Viewers']], 
        left_index=True, 
        right_index=True, 
        how='left'
    )
    
    # Fill initial NaN values with 0
    df['Twitch_Avg_Viewers'] = df['Twitch_Avg_Viewers'].fillna(0)
    
    # Drop Twitch_Peak_Viewers to avoid multicollinearity
    if 'Twitch_Peak_Viewers' in df.columns:
        df.drop(columns=['Twitch_Peak_Viewers'], inplace=True)
        
    # Apply log transformation
    # Using np.log1p for log(1 + x) to handle zeroes safely, ensuring float type
    df['Log_Twitch_Avg'] = np.log1p(df['Twitch_Avg_Viewers'].astype(float))

    # ==========================================
    # 4. Game Updates (The 'Halo Effect' Window)
    # ==========================================
    # Initialize update columns
    df['Is_Major_Update'] = 0
    df['Is_Minor_Update'] = 0
    
    update_df['Date'] = pd.to_datetime(update_df['Date'])
    
    # Iterate through update data to mark valid windows
    for _, row in update_df.iterrows():
        update_date = row['Date']
        significance = str(row.get('Significance', '')).strip().upper()
        
        if significance in ['VERY HIGH', 'HIGH']:
            # Active for the update Date AND the next 7 consecutive days
            end_date = update_date + timedelta(days=7)
            mask = (df.index >= update_date) & (df.index <= end_date)
            df.loc[mask, 'Is_Major_Update'] = 1
            
        elif significance == 'MEDIUM':
            # Active for the update Date AND the next 3 consecutive days
            end_date = update_date + timedelta(days=3)
            mask = (df.index >= update_date) & (df.index <= end_date)
            df.loc[mask, 'Is_Minor_Update'] = 1

    # ==========================================
    # 5. Final Feature Scaling (For Robust OLS Model)
    # ==========================================
    # Safely create derived features if their foundational columns exist
    if 'Discount_%' in df.columns:
        df['Discount_Ratio'] = df['Discount_%'] / 100.0
    else:
        df['Discount_Ratio'] = np.nan
        
    if 'Days_Since_Release' in df.columns:
        df['Years_Since_Release'] = df['Days_Since_Release'] / 365.0
    else:
        df['Years_Since_Release'] = np.nan
        
    if 'Avg_Player' in df.columns:
        # Cast to float to avoid numpy object array errors
        df['Log_Player'] = np.log1p(df['Avg_Player'].astype(float))
        df['Lag_Player'] = df['Log_Player'].shift(1)
    else:
        df['Log_Player'] = np.nan
        df['Lag_Player'] = np.nan

    # ==========================================
    # 6. Export
    # ==========================================
    # Drop rows with NaN created by the .shift(1) operation
    df = df.dropna(subset=['Lag_Player'])
    
    # Define exact necessary columns
    target_columns = [
        'Avg_Player', 'Log_Player', 'Lag_Player', 'Discount_Ratio', 'Base_Price', 
        'Is_Weekend', 'Is_Holiday', 'Years_Since_Release', 'Trend_Index', 
        'Log_Twitch_Avg', 'Is_Major_Update', 'Is_Minor_Update'
    ]
    
    # Keep only those columns that exist in the dataframe
    final_columns = [col for col in target_columns if col in df.columns]
    df_final = df[final_columns].copy()
    
    return df_final


def main():
    # ---------------------------------------------------------
    # Paths Definition
    # ---------------------------------------------------------
    RAW_DIR = "/home/honganh/OOD/ts/time_series_project/data/raw/"
    PROCESSED_DIR = "/home/honganh/OOD/ts/time_series_project/data/processed/"
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # ==========================================
    # Pipeline: Elden Ring
    # ==========================================
    print("Processing Elden Ring Data...")
    try:
        er_base = pd.read_csv(os.path.join(RAW_DIR, "EldenRing_Cleaned.csv"))
        er_trend = pd.read_csv(os.path.join(RAW_DIR, "EldenRing_trend.csv"))
        er_update = pd.read_csv(os.path.join(RAW_DIR, "eldenring_upd.csv"))
        
        # Note: Twitch data assumed available per requirements
        # If it doesn't exist under this name, an empty dataframe proxy is used.
        er_twitch_path = os.path.join(RAW_DIR, "EldenRing_twitch.csv")
        er_twitch = pd.read_csv(er_twitch_path) if os.path.exists(er_twitch_path) else pd.DataFrame(columns=['Date', 'Twitch_Avg_Viewers', 'Twitch_Peak_Viewers'])
        
        er_final = clean_and_engineer_features(er_base, er_trend, er_twitch, er_update)
        er_export_path = os.path.join(PROCESSED_DIR, "EldenRing_Final_Merged.csv")
        er_final.to_csv(er_export_path)
        print(f" -> Successfully saved: {er_export_path}")
    except Exception as e:
        print(f" -> Error processing Elden Ring: {e}")

    # ==========================================
    # Pipeline: Ready Or Not
    # ==========================================
    print("\nProcessing Ready Or Not Data...")
    try:
        ron_base = pd.read_csv(os.path.join(RAW_DIR, "ReadyorNot_Cleaned.csv"))
        ron_trend = pd.read_csv(os.path.join(RAW_DIR, "ReadyOrNot_trend.csv"))
        ron_update = pd.read_csv(os.path.join(RAW_DIR, "readyupdate.csv"))
        
        # Note: Twitch data assumed available per requirements 
        ron_twitch_path = os.path.join(RAW_DIR, "ReadyOrNot_twitch.csv")
        ron_twitch = pd.read_csv(ron_twitch_path) if os.path.exists(ron_twitch_path) else pd.DataFrame(columns=['Date', 'Twitch_Avg_Viewers', 'Twitch_Peak_Viewers'])
        
        ron_final = clean_and_engineer_features(ron_base, ron_trend, ron_twitch, ron_update)
        ron_export_path = os.path.join(PROCESSED_DIR, "ReadyOrNot_Final_Merged.csv")
        ron_final.to_csv(ron_export_path)
        print(f" -> Successfully saved: {ron_export_path}")
    except Exception as e:
        print(f" -> Error processing Ready Or Not: {e}")

if __name__ == "__main__":
    main()
