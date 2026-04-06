import pandas as pd
import numpy as np
import os
from datetime import timedelta

def clean_and_engineer_features(
    game_name: str,
    base_df: pd.DataFrame, 
    trend_df: pd.DataFrame, 
    twitch_df: pd.DataFrame, 
    update_df: pd.DataFrame
) -> pd.DataFrame:
    
    # 1. Base Data Setup
    df = base_df.copy()
    # Ép kiểu ngày tháng với dayfirst=True để xử lý định dạng VN/Châu Âu (17-12-2021)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce').dt.normalize()
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)

    # 2. Google Trends
    trend_work = trend_df.copy()
    t_date_col = 'Date' if 'Date' in trend_work.columns else ('Month' if 'Month' in trend_work.columns else trend_work.columns[0])
    trend_work['Date'] = pd.to_datetime(trend_work[t_date_col], dayfirst=True, errors='coerce').dt.normalize()
    trend_work.set_index('Date', inplace=True)
    
    if 'Trend_Value' not in trend_work.columns:
        num_cols = trend_work.select_dtypes(include='number').columns
        if len(num_cols) > 0:
            trend_work.rename(columns={num_cols[0]: 'Trend_Value'}, inplace=True)
            
    df = df.merge(trend_work[['Trend_Value']], left_index=True, right_index=True, how='left')
    df['Trend_Value'] = df['Trend_Value'].ffill().bfill()
    df['Trend_Index'] = df['Trend_Value'] / 100.0

# ==========================================
    # 3. Twitch Data Processing (Robust Fix)
    # ==========================================
    tw_work = twitch_df.copy()
    if 'Date' in tw_work.columns:
        tw_work['Date'] = pd.to_datetime(tw_work['Date'], dayfirst=True, errors='coerce').dt.normalize()
        tw_work.set_index('Date', inplace=True)
        
    # Merge lấy Avg
    df = df.merge(tw_work[['Twitch_Avg_Viewers']], left_index=True, right_index=True, how='left')
    
    # --- XỬ LÝ ROBUST: COI 0 CŨNG LÀ NaN ĐỂ LẤP ĐẦY ---
    # Vì với game lớn, 0 viewers chắc chắn là lỗi tracking
    df.loc[df['Twitch_Avg_Viewers'] == 0, 'Twitch_Avg_Viewers'] = np.nan
    
    # B1: Nội suy các lỗ hổng ở giữa (Gaps)
    df['Twitch_Avg_Viewers'] = df['Twitch_Avg_Viewers'].interpolate(method='linear')
    
    # B2: Cực kỳ quan trọng - Forward Fill (ffill) cho đoạn cuối bị cụt
    # Nó sẽ lấy giá trị ngày 2026-03-04 (9.09) đắp cho cả tuần tiếp theo
    df['Twitch_Avg_Viewers'] = df['Twitch_Avg_Viewers'].ffill()
    
    # B3: Backward Fill (bfill) cho đoạn đầu bị thiếu
    df['Twitch_Avg_Viewers'] = df['Twitch_Avg_Viewers'].bfill()
    
    # B4: Nếu cuối cùng vẫn còn (trường hợp cực hiếm) thì mới để 0
    df['Twitch_Avg_Viewers'] = df['Twitch_Avg_Viewers'].fillna(0)
    
    # Log transformation (ln(x+1))
    df['Log_Twitch_Avg'] = np.log1p(df['Twitch_Avg_Viewers'].astype(float))
    df.drop(columns=['Twitch_Avg_Viewers'], inplace=True, errors='ignore')
    
    # 4. Game Updates
    df['Is_Major_Update'] = 0
    df['Is_Minor_Update'] = 0
    upd_work = update_df.copy()
    upd_work['Date'] = pd.to_datetime(upd_work['Date'], dayfirst=True, errors='coerce').dt.normalize()
    
    for _, row in upd_work.iterrows():
        u_date = row['Date']
        if pd.isna(u_date): continue
        sig = str(row.get('Significance', '')).strip().upper()
        if sig in ['VERY HIGH', 'HIGH']:
            end_d = u_date + timedelta(days=7)
            df.loc[(df.index >= u_date) & (df.index <= end_d), 'Is_Major_Update'] = 1
        elif sig == 'MEDIUM':
            end_d = u_date + timedelta(days=3)
            df.loc[(df.index >= u_date) & (df.index <= end_d), 'Is_Minor_Update'] = 1

    # 5. Final Scaling
    if 'Discount_%' in df.columns:
        df['Discount_Ratio'] = df['Discount_%'] / 100.0
    if 'Days_Since_Release' in df.columns:
        df['Years_Since_Release'] = df['Days_Since_Release'] / 365.0
    if 'Avg_Player' in df.columns:
        df['Log_Player'] = np.log1p(df['Avg_Player'].astype(float))
        df['Lag_Player'] = df['Log_Player'].shift(1)

    # 6. Export
    df = df.dropna(subset=['Lag_Player'])
    cols_to_keep = [
        'Avg_Player', 'Log_Player', 'Lag_Player', 'Discount_Ratio', 'Base_Price', 
        'Is_Weekend', 'Is_Holiday', 'Years_Since_Release', 'Trend_Index', 
        'Log_Twitch_Avg', 'Is_Major_Update', 'Is_Minor_Update'
    ]
    return df[[c for c in cols_to_keep if c in df.columns]].copy()

def main():
    RAW_DIR = "/home/honganh/OOD/ts/time_series_project/data/raw/"
    PROCESSED_DIR = "/home/honganh/OOD/ts/time_series_project/data/processed/"
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    games = {
        "EldenRing": {
            "base": "EldenRing_Cleaned.csv", "trend": "EldenRing_trend.csv",
            "upd": "eldenring_upd.csv", "twitch": "elden_ring_twitch.csv"
        },
        "ReadyOrNot": {
            "base": "ReadyorNot_Cleaned.csv", "trend": "ReadyOrNot_trend.csv",
            "upd": "readyupdate.csv", "twitch": "ready_or_not_twitch.csv"
        }
    }

    for game, files in games.items():
        print(f"--- Processing {game} ---")
        try:
            b = pd.read_csv(os.path.join(RAW_DIR, files['base']))
            t = pd.read_csv(os.path.join(RAW_DIR, files['trend']))
            u = pd.read_csv(os.path.join(RAW_DIR, files['upd']))
            tw_path = os.path.join(RAW_DIR, files['twitch'])
            tw = pd.read_csv(tw_path)
            
            final_df = clean_and_engineer_features(game, b, t, tw, u)
            out_path = os.path.join(PROCESSED_DIR, f"{game}_Final_Merged.csv")
            final_df.to_csv(out_path)
            print(f"SUCCESS: Saved {len(final_df)} rows to {out_path}\n")
        except Exception as e:
            print(f"FAILED at {game}: {e}\n")

if __name__ == "__main__":
    main()