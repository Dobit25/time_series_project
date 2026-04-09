# Cell 1: Setup & Data Loading


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Set a professional seaborn style
sns.set_theme(style='whitegrid')

# File paths
er_path = '/home/honganh/OOD/ts/time_series_project/data/processed/EldenRing_Final_Merged.csv'
ron_path = '/home/honganh/OOD/ts/time_series_project/data/processed/ReadyOrNot_Final_Merged.csv'

# Load both CSVs, parse Date as datetime, and set it as the index
games = {
    'Elden Ring': pd.read_csv(er_path, parse_dates=['Date'], index_col='Date'),
    'Ready Or Not': pd.read_csv(ron_path, parse_dates=['Date'], index_col='Date')
}

print('Data Loading Complete!')

```

    Data Loading Complete!
    

# Cell 2: Data Quality & Summary Statistics


```python
for game_name, df in games.items():
    print(f"\n{'='*40}")
    print(f"Data Quality & Summary for: {game_name}")
    print(f"{'='*40}")
    
    print("--- Missing Values (isnull().sum()) ---")
    print(df.isnull().sum())
    print("\n--- Summary Statistics (describe()) ---")
    display(df.describe())

```

    
    ========================================
    Data Quality & Summary for: Elden Ring
    ========================================
    --- Missing Values (isnull().sum()) ---
    Avg_Player             0
    Log_Player             0
    Lag_Player             0
    Discount_Ratio         0
    Base_Price             0
    Is_Weekend             0
    Is_Holiday             0
    Years_Since_Release    0
    Trend_Index            0
    Log_Twitch_Avg         0
    Is_Major_Update        0
    Is_Minor_Update        0
    dtype: int64
    
    --- Summary Statistics (describe()) ---
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Avg_Player</th>
      <th>Log_Player</th>
      <th>Lag_Player</th>
      <th>Discount_Ratio</th>
      <th>Base_Price</th>
      <th>Is_Weekend</th>
      <th>Is_Holiday</th>
      <th>Years_Since_Release</th>
      <th>Trend_Index</th>
      <th>Log_Twitch_Avg</th>
      <th>Is_Major_Update</th>
      <th>Is_Minor_Update</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1496.000000</td>
      <td>1496.000000</td>
      <td>1496.000000</td>
      <td>1496.000000</td>
      <td>1.496000e+03</td>
      <td>1496.000000</td>
      <td>1496.000000</td>
      <td>1496.000000</td>
      <td>1496.000000</td>
      <td>1496.000000</td>
      <td>1496.000000</td>
      <td>1496.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>77955.512032</td>
      <td>10.903741</td>
      <td>10.905471</td>
      <td>0.031964</td>
      <td>5.999000e+01</td>
      <td>0.286096</td>
      <td>0.032086</td>
      <td>2.050685</td>
      <td>0.147567</td>
      <td>8.914903</td>
      <td>0.040775</td>
      <td>0.008021</td>
    </tr>
    <tr>
      <th>std</th>
      <td>117676.172873</td>
      <td>0.665998</td>
      <td>0.668686</td>
      <td>0.101521</td>
      <td>7.107803e-15</td>
      <td>0.452086</td>
      <td>0.176286</td>
      <td>1.183568</td>
      <td>0.036947</td>
      <td>0.981889</td>
      <td>0.197836</td>
      <td>0.089232</td>
    </tr>
    <tr>
      <th>min</th>
      <td>23082.000000</td>
      <td>10.046852</td>
      <td>10.046852</td>
      <td>0.000000</td>
      <td>5.999000e+01</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.002740</td>
      <td>0.100000</td>
      <td>3.465736</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>35815.000000</td>
      <td>10.486150</td>
      <td>10.486150</td>
      <td>0.000000</td>
      <td>5.999000e+01</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.026712</td>
      <td>0.120000</td>
      <td>8.623173</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>47350.500000</td>
      <td>10.765353</td>
      <td>10.767063</td>
      <td>0.000000</td>
      <td>5.999000e+01</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.050685</td>
      <td>0.120000</td>
      <td>8.961257</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>65198.000000</td>
      <td>11.085199</td>
      <td>11.086866</td>
      <td>0.000000</td>
      <td>5.999000e+01</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>3.074658</td>
      <td>0.160000</td>
      <td>9.308997</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>953426.000000</td>
      <td>13.767818</td>
      <td>13.767818</td>
      <td>0.400067</td>
      <td>5.999000e+01</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>4.098630</td>
      <td>0.680000</td>
      <td>12.624796</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>


    
    ========================================
    Data Quality & Summary for: Ready Or Not
    ========================================
    --- Missing Values (isnull().sum()) ---
    Avg_Player             0
    Log_Player             0
    Lag_Player             0
    Discount_Ratio         0
    Base_Price             0
    Is_Weekend             0
    Is_Holiday             0
    Years_Since_Release    0
    Trend_Index            0
    Log_Twitch_Avg         0
    Is_Major_Update        0
    Is_Minor_Update        0
    dtype: int64
    
    --- Summary Statistics (describe()) ---
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Avg_Player</th>
      <th>Log_Player</th>
      <th>Lag_Player</th>
      <th>Discount_Ratio</th>
      <th>Base_Price</th>
      <th>Is_Weekend</th>
      <th>Is_Holiday</th>
      <th>Years_Since_Release</th>
      <th>Trend_Index</th>
      <th>Log_Twitch_Avg</th>
      <th>Is_Major_Update</th>
      <th>Is_Minor_Update</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1564.000000</td>
      <td>1564.000000</td>
      <td>1564.000000</td>
      <td>1564.000000</td>
      <td>1564.000000</td>
      <td>1564.000000</td>
      <td>1564.000000</td>
      <td>1564.000000</td>
      <td>1564.000000</td>
      <td>1564.000000</td>
      <td>1564.000000</td>
      <td>1564.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>7842.713555</td>
      <td>8.770761</td>
      <td>8.770493</td>
      <td>0.059482</td>
      <td>43.509182</td>
      <td>0.285806</td>
      <td>0.034527</td>
      <td>2.143836</td>
      <td>0.137807</td>
      <td>5.256815</td>
      <td>0.029412</td>
      <td>0.002558</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6290.188045</td>
      <td>0.584052</td>
      <td>0.583692</td>
      <td>0.138224</td>
      <td>6.982953</td>
      <td>0.451942</td>
      <td>0.182636</td>
      <td>1.237349</td>
      <td>0.067660</td>
      <td>1.184390</td>
      <td>0.169012</td>
      <td>0.050524</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1848.000000</td>
      <td>7.522400</td>
      <td>7.522400</td>
      <td>0.000000</td>
      <td>35.990000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.002740</td>
      <td>0.020000</td>
      <td>3.044522</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4187.750000</td>
      <td>8.340158</td>
      <td>8.340158</td>
      <td>0.000000</td>
      <td>35.990000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.073288</td>
      <td>0.080000</td>
      <td>4.430817</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>6144.000000</td>
      <td>8.723394</td>
      <td>8.723394</td>
      <td>0.000000</td>
      <td>49.990000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.143836</td>
      <td>0.160000</td>
      <td>4.969813</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>8917.500000</td>
      <td>9.095883</td>
      <td>9.095883</td>
      <td>0.000000</td>
      <td>49.990000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>3.214384</td>
      <td>0.200000</td>
      <td>5.804374</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>55174.000000</td>
      <td>10.918265</td>
      <td>10.918265</td>
      <td>0.500100</td>
      <td>49.990000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>4.284932</td>
      <td>1.000000</td>
      <td>10.409401</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>


# Cell 3: Correlation Heatmap (Multicollinearity Check)


```python
cols_to_corr = ['Log_Player', 'Lag_Player', 'Discount_Ratio', 'Years_Since_Release', 
                'Trend_Index', 'Log_Twitch_Avg', 'Is_Major_Update', 'Is_Weekend']

for game_name, df in games.items():
    # Maintain requested columns that exist in the dataframe
    valid_cols = [c for c in cols_to_corr if c in df.columns]
    if not valid_cols:
        continue
        
    corr_matrix = df[valid_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap='coolwarm', 
        vmin=-1, 
        vmax=1, 
        square=True, 
        linewidths=.5
    )
    plt.title(f'Correlation Heatmap - {game_name}', pad=20, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

```


    
![png](eda_files/eda_5_0.png)
    



    
![png](eda_files/eda_5_1.png)
    


# Cell 4: Dual-Axis Time Series Visualization (Storytelling)


```python
for game_name, df in games.items():
    fig, ax1 = plt.subplots(figsize=(16, 7))
    
    # --- Left Y-Axis: Log_Player ---
    color1 = '#2c3e50' # Dark Blue/Navy solid
    ax1.plot(df.index, df['Log_Player'], color=color1, linestyle='-', linewidth=2.5, label='Log Player Count')
    ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Log of Player Count', color=color1, fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # --- Right Y-Axis: Log_Twitch_Avg ---
    ax2 = ax1.twinx()
    color2 = '#e74c3c' # Vibrant Red dashed
    ax2.plot(df.index, df['Log_Twitch_Avg'], color=color2, linestyle='--', linewidth=2, label='Log Twitch Avg Viewers', alpha=0.8)
    ax2.set_ylabel('Log of Twitch Avg Viewers', color=color2, fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # --- Shaded Regions for Major Updates ---
    if 'Is_Major_Update' in df.columns:
        is_update = df['Is_Major_Update'] == 1
        update_dates = df[is_update].index
        for i, dt in enumerate(update_dates):
            ax1.axvspan(
                dt, 
                dt + pd.Timedelta(days=1), 
                color='#f1c40f', 
                alpha=0.3, 
                label='Major Update Window' if i == 0 else ""
            )

    # Titles and Legends
    plt.title(f'Dual-Axis Plot: Players vs Twitch Viewers - {game_name}', fontsize=16, fontweight='bold', pad=20)
    
    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', frameon=True, shadow=True)
    
    plt.grid(False) # Turn off automatic grid for dual-axis clarity
    plt.tight_layout()
    plt.show()

```


    
![png](eda_files/eda_7_0.png)
    



    
![png](eda_files/eda_7_1.png)
    


# Cell 5: Seasonality Diagnostics (ACF/PACF & Boxplots)


```python
for game_name, df in games.items():
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Drop NA to avoid ACF/PACF errors
    log_player_clean = df['Log_Player'].dropna()
    
    # --- Subplot 1: ACF of Log_Player ---
    plot_acf(log_player_clean, lags=35, ax=axes[0], title='Autocorrelation (ACF) of Log_Player', color='#3498db')
    axes[0].set_xlabel('Lags (Days)')
    axes[0].set_ylabel('Autocorrelation')
    
    # --- Subplot 2: PACF of Log_Player ---
    plot_pacf(log_player_clean, lags=35, ax=axes[1], title='Partial Autocorrelation (PACF) of Log_Player', color='#9b59b6', method='ywm')
    axes[1].set_xlabel('Lags (Days)')
    axes[1].set_ylabel('Partial Autocorrelation')
    
    # --- Subplot 3: Boxplot Is_Weekend vs Log_Player ---
    if 'Is_Weekend' in df.columns:
        sns.boxplot(data=df, x='Is_Weekend', y='Log_Player', ax=axes[2], palette='Set2')
        axes[2].set_title('Log_Player: Weekday (0) vs Weekend (1)')
        axes[2].set_xlabel('Is Weekend')
        axes[2].set_ylabel('Log Player Count')
    
    # General figure title
    fig.suptitle(f'Seasonality & Diagnostics - {game_name}', fontsize=18, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.show()

```

    /tmp/ipykernel_2744822/776366727.py:19: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.
    
      sns.boxplot(data=df, x='Is_Weekend', y='Log_Player', ax=axes[2], palette='Set2')
    


    
![png](eda_files/eda_9_1.png)
    


    /tmp/ipykernel_2744822/776366727.py:19: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.
    
      sns.boxplot(data=df, x='Is_Weekend', y='Log_Player', ax=axes[2], palette='Set2')
    


    
![png](eda_files/eda_9_3.png)
    

