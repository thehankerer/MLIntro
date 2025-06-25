import pandas as pd

def merge_currency_data(sgd_inr_path, usd_inr_path, xau_inr_path, output_path):
    # Read CSV files
    sgd_df = pd.read_csv(sgd_inr_path, usecols=['Date', 'Price'])
    usd_df = pd.read_csv(usd_inr_path, usecols=['Date', 'Price'])
    xau_df = pd.read_csv(xau_inr_path, usecols=['Date', 'Price'])

    # Rename price columns for clarity
    sgd_df = sgd_df.rename(columns={'Price': 'sgd_inr'})
    usd_df = usd_df.rename(columns={'Price': 'usd_inr'})
    xau_df = xau_df.rename(columns={'Price': 'xau_inr'})

    # Merge on 'date'
    merged = sgd_df.merge(usd_df, on='Date', how='inner').merge(xau_df, on='Date', how='inner')

    # Save to CSV
    merged.to_csv(output_path, index=False)

# Example usage:
#merge_currency_data('SGD_INR Historical Data.csv', 'USD_INR Historical Data.csv', 'XAU_INR Historical Data.csv', 'merged_currencies.csv')

def preprocess_currency_data(df):
    # Convert 'Date' to datetime and set as index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Sort by index
    df.sort_index(inplace=True)
    
    # Drop rows with any NaN values
    df.dropna(inplace=True)
    
    return df   

def describe_currency_data(df):
    print("Data Description:")
    print(df.describe())
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum())
    
def feature_engineer_currency_data(df): 
    # Create target variable: 1 if price will increase in the next 15 days, else 0
    df['Target'] = (df['sgd_inr'].shift(-15) > df['sgd_inr']).astype(int)
    
    # Calculate daily range and volatility
    df['Daily_Range'] = df['sgd_inr'].rolling(window=1).max() - df['sgd_inr'].rolling(window=1).min()
    df['Volatility'] = (df['sgd_inr'].rolling(window=1).max() - df['sgd_inr'].rolling(window=1).min()) / df['sgd_inr']
    
    # Ensure xau_inr is float (remove commas if present)
    df['xau_inr'] = df['xau_inr'].replace({',': ''}, regex=True).astype(float)
    # Moving averages
    df['MA_25'] = df['sgd_inr'].rolling(window=25).mean()
    df['MA_75'] = df['sgd_inr'].rolling(window=75).mean()
    df['MA_Ratio'] = df['MA_25'] / df['MA_75']
    
    # Percentage change
    df['Change'] = df['sgd_inr'].pct_change()
    
    # Lag features
    for lag in [1, 2]:
        df[f'Price_Lag{lag}'] = df['sgd_inr'].shift(lag)
        df[f'Change_Lag{lag}'] = df['Change'].shift(lag)
    
    # Drop NaN values after feature engineering
    df.dropna(inplace=True)
    
    return df

# Example usage:
df = pd.read_csv('merged_currencies.csv')
df = preprocess_currency_data(df)
describe_currency_data(df)


df = feature_engineer_currency_data(df)
print(df.head())
df.to_csv('processed_currencies.csv', index=False)
# This code merges currency data from three CSV files, preprocesses it, describes it, and
# performs feature engineering to create new features for analysis or modeling.
# The final processed data can be saved to a new CSV file for further use.
# The functions are modular to allow for easy testing and reuse.
# The example usage at the end shows how to use these functions in practice.
