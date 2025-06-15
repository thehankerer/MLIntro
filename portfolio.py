import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('SGD_INR Historical Data.csv')

def preprocess_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    df = df.dropna()
    return df

def describe_data(df):
    print("Data Description:")
    print(df.describe())
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum())
    
# Convert volume strings like '28.79K' to numeric
def convert_volume(val):
    if isinstance(val, str):
        val = val.replace(',', '').strip()
        if val.endswith('K'):
            return float(val[:-1]) * 1_000
        elif val.endswith('M'):
            return float(val[:-1]) * 1_000_000
        elif val.endswith('B'):
            return float(val[:-1]) * 1_000_000_000
        else:
            try:
                return float(val)
            except:
                return np.nan
    return val


    
def feature_engineering(df):
    df['Target'] = (df['Price'].shift(-1) > df['Price']).astype(int)
    df['Daily_Range'] = df['High'] - df['Low']
    df['Volatility'] = (df['High'] - df['Low']) / df['Open']
    df['MA_25'] = df['Price'].rolling(window=25).mean()
    df['MA_75'] = df['Price'].rolling(window=75).mean()
    df['MA_Ratio'] = df['MA_25'] / df['MA_75']
    df.dropna(inplace=True)
    return df
describe_data(df)
df = preprocess_data(df)
print(df.head())
df = feature_engineering(df)
print(df.head())

def plot_data(df):
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Price'], label='Price', color='blue')
    plt.plot(df.index, df['MA_25'], label='25-Day MA', color='orange')
    plt.plot(df.index, df['MA_75'], label='75-Day MA', color='green')
    plt.title('SGD to INR Price with Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price (INR)')
    plt.legend()
    plt.grid()
    plt.savefig("sgd_inr_prediction.png")
    
def datasplit(df):
    features = ['Open', 'High', 'Low', 'Change', 'Daily_Range', 'Volatility', 'MA_3', 'MA_7', 'MA_Ratio']
    X = df[features]
    y = df['Target']
    train_size = int(len(df) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    return X_train, X_test, y_train, y_test
def plot_correlation_matrix(df):
    feature = ['Open', 'High', 'Low', 'Daily_Range', 'Volatility', 'MA_25', 'MA_75', 'MA_Ratio','Target']
    plt.figure(figsize=(12, 8))
    x = df[feature]
    corr = x.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Correlation Matrix')
    plt.savefig("correlation_matrix.png")
    
    
plot_correlation_matrix(df)

