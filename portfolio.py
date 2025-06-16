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
    df['Target'] = (df['Price'].shift(-15) > df['Price']).astype(int)
    df['Daily_Range'] = df['High'] - df['Low']
    df['Volatility'] = (df['High'] - df['Low']) / df['Open']
    df['MA_25'] = df['Price'].rolling(window=25).mean()
    df['MA_75'] = df['Price'].rolling(window=75).mean()
    df['MA_Ratio'] = df['MA_25'] / df['MA_75']
    df['Change'] = df['Price'].pct_change()
    df['Price_Lag1'] = df['Price'].shift(1)
    df['Price_Lag2'] = df['Price'].shift(2)
    df['Change_Lag1'] = df['Change'].shift(1)
    df['Change_Lag2'] = df['Change'].shift(2)

    df.dropna(inplace=True)
    return df
describe_data(df)
#df = preprocess_data(df)
#print(df.head())
#df = feature_engineering(df)
#print(df.head())

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
    features = ['Open', 'High', 'Low', 'Daily_Range', 'Volatility', 'MA_25', 'MA_75', 'MA_Ratio','Price_Lag1', 'Price_Lag2', 'Change_Lag1', 'Change_Lag2']
    X = df[features]
    y = df['Target']
    ''''
    train_size = int(len(df) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    '''
    
    from sklearn.model_selection import TimeSeriesSplit

    tscv = TimeSeriesSplit(n_splits=5)
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    return X_train, X_test, y_train, y_test

def plot_correlation_matrix(df):
    feature = ['Open', 'High', 'Low', 'Daily_Range', 'Volatility', 'MA_25', 'MA_75', 'MA_Ratio','Target']
    plt.figure(figsize=(12, 8))
    x = df[feature]
    corr = x.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Correlation Matrix')
    plt.savefig("correlation_matrix.png")
    
    
#plot_correlation_matrix(df)

def train_model(X_train, y_train):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42,class_weight='balanced')
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    from sklearn.metrics import classification_report, accuracy_score
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    
def plot_predictions(y_test, y_pred, df):
    # Create a DataFrame to align index and plot
    results = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': y_pred
    }, index=y_test.index)

    plt.figure(figsize=(14, 6))
    plt.plot(results.index, results['Actual'], label='Actual', alpha=0.7, color='blue')
    plt.plot(results.index, results['Predicted'], label='Predicted', alpha=0.7, color='red')
    plt.title('Actual vs Predicted Target Values')
    plt.xlabel('Date')
    plt.ylabel('Target')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png")


def main():
    df = pd.read_csv('SGD_INR Historical Data.csv')
    df = preprocess_data(df)
    df = feature_engineering(df)
    plot_data(df)
    X_train, X_test, y_train, y_test = datasplit(df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    
    y_pred = model.predict(X_test)
    plot_predictions(y_test, y_pred, df)
    from sklearn.feature_selection import mutual_info_classif
    mi = mutual_info_classif(X_train, y_train)
    print(pd.Series(mi, index=X_train.columns).sort_values(ascending=False))
    
if __name__ == "__main__":
    main()
    print("Model training and evaluation completed.")
    print("Plots saved as 'sgd_inr_prediction.png' and 'correlation_matrix.png'.")
    

    
