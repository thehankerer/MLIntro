import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

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
    df['Target'] = (df['sgd_inr'].shift(-1) > df['sgd_inr']).astype(int)
    
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


# Select features and target
# Load processed features directly from CSV
processed_df = pd.read_csv('processed_currencies.csv')

features = [
    'sgd_inr', 'usd_inr', 'xau_inr', 'Daily_Range', 'Volatility',
    'MA_25', 'MA_75', 'MA_Ratio', 'Change',
    'Price_Lag1', 'Price_Lag2', 'Change_Lag1', 'Change_Lag2'
]
target = 'Target'

X = processed_df[features]
y = processed_df[target]

# Split data (time-series aware: no shuffling)
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

def linear_regression_usd_xau_to_sgd(df, plot=True):
    """
    Fit a linear regression model using usd_inr and xau_inr to predict sgd_inr.
    Prints coefficients and R^2 score. Optionally plots predictions.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    import matplotlib.pyplot as plt

    X = df[['usd_inr', 'xau_inr']]
    y = df['sgd_inr']

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    print("Linear Regression Coefficients:", lr.coef_)
    print("Intercept:", lr.intercept_)
    print("R^2 Score:", r2_score(y_test, y_pred))

    if plot:
        plt.figure(figsize=(14, 6))
        plt.plot(y_test.index, y_test.values, label='Actual SGD/INR', color='blue')
        plt.plot(y_test.index, y_pred, label='Predicted SGD/INR', color='red', alpha=0.7)
        plt.title('Linear Regression: Actual vs Predicted SGD/INR')
        plt.xlabel('Index')
        plt.ylabel('SGD/INR')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("linear_regression_sgd_inr.png")
        plt.show()

    return lr, X_test, y_test, y_pred

# Example usage:
linear_regression_usd_xau_to_sgd(processed_df)

# Predict
y_pred = model.predict(X_test)

def plot_feature_importances(model, feature_names):
    import matplotlib.pyplot as plt
    import numpy as np
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), np.array(feature_names)[indices], rotation=90)
    plt.xlim([-1, len(importances)])
    plt.tight_layout()
    plt.savefig("feature_importances.png")
    plt.show()
    
plot_feature_importances(model, features)

def plot_correlation_matrix(df):
    import seaborn as sns
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Correlation Matrix')
    plt.savefig("correlation_matrix_gold.png")
    plt.show()

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
def plot_predictions(y_test, y_pred):
    import matplotlib.pyplot as plt
    
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
    plt.savefig("predictions_plot.png")
    plt.show()
    
plot_correlation_matrix(processed_df)
plot_predictions(y_test, y_pred)
