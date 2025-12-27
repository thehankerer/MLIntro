import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


processed_df = pd.read_csv('processed_currencies.csv')


def feature_engineering(df):
    """
    Perform feature engineering on the dataframe.
    Adds lagged features and movement classification.
    """
    df = df.copy()
    # Movement should be 1 when next day's price is higher than current day's price
    df['sgd_movement'] = (df['Price_Lag1'] > df['sgd_inr']).astype(int)
    
    #lead features for sgd_inr
    df['Price_Lead1'] = df['sgd_inr'].shift(-1)
    df['Price_Lead2'] = df['sgd_inr'].shift(-2)
    df.dropna(inplace=True)
    return df

def linear_regression_usd_xau_to_sgd(df, plot=True):
    """
    Fit a linear regression model using usd_inr and xau_inr to predict sgd_inr.
    Prints coefficients and R^2 score. 
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt

    X = df[['usd_inr', 'xau_inr']]
    y = df['Price_Lead1']  # Predicting next day's sgd_inr

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    print("Linear Regression Coefficients:", lr.coef_)
    print("Intercept:", lr.intercept_)
    print("R^2 Score:", r2_score(y_test, y_pred))
    print("Accuracy Score:", lr.score(X_test, y_test))

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
        plt.savefig("linear_regression_sgd_inr_corrected.png")
        plt.show()

    return lr, X_test, y_test, y_pred



def logistic_regression_usd_xau_to_sgd(df):
    """
    Fit a logistic regression model using usd_inr and xau_inr to classify sgd_inr movements.
    Prints classification report and accuracy score.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, accuracy_score

    X = df[['usd_inr', 'xau_inr']]
    y = df['sgd_movement']  # 1 for up, 0 for down

    # Print overall distribution
    print("Movement distribution (overall):")
    print(y.value_counts())

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Use a pipeline with scaling to ensure features are comparable
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
    import numpy as np

    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    pipe.fit(X_train, y_train)

    probs = pipe.predict_proba(X_test)[:, 1]
    y_pred = (probs > 0.5).astype(int)

    # Diagnostics
    print("Movement distribution (train):")
    print(y_train.value_counts())
    print("Movement distribution (test):")
    print(y_test.value_counts())
    import pandas as _pd
    print("Predicted distribution (threshold=0.5):")
    print(_pd.Series(y_pred).value_counts())

    print("ROC AUC:", roc_auc_score(y_test, probs))
    print("Classification report (threshold=0.5):")
    print(classification_report(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Find best threshold by Youden's J (maximize TPR - FPR)
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    best_idx = np.argmax(tpr - fpr)
    best_thresh = thresholds[best_idx]
    print("Best threshold by Youden J:", best_thresh)
    y_pred_best = (probs > best_thresh).astype(int)
    print("Classification report (best threshold):")
    print(classification_report(y_test, y_pred_best))
    

    # Final return: return the trained pipeline and test data
    return pipe, X_test, y_test, y_pred,y_pred_best
    
def plot_predictions(y_test, y_pred):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    # Use scatter plot for discrete movement visualization
    plt.scatter(y_test.index, y_test.values, label='Actual Movement', color='green', s=30)
    plt.scatter(y_test.index, y_pred, label='Predicted Movement', color='red', alpha=0.8, s=30)
    plt.title('Logistic Regression: Actual vs Predicted SGD/INR Movement')
    plt.xlabel('Index')
    plt.ylabel('Movement (1=Up, 0=Down)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #plt.savefig("logistic_regression_sgd_movement.png")
    plt.show()
    

featured_df = feature_engineering(processed_df)    
log_reg, X_test, y_test, y_pred, y_pred_best = logistic_regression_usd_xau_to_sgd(featured_df)
plot_predictions(y_test, y_pred_best)
#plot_predictions(y_test, y_pred) to plot with default 0.5 threshold
linear_regression_usd_xau_to_sgd(featured_df)
