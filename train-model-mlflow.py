# import libraries
import mlflow
import argparse
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.metrics import f1_score
import xgboost as xgb
import matplotlib.pyplot as plt

def main(args):
    # enable autologging
    mlflow.autolog()
    
    # Read data asset
    print("Reading data...")
    df = pd.read_csv(args.training_data)

    # Read model hyperparameters from json file
    with open(args.hparam_file) as f:
        hyperparameters = json.load(f)

    # Label Encoding
    le = LabelEncoder()
    df['Month'] = le.fit_transform(df['Month'])

    # One hot encoding
    encoded_df = pd.get_dummies(df, columns=['Occupation', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour'])
    
    # Scaling numerical features
    scaler = MinMaxScaler()
    cols = ['Month','Age','Annual_Income','Monthly_Inhand_Salary','Num_Bank_Accounts','Num_Credit_Card','Interest_Rate',
        'Num_of_Loan', 'Delay_from_due_date','Num_of_Delayed_Payment','Changed_Credit_Limit','Num_Credit_Inquiries',
        'Outstanding_Debt', 'Credit_Utilization_Ratio','Credit_History_Age','Total_EMI_per_month','Amount_invested_monthly']

    encoded_df[cols] = scaler.fit_transform(encoded_df[cols])

    # split data
    print("Splitting data...")
    y = encoded_df['Credit_Score']
    y = y.map({'Good':2, 'Standard':1, 'Poor':0})
    X = encoded_df.drop(['Credit_Score'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=587)

    # train model
    print("Training model...")
    model = xgb.XGBClassifier(**hyperparameters)
    model.fit(X_train, y_train)

    # evaluate model
    y_hat = model.predict(X_test)
    acc = np.average(y_hat == y_test)
    print('Accuracy:', acc)
    mlflow.log_metric("Accuracy", acc)

    f1 = f1_score(y_test, y_hat, average='weighted')
    print('F1 score:', f1)
    mlflow.log_metric("F1-score", f1)

    y_scores = model.predict_proba(X_test)

    # Feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = X_train.columns

    fig = plt.figure(figsize=(6, 4))
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation='vertical')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('XGBoost Feature Importance')
    plt.savefig("xgb_feature_importance.png")
    mlflow.log_artifact("xgb_feature_importance.png") 


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data", dest='training_data', type=str)
    parser.add_argument("--json_file", dest="hparam_file", type=str, help="Path to the JSON file containing hyperparameters")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    print("\n\n")
    print("*" * 60)

    args = parse_args()
    main(args)

    print("*" * 60)
    print("\n\n")
