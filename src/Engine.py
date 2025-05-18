import pandas as pd
import subprocess
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ML_Pipeline.Preprocess import apply
from ML_Pipeline.Train_Model import fit
from ML_Pipeline.Predict import init as predict_init
from ML_Pipeline.Utils import save_model, load_model, plot_results

MODEL_PATH = '../output/deep-ae-model'
THRESHOLD_PATH = '../output/threshold.pkl'

def main():
    val = int(input("Train - 0\nPredict - 1\nDeploy - 2\nEnter your value: "))

    if val == 0:
        # TRAIN
        data = pd.read_csv('../input/paysim.csv')
        print("Loaded raw data, shape=", data.shape)
        
        # Process data and split into train/test
        train_data, test_data = apply(data, is_train=True)
        print(f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")
        
        # Train model and save
        model, history = fit(train_data)
        save_model(model, train_data.columns.drop('isFraud').tolist())
        
        # Evaluate model performance on test set
        test_X = test_data.drop('isFraud', axis=1).values
        test_y = test_data['isFraud'].values
        
        # Get reconstruction errors
        recon = model.predict(test_X)
        mse = np.mean(np.square(recon - test_X), axis=1)
        
        # Find optimal threshold (95th percentile of non-fraud transactions)
        normal_errors = mse[test_y == 0]
        threshold = np.percentile(normal_errors, 95)
        
        with open(THRESHOLD_PATH, 'wb') as f:
            pickle.dump(threshold, f)
        print(f"Model and threshold saved (95th percentile = {threshold:.6f})")
        
        # Plot training results
        plot_results(model, history, test_X, test_y, mse, threshold)

    elif val == 1:
        # PREDICT on fictional/test data
        model, columns = load_model(MODEL_PATH)
        
        # Load threshold
        with open(THRESHOLD_PATH, 'rb') as f:
            threshold = pickle.load(f)
            print(f"Loaded threshold value: {threshold:.6f}")

        # Example test set - using your provided test scenarios
        test_df = pd.DataFrame([
    # === Normal Transactions ===
    {
        'amount': 500.0,
        'oldbalanceOrg': 10000.0,
        'newbalanceOrig': 9500.0,
        'oldbalanceDest': 5000.0,
        'newbalanceDest': 5500.0,
        'type': 'TRANSFER'
    },
    {
        'amount': 120.0,
        'oldbalanceOrg': 2500.0,
        'newbalanceOrig': 2380.0,
        'oldbalanceDest': 800.0,
        'newbalanceDest': 920.0,
        'type': 'PAYMENT'
    },
    {
        'amount': 1000.0,
        'oldbalanceOrg': 5000.0,
        'newbalanceOrig': 4000.0,
        'oldbalanceDest': 2000.0,
        'newbalanceDest': 3000.0,
        'type': 'TRANSFER'
    },
    {
        'amount': 300.0,
        'oldbalanceOrg': 1500.0,
        'newbalanceOrig': 1200.0,
        'oldbalanceDest': 400.0,
        'newbalanceDest': 700.0,
        'type': 'PAYMENT'
    },
    {
        'amount': 750.0,
        'oldbalanceOrg': 5000.0,
        'newbalanceOrig': 4250.0,
        'oldbalanceDest': 1000.0,
        'newbalanceDest': 1750.0,
        'type': 'TRANSFER'
    },

    # === Fraudulent Transactions ===
    {
        'amount': 100000.0,
        'oldbalanceOrg': 2600000.0,
        'newbalanceOrig': 2470000.0,  # mismatch
        'oldbalanceDest': 2100000.0,
        'newbalanceDest': 5700000.0,  # suspicious increase
        'type': 'CASH_IN'
    },
    {
        'amount': 800000.0,
        'oldbalanceOrg': 800000.0,
        'newbalanceOrig': 1400.0,  # almost drained
        'oldbalanceDest': 1300000.0,
        'newbalanceDest': 160000.0,  # decrease?
        'type': 'TRANSFER'
    },
    {
        'amount': 390000.0,
        'oldbalanceOrg': 270000.0,
        'newbalanceOrig': 1400000.0,  # increase?
        'oldbalanceDest': 4500000.0,
        'newbalanceDest': 13800000.0,  # massive increase
        'type': 'CASH_IN'
    },
    {
        'amount': 75000.0,
        'oldbalanceOrg': 80000.0,
        'newbalanceOrig': 5000.0,
        'oldbalanceDest': 10000.0,
        'newbalanceDest': 85000.0,
        'type': 'DEBIT'  # logically incorrect
    },
    {
        'amount': 25000.0,
        'oldbalanceOrg': 100000.0,
        'newbalanceOrig': 75000.0,
        'oldbalanceDest': 50000.0,
        'newbalanceDest': 75000.0,
        'type': 'TRANSFER'  # circular pattern
    }
])
        processed = apply(test_df, is_train=False)
        
        print("\nEvaluating test cases:")
        print("-" * 65)
        print(f"{'Index':<6}{'Transaction Type':<15}{'Amount':<12}{'MSE':<12}{'Status':<8}")
        print("-" * 65)
        
        for i in processed.index:
            score = predict_init(processed.loc[[i]], model, columns)
            status = 'FRAUD' if score > threshold else 'OK'
            print(f"{i:<6}{test_df.loc[i, 'type']:<15}{test_df.loc[i, 'amount']:<12.2f}{score:<12.6f}{status:<8}")

    else:
        # DEPLOY
        subprocess.Popen(['gunicorn', 'wsgi:app', '--bind', '0.0.0.0:5001'])
        print("API deployed on port 5001")

if __name__ == '__main__':
    main()