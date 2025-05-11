import pandas as pd
import subprocess
import pickle
import numpy as np
from ML_Pipeline.Preprocess import apply
from ML_Pipeline.Train_Model import fit
from ML_Pipeline.Predict import init as predict_init
from ML_Pipeline.Utils import save_model, load_model

THRESHOLD_PATH = '../output/threshold.pkl'

def main():
    val = int(input("Train - 0\nPredict - 1\nDeploy - 2\nEnter your value: "))

    if val == 0:
        # TRAIN
        data = pd.read_csv('../input/paysim.csv')  # Updated filename to match yours
        print("Loaded raw data, shape=", data.shape)
        processed = apply(data, is_train=True)
        model, columns = fit(processed)
        save_model(model, columns)

        # Compute threshold on non-fraud training set
        train_X = processed[processed['isFraud']==0].drop('isFraud', axis=1).values
        recon = model.predict(train_X)
        mse = np.mean(np.square(recon - train_X), axis=1)
        threshold = np.percentile(mse, 95)
        with open(THRESHOLD_PATH, 'wb') as f:
            pickle.dump(threshold, f)
        print(f"Model and threshold saved (95th percentile = {threshold:.6f})")

    elif val == 1:
        # PREDICT on fictional/test data
        model, columns = load_model('../output/deep-ae-model')
        
        # Load threshold properly from the pickle file
        with open(THRESHOLD_PATH, 'rb') as f:
            threshold = pickle.load(f)
            print(f"Loaded threshold value: {threshold:.6f}")

        # Example test set - you can customize this with your test data
        test_df = pd.DataFrame([
    # Normal transaction for reference
    {
        'step': 1, 
        'amount': 500.0, 
        'oldbalanceOrg': 10000.0, 
        'newbalanceOrig': 9500.0,
        'oldbalanceDest': 5000.0, 
        'newbalanceDest': 5500.0, 
        'type': 'TRANSFER'
    },
    
    # Potentially fraudulent transactions
    
    # 1. Very large amount transaction
    {
        'step': 1, 
        'amount': 9999999999.0,  # Very large amount
        'oldbalanceOrg': 1000000.0, 
        'newbalanceOrig': 1.0,  # Almost emptied the account
        'oldbalanceDest': 0.0, 
        'newbalanceDest': 999999.0, 
        'type': 'TRANSFER'  
    },
    
    # 2. Transaction that doesn't balance (money disappears)
    {
        'step': 2, 
        'amount': 5000.0, 
        'oldbalanceOrg': 10000.0, 
        'newbalanceOrig': 5000.0,
        'oldbalanceDest': 1000.0, 
        'newbalanceDest': 1000.0,  # Money didn't arrive at destination
        'type': 'TRANSFER'
    },
    
    # 3. Cash out of entire balance
    {
        'step': 3, 
        'amount': 50000.0, 
        'oldbalanceOrg': 50000.0, 
        'newbalanceOrig': 0.0,  # Complete account drainage
        'oldbalanceDest': 0.0, 
        'newbalanceDest': 50000.0, 
        'type': 'CASH_OUT'
    },
    
    # 4. Multiple quick small transfers (structuring)
    {
        'step': 4, 
        'amount': 9000.0,  # Just under 10k reporting threshold
        'oldbalanceOrg': 100000.0, 
        'newbalanceOrig': 91000.0,
        'oldbalanceDest': 5000.0, 
        'newbalanceDest': 14000.0, 
        'type': 'TRANSFER'
    },
    
    # 5. Payment to never-seen-before merchant (unusual pattern)
    {
        'step': 5, 
        'amount': 15735.62,  # Specific odd amount
        'oldbalanceOrg': 20000.0, 
        'newbalanceOrig': 4264.38,
        'oldbalanceDest': 1000000.0,  # Very high merchant balance
        'newbalanceDest': 1015735.62, 
        'type': 'PAYMENT'
    },
    
    # 6. Inconsistent values (amount withdrawn doesn't match balance change)
    {
        'step': 6, 
        'amount': 2500.0, 
        'oldbalanceOrg': 10000.0, 
        'newbalanceOrig': 5000.0,  # Should be 7500 if withdrawal was 2500
        'oldbalanceDest': 1000.0, 
        'newbalanceDest': 3500.0, 
        'type': 'TRANSFER'
    },
    
    # 7. Very unusual transaction type with high amount
    {
        'step': 7, 
        'amount': 75000.0, 
        'oldbalanceOrg': 80000.0, 
        'newbalanceOrig': 5000.0,
        'oldbalanceDest': 10000.0, 
        'newbalanceDest': 85000.0, 
        'type': 'DEBIT'  
    }
])
        
        processed = apply(test_df, is_train=False)

        for i in processed.index:
            score = predict_init(processed.loc[[i]], model, columns)
            status = 'FRAUD' if score > threshold else 'OK'
            print(f"Record {i}: MSE={score:.6f} → {status}")

    else:
        # DEPLOY
        subprocess.Popen(['gunicorn', 'wsgi:app', '--bind', '0.0.0.0:5001'])
        print("API deployed on port 5001")

if __name__ == '__main__':
    main()