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
from sklearn.metrics import precision_recall_curve

MODEL_PATH = '../output/banksafenet-model'
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
        
        # Train BankSafeNet model (dual autoencoders + transformer)
        model_dict, history = fit(train_data)
        save_model(model_dict, train_data.columns.drop('isFraud').tolist())
        
        # Evaluate model performance on test set
        test_X = test_data.drop('isFraud', axis=1).values
        test_y = test_data['isFraud'].values
        
        # Get fraud scores from BankSafeNet
        fraud_scores = predict_init(test_data.drop('isFraud', axis=1), model_dict, 
                                  train_data.columns.drop('isFraud').tolist())
        
        # CORRECTED THRESHOLD COMPUTATION using precision-recall curve
        # This finds the threshold that maximizes F1 score (better for imbalanced data)
        precision, recall, thresholds = precision_recall_curve(test_y, fraud_scores)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        threshold = thresholds[optimal_idx]
        
        with open(THRESHOLD_PATH, 'wb') as f:
            pickle.dump(threshold, f)
        print(f"Optimal threshold found: {threshold:.6f} (F1-score: {f1_scores[optimal_idx]:.4f})")
        
        # Plot training results
        plot_results(model_dict, history, test_X, test_y, fraud_scores, threshold)

    elif val == 1:
        # PREDICT (same as before)
        model_dict, columns = load_model(MODEL_PATH)
        
        with open(THRESHOLD_PATH, 'rb') as f:
            threshold = pickle.load(f)
            print(f"Loaded threshold value: {threshold:.6f}")

        # Example test cases (same as before)
        test_df = pd.DataFrame([
            {
                'amount': 500.0,
                'oldbalanceOrg': 10000.0,
                'newbalanceOrig': 9500.0,
                'oldbalanceDest': 5000.0,
                'newbalanceDest': 5500.0,
                'type': 'TRANSFER'
            },
            {
                'amount': 100000.0,
                'oldbalanceOrg': 2600000.0,
                'newbalanceOrig': 2470000.0,
                'oldbalanceDest': 2100000.0,
                'newbalanceDest': 5700000.0,
                'type': 'CASH_IN'
            }
        ])
        
        processed = apply(test_df, is_train=False)
        
        print("\nEvaluating test cases:")
        print("-" * 70)
        print(f"{'Index':<6}{'Transaction Type':<15}{'Amount':<12}{'Fraud Score':<12}{'Status':<8}")
        print("-" * 70)
        
        scores = predict_init(processed, model_dict, columns)
        for i, score in enumerate(scores):
            status = 'FRAUD' if score > threshold else 'OK'
            print(f"{i:<6}{test_df.loc[i, 'type']:<15}{test_df.loc[i, 'amount']:<12.2f}{score:<12.6f}{status:<8}")

    else:
        # DEPLOY
        subprocess.Popen(['gunicorn', 'wsgi:app', '--bind', '0.0.0.0:5001'])
        print("API deployed on port 5001")

if __name__ == '__main__':
    main()