import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

# Preprocess PaySim transactions for training and inference
def apply(df: pd.DataFrame, is_train: bool = False):
    """
    Enhanced preprocessing for PaySim transaction data
    - Improved feature engineering
    - Robust scaling for outlier handling
    - Better handling of transaction types
    - Balance and amount ratio features
    """
    # Drop identifier and auxiliary columns
    drop_cols = ['nameOrig', 'nameDest', 'isFlaggedFraud', 'step']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    # Create a copy to avoid modifying the original dataframe
    df_processed = df.copy()
    
    # Feature Engineering - Add transaction balance features
    # These better capture discrepancies that might indicate fraud
    df_processed['balanceDiffOrig'] = df_processed['oldbalanceOrg'] - df_processed['newbalanceOrig']
    df_processed['balanceDiffDest'] = df_processed['newbalanceDest'] - df_processed['oldbalanceDest']
    
    # Check for balance inconsistency (expected vs actual change)
    df_processed['origBalanceConsistency'] = abs(df_processed['balanceDiffOrig'] - df_processed['amount']) / (df_processed['amount'] + 1e-8)
    df_processed['destBalanceConsistency'] = abs(df_processed['balanceDiffDest'] - df_processed['amount']) / (df_processed['amount'] + 1e-8)
    
    # Cap extreme values for consistency features
    df_processed['origBalanceConsistency'] = df_processed['origBalanceConsistency'].clip(0, 5)
    df_processed['destBalanceConsistency'] = df_processed['destBalanceConsistency'].clip(0, 5)
    
    # Amount-to-balance ratios
    df_processed['amountToOldBalanceOrig'] = df_processed['amount'] / (df_processed['oldbalanceOrg'] + 1e-8)
    df_processed['amountToNewBalanceOrig'] = df_processed['amount'] / (df_processed['newbalanceOrig'] + 1e-8)
    df_processed['amountToOldBalanceDest'] = df_processed['amount'] / (df_processed['oldbalanceDest'] + 1e-8)
    df_processed['amountToNewBalanceDest'] = df_processed['amount'] / (df_processed['newbalanceDest'] + 1e-8)
    
    # Cap these ratios to reasonable values
    for col in ['amountToOldBalanceOrig', 'amountToNewBalanceOrig', 
                'amountToOldBalanceDest', 'amountToNewBalanceDest']:
        df_processed[col] = df_processed[col].clip(0, 10)
    
    # Zero balance flags - these can be important indicators
    df_processed['isOldBalanceOrigZero'] = (df_processed['oldbalanceOrg'] == 0).astype(int)
    df_processed['isNewBalanceOrigZero'] = (df_processed['newbalanceOrig'] == 0).astype(int)
    df_processed['isOldBalanceDestZero'] = (df_processed['oldbalanceDest'] == 0).astype(int)
    df_processed['isNewBalanceDestZero'] = (df_processed['newbalanceDest'] == 0).astype(int)
    
    # Handle the transaction type with better encoding
    df_processed = pd.get_dummies(df_processed, columns=['type'], prefix='type')
    
    # Separate the fraud label if present
    if 'isFraud' in df_processed.columns:
        target = df_processed['isFraud'].copy()
        features = df_processed.drop('isFraud', axis=1)
    else:
        target = None
        features = df_processed.copy()
    
    # Identify numeric feature columns
    num_cols = features.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Use RobustScaler instead of StandardScaler to handle outliers better
    scaler_path = '../output/scaler.pkl'
    
    if is_train:
        # Use RobustScaler which is less sensitive to outliers
        scaler = RobustScaler().fit(features[num_cols])
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
    else:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    
    # Apply scaling
    features[num_cols] = scaler.transform(features[num_cols])
    
    # Handle consistent feature columns
    features_path = '../output/feature_names.pkl'
    
    if is_train:
        # Save the list of columns after dummies & scaling
        with open(features_path, 'wb') as f:
            pickle.dump(features.columns.tolist(), f)
            
        # Stratified split to maintain fraud distribution
        X_train, X_test = train_test_split(
            df_processed, 
            test_size=0.2, 
            random_state=42, 
            stratify=df_processed['isFraud'] if 'isFraud' in df_processed.columns else None
        )
        
        # Train only on non-fraudulent data (autoencoder approach)
        X_train_normal = X_train[X_train.isFraud == 0].copy()
        
        # Process the training and test data consistently
        train_features = X_train_normal.drop('isFraud', axis=1)
        train_features[num_cols] = scaler.transform(train_features[num_cols])
        X_train_normal = pd.concat([train_features, X_train_normal['isFraud']], axis=1)
        
        test_features = X_test.drop('isFraud', axis=1)
        test_features[num_cols] = scaler.transform(test_features[num_cols])
        X_test = pd.concat([test_features, X_test['isFraud']], axis=1)
        
        return X_train_normal, X_test
    else:
        # Load the training feature list and reindex
        with open(features_path, 'rb') as f:
            feature_list = pickle.load(f)
        features = features.reindex(columns=feature_list, fill_value=0)
        
        # Reattach target if present
        if target is not None:
            features['isFraud'] = target.values
            
        return features