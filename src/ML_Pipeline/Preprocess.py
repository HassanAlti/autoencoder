import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def apply(df: pd.DataFrame, is_train: bool = False):
    """
    Enhanced BankSafeNet preprocessing with better feature engineering
    """
    # Drop identifier columns as per paper
    drop_cols = ['nameOrig', 'nameDest', 'isFlaggedFraud', 'step']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    # Create a copy to avoid modifying original
    df_processed = df.copy()
    
    # ENHANCED FEATURE ENGINEERING (Key for better performance)
    if 'amount' in df_processed.columns:
        # Log transform for amount (handles large value ranges)
        df_processed['amount_log'] = np.log1p(df_processed['amount'])
        
        # Balance difference features (key fraud indicators)
        df_processed['balance_diff_orig'] = df_processed['oldbalanceOrg'] - df_processed['newbalanceOrig']
        df_processed['balance_diff_dest'] = df_processed['newbalanceDest'] - df_processed['oldbalanceDest']
        
        # Amount to balance ratios (key fraud indicators)
        df_processed['amount_to_old_balance_orig'] = df_processed['amount'] / (df_processed['oldbalanceOrg'] + 1)
        df_processed['amount_to_old_balance_dest'] = df_processed['amount'] / (df_processed['oldbalanceDest'] + 1)
        
        # Anomaly indicators (mentioned in paper)
        df_processed['is_zero_balance_orig'] = (df_processed['oldbalanceOrg'] == 0).astype(int)
        df_processed['is_zero_balance_dest'] = (df_processed['oldbalanceDest'] == 0).astype(int)
        df_processed['balance_error_orig'] = (df_processed['balance_diff_orig'] != df_processed['amount']).astype(int)
    
    # Handle missing values with mean imputation (as per paper)
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'isFraud':  # Don't impute target variable
            mean_val = df_processed[col].mean()
            df_processed[col] = df_processed[col].fillna(mean_val)
    
    # One-hot encoding for categorical features
    df_processed = pd.get_dummies(df_processed, columns=['type'], prefix='type')
    
    # Separate fraud label if present
    if 'isFraud' in df_processed.columns:
        target = df_processed['isFraud'].copy()
        features = df_processed.drop('isFraud', axis=1)
    else:
        target = None
        features = df_processed.copy()
    
    # Use StandardScaler instead of MinMaxScaler for better performance with deep learning
    scaler_path = '../output/scaler.pkl'
    
    if is_train:
        # Fit scaler on training data
        scaler = StandardScaler()
        features_scaled = pd.DataFrame(
            scaler.fit_transform(features), 
            columns=features.columns, 
            index=features.index
        )
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
    else:
        # Load existing scaler
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        features_scaled = pd.DataFrame(
            scaler.transform(features), 
            columns=features.columns, 
            index=features.index
        )
    
    # Handle feature consistency for inference
    features_path = '../output/feature_names.pkl'
    
    if is_train:
        # Save feature names
        with open(features_path, 'wb') as f:
            pickle.dump(features_scaled.columns.tolist(), f)
            
        # Split data maintaining fraud distribution
        if target is not None:
            final_data = pd.concat([features_scaled, target], axis=1)
            X_train, X_test = train_test_split(
                final_data, 
                test_size=0.2, 
                random_state=42, 
                stratify=final_data['isFraud']
            )
            return X_train, X_test
        else:
            return features_scaled, None
    else:
        # Load feature names and ensure consistency
        with open(features_path, 'rb') as f:
            feature_list = pickle.load(f)
        features_scaled = features_scaled.reindex(columns=feature_list, fill_value=0)
        
        # Reattach target if present
        if target is not None:
            features_scaled['isFraud'] = target.values
            
        return features_scaled