import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Preprocess PaySim transactions for training and inference
def apply(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    # Drop identifier and auxiliary columns
    drop_cols = ['nameOrig', 'nameDest', 'isFlaggedFraud']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # One-hot encode the transaction type
    df = pd.get_dummies(df, columns=['type'], prefix='type')

    # Separate the fraud label if present
    if 'isFraud' in df.columns:
        target = df['isFraud']
        df = df.drop('isFraud', axis=1)
    else:
        target = None

    # Identify numeric feature columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Fit or load scaler
    scaler_path = '../output/scaler.pkl'
    if is_train:
        scaler = StandardScaler().fit(df[num_cols])
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
    else:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

    # Apply scaling
    df[num_cols] = scaler.transform(df[num_cols])

    # Ensure consistent feature columns
    features_path = '../output/feature_names.pkl'
    if is_train:
        # Save the list of columns after dummies & scaling
        with open(features_path, 'wb') as f:
            pickle.dump(df.columns.tolist(), f)
    else:
        # Load the training feature list and reindex
        with open(features_path, 'rb') as f:
            feature_list = pickle.load(f)
        df = df.reindex(columns=feature_list, fill_value=0)

    # Re-attach the target label
    if target is not None:
        df['isFraud'] = target.values

    return df