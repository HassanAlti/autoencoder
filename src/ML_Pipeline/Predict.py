import numpy as np

def init(test_df, model, columns):
    """
    Calculate reconstruction error (MSE) for given samples
    
    Args:
        test_df: DataFrame with samples to evaluate
        model: trained autoencoder model
        columns: feature columns in correct order
    
    Returns:
        MSE values for each sample
    """
    # Ensure correct column order
    if 'isFraud' in test_df.columns:
        X = test_df.drop('isFraud', axis=1)[columns].values
    else:
        X = test_df[columns].values
    
    # Get reconstructions
    preds = model.predict(X)
    
    # Calculate mean squared error per sample
    mse = np.mean(np.square(preds - X), axis=1)
    
    return mse[0] if len(mse) == 1 else mse
