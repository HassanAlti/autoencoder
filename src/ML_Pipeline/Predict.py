import numpy as np

def init(test_df, model, columns):
    # ensure correct col order
    X = test_df[columns].values
    preds = model.predict(X)
    mse = np.mean(np.square(preds - X), axis=1)
    return mse[0] if len(mse)==1 else mse