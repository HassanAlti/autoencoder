from tensorflow import keras


def fit(df):
    # Train only on non-fraudulent rows
    train_df = df[df['isFraud']==0].drop('isFraud', axis=1)
    X = train_df.values

    input_dim = X.shape[1]
    inp = keras.layers.Input(shape=(input_dim,))
    encoded = keras.layers.Dense(32, activation='relu')(inp)
    decoded = keras.layers.Dense(input_dim, activation='linear')(encoded)

    model = keras.Model(inputs=inp, outputs=decoded)
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, X, epochs=20, batch_size=256, validation_split=0.1)

    return model, train_df.columns.tolist()