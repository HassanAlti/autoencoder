import pickle
from tensorflow import keras

# You’ll overwrite this list in Preprocess, but useful as placeholder
PREDICTORS = []
TARGET = ['isFraud']


def save_model(model, columns, output_dir='../output'):
    model.save(f"{output_dir}/deep-ae-model")
    with open(f"{output_dir}/columns.mapping", 'wb') as f:
        pickle.dump(columns, f)
    return True


def load_model(model_path, output_dir='../output'):
    model = keras.models.load_model(model_path)
    with open(f"{output_dir}/columns.mapping", 'rb') as f:
        columns = pickle.load(f)
    return model, columns