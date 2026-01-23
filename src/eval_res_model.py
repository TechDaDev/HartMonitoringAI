import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import os

def relabel(y):
    return np.where(y == 0, 0, 1)

def normalize_per_sample(X):
    X_norm = np.zeros_like(X, dtype=np.float32)
    for i in range(X.shape[0]):
        row = X[i, :].reshape(-1, 1)
        scaler = StandardScaler()
        X_norm[i, :] = scaler.fit_transform(row).ravel()
    return X_norm

def main():
    test_path = "./data/mitbih_test.csv"
    test_df = pd.read_csv(test_path, header=None)
    X_test = test_df.iloc[:, :-1].values
    y_test = relabel(test_df.iloc[:, -1].values)
    X_test_cnn = normalize_per_sample(X_test)[..., np.newaxis]

    model = tf.keras.models.load_model("mitbih_residual_cnn.h5")
    
    print("\n--- Final Evaluation ---")
    results = model.evaluate(X_test_cnn, y_test, verbose=0)
    
    print(f"Test Accuracy  : {results[1]:.4f}")
    print(f"Test Recall    : {results[2]:.4f}")
    print(f"Test Precision : {results[3]:.4f}")

if __name__ == "__main__":
    main()
