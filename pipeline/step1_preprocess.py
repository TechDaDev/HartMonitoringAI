import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os

# Set matplotlib to non-interactive mode
plt.switch_backend('Agg')

def main():
    # Step 1.1 — Load training & test data
    print("--- Step 1.1: Loading Data ---")
    train_path = "./data/mitbih_train.csv"
    test_path = "./data/mitbih_test.csv"

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Error: Dataset files not found in ./data/")
        return

    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    # Step 1.2 — Split signals and labels
    print("\n--- Step 1.2: Splitting Signals and Labels ---")
    X_train = train_df.iloc[:, :-1].values
    y_train_raw = train_df.iloc[:, -1].values

    X_test = test_df.iloc[:, :-1].values
    y_test_raw = test_df.iloc[:, -1].values

    print("X_train shape:", X_train.shape)
    print("y_train_raw shape:", y_train_raw.shape)

    # Step 1.3 — Relabel (Normal vs Abnormal)
    print("\n--- Step 1.3: Relabeling (Normal vs Abnormal) ---")
    def relabel(y):
        return np.where(y == 0, 0, 1)

    y_train = relabel(y_train_raw)
    y_test = relabel(y_test_raw)

    print("Train distribution:", Counter(y_train))
    print("Test distribution:", Counter(y_test))

    # Step 1.4 — Basic signal sanity check
    print("\n--- Step 1.4: Sanity Check (Plotting) ---")
    plt.figure(figsize=(10, 4))
    plt.plot(X_train[0])
    plt.title("Example ECG Beat (Normal)")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    # Save the plot instead of showing it
    plot_filename = "plots/ecg_sanity_check.png"
    plt.savefig(plot_filename)
    print(f"Sanity check plot saved as {plot_filename}")

    print("\n✅ Step 1 Accomplished Successfully")

if __name__ == "__main__":
    main()
