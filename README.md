# 🫀 Heart Condition Detection using AI

This project implements a deep learning pipeline for detecting cardiac abnormalities from ECG signals using the MIT-BIH Arrhythmia Dataset. It features a Paper-Inspired Deep Residual 1D CNN and a real-time monitoring dashboard.

## 📂 Project Structure

- **`data/`**: Raw MIT-BIH and PTBDB CSV datasets.
- **`models/`**: Trained model artifacts (`.h5` files).
- **`pipeline/`**: Phase-by-phase development scripts (Step 1 to 5).
- **`plots/`**: Generated charts and confusion matrices.
- **`src/`**: Utility scripts and evaluation tools.
- **`app.py`**: The main Streamlit dashboard for real-time monitoring.
- **`requirements.txt`**: List of Python dependencies.

## 🚀 How to Run

### 1. Installation
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Training Pipeline (Optional)
The pipeline is pre-run, but you can re-run any step from the root:
```bash
python pipeline/step4_residual_cnn.py
```

### 3. Launch the Monitoring App
Run the real-time simulation dashboard:
```bash
streamlit run app.py
```

## 🧠 Model Architecture
The system uses a **Deep Residual 1D Convolutional Neural Network** with:
- 5 Residual Blocks with skip connections.
- Batch Normalization and ReLU activation.
- Dropout for regularization.
- Binary sigmoid output for Normal vs. Abnormal classification.

## 🩺 Clinical Validation
The model achieves:
- **Accuracy**: ~98.0%
- **Sensitivity (Recall)**: ~97.0% (at threshold 0.2)
- **Specificity**: ~96.8%

*Note: This project is for educational purposes and is not a medical device.*
