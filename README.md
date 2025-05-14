# MuscleStiffness

**Authors:**  
- Siddharth Krishna  
- Advisor: Professor Shijia Pan  
- Lab: Persuasive Autonomous Networks Lab  
- PhD Student: Shubham Rohal  

A comprehensive, end-to-end system for measuring, modeling, and visualizing muscle stiffness. It integrates:

1. **MATLAB** preprocessing of raw vibration/IMU recordings (FinePose pipeline)  
2. **Python/PyTorch** unified regression model training & verification  
3. **Arduino Nano 33 BLE** peripheral sketch (vibration excitation → IMU sampling → BLE notification)  
4. **Android Jetpack Compose** central app (BLE receive → normalization → model inference → UI visualization)  

---

## 📂 File & Component Breakdown

### Data & MATLAB Original Code

| File                                               | Purpose                                                                                 |
|----------------------------------------------------|-----------------------------------------------------------------------------------------|
| `data/matlab_original_code/FinePose.pdf`           | Research paper describing the FinePose method                                           |
| `data/matlab_original_code/data_process.m`         | Segment raw 100 Hz CSV into vibration-only windows (`*_VIB_ONLY.mat`)                   |
| `data/matlab_original_code/features.m`             | Extract per-event features (autocorr, SNR, freq-energy → 10-D vectors)                  |
| `data/matlab_original_code/regression.m`           | Fit per-subject OLS models: stiffness = a·x + b                                         |
| `data/matlab_original_code/selfCorr.m`, ...        | Auxiliary scripts for cross-validation, error metrics, exploratory analyses             |

### Processed MATLAB Data

- `data/processed_matlab_data/*.mat`  
  `.mat` files containing segmented vibration windows  

- `data/processed_matlab_data/*_preprocessed.csv`  
  CSVs of 10-D feature vectors ready for Python ingestion  

### Python Model Training & Visualization

_All located under `data_conversion_code/`_

```text
pythonprocessing.py    # Merge all CSVs → master DataFrame + train/test splits
pytorchmodel.py        # Define & train feed-forward regression network → universal_model.pt
verify.py              # Evaluate on held-out data (R², MAE)
visualization.py       # Plot training/validation loss and feature importance
debug_data.py          # Sanity-check data distributions and anomalies



