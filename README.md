MuscleStiffness

Authors: Siddharth KrishnaAdvisor: Professor Shijia PanLab: Persuasive Autonomous Networks LabPhD Student: Shubham Rohal

A comprehensive, end-to-end system for measuring, modeling, and visualizing muscle stiffness. It integrates:

MATLAB preprocessing of raw vibration/IMU recordings (FinePose pipeline)

Python/PyTorch unified regression model training & verification

Arduino Nano 33 BLE peripheral sketch (vibration excitation → IMU sampling → BLE notification)

Android Jetpack Compose central app (BLE receive → normalization → model inference → UI visualization)

File & Component Breakdown

Data & MATLAB Original Code

data/matlab_original_code/FinePose.pdfResearch paper describing the FinePose method for haptic vibration and muscle profiling.

data/matlab_original_code/data_process.mSegments raw 100 Hz CSV recordings into vibration-only windows (*_VIB_ONLY.mat) via high-pass filtering and sliding-window event detection.

data/matlab_original_code/features.mExtracts per-event features: autocorrelation lags, signal‑to‑noise ratios across channels, and frequency‑energy bands; outputs 10-dimensional vectors.

data/matlab_original_code/regression.mFits ordinary‐least‐squares (OLS) models per subject: stiffness = a·x + b, where x is the feature vector.

Auxiliary scripts (selfCorr.m, new_ECB_factor.m, exFitting.m) for cross-validation, error metrics, and exploratory analyses.

Processed MATLAB Data

data/processed_matlab_data/*.mat.mat files containing segmented vibration windows.

data/processed_matlab_data/*_preprocessed.csvCSV exports of feature vectors (10 floats per event) ready for Python ingestion.

Python Model Training & Visualization

Located under data_conversion_code/:

pythonprocessing.pyLoads all preprocessed CSVs, merges into a master dataset, and prepares train/test splits.

pytorchmodel.pyDefines a feed‐forward regression network in PyTorch, trains to minimize mean squared error, and saves universal_model.pt.

verify.pyLoads the checkpoint, evaluates on held‑out data, and reports R² and MAE.

visualization.pyGenerates plots of training/validation loss curves and feature importance histograms.

debug_data.pyUtility for sanity‐checking data distributions and catching anomalies.

Arduino Peripheral Sketch

Located under arduino_code/musclestiff.ino:

Hardware: Arduino Nano 33 BLE (LSM9DS1 IMU onboard) + coin vibration motor on pin D9.

Workflow:

Advertise BLE service 0000FFFF-0000-1000-8000-00805F9B34FB with characteristic 0000FFFE-0000-1000-8000-00805F9B34FB (40 bytes for ten floats).

Every 5 s, turn motor ON for 1 s, then OFF.

After motor OFF, sample IMU acceleration 10×, compute magnitude √(x²+y²+z²), and store in features[10].

Write & notify the characteristic with the 40-byte buffer.

Note: Default BLE MTU=23 bytes will truncate 40 bytes; central must request a larger MTU.

Android Jetpack Compose App

Core Kotlin Classes

BLEManager.kt (com.example.musclestiffness.utils)

Manages scanning (BluetoothLeScanner), connect via device.connectGatt(..., gattCallback), and BLE notifications.

Negotiates MTU (gatt.requestMtu(64)) in onConnectionStateChange, waits for onMtuChanged before service discovery.

Parses incoming 40 B payload into FloatArray(10) and forwards via a callback.

BleViewModel.kt (com.example.musclestiffness.ble)

Holds a MutableStateFlow<FloatArray?> called features.

Exposes startScan() (called post-permissions) and disconnect() in onCleared().

ModelHelper.kt (com.example.musclestiffness.utils)

Loads PyTorch model from assets/universal_model.pt into internal storage.

predict(input: FloatArray) → FloatArray?: converts to tensor, runs inference, clamps output [0,1].

MainActivity.kt

Requests BLUETOOTH_SCAN, BLUETOOTH_CONNECT, ACCESS_FINE_LOCATION at runtime.

On grant → bleVm.startScan(); otherwise shows permission denial.

Sets Compose content via AppNavigation(autoFeatures = featuresFlow).

UI Composables (ui/)

AppNavigation: three-step onboarding (Welcome, Connect, Ready) before MuscleStiffnessUI.

MuscleStiffnessUI(autoFeatures: FloatArray):

Normalizes features via stored featureMins & featureMaxs.

Auto-infers on BLE arrival (LaunchedEffect).

Manual fallback: OutlinedTextField accepts 10 comma-separated floats; “Check Stiffness” button enables on non-blank input and enforces exactly 10 values.

CircularGaugeCompose & BodyMap: visualize probability (%) and allow region-specific advice popups.

Build & Dependencies

Gradle plugins: Android application, Kotlin Android, Kotlin Compose.

Compose BOM + Material3 + lifecycle + activityCompose.

PyTorch Android (org.pytorch:pytorch_android, torchvision).

Min SDK: 27, Target SDK: 35, Kotlin JVM 11, Compose compiler 1.4.0.

Features & Functionality

✅ Implemented & Working

MATLAB → Python reproducibility: feature vectors, model training, checkpointing.

Arduino: vibration motor control, IMU sampling, BLE notifications (with MTU negotiation).

Android:

Permissions flow and BLE scan/connect.

MTU negotiation ensures full 10‐float payload.

Real‐time inference via PyTorch Android.

Manual input fallback.

Interactive UI: gauge + body‐map with advice.

⚠️ Known Issues & Roadmap

BLE connection stability:

Symptom: peripheral disconnects quickly; Android stops receiving notifications.

Diagnosis: using autoConnect=false, no rescan after connect.

Next Steps: switch to autoConnect=true, implement background watchdog/periodic rescans.

MTU negotiation timing:

Symptom: truncated payloads if discoverServices() runs before MTU change.

Validation: Logs in onMtuChanged should precede onServicesDiscovered.

UX improvements:

Display scan/connection status to user.

Add persistent logs of received features (for later offline debugging).

Implement retry/backoff when permissions denied or Bluetooth OFF.

Model enhancements:

Evaluate non‐linear regressors (Random Forest, small neural networks).

On-device fine‐tuning for subject personalization.

How to Run

Preprocess & trainFollow the MATLAB and Python steps above to regenerate universal_model.pt.

Flash ArduinoUpload arduino_code/musclestiff.ino to Nano 33 BLE; verify Serial logs.

Build & install Android

Open code/latest_code/app/ in Android Studio.

Place universal_model.pt in app/src/main/assets/.

Run on a physical device, grant BLE permissions, and test.

Acknowledgements & License

Chagas et al., “FinePose: Fine-grained Haptic Vibration for Muscle Profiling,” ACM BodySys ’22.

ArduinoBLE & PyTorch Android libraries.

Persuasive Autonomous Networks Lab, Professor Shijia Pan, PhD student Shubham Rohal.

License: MIT © Siddharth Krishna. Feel free to fork and adapt for research.

