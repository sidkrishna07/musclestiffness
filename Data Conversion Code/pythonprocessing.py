import scipy.io
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import MinMaxScaler

# High-pass filter function
def highpass_filter(data, cutoff, fs):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(4, normal_cutoff, btype='high', analog=False)  # Using a 4th order Butterworth filter
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# Lag calculation function
def find_lag(x, y):
    corr = np.correlate(x - np.mean(x), y - np.mean(y), mode='full')
    lag = np.argmax(corr) - len(x) + 1
    return lag

# Exponential fitting function (adjusted for accurate fitting)
def exponential_fitting(d1, d2, d3, fs):
    decay_rate = np.mean([
        np.polyfit(np.arange(len(d1)), np.log(np.abs(d1) + 1e-10), 1)[0],
        np.polyfit(np.arange(len(d2)), np.log(np.abs(d2) + 1e-10), 1)[0],
        np.polyfit(np.arange(len(d3)), np.log(np.abs(d3) + 1e-10), 1)[0]
    ])
    amp = np.max([np.max(np.abs(d1)), np.max(np.abs(d2)), np.max(np.abs(d3))])
    return amp, decay_rate

# Bandwidth calculation using power spectral density
def calculate_bandwidth(signal):
    f_signal = np.fft.fft(signal)
    power_spectrum = np.abs(f_signal)**2
    freqs = np.fft.fftfreq(len(signal), d=1/540)
    positive_freqs = freqs[:len(freqs)//2]
    power_spectrum = power_spectrum[:len(power_spectrum)//2]
    bandwidth = positive_freqs[np.argmax(power_spectrum)]
    return bandwidth

# Feature extraction function
def extract_features(d1, d2, d3, fs):
    corrs = [np.corrcoef(d1, d2)[0, 1], np.corrcoef(d2, d3)[0, 1], np.corrcoef(d1, d3)[0, 1]]
    lags = [find_lag(d1, d2), find_lag(d2, d3), find_lag(d1, d3)]
    amp, decay_var = exponential_fitting(d1, d2, d3, fs)
    bandwidths = [calculate_bandwidth(d1), calculate_bandwidth(d2), calculate_bandwidth(d3)]
    e12 = np.sum(d1**2) / (np.sum(d2**2) + 1e-10)
    e23 = np.sum(d2**2) / (np.sum(d3**2) + 1e-10)
    e13 = np.sum(d1**2) / (np.sum(d3**2) + 1e-10)
    
    features = [np.mean(corrs), np.mean(lags), amp, decay_var, *bandwidths, e12, e23, e13]
    return features

# Load and process data
file_paths = [
    "kevin2/0420_kevin_postural_0_VIB_ONLY.mat",
    "kevin2/0420_kevin_postural_10_VIB_ONLY.mat",
    "kevin2/0420_kevin_postural_20_VIB_ONLY.mat",
    "kevin2/0420_kevin_postural_30_VIB_ONLY.mat",
    "kevin2/0420_kevin_postural_40_VIB_ONLY.mat",
    "kevin2/0420_kevin_postural_50_VIB_ONLY.mat"
]

Fs = 540  # Sampling frequency
pass_freq = 50  # High-pass filter cutoff

all_features = []
all_labels = []

for label, file_path in enumerate(file_paths):
    mat_data = scipy.io.loadmat(file_path)
    
    D1, D2, D3 = mat_data['D1'], mat_data['D2'], mat_data['D3']
    
    for i in range(D1.shape[0]):
        d1 = highpass_filter(D1[i, :], pass_freq, Fs)
        d2 = highpass_filter(D2[i, :], pass_freq, Fs)
        d3 = highpass_filter(D3[i, :], pass_freq, Fs)
        
        features = extract_features(d1, d2, d3, Fs)
        all_features.append(features)
        all_labels.append(label * 10)

# Normalize features
scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(all_features)
feature_mins = scaler.data_min_
feature_maxs = scaler.data_max_
print("Feature mins:", feature_mins)
print("Feature maxs:", feature_maxs)


# Save to CSV
output_df = pd.DataFrame(normalized_features, columns=['Corr', 'Lag', 'Amp', 'Decay_Var', 'BW1', 'BW2', 'BW3', 'E12', 'E23', 'E13'])
output_df['Label'] = all_labels
output_df.to_csv('adjusted_preprocessed_kevin2_data.csv', index=False)
