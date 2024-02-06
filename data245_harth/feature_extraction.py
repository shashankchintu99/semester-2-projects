import os
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn import manifold, preprocessing


class_code_to_class_name = {
    1: 'walking',
    2: 'running',
    3: 'shuffling',
    4: 'stairs (ascending)',
    5: 'stairs (descending)',
    6: 'standing',
    7: 'sitting',
    8: 'lying',
    13: 'cycling (sit)',
    14: 'cycling (stand)',
    130: 'cycling (sit, inactive)',
    140: 'cycling (stand, inactive)',
}

class_code_to_id = {}
for i, code in enumerate(class_code_to_class_name.keys()):
    class_code_to_id[code] = i

def get_csvs(data_path):
    '''Gets all the csvs from a directory.'''
    files = os.listdir(data_path)
    csv_paths = []
    for f in files:
        path = os.path.join(data_path, f)
        csv_paths.append(path)
    return csv_paths


def fft_mag(X):
    fft = np.fft.rfft(X, axis=0)
    mag = np.abs(fft)
    return mag

def create_windowed_features(X, window_size, step_size):
    outputs = []
    N = len(X)
    for i in range(0, N, step_size):
        features = []
        window = X[i: i + window_size]

        # Compute norm of the acceleration.
        back_magnitude = np.linalg.norm(window[:, :3], axis=1, keepdims=True)
        thigh_magnitude = np.linalg.norm(window[:, 3:], axis=1, keepdims=True)

        # Concatenate with the original feature.
        window = np.concatenate([window, back_magnitude, thigh_magnitude], axis=1)

        # Calculate gravity using low pass filter.
        b, a = scipy.signal.butter(4, 1, fs=50, btype='lowpass')
        gravity = scipy.signal.filtfilt(b, a, window, axis=0)
        
        # Remove gravity.
        window = window - gravity

        #####  Create features #####

        # Mean of gravity.
        gravity_mean = np.mean(gravity, axis=0)
        features.append(gravity_mean)

        # Standard deviation of gravity.
        gravity_std = np.std(gravity, axis=0)
        features.append(gravity_std)

        # Coefficient of variation of gravity.
        # The coefficient of variation is the standard deviation divided by the mean.
        variation_gravity = scipy.stats.variation(gravity, axis=0)
        variation_gravity[np.isnan(variation_gravity)] = 0
        features.append(variation_gravity)

        # Total energy of the signal.
        energy = np.sum(window ** 2, axis=0)
        features.append(energy)

        # Mean of the signal.
        mean = np.mean(window, axis=0)
        features.append(mean)

        # Standard deviation of the signal.
        std = np.std(window, axis=0)
        features.append(std)

        # Fourier transform.
        fft = fft_mag(window)

        # Total frequency power.
        fft_power = np.sum(fft ** 2, axis=0)
        features.append(fft_power)

        # fft magnitude mean.
        fft_mean = np.mean(fft, axis=0)
        features.append(fft_mean)

        # fft magnitude std.
        fft_std = np.std(fft, axis=0)
        features.append(fft_std)

        # Frequency at which fft is maximum.
        fft_freq = np.fft.rfftfreq(window.shape[0])
        fft_max_freq = fft_freq[fft.argmax(axis=0)]
        features.append(fft_max_freq)

        # Maximum fft.
        fft_max = fft.max(axis=0)
        features.append(fft_max)

        features = np.concatenate(features)
        outputs.append(features)
    return np.array(outputs)

def create_windowed_labels(y, window_size, step_size):
    labels = []
    N = len(X)
    for i in range(0, N, step_size):
        features = []
        window = y[i: i + window_size]
        counts = np.bincount(window)
        label = counts.argmax()
        labels.append(label)
        
    return np.array(labels)

data_path = './harth/'
csv_paths = get_csvs(data_path)
feature_columns=['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']
window_size = 250
step_size = 250
sample_rate = 50

features = []
labels = []
for path in csv_paths:
    print(f'Processing {path}')
    df = pd.read_csv(path)
    X = df[feature_columns].to_numpy()
    y = df['label'].to_numpy()

    X_feat = create_windowed_features(X, window_size, step_size)
    features.append(X_feat)
    y = create_windowed_labels(y, window_size, step_size)
    labels.append(y)
    
features = np.concatenate(features, axis=0)
labels = np.concatenate(labels, axis=0)

print(features.shape)
print(labels.shape)
np.save('features.npy', features)
np.save('labels.npy', labels)