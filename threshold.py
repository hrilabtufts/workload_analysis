import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy.io
from scipy.signal import butter, lfilter, find_peaks

from plotly.subplots import make_subplots

import numpy as np
import pandas as pd

import random

import pathlib
import imageio

import glob
import csv
import joblib
import math as math

from argparse import ArgumentParser
from os.path import join, exists, isfile, isdir, basename
from os import listdir

parser = ArgumentParser(description='Script for calculating threshold value from eye tracking data')
parser.add_argument('client_data', type=str, help='Client data directory to process events files from')
parser.add_argument('server_data', type=str, help='Server data directory to process events files from')
args = parser.parse_args()

if not exists(args.client_data) :
    print(f'ERROR: Directory {args.client_data} does not exist')
    exit(1)

if not isdir(args.client_data) :
    print(f'ERROR: {args.client_data} is not a directory')
    exit(2)

if not exists(args.server_data) :
    print(f'ERROR: Directory {args.server_data} does not exist')
    exit(3)

if not isdir(args.server_data) :
    print(f'ERROR: {args.server_data} is not a directory')
    exit(4)

files = listdir(args.client_data)
globals = {
	'predict' : None
}

for file in files :
	if '_Eye_Tracking_Raw' in file :
		args.eye_tracking_raw = join(args.client_data, file)
	if '_Eye_Tracking_Events' in file :
		args.eye_tracking_events = join(args.client_data, file)

files = listdir(args.server_data)

for file in files :
	if '_Space_Station_Events' in file :
		args.station_events = join(args.server_data, file)

#print(basename(args.eye_tracking_raw))
#print(basename(args.eye_tracking_events))
#print(basename(args.station_events))

def find_closest_val(arr, val):
    n = [abs(k-val) for k in arr]
    ind = n.index(min(n))
    return arr[ind], ind

def preprocess_pupil(y):
    
    # First, use thresholding for logical pupil dilation
    for k in range(0, len(y)):
        if (y[k] < 0.8) or (y[k] > 10):
            y[k] = 'nan'

    # Then, use interpolation to fill out NaN values
    y_interpolated = np.interp(np.arange(len(y)), np.arange(len(y))[np.isnan(y) == False], y[np.isnan(y) == False])
    
    return y_interpolated

def fit_luminance_function(gaze_array):
    x = np.linspace(0, 1, len(gaze_array), endpoint=True)
    y = gaze_array
    
    # Using polyfit() to generate coefficients
    coeffs = np.polyfit(x, y, 3)  # Last argument is degree of polynomial

    # Define a polynomial function based on the coeffs
    globals['predict'] = np.poly1d(coeffs)

def predict_pupil_size (luminance) :
	return globals['predict'](luminance)

calibration_steps=[]
with open(args.eye_tracking_events, 'r') as file :
    for l in file:
        line = l.strip()
        cols = line.split(',')
        if cols[2].startswith('pupil_calibration_') :
            if '_started' in cols[2] :
                continue
            parts = cols[2].split('_')
            if parts[2] == '0' :
            	continue
            if parts[2] == 'ended' :
            	parts[2] = '270'
            calibration_step = {
                'time' : float(cols[0]),
                'step' : int(parts[2]) - 15
            }
            calibration_steps.append(calibration_step)

raw=[]
pupil_at_incrementation = []
with open(args.eye_tracking_raw, 'r') as file :
    for l in file:
        line = l.strip()
        if line.startswith('unityClientTimestamp,') :
            continue
        cols = line.split(',')
        raw.append([ float(cols[0]), float(cols[15]) ])

c = 0
for i in range(len(raw)) :
    cols = raw[i]
    if c == len(calibration_steps) :
        break
    if cols[0] >= calibration_steps[c]['time'] :
        pupil = cols[1]
        if pupil < 0 :
            j = 1
            while pupil < 0 :
                pupil = raw[i-j][1]
                j += 1
        pupil_at_incrementation.append(pupil)
        c+=1

print(pupil_at_incrementation)

space_station_events = [];

with open(args.station_events, mode ='r') as file:
    csvFile = csv.reader(file)
    count = 0;
    for lines in csvFile:
        space_station_events.append([lines[0], lines[2]])

space_station_events = np.array(space_station_events)

for i in range(0, len(space_station_events)):
    if space_station_events[i, 1] == 'Trial_3 started':
        start_trial_time = float(space_station_events[i][0])
    else:
        continue

#print(start_trial_time)

eye_gaze_left = []
eye_gaze_right = []
luminance = []
LSL = []

with open(args.eye_tracking_raw, mode ='r') as file:
    csvFile = csv.reader(file)
    count = 0;
    for lines in csvFile:
        if count > 0:
            LSL.append(float(lines[0]))
            eye_gaze_left.append(float(lines[15]))
            eye_gaze_right.append(float(lines[16]))
            luminance.append(float(lines[23]))
        count += 1;
        
eye_gaze_left = np.array(eye_gaze_left)
eye_gaze_right = np.array(eye_gaze_right)
eye_gaze_left_preprocessed = preprocess_pupil(eye_gaze_left)
eye_gaze_right_preprocessed = preprocess_pupil(eye_gaze_right)
luminance = np.array(luminance)
LSL = np.array(LSL)

start_trial_LSL, ind_start_trial_LSL = find_closest_val(LSL, start_trial_time)

WL_LSL = LSL[ind_start_trial_LSL:-1]
WL_luminance = luminance[ind_start_trial_LSL:-1]
WL_left_eye_gaze = eye_gaze_left[ind_start_trial_LSL:-1]
WL_right_eye_gaze = eye_gaze_right[ind_start_trial_LSL:-1]

WL_left_eye_gaze_preprocessed = preprocess_pupil(WL_left_eye_gaze)
WL_right_eye_gaze_preprocessed = preprocess_pupil(WL_right_eye_gaze)

WL_left_eye_gaze_without_lum = []
WL_right_eye_gaze_without_lum = []

fit_luminance_function(pupil_at_incrementation)

# For every point of the workload, extract the luminance effect
for s in range(0, len(WL_left_eye_gaze)):
    
    # Calculate the RGB value of the screen luminance
    RGB_value = WL_luminance[s]
    # print(RGB_value)
    # print(pupil_at_incrementation)
    
    # Calculate the corresponding luminance effect on eye gaze using fit function
    eye_gaze_pred = predict_pupil_size(RGB_value)
    # print("Predicted Eye Gaze: ", eye_gaze_pred)

    # Now, extract this value from the real workload eye gaze value
    WL_left_eye_gaze_without_lum.append(WL_left_eye_gaze_preprocessed[s] - eye_gaze_pred)
    WL_right_eye_gaze_without_lum.append(WL_right_eye_gaze_preprocessed[s] - eye_gaze_pred)

WL_left_eye_gaze_without_lum = np.array(WL_left_eye_gaze_without_lum)
WL_right_eye_gaze_without_lum = np.array(WL_right_eye_gaze_without_lum)

N=150
y_padded_left = np.pad(WL_left_eye_gaze_without_lum, (N//2, N-1-N//2), mode='edge')
eye_gaze_moving_avg_left = np.convolve(y_padded_left, np.ones((N,))/N, mode='valid')

y_padded_right = np.pad(WL_right_eye_gaze_without_lum, (N//2, N-1-N//2), mode='edge')
eye_gaze_moving_avg_right = np.convolve(y_padded_right, np.ones((N,))/N, mode='valid')

window_size = 50

# Modify the signal to ignore negative parts
modified_signal = np.where(eye_gaze_moving_avg_left > 0, eye_gaze_moving_avg_left, 0)

# Compute power of the modified signal
power = modified_signal**2
moving_avg_power = np.convolve(power, np.ones(window_size)/window_size, mode='valid')

# Define threshold
threshold = np.mean(moving_avg_power) + 2 * np.std(moving_avg_power)

# Find peaks in the moving average power
peaks, _ = find_peaks(moving_avg_power, height=threshold)

# Plot results
plt.figure(figsize=(20, 12))

# Plot time series
plt.subplot(2, 1, 1)
plt.plot(eye_gaze_moving_avg_left, label='Original Time Series', color='grey')
plt.title('Original Time Series', fontsize=24)
plt.xticks(fontsize=18); plt.yticks(fontsize=18)
plt.legend()

# Plot moving average power
plt.subplot(2, 1, 2)
x_vals = np.arange(window_size-1, len(moving_avg_power)+window_size-1)
plt.plot(x_vals, moving_avg_power, label='Moving Average Power', color='orange')
plt.scatter(x_vals[peaks], moving_avg_power[peaks], color='red', label='Detected Peaks', s=100)
plt.axhline(y=threshold, color='green', linestyle='--', label='Threshold')
formatter = mpl.ticker.StrMethodFormatter("{x:.0f}")
plt.title('Moving Average Power with Detected Peaks', fontsize=24)
plt.xticks(fontsize=18); plt.yticks(fontsize=18)
plt.legend()

plt.tight_layout()
plt.show()

print(f'Threshold: {threshold}')
