import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy.signal import butter, sosfiltfilt
from datetime import datetime
import scipy.signal as sig
import pylsl
import collections
from pynput.keyboard import Key, Controller

max_len = 180 # modify the number of samples for analysis here
threshold_after_first_peak = 60 # how many data points do we search after the first strong peak above threshold
threshold = 100
def mark_blink(peaks):
    """
    Applies a high pass filter to the voltage data in the dataframe.

    Parameters:
    data: A 1D list containing the indecies of the peak.

    Returns:
    A list containing the start time and end time of a peak
    """
    blink_times = []
    for i in range(len(peaks)):
        start = peaks[i]
        if i + 1 < len(peaks):
            end = peaks[i + 1]
        else:
            end = peaks[0] + max_len
        blink_times.append((start, end))
    return blink_times
def extract_features(blink_times, time, eeg_data):
    """
    Extract different features of the peaks.

    Parameters:
    data: A 1D list containing the start time and end time of a peak, the time data, and the 
    corresponding values of data

    Returns:
    A list containing the duration, the peak, and the number of points in the interval
    """
    blink_features = []
    for blink in blink_times:
        duration = time[blink[1]] - time[blink[0]]
        peak = eeg_data[blink[0]]
        frequency = blink[1] - blink[0]
        blink_features.append((duration, peak, frequency))
    return blink_features
def separate_blink(blink_features):
    """
    Separate single blink from double blink.

    Parameters:
    data: blink_features

    Returns:
    A list that stores all single blink, a list that stores all double blink.
    """
    single_blink = []
    double_blink = [] 
    for i in range(len(blink_features)):
        if blink_features[i][0] < 1:
            if i + 1 < len(blink_features) and blink_features[i+1][0] < 1:
                duration = blink_features[i][0] + blink_features[i+1][0]
                peak = max(blink_features[i][1], blink_features[i+1][1])
                frequency = blink_features[i][2] + blink_features[i+1][2]
                double_blink.append((duration, peak, frequency))
            elif i + 1 < len(blink_features) and blink_features[i+1][0] >= 1:
                duration = blink_features[i][0]
                peak = max(blink_features[i][1], blink_features[i+1][1])
                frequency = blink_features[i][2]
                single_blink.append((duration, peak, frequency))
    return single_blink, double_blink

# extract data from the file 
def invert_data(data):
    # negative values become positive, while positive values become negative in channel 1 and channel 2
    clean_data = [[], [], []]
    for i in range(len(data)):
        ch1 = data[i][1]
        ch2 = data[i][2]
        time = data[i][-2]
        clean_data[0].append(-ch1)
        clean_data[1].append(-ch2)
        clean_data[2].append(time)
    return clean_data

def find_peak(data, threshold):
    peaks_1, _ = sig.find_peaks(data[0], height = threshold)
    peaks_2, _ = sig.find_peaks(data[1], height = threshold)
    return peaks_1, peaks_2


def statistical_classification(peaks_1, peaks_2, threshold, data):
    # peaks_1, peaks_2 stores the indecies of the array, in which indecies a peak occurs
    peaks_1 = np.array(peaks_1)
    peaks_2 = np.array(peaks_2)
    # store the peak and its respective index into a a list of tuple
    data_ch1 = [(data[0][i], i) for i in peaks_1]
    data_ch2 = [(data[1][i], i) for i in peaks_2]
    # find the maximum of the the value and return index of the peak in this list of tuple
    max_ch1_idx = np.argmax(data_ch1)[0]
    max_ch2_idx = np.argmax(data_ch2)[0]
    # use the peak index to find the initial index of the peak in the data
    ch1_idx = data_ch1[max_ch1_idx][1]
    ch2_idx = data_ch2[max_ch2_idx][1]

    # look for the biggest peak and check the peak right after it is above threshold or not 
    if data[ch1_idx] >= threshold:
        next_peak = np.argmax(data[ch1_idx : np.min(ch1_idx+threshold_after_first_peak, len(data))])
        if data[next_peak] >= threshold:
            return 1
    # look for the biggest peak and check the peak right after it is above threshold or not 
    if data[ch2_idx] >= threshold:
        next_peak = np.argmax(data[ch2_idx : np.min(ch2_idx+threshold_after_first_peak, len(data))])
        if data[next_peak] >= threshold:
            return 1
    
    return 0

def lsl_inlet(name):
    inlet = None
    tries = 0
    info = pylsl.resolve_stream('name', name)
    inlet = pylsl.stream_inlet(info[0], recover = False)
    print(f'backend has received the {info[0].type()} inlet.')
    return inlet

def main():
    terminate_backend = False
    keyboard = Controller() # setup virtual keyboard
    # Wait for a marker, then start recording EEG data
    data = collections.deque(maxlen=max_len) # fast datastructure for appending/popping in either direction
    #subject_threshold = pd.read_csv('Subject_1_average_threashold.csv')
    #max_threshold = subject_threshold['']
    
    print('main function started')
    while True and terminate_backend == False:
        # Constantly check for a marker
        eeg, t_eeg = eeg_in.pull_sample(timeout=0)
        if eeg is not None:
            data.append(eeg)
        if len(data) == max_len:
            # classify this chunk
            #------code starts here------#
            data_processed = invert_data(data)
            peaks_1, peaks_2 = find_peak(data_processed)
            label = statistical_classification(peaks_1, peaks_2, 250, data_processed)
            # if it returns 1, it will press the spacebar
            if label == 1: # some function that return label of the data
                keyboard.press(Key.space)
                keyboard.release(Key.space)   
             # otherwise do nothing
            data = collections.deque(maxlen=max_len)

# initialize variables to store stream
eeg_in = None

if __name__ == "__main__":
    # Initialize our streams
    eeg_in = lsl_inlet('dino_EEG')
    # Run out main function
    main()
