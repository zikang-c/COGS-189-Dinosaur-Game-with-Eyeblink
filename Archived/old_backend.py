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

max_len = 375# modify the number of samples for analysis here
threshold_after_first_peak = 88 # how many data points do we search after the first strong peak above threshold
subject = 'Filtered_Subject_1\Subject_1_average_threashold.csv'

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
    clean_data = [[], []]
    for i in range(len(data)):
        ch1 = data[i][0]
        ch2 = data[i][1]
        clean_data[0].append(-ch1)
        clean_data[1].append(-ch2)
    return clean_data

def find_peak(data, threshold):
    peaks_1, _ = sig.find_peaks(data[0], height = threshold[0])
    peaks_2, _ = sig.find_peaks(data[1], height = threshold[1])
    print("peaks_1 in find_peak: ", peaks_1)
    return peaks_1, peaks_2


def statistical_classification(peaks_1, peaks_2, threshold, data):
    # peaks_1, peaks_2 stores the indecies of the array, in which indecies a peak occurs
    # peaks_1 = np.array(peaks_1)
    # peaks_2 = np.array(peaks_2)
    if len(peaks_1) == 0 or len(peaks_2) == 0:
        return 0
    print("peaks_1 in statistical_classification: ", peaks_1)
    # store the peak and its respective index into a a list of tuple
    data_ch1 = [(data[0][i], i) for i in peaks_1]
    data_ch2 = [(data[1][i], i) for i in peaks_2]
    data_ch1 = np.array(data_ch1)
    data_ch2 = np.array(data_ch2)
    print("data_ch1 in statistical_classification: ", data_ch1)
    print("data_ch2 in statistical_classification: ", data_ch2)
    # find the maximum of the the value and return index of the peak in this list of tuple
    
    max_ch1_idx = np.argmax(data_ch1[:,0])

   
    max_ch2_idx = np.argmax(data_ch2[:,0])
    # max_ch1_idx = np.argmax([data_ch1[0]])
    # max_ch2_idx = np.argmax([data_ch2[0]])
    # use the peak index to find the initial index of the peak in the data
    print(max_ch1_idx)
    print(len(data_ch1))
    print(data_ch1[max_ch1_idx])
    ch1_idx = int(data_ch1[max_ch1_idx][1])
    ch2_idx = int(data_ch2[max_ch2_idx][1])
    print("ch1_idx is", ch1_idx)
    #print("data is", data)
    # look for the biggest peak and check the peak right after it is above threshold or not 
    if data[0][ch1_idx] >= threshold[0]:
        print("start idx is ", ch1_idx)
        
        print("length of data[0]", len(data[0]))
        end_idx = min(ch1_idx+threshold_after_first_peak, len(data[0]))
        print("end_idx is", end_idx)
        print(data[0][ch1_idx : end_idx])
        next_peak = np.argmax(data[0][ch1_idx : end_idx]) + ch1_idx
        print("the peak after tha strongest peak is ", data[0][next_peak])
        if data[0][next_peak] >= 50:
            return 1
    # look for the biggest peak and check the peak right after it is above threshold or not 
    if data[1][ch2_idx] >= threshold[1] - 20:
        next_peak = np.argmax(data[1][ch2_idx : min(ch2_idx+threshold_after_first_peak, len(data[1]))])
        if data[1][next_peak] >= 50: 
            return 1
    return 0


def filter(data):
    # Define filter parameters
    low_cut = 1  
    high_cut = 5  
    fs = 250 
    filter_order = 8

    # Calculate filter coefficients
    nyquist_freq = 0.5 * fs
    a = butter(filter_order, [low_cut / nyquist_freq, high_cut / nyquist_freq], btype='bandpass', output='sos') 
    
    data[0] = sosfiltfilt(a, data[0])
    data[1] = sosfiltfilt(a, data[1])
    return
 

def lsl_inlet(name):
    inlet = None
    tries = 0
    print('before resolve stream')
    info = pylsl.resolve_stream('type', 'EEG')
    # error
    print('enter lsl, before inlet')
    inlet = pylsl.stream_inlet(info[0], recover = False)
    print(f'backend has received the {info[0].type()} inlet.')
    return inlet

def main():
    print("Enter main")
    terminate_backend = False
    keyboard = Controller() # setup virtual keyboard
    # Wait for a marker, then start recording EEG data
    data = collections.deque(maxlen=max_len) # fast datastructure for appending/popping in either direction
    subject_threshold = pd.read_csv(subject)
    # threshold <- [average max-threash of channel 1, average max-threash of channel 2]
    # threshold = [-subject_threshold.iloc[0, 0], -subject_threshold.iloc[0, 1]]
    threshold = [120, 120]
    print(threshold)
    print('main function started')
    while True and terminate_backend == False:
        # Constantly check for a marker
        eeg, t_eeg = eeg_in.pull_sample(timeout=0)
        if eeg is not None:
            data.append(eeg)
        if len(data) == max_len:
            # print(data)
            # classify this chunk
            #------code starts here------#
            
            data_processed = invert_data(data)
            filter(data_processed)
            #print(data_processed)
            peaks_1, peaks_2 = find_peak(data_processed, threshold)
            label = statistical_classification(peaks_1, peaks_2, threshold, data_processed)
            # if it returns 1, it will press the spacebar
            if label == 1: # some function that return label of the data
                keyboard.press(Key.space)
                keyboard.release(Key.space)  
                print("jump!")
             # otherwise do nothing
            data = collections.deque(maxlen=max_len)

# initialize variables to store stream
eeg_in = None

if __name__ == "__main__":
    print("Hello")
    # Initialize our streams
    eeg_in = lsl_inlet('dino_EEG')
    # Run out main function
    main()
