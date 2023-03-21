import numpy as np
import pandas as pd
import scipy.signal as sig
import pylsl
import collections
from pynput.keyboard import Key, Controller
import offline_preprocess as ofp

max_len = 375 # modify the number of samples for analysis here
threshold_after_first_peak = 150 # how many data points do we search after the first strong peak above threshold
peak_residue_window = 0
#subject_threshold = [100, 100]
subject = 'Filtered_Subject_4\Subject_4_average_threashold.csv'

# extract data from the file 
def invert_data(data):
    """
    Inverts the input data by changing the signs of the values.
    
    Args:
        data (list): A list containing two lists, one for each channel's data.
        
    Returns:
        clean_data (list): A list containing two lists of inverted data.
    """
    # negative values become positive, while positive values become negative in channel 1 and channel 2
    clean_data = [[], []]
    for i in range(len(data)):
        ch1 = data[i][0]
        ch2 = data[i][1]
        clean_data[0].append(-ch1)
        clean_data[1].append(-ch2)
    return clean_data

def find_peak(data, threshold):
    """
    Finds the peaks in the input data based on the given threshold.
    
    Args:
        data (list): A list containing two lists, one for each channel's data.
        threshold (list): A list containing two threshold values, one for each channel.
        
    Returns:
        peaks_1 (array): An array of indices where peaks are found in the first channel.
        peaks_2 (array): An array of indices where peaks are found in the second channel.
    """
    peaks_1, _ = sig.find_peaks(data[0], height=(-threshold[0]))
    peaks_2, _ = sig.find_peaks(data[1], height=(-threshold[1]))
    print("peaks_1 in find_peak: ", peaks_1)
    print("peaks_2 in find_peak: ", peaks_2)
    return peaks_1, peaks_2


def statistical_classification(peaks_1, peaks_2, data):
    """
    Classifies the input data by comparing the peaks found in the data.
    
    Args:
        peaks_1 (array): An array of indices where peaks are found in the first channel.
        peaks_2 (array): An array of indices where peaks are found in the second channel.
        data (list): A list containing two lists, one for each channel's data.
        
    Returns:
        int: 1 if a double eyeblink is detected, 0 otherwise.
    """
    # peaks_1, peaks_2 stores the indecies of the array, in which indecies a peak occurs
    global peak_residue_window

    if len(peaks_1) == 0 or len(peaks_2) == 0:
        peak_residue_window = 0
        return 0
    # store the peak and its respective index into a a list of tuple
    data_ch1 = np.array([(data[0][i], i) for i in peaks_1])
    data_ch2 = np.array([(data[1][i], i) for i in peaks_2])

    #print("data_ch1 in statistical_classification: ", data_ch1) 
    #print("data_ch2 in statistical_classification: ", data_ch2)

    # If there's only one peak in the first channel
    if (len(data_ch1) == 1):
        # If the peak index is less than the peak_residue_window, it means that the peak in the
        # last epoch and the peak in the current epoch belong to one double blink.
        # Reset the peak_residue_window and return 1
        # (indicating a double eyeblink is detected)
        if (data_ch1[0][1] < peak_residue_window): 
            peak_residue_window = 0
            return 1
        # If the peak index is close to the end of the window, update peak_residue_window
        # to account for the remaining peak search area
        # and return 0 (indicating no double eyeblink is detected)
        elif data_ch1[0][1] > max_len - threshold_after_first_peak:
            peak_residue_window = data_ch1[0][1] + threshold_after_first_peak - max_len
            return 0
    # If there's only one peak in the second channel
    elif (len(data_ch2) == 1):
        # Similar logic to the first channel
        if (data_ch2[0][1] < peak_residue_window): 
            peak_residue_window = 0
            return 1
        elif data_ch2[0][1] > max_len - threshold_after_first_peak:
            peak_residue_window = data_ch2[0][1] + threshold_after_first_peak - max_len
            return 0
        
    # Iterate through the peaks in the first channel    
    for idx in range(len(data_ch1) - 1):
        # If the distance between two consecutive peaks is less than the threshold_after_first_peak,
        # reset the peak_residue_window and return 1 (indicating a double eyeblink is detected)
        if data_ch1[idx + 1][1] - data_ch1[idx][1] < threshold_after_first_peak:
            peak_residue_window = 0
            return 1
    # Similar logic to the first channel
    for idx in range(len(data_ch2) - 1):
        if data_ch2[idx + 1][1] - data_ch2[idx][1] < threshold_after_first_peak:
            peak_residue_window = 0
            return 1   
    # If no double eyeblink is detected, return 0
    return 0


def filter(data):
    """
    Applies high-pass and low-pass filters to the input data.
    
    Args:
        data (list): A list containing two lists, one for each channel's data.
        
    Returns:
        list: A list containing two filtered lists, one for each channel.
    """
    #return [data[0], data[1]]
    filtered = ofp.high_pass(pd.DataFrame({'FP1': data[0], 'FP2': data[1]}))
    filtered = ofp.low_pass(filtered)
    return [filtered['FP1 (channel 1)'].values, filtered['FP2 (channel 2)'].values]
 

def lsl_inlet(name):
    """
    Sets up the LSL inlet for receiving data from the BCI device.
    
    Args:
        name (str): The name of the LSL stream.
        
    Returns:
        inlet (pylsl.StreamInlet): The LSL stream inlet object.
    """
    inlet = None
    tries = 0
    print('before resolve stream')
    info = pylsl.resolve_stream('name', name)
    # error
    print('enter lsl, before inlet')
    inlet = pylsl.stream_inlet(info[0], recover = False)
    print(f'backend has received the {info[0].type()} inlet.')
    return inlet

def main():
    """
    The main function of the script. It runs in a loop, pulling chunks of data from the
    BCI device, processing the data, and classifying it. If the classification result is
    positive for a double eyeblink, it simulates a spacebar press.
    """
    terminate_backend = False
    keyboard = Controller() # setup virtual keyboard
    subject_threshold = pd.read_csv(subject).iloc[0].values

    while True and terminate_backend == False:
        chunk, timestamps = eeg_in.pull_chunk(timeout=max_len, max_samples=max_len)
        if chunk is not None:
            if len(chunk) != 0:
                # classify this chunk
                data_processed = invert_data(chunk)
                data_processed = filter(data_processed)
                peaks_1, peaks_2 = find_peak(data_processed, subject_threshold)
                label = statistical_classification(peaks_1, peaks_2, data_processed)
                # if it returns 1, it will press the spacebar
                if label == 1: 
                    keyboard.press(Key.space)
                    keyboard.release(Key.space)  
                    print("JUMP!")

# initialize variables to store stream
eeg_in = None

if __name__ == "__main__":
    print("Hello")
    # Initialize our streams
    eeg_in = lsl_inlet('dino_EEG')
    # Run out main function
    main()
