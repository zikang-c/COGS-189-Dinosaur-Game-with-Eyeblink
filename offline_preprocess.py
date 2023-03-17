import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy.signal import butter, sosfiltfilt
import os

epoch_seconds = 5.2
epoch = int(epoch_seconds / 0.004)

def high_pass(data):
    """
    Applies a high pass filter to the voltage data in the dataframe.

    Parameters:
    data: A Pandas dataframe containing the voltage data.

    Returns:
    A Pandas dataframe containing the flitered voltages of the two channels
    """
    # Define the cutoff frequency (in Hz) for the high pass filter
    cutoff_freq = 1

    # Define the filter order
    filter_order = 8

    # Define the sampling rate (in Hz) for the voltage data
    sampling_rate = 250

    # Define the voltage variable
    voltage_chan_1 = data.iloc[:, 0].values
    voltage_chan_2 = data.iloc[:, 1].values

    # Define the filter coefficients
    nyquist_freq = sampling_rate / 2
    normalized_cutoff_freq = cutoff_freq / nyquist_freq
    a = butter(filter_order, normalized_cutoff_freq, btype='highpass', output='sos')

    # Apply the filter to the voltage data
    voltage_chan_1 = sosfiltfilt(a, voltage_chan_1)
    voltage_chan_2 = sosfiltfilt(a, voltage_chan_2)

    return pd.DataFrame({'FP1 (channel 1)': voltage_chan_1, 'FP2 (channel 2)': voltage_chan_2})


def low_pass(data):
    """
    Applies a low pass filter to the voltage data in a Pandas dataframe.

    Parameters:
    data: A Pandas dataframe containing the voltage data.

    Returns:
    A Pandas dataframe containing the flitered voltages of the two channels
    """
    # Define the cutoff frequency (in Hz) for the low pass filter
    cutoff_freq = 5

    # Define the filter order
    filter_order = 8

    # Define the sampling rate (in Hz) for the voltage data
    sampling_rate = 250

    # Define the voltage variable
    voltage_chan_1 = data.iloc[:, 0].values
    voltage_chan_2 = data.iloc[:, 1].values

    # Define the filter coefficients
    nyquist_freq = sampling_rate / 2
    normalized_cutoff_freq = cutoff_freq / nyquist_freq
    a = butter(filter_order, normalized_cutoff_freq, btype='lowpass', output='sos')

    # Apply the filter to the voltage data
    voltage_chan_1 = sosfiltfilt(a, voltage_chan_1)
    voltage_chan_2 = sosfiltfilt(a, voltage_chan_2)

    return pd.DataFrame({'FP1 (channel 1)': voltage_chan_1, 'FP2 (channel 2)': voltage_chan_2})


def average_min_max(data):
    """
    Calculate the average minimum and maximum filtered and normalized voltage 
    value for channel 1 and 2 based on a manually set threashold.

    Parameters:
    data: A Pandas dataframe containing the filtered voltage data.

    Returns:
    min_1, min_2: averaged minimum value of channel 1 and 2.
    max_1, max_2: averaged maximum value of channel 1 and 2.
    """  
    min_1, max_1, num_min_1, num_max_1= 0, 0, 0, 0
    min_2, max_2, num_min_2, num_max_2= 0, 0, 0, 0
    voltage_chan_1 = data.iloc[:, 0].values
    voltage_chan_2 = data.iloc[:, 1].values

    for i in range(len(voltage_chan_1)):
        # Calculation for channel 1 based on manually set threashold
        if (voltage_chan_1[i] <= -100):
            min_1 += voltage_chan_1[i]
            num_min_1 += 1
        elif (voltage_chan_1[i] >= 70):
            max_1 += voltage_chan_1[i]
            num_max_1 += 1

        # Calculation for channel 2 based on manually set threashold
        if (voltage_chan_2[i] <= -100):
            min_2 += voltage_chan_2[i]
            num_min_2 += 1
        elif (voltage_chan_2[i] >= 70):
            max_2 += voltage_chan_2[i]
            num_max_2 += 1

    return (min_1 / num_min_1), (max_1 / num_max_1), (min_2 / num_min_2), (max_2 / num_max_2)


def preprocess(df):
    df = df.loc[6:,:]
    data = df.iloc[:,1:3]
    time = df.iloc[:, -2]

    filtered = high_pass(data)
    filtered = low_pass(filtered)
    #print("Average minimum and maximum voltage values found for FP1 (channel 1): ", avg_min_1, avg_max_1)
    #print("Average minimum and maximum voltage values found for FP2 (channel 2): ", avg_min_2, avg_max_2)

    return filtered

def divide_dataset(data, time):
    num_epoch = math.ceil(len(time) / epoch)
    data_subsets = []
    time_subsets = []
    start = 0
    for i in range(num_epoch):
        if start + epoch >= len(time):
            data_subsets.append(data.iloc[start:,:])
            time_subsets.append(time[start:])       
        else:
            data_subsets.append(data.iloc[start:start + epoch,:])
            time_subsets.append(time[start:start + epoch])
            start = start + epoch
    return data_subsets, time_subsets

def main():
    i = 1
    folder_directories = ['Subject_1/CSV', 'Subject_2/CSV', 'Subject_3/CSV', 'Subject_4/CSV']
    for folder in folder_directories:
        csv_files = [f for f in os.listdir(folder) if f.endswith('.csv')]

        j = 1
        avg_min_1, avg_max_1, avg_min_2, avg_max_2 = 0,0,0,0
        for csv_file in csv_files:
            df = pd.read_csv(os.path.join(folder, csv_file), delimiter='\t')

            time = df.iloc[:, -2]
            filtered = preprocess(df)
            filtered['time'] = df.iloc[:, -2]

            tmp_avg_min_1, tmp_avg_max_1, tmp_avg_min_2, tmp_avg_max_2 = average_min_max(filtered)
            avg_min_1 += tmp_avg_min_1
            avg_max_1 += tmp_avg_max_1
            avg_min_2 += tmp_avg_min_2
            avg_max_2 += tmp_avg_max_2


            new_file_name = f"Subject_{i}_filtered_{j}.csv"
            filtered.to_csv(new_file_name, index=False)
            j += 1

        avg_min_1 /= 5
        avg_max_1 /= 5
        avg_min_2 /= 5
        avg_max_2 /= 5
        FP1_channel_1 = np.array([avg_min_1, avg_max_1])
        FP2_channel_2 = np.array([avg_min_2, avg_max_2])
        subject_threashold = pd.DataFrame({'FP1': FP1_channel_1, 'FP2 (channel 2)': FP2_channel_2})

        new_file_name = f"Subject_{i}_average_threashold.csv"
        subject_threashold.to_csv(new_file_name, index=False)
        i += 1

if __name__ == "__main__":
    main()
