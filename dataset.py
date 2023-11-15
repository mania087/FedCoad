import numpy as np
import pandas as pd
import re
import glob
import os
import collections
import csv
import scipy.io as sio


def load_wisdm(loc, freq=25, sec=8, channel_first=True):
    # WISDM dataset
    # Hz = 20, sec =10

    columns = ['user','activity','timestamp', 'x-axis', 'y-axis', 'z-axis']
    activity_remap = {
        "Walking":0,
        "Jogging":1,
        "Upstairs":2,
        "Downstairs":3,
        "Sitting":4,
        "Standing":5
    }
    df = pd.read_csv(loc, header = None, names = columns)
    
    # transform from object to float64
    df['z-axis'] = df['z-axis'].map(lambda x: str(re.findall("\d+\.\d+", str(x))))
    df['z-axis'] = df['z-axis'].map(lambda x: x[2:-2])
    df['z-axis'] = pd.to_numeric(df['z-axis'],errors='coerce')

    # drop NA
    df.dropna(axis=0, how='any', inplace=True)

    # rename activity
    df.replace({"activity":activity_remap}, inplace=True)


    # df sort subject & timestamp
    df.sort_values(by=['user', 'timestamp'],inplace=True)

    window_length = freq*sec
    subjects_dataset = []
    subjects_label = []
    # list of subject
    subjects = df["user"].unique()
    for subject in subjects:
        data = []
        label = []
        # get subject df 
        subject_df = df[df["user"]==subject]
        # length of subject data
        length_of_data = len(subject_df)

        for i in range(0,length_of_data - window_length + 1,window_length // 2):
            start_index = i
            end_index = i + window_length
            windows = subject_df[["x-axis", "y-axis","z-axis"]][start_index:end_index].to_numpy()
            # reshape
            if channel_first:
                windows = windows.transpose()
            labels = np.argmax(np.bincount(subject_df["activity"][start_index:end_index]))
            data.append(windows)
            label.append(labels)

        # append to dataset    
        subjects_dataset.append(np.array(data))
        subjects_label.append(np.array(label, dtype=np.int64))
    
    return subjects_dataset, subjects_label

def load_motionsense(loc, freq=50, sec=4, channel_first=True):
    # label_set = {}
    user_datasets = {}
    all_trials_folders = sorted(glob.glob(loc + "/*"))

    # Loop through every trial folder
    for trial_folder in all_trials_folders:
        trial_name = os.path.split(trial_folder)[-1]

        # label of the trial is given in the folder name, separated by underscore
        label = trial_name.split("_")[0]
        # label_set[label] = True
        
        # Loop through files for every user of the trail
        for trial_user_file in sorted(glob.glob(trial_folder + "/*.csv")):

            # use regex to match the user id
            user_id_match = re.search(r'(?P<user_id>[0-9]+)\.csv', os.path.split(trial_user_file)[-1])
            
            
            if user_id_match is not None:
                user_id = int(user_id_match.group('user_id'))

                # Read file
                user_trial_dataset = pd.read_csv(trial_user_file)
                user_trial_dataset.dropna(how = "any", inplace = True)

                # Extract the x, y, z channels
                values = user_trial_dataset[["x", "y", "z"]].values

                # the label is the same during the entire trial, so it is repeated here to pad to the same length as the values
                labels = np.repeat(label, values.shape[0])

                if user_id not in user_datasets:
                    user_datasets[user_id] = []
                user_datasets[user_id].append((values, labels))
            else:
                print("[ERR] User id not found", trial_user_file)

    # sort the dictionary    
    user_datasets = collections.OrderedDict(sorted(user_datasets.items()))
    window_length = freq * sec
    stride = window_length // 2

    activity_remap = {
        "wlk":0,
        "jog":1,
        "ups":2,
        "dws":3,
        "sit":4,
        "std":5
    }

    windowed_user_data = []
    windowed_user_label = []
    for k, v in user_datasets.items():
        user_data = [] 
        user_label = []
        for data in v:
            features = data[0]
            list_of_labels = [activity_remap[label] for label in data[1]]
           
            for i in range(0,len(data[0]) - window_length + 1, stride):
                start_index = i
                end_index = i + window_length
                
                windows = features[start_index:end_index]
                
                if channel_first:
                    # reshape, for example (400,3) to (3,400)
                    windows = windows.transpose()
                
                labels = np.argmax(np.bincount(list_of_labels[start_index:end_index]))
                user_data.append(windows)
                user_label.append(labels)
        # change to np array
        user_data = np.array(user_data)
        user_label = np.array(user_label)

        windowed_user_data.append(user_data)
        windowed_user_label.append(user_label)
    return windowed_user_data, windowed_user_label
