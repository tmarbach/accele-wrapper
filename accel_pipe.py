#!/usr/bin/env python3


from distutils.command.clean import clean
import pandas as pd
import numpy as np
import argparse
import csv
import os
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

#FOR ACCELERATER
#TODO: add flag for wild data X
    # script checks for flag before processing the input file, checks input file format
    # if dir then concats all files or just processes one at a time
    # cleans by checking that xyz columns have no missing values X
    # break into windows of window_size flag
    # output file as outputfilename flag

def arguments():
    parser = argparse.ArgumentParser(
            prog='acceleRater_pipeline', 
            description="Preps data for use in AcceleRater",
            epilog=""
                 ) 
    parser.add_argument(
            "-i",
            "--raw-accel-csv",
            type=str,
            help = "input the path to the csv file of accelerometer data that requires cleaning, or prepped_ csv file."
            )
    parser.add_argument(
            "-r",
            "--wild-data",
            help="Process wild data for use after the model is train. No behaviors in the input",
            action="store_true"
            )
    parser.add_argument(
            "-o",
            "--oversample",
            help = "Flag to oversample the minority classes: o -- oversample, s -- SMOTE, or a -- ADASYN ",
            default=False, 
            type=str 
            )
    parser.add_argument(
            "-w",
            "--window-size",
            help="Number of rows to include in each data point (25 rows per second)",
            type=int
            )
    parser.add_argument(
            "-c",
            "--classes-of-interest",
            help="Define the classes of interest",
            default=False, 
            type=str
            )
    parser.add_argument(
            "-f",
            "--acceleRater-data-output-file",
            help="Directs the acceleRater data output to a filename of your choice",
            default=False
            )
    return parser.parse_args()


def accel_data_csv_cleaner(accel_data_csv):
    if accel_data_csv.endswith(".xlsx"):
        df = pd.read_excel(accel_data_csv)
    else:
        df = pd.read_csv(accel_data_csv)
    if 'Behavior' not in df.columns:
        raise ValueError("'Behavior' column is missing")
    if 'accX' not in df.columns:
        raise ValueError("'accX' column is missing")
    if 'accY' not in df.columns:
        raise ValueError("'accY' column is missing")
    if 'accZ' not in df.columns:
        raise ValueError("'accZ' column is missing")
    df['input_index'] = df.index
    cols_at_front = ['Behavior',
                     'accX', 
                     'accY', 
                     'accZ']
    df = df[[c for c in cols_at_front if c in df]+
            [c for c in df if c not in cols_at_front]]
    df= df.dropna(subset=['Behavior'])
    df = df.loc[df['Behavior'] != 'n']
    
    return df

# def accel_data_dir_cleaner(accel_data_csv):
#     #TODO: allow for xlsx files
#     files = [f for f in os.listdir(accel_data_csv) if f.endswith('.csv')]
#     fulldf = []
#     for csv in files:
#         df = accel_data_csv_cleaner(csv)
#         fulldf.append(df)
#     alldf = pd.concat(fulldf)
#     return alldf


def accel_data_dir_cleaner(accel_data_csv):
    #TODO: allow for xlsx files
    files = [f for f in os.listdir(accel_data_csv) if f.endswith('.csv')]
    fulldf = []
    for csv in files:
        adf = pd.read_csv(accel_data_csv + csv, low_memory=False)
        if 'Behavior' not in adf.columns:
            raise ValueError("'Behavior' column is missing")
        if 'accX' not in adf.columns:
            raise ValueError("'accX' column is missing")
        if 'accY' not in adf.columns:
            raise ValueError("'accY' column is missing")
        if 'accZ' not in adf.columns:
            raise ValueError("'accZ' column is missing")
        fulldf.append(adf)
    df = pd.concat(fulldf)    
    df['input_index'] = df.index
    cols_at_front = ['Behavior',
                     'accX', 
                     'accY', 
                     'accZ']
    df = df[[c for c in cols_at_front if c in df]+
            [c for c in df if c not in cols_at_front]]
    df= df.dropna(subset=['Behavior'])
    df = df.loc[df['Behavior'] != 'n']
    return df


def accel_wilddata_csv_cleaner(accel_data_csv):
    if accel_data_csv.endswith(".xlsx"):
        df = pd.read_excel(accel_data_csv)
    else:
        df = pd.read_csv(accel_data_csv)
    if 'accX' not in df.columns:
        raise ValueError("'accX' column is missing")
    if 'accY' not in df.columns:
        raise ValueError("'accY' column is missing")
    if 'accZ' not in df.columns:
        raise ValueError("'accZ' column is missing")
    df['input_index'] = df.index
    cols_at_front = ['accX', 
                     'accY', 
                     'accZ']
    df = df[[c for c in cols_at_front if c in df]+
            [c for c in df if c not in cols_at_front]]
    clean_df= df.dropna(subset=['accX','accY','accZ'])
    if clean_df.shape[0] != df.shape[0]:
        print("Null values removed from wild data accX, accY, accZ columns.")
    else:
        print("Wild data has all accX, accY, accZ values.")
    return clean_df


def wild_leaping_window(df, window_size):
    """
    Input: 
    df -- dataframe of cleaned input data, likely from a csv
    window_size -- number of rows of data to convert to 1 row for AcceleRater (25 = 1sec)
    Output:
    windows -- list of lists of unlabeled accel data (EX:[x,y,z,...,x,y,z])
    """
    windows = []
    number_of_rows_minus_window = df.shape[0] - window_size + 1
    if window_size > df.shape[0]:
        raise ValueError('Window larger than data given')
    for i in range(0, number_of_rows_minus_window, window_size):
        window = df[i:i+window_size]
        if len(set(np.ediff1d(window.input_index))) != 1:
            continue
        windows.append(window)    
    
    print("Windows pulled")
    return windows


def wild_flattener(windows):
    """
    Input:
        windows -- list of dataframes of all one class 
    Output:
        Xdata -- arrays of xyz data of each window stacked together
        ydata -- integer class labels for each window
    """
    positions = ['accX', 'accY', 'accZ']
    xyzdata = []
    for window in windows:
        windowdata = window[positions].to_numpy()
        xlist = windowdata.tolist()
        flat_list = [accel_value for datarow in xlist for accel_value in datarow]
        xyzdata.append(flat_list)
    print("Windows flattened")
    return xyzdata




def output_prepped_data(original_data, clean_csv_df):
    filename = os.path.basename(original_data)
    clean_csv_df.to_csv('prepped_'+filename, index=False)


def class_identifier(df, c_o_i):
    if c_o_i == False:
        bdict = dict(zip(list(df.Behavior.unique().sum()), range(1, len(list(df.Behavior.unique().sum()))+1)))
        coi_list = list(bdict.keys())
    else:
        blist = list(df.Behavior.unique().sum())
        coi_list = ['other-classes'] + [bclass for bclass in c_o_i]
        bdict = {x: 0 for x in blist}
        count = 0
        for bclass in c_o_i:
            count +=1
            bdict[bclass] = count
        diff = list(set(c_o_i)-set(blist))
        if len(diff) > 0:
            missingclasses = ','.join(str(c) for c in diff)
            print("Classes " + missingclasses + " not found in input data.")

    return bdict, coi_list


def singleclass_leaping_window_exclusive(df, window_size, coi=False):
    """
    Input: 
    df -- dataframe of cleaned input data, likely from a csv
    window_size -- number of rows of data to convert to 1 row for AcceleRater (25 = 1sec)
    Output:
    windows -- list of lists of accel data (EX:[x,y,z,...,x,y,z,class_label])
    allclasses -- list of the behavior classes that are present in the windows
    """
    classes = []
    windows = []
    number_of_rows_minus_window = df.shape[0] - window_size + 1
    if window_size > df.shape[0]:
        raise ValueError('Window larger than data given')
    for i in range(0, number_of_rows_minus_window, window_size):
        window = df[i:i+window_size]
        if len(set(window.Behavior)) != 1:
            continue
        if len(set(np.ediff1d(window.input_index))) != 1:
            continue
        if coi:
            coiclasses = list(coi)
            if window['Behavior'].iloc[0] not in list(coiclasses):
                continue
        windows.append(window)
        classes.append(window['Behavior'].iloc[0])
    allclasses = set(classes)
    if coi:
        diff = list(set(coi)-allclasses)
        if len(diff) > 0:
            missingclasses = ','.join(str(c) for c in diff)
            print("Classes " + missingclasses + " not found in any window.")
    
    print("Windows pulled")
    return windows, list(allclasses)


def accel_singlelabel_xy(windows):
    """
    Input:
        windows -- list of dataframes of all one class 
    Output:
        Xdata -- arrays of xyz data of each window stacked together
        ydata -- integer class labels for each window
    """
    positions = ['accX', 'accY', 'accZ']
    strikes = ['h', 'm', 'f']
    Xdata, ydata = [], []

    for window in windows:
        behavior = window['Behavior'].iloc[0]
        if behavior in strikes:
            behavior  = 't'
        Xdata.append(window[positions].to_numpy())
        ydata.append(behavior)
    return np.stack(Xdata), np.asarray(ydata)



def accel_oversampler(Xdata,ydata, sampler_flag = False):
    """
    Reduces the dimensions of inout data, splits into train/test set,
    then randomly oversamples the minority classes to match the majority class"""
    nsamples, nx, ny = Xdata.shape
    Xdata2d = Xdata.reshape((nsamples,nx*ny))
    if sampler_flag == 'o':
        ros = RandomOverSampler(random_state=0)
        X_resampled, y_resampled = ros.fit_resample(Xdata2d, ydata)
    elif sampler_flag == 's':
        X_resampled, y_resampled = SMOTE(k_neighbors=2).fit_resample(Xdata2d, ydata)
    elif sampler_flag == 'a':
        X_resampled, y_resampled = ADASYN(n_neighbors=2).fit_resample(Xdata2d, ydata)
    else:
        X_resampled, y_resampled = Xdata2d, ydata

    tupdata = zip(X_resampled,y_resampled)
    total_oversample_data = [list(elem) for elem in tupdata]
    clean_data = [np.append(sample[0], sample[1]) for sample in total_oversample_data]

    print("windows flattened and labeled")
    return clean_data



def output_data(totaldata, windowsize, coi = False, oversampling = False, output_filename = False):
    if coi:
        coi_label = str(coi)
    coi_label = 'all_classes'
    label = coi_label + str(windowsize) + "_acceleRater.csv"
    if output_filename == False and oversampling == False:
        with open(label, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(totaldata) 
    elif output_filename == False and oversampling == True:
        with open(str(oversampling) + '_' + label, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(totaldata)
    else:
        with open(output_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(totaldata)



def main():
    args = arguments()
    if args.wild_data:
        df = accel_wilddata_csv_cleaner(args.raw_accel_csv)
        windows = wild_leaping_window(df, args.window_size)
        flat_data_list = wild_flattener(windows)
        with open(args.acceleRater_data_output_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(flat_data_list)
    else:
        if args.raw_accel_csv.endswith('/'):
            df = accel_data_dir_cleaner(args.raw_accel_csv)
        elif "prepped_" in os.path.basename(args.raw_accel_csv):
            df = pd.read_csv(args.raw_accel_csv)
        else:
            df = accel_data_csv_cleaner(args.raw_accel_csv)
            output_prepped_data(args.raw_accel_csv,df)
        classdict, coilist = class_identifier(df, args.classes_of_interest) # for indentifying missing classes
        windows, all_classes = singleclass_leaping_window_exclusive(df, int(args.window_size), args.classes_of_interest)
        Xdata, ydata = accel_singlelabel_xy(windows)
        total_data = accel_oversampler(Xdata, ydata, args.oversample)      

        output_data(total_data, args.window_size, args.classes_of_interest, args.oversample, args.acceleRater_data_output_file)
    



if __name__ == "__main__":
    main()
    