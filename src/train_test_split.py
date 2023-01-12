import os
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import argparse

def delete_old_imgs(path):

    for f in os.listdir(path):
        os.remove(path / f)

    return None

def copy_over_files(files, output_folder):

    for file in files:
        file_npy = f'{file[:-3]}npy'
        shutil.copy(data_path / file_npy, main_folder / 'data' / output_folder / file_npy)

    print(f'Copied files to output folder {output_folder}')

    return None

if __name__ == '__main__':

    # Create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--resize', type=int)
    parser.add_argument('--train-split', type=float, default=0.7)
    parser.add_argument('--data-path', type=str, default='raw_data_files/AFF_large_processed')
    parser.add_argument('--output-path', type=str, default='data')
    
    opt = parser.parse_args()
    # Hyperparameters
    RESIZE_SIZE = opt.resize
    IMG_FOLDER = f'AFF_{RESIZE_SIZE}x{RESIZE_SIZE}'
    if opt.train_split <= 1 and opt.train_split > 0:
        TRAIN_PERCENTAGE = opt.train_split
    else:
        raise ValueError('Invalid train split')

    # Get paths
    cwd = Path.cwd()
    main_folder = cwd.parent
    data_path = main_folder / opt.data_path / IMG_FOLDER
    #model_path = cwd / 'Resize_CNN'

    # List images
    lst_img = os.listdir(data_path)
    # Read annotations dataframe
    df = pd.read_json(data_path / f'{RESIZE_SIZE}_annotations.json')

    # Get the patient ID
    df = df.assign(
    patient_id = df.file_name.apply(lambda x: x.split('_')[1]))

    print('Performing train, val, test split ... ')
    # Compute the number of observations for train/val/test
    Train_n = int(np.floor(len(df.patient_id.unique()) * TRAIN_PERCENTAGE))
    Rest_n = len(df.patient_id.unique()) - Train_n
    Val_n = int(np.floor(Rest_n*0.5))
    Test_n = Rest_n - Val_n

    # Get the patient id random sequences
    Pat_seq = np.arange(len(df.patient_id.unique()))
    Train_pat = np.random.choice(Pat_seq, size = Train_n, replace=False)
    Rest_pat = Pat_seq[~np.isin(Pat_seq, Train_pat)]
    Val_pat = np.random.choice(Rest_pat, size = Val_n, replace=False)
    Test_pat = Rest_pat[~np.isin(Rest_pat, Val_pat)]

    # Make strings
    Train_pat_str = [str(pat) for pat in Train_pat]
    Val_pat_str = [str(pat) for pat in Val_pat]
    Test_pat_str = [str(pat) for pat in Test_pat]

    # Setup the split annotation files
    Train_annotations = df.loc[df.patient_id.isin(Train_pat_str)]
    Val_annotations = df.loc[df.patient_id.isin(Val_pat_str)]
    Test_annotations = df.loc[df.patient_id.isin(Test_pat_str)]

    # Drop any rows with mising values
    Train_annotations = Train_annotations.dropna(axis=0)
    Val_annotations = Val_annotations.dropna(axis = 0)
    Test_annotations = Test_annotations.dropna(axis = 0)

    # Save them to data folder
    Train_annotations.to_json(main_folder / opt.output_path / 'training_annotation.json')
    Val_annotations.to_json(main_folder / opt.output_path / 'validation_annotation.json')
    Test_annotations.to_json(main_folder / opt.output_path / 'testing_annotation.json')

    Train_files = Train_annotations.file_name.values
    Val_files = Val_annotations.file_name.values
    Test_files = Test_annotations.file_name.values

    print('Deleting old images ... ')
    # Delete old images
    delete_old_imgs(main_folder / opt.output_path / 'train_img')
    delete_old_imgs(main_folder / opt.output_path / 'val_img')
    delete_old_imgs(main_folder / opt.output_path / 'test_img')

    print('Copying over new images ... ')
    # Copy over new images
    copy_over_files(Train_files, 'train_img')
    copy_over_files(Val_files, 'val_img')
    copy_over_files(Test_files, 'test_img')



