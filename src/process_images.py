import os
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from skimage import io
import argparse

def read_img(img_path):
    img = io.imread(img_path, as_gray = True)
    return img

def normalize_img(img):
    
    img = img / np.percentile(img, 99)
    img[img>1] = 1

    return img

def resize_img(img):
    
    img_resized = cv2.resize(img, (RESIZE_SIZE,RESIZE_SIZE))
    img_resized = img_resized.astype('float32')

    return img_resized

def get_img_info(img, width_percentage, height_percentage):

    img_width = img.shape[1]
    img_height = img.shape[0]
    
    # Get the fractions
    width_fraction = (width_percentage / 100)
    height_fraction = (height_percentage / 100)

    # True coordinates
    width_coord = img_width * width_fraction
    height_coord = img_height * height_fraction

    # Get the factors of resizing
    width_factor = RESIZE_SIZE / img_width
    height_factor = RESIZE_SIZE / img_height

    # Get the label in the resized image
    resized_width_fraction = width_coord * width_factor / RESIZE_SIZE
    resized_height_fraction = height_coord * height_factor / RESIZE_SIZE

    info_dict = {'width_coord': width_coord,
                'height_coord' : height_coord,
                'width_factor' : width_factor,
                'height_factor' : height_factor,
                'resized_width_fraction': resized_width_fraction,
                'resized_height_fraction': resized_height_fraction,
                }

    return info_dict


def process_img(img_path, img_file_name, width_percentage, height_percentage):

    img = read_img(img_path)
    img = normalize_img(img)

    img_info = get_img_info(img, width_percentage, height_percentage)

    resized_img = resize_img(img)

    # Store resized img
    #print('Writing resized image in .npy format')
    np.save(processed_img_path / f'{img_file_name[:-4]}.npy', resized_img)

    return img_info

def process_imgs(df):

    # Make new columns for this batch of preprocessing

    df = df.assign(
        width_coord = 0,
        height_coord = 0,
        width_factor = 0,
        height_factor = 0,
        resized_width_fraction = 0,
        resized_height_fraction = 0)
    
    # Loop over rows
    for ind in range(df.shape[0]):
        
        img_file_name = df.iloc[ind, 0]
        width_percentage = df.iloc[ind, 3]
        height_percentage = df.iloc[ind, 4]

        img_path = original_img_path / img_file_name

        img_info = process_img(img_path, img_file_name, width_percentage, height_percentage)

        # Enter img information into the dataframe
        df.iloc[ind, 5] = img_info['width_coord']
        df.iloc[ind, 6] = img_info['height_coord']
        df.iloc[ind, 7] = img_info['width_factor']
        df.iloc[ind, 8] = img_info['height_factor']
        df.iloc[ind, 9] = img_info['resized_width_fraction']
        df.iloc[ind, 10] = img_info['resized_height_fraction']

    return df

if __name__ == '__main__':

    # Create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--resize', type=int)
    parser.add_argument('--img_folder', type=str)
    parser.add_argument('--annotations_path', type=str)
    opt = parser.parse_args()
    # Hyperparameters
    RESIZE_SIZE = opt.resize
    IMG_FOLDER = f'AFF_{RESIZE_SIZE}x{RESIZE_SIZE}'
    # Define paths
    cwd = Path.cwd()
    main_folder = cwd.parent
    annotations_path = main_folder / opt.annotations_path
    processed_img_path = main_folder / opt.img_folder / 'AFF_large_processed' / IMG_FOLDER
    original_img_path = main_folder  / opt.img_folder / 'AFF_large'

    # Read annotations file
    print('Reading annotations ...')
    df = pd.read_json(annotations_path / 'annotations.json')

    # List of already processed images
    lst_processed = os.listdir(processed_img_path)

    print('Filtering annotations ...')

    # Get only files that have not been processed
    df = df[~df.file_name.isin(lst_processed)]

    print('Processing images ...')
    # Process images and get result
    df_processed = process_imgs(df)

    # Store the dataframe
    df_processed.to_json(processed_img_path / f'{RESIZE_SIZE}_annotations.json')

    print('Done')





