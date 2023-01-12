import os
import cv2
import torch
import torchvision
import torch.nn as nn
import pandas as pd
import numpy as np
import skimage.io as io
from pathlib import Path
from torchvision.transforms import ToTensor
from resize_dataset import FracturesDataSet
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    training_annotation,
    val_dir,
    validation_annotation,
    batch_size,
    validation_size,
    num_workers=4,
    pin_memory=True,
    transform = False
):
    train_ds = FracturesDataSet(
        json_file = training_annotation,
        root_dir = train_dir,
        transform=transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
  #      collate_fn=my_collate
    )

    val_ds = FracturesDataSet(
        json_file = validation_annotation,
        root_dir = val_dir,
        transform=None
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=validation_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
 #       collate_fn=my_collate
    )

    return train_loader, val_loader

def save_predictions_as_csv(loader, model, model_folder, device="cuda"):

    model.eval()

    cwd = Path.cwd()

    folder = cwd / model_folder

    # Initialize the csv with a header
    with open(f'{folder}/val_pred.csv', 'w', encoding='UTF8') as f:
        f.write(f'image_name,predicted_width_fraction,predicted_height_fraction,width_factor,height_factor,target_width_fraction,target_height_fraction\n')

    for idx, data in enumerate(loader):

        img = data['image'].to(device)
        target_width = data['resized_width'].float().to(device)
        target_height = data['resized_height'].float().to(device)
        width_factor = data['width_factor'].to(device)
        height_factor = data['height_factor'].to(device)
        img_name = data['image_name']

        with torch.no_grad():
            preds = model(img)
            width_hat = preds['label_width'].float()
            height_hat = preds['label_height'].float()
        
        with open(f'{folder}/val_pred.csv', 'a', encoding='UTF8') as f:
                for ind, name in enumerate(img_name):
                    f.write(f'{name},{width_hat[ind].item()},{height_hat[ind].item()},{width_factor[ind].item()},{height_factor[ind].item()},{target_width[ind].item()},{target_height[ind].item()}\n')

def save_validation_inference_imgs(model_folder, resize):

    cwd = Path.cwd()

    folder = cwd / model_folder

    # Read the validation results csv and the trainign params json
    validation_results = pd.read_csv(f'{folder}/val_pred.csv')
    annotation = pd.read_json(cwd / 'curr_annotations' / 'annotations.json')

    # Apply the scale factors to transform the prediction back to the original shape
    validation_results = validation_results.assign(
        predicted_width_orig = validation_results.predicted_width_fraction * resize / validation_results.width_factor,
        target_width_orig = validation_results.target_width_fraction * resize / validation_results.width_factor,

        predicted_height_orig = validation_results.predicted_height_fraction * resize / validation_results.height_factor,
        target_height_orig = validation_results.target_height_fraction * resize / validation_results.height_factor
    )

    # Select 10 random images from the validation set

    indexes = np.random.randint(0, validation_results.shape[0], size = 10)

    for ind in indexes:
        # Get the image data and load the validation image
        img_data = validation_results.iloc[ind,]
        img_name = img_data.image_name # .png name

        # Get the true label
        row = annotation.loc[annotation.file_name == img_name]

        img_path = cwd / 'raw_data_files' / 'AFF_large' / img_name
        img = io.imread(img_path, as_gray = True)

        # Normalize the image
        norm = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # Make color image
        norm = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)

        img_drawn = cv2.circle(norm, (int(img_data.predicted_width_orig), int(img_data.predicted_height_orig)), radius=25, color=(0,0,255), thickness=-1)
        img_drawn = cv2.circle(img_drawn, (int(img_data.target_width_orig), int(img_data.target_height_orig)), radius=25, color=(0,255,0), thickness=-1)
        #img_drawn = cv2.circle(img_drawn, (int(row.original_width * row.x / 100), int(row.original_height * row.y / 100)), radius=25, color=(255,0,0), thickness=-1)

        print(f'True Width {int(row.original_width * row.x / 100)}')
        print(f'Target Width {int(img_data.target_width_orig)}')
        print(f'Predicted Width {int(img_data.predicted_width_orig)}')

        cv2.imwrite(str(cwd / model_folder / f'val_{img_name}'), img_drawn)