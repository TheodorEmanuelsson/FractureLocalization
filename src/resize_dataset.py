import os
import pandas as pd
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
import albumentations as A


class FracturesDataSet(Dataset):
    def __init__(self, json_file, root_dir, transform=None):
        self.annotations = pd.read_json(json_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):

        # Format the image name
        img_file = f'{self.annotations.iloc[index, 0][:-3]}npy'
        # Get the image path
        img_path = os.path.join(self.root_dir, img_file)
        # Read the image
        img = np.load(img_path)

        # Find pixel label in resized image
        width_factor = self.annotations.iloc[index, 7]
        height_factor = self.annotations.iloc[index, 8]
        # Get normalized resized labels
        resized_label_width = np.array([self.annotations.iloc[index, 9]]).reshape(-1,1)
        resized_label_height = np.array([self.annotations.iloc[index, 10]]).reshape(-1,1)

        # Perform any pytorch transformations
        if self.transform is not None:
            # Compute the coordinates in the resized image
            width_x = int(img.shape[1]*resized_label_width)
            height_y = int(img.shape[0]*resized_label_height)

            transforms = A.Compose([
                A.Rotate(limit = 40, p=0.5, border_mode = cv2.BORDER_CONSTANT),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5)
                ], keypoint_params=A.KeypointParams(format='xy'))
            transformed = transforms(image=img,keypoints=[(width_x, height_y)])
            transformed_image = transformed['image']
            try: 
                transformed_width, transformed_height = transformed['keypoints'][0]
                resized_label_width = np.array([transformed_width / img.shape[1]]).reshape(-1,1)
                resized_label_height = np.array([transformed_height / img.shape[0]]).reshape(-1,1)
                img = transformed_image
            except:
                # Keypoint gets out of the frame sometimes.
                print('\nError in augmentation. Processing with original image')
                pass
        
        # Convert to torch tensor
        resized_label_width = torch.from_numpy(resized_label_width)
        resized_label_height = torch.from_numpy(resized_label_height)

        img = torch.from_numpy(img)
        img = img.unsqueeze(0)

        return {'image':img, 'resized_width': resized_label_width, 'resized_height': resized_label_height,
                'width_factor': width_factor, 'height_factor': height_factor,
                'image_name': self.annotations.iloc[index, 0]}
        





    




