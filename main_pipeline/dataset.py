import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class CustomDepthDataset(Dataset):
    def __init__(
        self,
        csv_path,
        transforms=None
        ) -> None:

        self.data = pd.read_csv(
            os.path.join(csv_path), 
            header=None, 
            sep=' ', 
            index_col=None
        )
        self.transforms = transforms


    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, idx) -> tuple:
        image_path, depth_path = self.data.iloc[idx]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        depth = cv2.imread(depth_path)[:, :, 0]

        if self.transforms:
            transformed = self.transforms(image=image, mask=depth)
            image, depth = transformed['image'], transformed['mask']

        return image, depth


class CustomSegmDataset(Dataset):
    def __init__(
            self,
            csv_path,
            transforms=None
            ) -> None:

        self.data = pd.read_csv(
            os.path.join(csv_path), 
            header=None, 
            sep=' ', 
            index_col=None
        )
        self.transforms = transforms


    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx) -> tuple:
        image_path, mask_path = self.data.iloc[idx]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = np.array(Image.open(mask_path))

        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image, mask = transformed['image'], transformed['mask']

        return image, mask


class CustomMultiDataset(Dataset):
    def __init__(
            self,
            csv_path,
            transforms=None
            ) -> None:

        self.data = pd.read_csv(
            os.path.join(csv_path), 
            header=None, 
            sep=' ', 
            index_col=None
        )
        self.transforms = transforms


    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx) -> tuple:
        image_path, mask_path, depth_path = self.data.iloc[idx]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = np.array(Image.open(mask_path))
        depth = cv2.imread(depth_path)[:, :, 0]

        if self.transforms:
            transformed = self.transforms(image=image, mask=mask, depth=depth)
            image, mask, depth = transformed['image'], transformed['mask'], transformed['depth']

        return image, mask, depth
