import pandas as pd
import numpy as np
import warnings
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

class SatelliteDatasetFromImages(Dataset):
    def __init__(self, root, gt_path, transform): 
        self.data = np.load(gt_path)#[:1]
        self.root = root
        self.gt_path = gt_path
        self.transform = transform
        
    def __getitem__(self, index):
        cluster = self.data[index][0]
        label = torch.from_numpy(np.array(self.data[index][1], dtype=np.float32))


        img_as_tensor = [self.transform(Image.open('{}/{}/{}_{}.jpeg'.format(self.root, 'sentinel_lowres', cluster, i))) for i in range(1, 289+1)]
        counts_as_tensor = [torch.from_numpy(np.load('{}/{}/{}_{}.npy'.format(self.root, 'counts_2x2', cluster, i))) for i in range(1, 289+1)]
        img_as_tensor = torch.stack(img_as_tensor, dim=0)
        counts_as_tensor = torch.stack(counts_as_tensor, dim=0)

        return img_as_tensor, label, counts_as_tensor, cluster

    def __len__(self):
        return len(self.data) 

class CustomDatasetFromImages(Dataset):
    def __init__(self, csv_path, transform):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.transforms = transform
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=None)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 1])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 0])
        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(single_image_name)
        # Transform the image
        img_as_tensor = self.transforms(img_as_img)
        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len
