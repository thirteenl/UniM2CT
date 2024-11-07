import os
import torch
import glob

import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class FASDataSet(Dataset):
    """ A data loader for Face PAD where samples are organized in this way
    Args:
        root (string): Root directory path where all images reside
        txt_file (string): Path to txt file containing image paths and labels
        depth_map_size (tuple): size of pixel-wise binary supervision map. The paper uses map_size=14
        transform: transform to be applied to the image
    """

    def __init__(self, root, txt_file, is_transform=True, depth_map_size=None):
        self.root = root
        self.txt_file = txt_file
        self.depth_map_size = depth_map_size
        self.is_transform = is_transform

        self.img_paths = []
        self.labels = []

        self.load_data()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float)

        img = Image.open(img_path).convert('RGB')
        if self.is_transform:
            img = self.transform(img)

        if self.depth_map_size is not None:
            # Generate depth map
            if label == 0:
                depth_map = np.ones(self.depth_map_size)
            else:
                depth_map = np.zeros(self.depth_map_size)

            sample = {'image': img, 'depth_map': torch.from_numpy(depth_map), 'label': label}
        else:
            sample = {'image': img, 'label': label}

        return sample

    def load_data(self):
        with open(self.txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                img_path, label = line.strip().split(" ")
                self.img_paths.append(os.path.join(self.root, img_path))
                self.labels.append(int(label))

    @staticmethod
    def transform(img):
        trans = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return trans(img)
