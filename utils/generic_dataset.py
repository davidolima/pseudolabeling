#!../bin/python3

"""
David de Oliveira Lima,
Fev, 2024
"""

import os
from pathlib import Path
from typing import List, Union, Tuple, Optional

import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torchvision
from torch.utils.data import Dataset


import sys
#  sys.path.append('..')

class GenericDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        subset: list[str] = ['train', 'test'],
        transforms: Optional[torchvision.transforms.Compose] = None,
        ignore_unlabelled: bool = True,
    ):
        super().__init__()

        self.root_dir = root_dir
        self.transforms = transforms
        self.ignore_unlabelled = ignore_unlabelled
        self.filepaths = []

        # Identify and enumerate classes.
        # Expects subfolder "train" containing class folders whithin.
        if not ignore_unlabelled:
            self.classes = ["unlabelled"]
        else:
            self.classes = []

        self.classes += os.listdir(os.path.join(root_dir, 'train'))
        self.classes = dict(enumerate(self.classes))
        self.classes = dict([(value, key) for (key, value) in self.classes.items()]) # invert dict

        # Finally, load images and print information.
        for s in subset:
            self._load_images(os.path.join(self.root_dir, s))

        print(f"> Successfully Loaded {len(self.filepaths)} images of {len(self.classes)} different classes:")
        for k,v in self.classes.items():
            labels = [x[1] for x in self.filepaths]
            print('  >', v, '-', f"'{k}' ({labels.count(k)} images)")

    def _load_images(self, load_path):
        for (path, _, images) in os.walk(load_path):
            if len(path.split('/')) > 4:
                label = path.split('/')[4]
            else:
                if self.ignore_unlabelled:
                    continue
                label = "unlabelled"
            self.filepaths += [(os.path.join(path, img), label) for img in images]

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, index: int):
        # image and label
        filepath, label = self.filepaths[index]

        # Encode label
        label = self.classes[label]
        label_tensor = torch.tensor(label, dtype=torch.uint8)

        image = Image.open(filepath).convert('RGB')

        # apply transforms
        if self.transforms:
            image = self.transforms(image)
        return image, label_tensor

if __name__ == "__main__":
    root = "/datasets/glomerulus-kaggle/"
    f = GenericDataset(root, ignore_unlabelled=False)
    print("[!] Successfully loaded full glomerulus dataset.")
    print(" > Sample batch:\n", f[np.random.randint(0, len(f))])
    f = GenericDataset(root, ignore_unlabelled=True)
    print("[!] Successfully loaded labelled images of glomerulus dataset.")
    print(" > Sample batch:\n", f[np.random.randint(0, len(f))])

    print("[!] All good!")
