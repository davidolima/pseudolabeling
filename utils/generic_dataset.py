#!../bin/python3

"""
David de Oliveira Lima,
Fev, 2024
"""

import os
from pathlib import Path
from typing import List, Union, Tuple, Optional, overload

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
        root_dir: Optional[str] = None,
        subset: Optional[list[str]] = ['train', 'test'],
        transforms: Optional[torchvision.transforms.Compose] = None,
        ignore_unlabelled: bool = True,

        # Poor man's overloading
        filepaths: Optional[str] = None,
        classes: Optional[list[str]] = None,


    ) -> None:
        super().__init__()

        if filepaths is None and root_dir is not None:
            self.__init_from_root_dir(
                root_dir=root_dir,
                subset=subset,
                transforms=transforms,
                ignore_unlabelled=ignore_unlabelled,
            )
        else:
            self.__init_from_filepaths(
                filepaths=filepaths,
                classes=classes,
                transforms=transforms,
            )

    def __init_from_root_dir(
        self,
        root_dir: str,
        subset: list[str],
        transforms: Optional[torchvision.transforms.Compose] = None,
            ignore_unlabelled: bool = True,
    ):
        self.transforms = transforms

        self.root_dir = root_dir
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

        self.print_status()

    def __init_from_filepaths(
        self,
        filepaths = list[Tuple[str, str]],
        classes = list[str],
        transforms: Optional[torchvision.transforms.Compose] = None,
    ):
        self.filepaths = filepaths
        self.transforms = transforms

        self.classes = dict(enumerate(classes))
        self.classes = dict([(value, key) for (key, value) in self.classes.items()]) # invert dict

        self.print_status()

    def _load_images(self, load_path):
        for (path, _, images) in os.walk(load_path):
            if len(path.split('/')) > 4:
                label = path.split('/')[4]
            else:
                if self.ignore_unlabelled:
                    continue
                label = "unlabelled"
            self.filepaths += [(os.path.join(path, img), label) for img in images]

    def print_status(self):
        print(f"> Successfully Loaded {len(self.filepaths)} images of {len(self.classes)} different classes:")
        for k,v in self.classes.items():
            labels = [x[1] for x in self.filepaths]
            print('  >', v, '-', f"'{k}' ({labels.count(k)} images)")

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, index: int) -> Tuple[Image.Image, torch.Tensor]:
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

    def shuffle(self) -> None:
        from random import shuffle
        shuffle(self.filepaths)

    def split(self, percentage: float, shuffle: bool = True) -> (Dataset, Dataset):
        assert 0 < percentage < 1, "Percentage must be between 0 and 1."

        if shuffle: self.shuffle()

        percent_idx = int(np.ceil(percentage * len(self.filepaths)))
        print(percent_idx)

        left = GenericDataset(
            filepaths  = self.filepaths[:percent_idx],
            classes    = self.classes.keys(),
            transforms = self.transforms
        )
        right = GenericDataset(
            filepaths  = self.filepaths[percent_idx:],
            classes    = self.classes.keys(),
            transforms = self.transforms
        )

        return (left, right)

if __name__ == "__main__":
    root = "/datasets/glomerulus-kaggle/"
    print("[+] Testing initialization and loading images.")
    f = GenericDataset(root, ["train","test"], ignore_unlabelled=False)
    print(" > Successfully loaded full glomerulus dataset.")
    print(" > Random sample batch:\n", f[np.random.randint(0, len(f))])
    f = GenericDataset(root, ["train","test"], ignore_unlabelled=True)
    print(" > Successfully loaded labelled images of glomerulus dataset.")
    print(" > Random sample batch:\n", f[np.random.randint(0, len(f))])

    print("[+] Testing Shuffle.")
    rand_idx = np.random.randint(0, len(f))
    print(f" > Batch at position {rand_idx}:\n", f[rand_idx])
    print("Shuffling...")
    f.shuffle()
    print(f" > Batch at position {rand_idx} (Should be different):\n", f[rand_idx])

    print("[+] Testing split.")

    rand_percent = np.random.random(1).item()
    print(f"Splitting in {rand_percent:.2f}% / {1-rand_percent:.2f}%")
    l,r = f.split(rand_percent, shuffle=True)
    print(
        f"Left: {len(l)} images ({len(l)/len(f)*100:.2f}%)"
    )
    print(
        f"Right: {len(r)} images ({len(r)/len(f)*100:.2f}%)"
    )

    print("[!] All good!")
