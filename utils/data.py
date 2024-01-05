import os
from pathlib import Path
from typing import List, Union, Tuple

import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Dataset

import torchvision.transforms as T

import sys
#  sys.path.append('..')

class FullRadiographDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        fold_nums: list,
        transforms,
        fold_txt_dir: str='splits',
        albumentations_package: bool=True
    ):
        super().__init__()

        self.root_dir = root_dir
        self.fold_nums = fold_nums
        self.transforms = transforms
        self.albumentations = albumentations_package

        # labels
        self.filepaths = []
        self._load_images(fold_txt_dir)

        # this maybe useful later for reproducibility
        self.filepaths.sort()

        print(f"> Successfully Loaded {len(self.filepaths)} images.")

    def _load_images(self,fold_txt_dir):
        for i in self.fold_nums:
            filepath = os.path.join(self.root_dir, fold_txt_dir, f'{i:02d}.txt')
            with open(filepath) as txt_file:
                for line in txt_file:
                    img_relpath = line.strip()
                    filename = img_relpath.split('/')[-1]
                    sex = filename.split('-')[10]
                    # if sex not in ['M', 'F']:
                    #     continue
                    self.filepaths.append(os.path.join(self.root_dir, img_relpath))

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, index: int):
        # image and label
        filepath = self.filepaths[index]
        filename = filepath.split('/')[-1]

        # get the labels
        sex = filename.split('-')[10]
        # age = filename.split('-')[-2][1:]
        # months = filename.split('-')[-1][1:3]

        assert sex in ['F', 'M', 'NA']
        if sex == 'F':
            label = 0
        elif sex == 'M':
            label = 1
        else:
            label = -1

        label_tensor = torch.tensor(label, dtype=torch.int64)

        image = Image.open(filepath).convert('RGB')

        # apply transforms
        if self.albumentations:
            #image = np.array(image)
            img_tensor = self.transforms(image)
        else:
            raise Exception('Not implemented yet.')

        return img_tensor, label_tensor


class LabelledSet(FullRadiographDataset):
    def __init__(self, root_dir, fold_nums, transforms):
        super().__init__(root_dir, fold_nums, transforms)

    def _load_images(self,fold_txt_dir):
        for i in self.fold_nums:
            filepath = os.path.join(self.root_dir, fold_txt_dir, f'{i:02d}.txt')
            with open(filepath) as txt_file:
                for line in txt_file:
                    img_relpath = line.strip()
                    filename = img_relpath.split('/')[-1]
                    sex = filename.split('-')[10]
                    if sex not in ['M', 'F']: # Assert all images are labelled.
                        continue
                    self.filepaths.append(os.path.join(self.root_dir, img_relpath))

    def __getitem__(self, index: int):
        # image and label
        filepath = self.filepaths[index]
        filename = filepath.split('/')[-1]

        # get the labels
        sex = filename.split('-')[10]
        # age = filename.split('-')[-2][1:]
        # months = filename.split('-')[-1][1:3]


        label = 0 if sex == "M" else 1
        label_tensor = torch.tensor(label, requires_grad=True, dtype=torch.float)

        image = Image.open(filepath)
        image = image.convert('RGB')

        # apply transforms
        if self.albumentations:
            # image = np.array(image)
            img_tensor = self.transforms(image)
        else:
            raise Exception('Not implemented yet.')

        return img_tensor, label_tensor

class UnlabelledSet(FullRadiographDataset):
    def __init__(self, root_dir, fold_nums, transforms):
        super().__init__(root_dir, fold_nums, transforms)

    def _load_images(self,fold_txt_dir):
        for i in self.fold_nums:
            filepath = os.path.join(self.root_dir, fold_txt_dir, f'{i:02d}.txt')
            with open(filepath) as txt_file:
                for line in txt_file:
                    img_relpath = line.strip()
                    filename = img_relpath.split('/')[-1]
                    sex = filename.split('-')[10]
                    self.filepaths.append(os.path.join(self.root_dir, img_relpath))

    def __getitem__(self, index: int):
        # image and label
        filepath = self.filepaths[index]
        image = Image.open(filepath)
        image = image.convert('RGB')

        # apply transforms
        if self.albumentations:
            # image = np.array(image)
            img_tensor = self.transforms(image)
        else:
            raise Exception('Not implemented yet.')

        return img_tensor

if __name__ == "__main__":
    root = "/datasets/pan-radiographs/"
    f = FullRadiographDataset(root, list(range(1,31)), None)
    print("[!] Successfully loaded full radiograph dataset.")
    u = UnlabelledSet(root, list(range(1,26)), None)
    print("[!] Successfully loaded unlabelled dataset.")
    l = LabelledSet(root, list(range(26,31)), None)
    print("[!] Successfully loaded labelled dataset.")

    print("[!] All good!")
