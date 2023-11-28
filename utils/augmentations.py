import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torchvision import transforms as T

import sys
sys.path.append('../')

# def get_transforms(inputs, subset='train'):
def get_transforms(
    aug_name,
    img_size,
    subset='train',
    mean=[0.485, 0.456, 0.406],
    stdv=[0.229, 0.224, 0.225]
):
    if aug_name == 'imagenet':
        print('Using ImageNet augmentations ...')
        if subset == 'train':
            return T.Compose([
                T.RandomResizedCrop(img_size),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                # TODO: add PCA noise
                T.ToTensor(),
                T.Normalize(mean, stdv)
            ])
        else:
            return T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean, stdv)
            ])
    elif aug_name == 'auto_trivial':
        if subset == 'train':
            return T.Compose([
                T.Resize((img_size, img_size)),
                T.TrivialAugmentWide(),
                T.ToTensor(),
                T.Normalize(mean, stdv)
            ])
        else:
            return T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean, stdv)
            ])
    elif aug_name == 'small_complexity':
        if subset == 'train':
            return A.Compose([
#                 A.Compose([
#                     A.GaussNoise(var_limit=(0, 100), p=0.8),
#                     A.Blur(blur_limit=15, always_apply=True),
#                     A.ShiftScaleRotate(
#                         shift_limit=0.03,
#                         scale_limit=0.05,
#                         rotate_limit=4,
#                         interpolation=cv2.INTER_CUBIC,
#                         p=1
#                     ),
#                     A.RandomBrightnessContrast(
#                         brightness_limit=(-0.2, 0.2),
#                         contrast_limit=(-0.2, 0.2),
#                         always_apply=True
#                     )
#                 ], p=0.9),
                A.ShiftScaleRotate(
                    shift_limit=0.03,
                    scale_limit=0.05,
                    rotate_limit=4,
                    interpolation=cv2.INTER_CUBIC,
                    p=0.9
                ),
                A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_CUBIC),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=mean, std=stdv),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_CUBIC),
                A.Normalize(mean=mean, std=stdv),
                ToTensorV2()
            ])
    elif aug_name == 'resize_only':
        return A.Compose([
            A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_CUBIC),
            A.Normalize(mean=mean, std=stdv),
            ToTensorV2()
        ])
    elif aug_name == 'simple_crop_att':
        print('Using simple crop augmentation "attention" augmentation')
        if subset == 'train':
            return A.Compose([
                        A.Resize(320, 320, interpolation=cv2.INTER_CUBIC),
                        A.Crop(48, 93, 272, 317),
                        A.HorizontalFlip(p=0.5),
                        A.Normalize(mean=mean, std=stdv),
                        ToTensorV2()
                    ])
        else:
            return A.Compose([
                        A.Resize(320, 320, interpolation=cv2.INTER_CUBIC),
                        A.Crop(48, 93, 272, 317),
                        A.Normalize(mean=mean, std=stdv),
                        ToTensorV2()
                    ])
    elif aug_name == 'proportional_336_crop':
        print('Using proportional 336 crop augmentation')
        a1 = 143/112

        resized_image = 336 #int(224*1.5)

        x1 = int((resized_image - 224)/2)
        x2 = int((resized_image - 224)/2 + 224)
        y1 = int(x1*a1)
        y2 = int(resized_image - y1)

        if subset == 'train':
            return A.Compose([
                A.Resize(resized_image, resized_image),
                A.Crop(x1, y1, x2, y2),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=mean, std=stdv),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(resized_image, resized_image),
                A.Crop(x1, y1, x2, y2),
                A.Normalize(mean=mean, std=stdv),
                ToTensorV2()
            ])

    else:
        print('Using only horizontal flip augmentation.')
        if subset == 'train':
            return A.Compose([
                A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_CUBIC),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=mean, std=stdv),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_CUBIC),
                A.Normalize(mean=mean, std=stdv),
                ToTensorV2()
            ])
