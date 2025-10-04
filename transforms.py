import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


def get_cifar10_stats():
    """CIFAR-10 dataset statistics"""
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    return mean, std


def get_train_transforms():
    """Training transforms with albumentation"""
    mean, std = get_cifar10_stats()
    
    # Convert mean to integers for fill_value (0-255 range)
    fill_value = tuple([int(m * 255) for m in mean])
    
    transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            p=0.5
        ),
        A.CoarseDropout(
            max_holes=1,
            max_height=16,
            max_width=16,
            min_holes=1,
            min_height=16,
            min_width=16,
            fill_value=fill_value,
            mask_fill_value=None,
            p=0.5
        ),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
    
    return transforms


def get_test_transforms():
    """Test transforms (only normalization)"""
    mean, std = get_cifar10_stats()
    
    transforms = A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
    
    return transforms


class AlbumentationTransform:
    """Wrapper to use Albumentations with PyTorch datasets"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        # Convert PIL image to numpy array
        if hasattr(img, 'numpy'):
            img = img.numpy()
        else:
            img = np.array(img)
        
        # Apply albumentations transform
        transformed = self.transform(image=img)
        return transformed['image']