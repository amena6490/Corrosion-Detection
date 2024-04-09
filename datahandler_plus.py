from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
import cv2
import torch
from torchvision import transforms, utils
import torch.nn.functional as F



class SegDataset(Dataset):
    """Segmentation Dataset"""

    def __init__(self, root_dir, imageFolder, maskFolder, transform=None, 
                 seed=None, fraction=None, subset=None):
        self.root_dir = root_dir
        self.transform = transform

        self.mapping = {(0,0,0): 0, (0,0,128): 1, (0,128,0): 2, (0,128,128): 3}
        
        # Determine if we are splitting the dataset by a percentage
        if not fraction:
            self.image_names = sorted(
                glob.glob(os.path.join(self.root_dir, imageFolder, '*')))
            self.mask_names = sorted(
                glob.glob(os.path.join(self.root_dir, maskFolder, '*')))
        else:
            assert(subset in ['Train', 'Test'])
            self.fraction = fraction
            self.image_list = np.array(
                sorted(glob.glob(os.path.join(self.root_dir, imageFolder, '*'))))
            self.mask_list = np.array(
                sorted(glob.glob(os.path.join(self.root_dir, maskFolder, '*'))))
            if seed:
                # see here that the seed value is just a value. 
                np.random.seed(seed)
                indices = np.arange(len(self.image_list))
                np.random.shuffle(indices)
                self.image_list = self.image_list[indices]
                self.mask_list = self.mask_list[indices]
            if subset == 'Train':
                self.image_names = self.image_list[:int(
                    np.ceil(len(self.image_list)*(1-self.fraction)))]
                self.mask_names = self.mask_list[:int(
                    np.ceil(len(self.mask_list)*(1-self.fraction)))]
            else:
                self.image_names = self.image_list[int(
                    np.ceil(len(self.image_list)*(1-self.fraction))):]
                self.mask_names = self.mask_list[int(
                    np.ceil(len(self.mask_list)*(1-self.fraction))):]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        
        # pass in the image
        img_name = self.image_names[idx]
        image = cv2.imread(img_name) # jpeg, RGB format
        
        # pass in the mask
        msk_name = self.mask_names[idx]
        mask_png = cv2.imread(msk_name) # jpeg, RGB format
        height, width, channels = image.shape
        
        target = torch.from_numpy(mask_png)
        
        # TODO: comment what this function is doing
        target = target.permute(2, 0, 1).contiguous()
        
        mask = torch.empty(height, width, dtype=torch.long)
        
        for k in self.mapping:
            # Get all indices for current class
            idx = (target==torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
            validx = (idx.sum(0) == 3)  # Check that all channels match
            mask[validx] = torch.tensor(self.mapping[k], dtype=torch.long)
        
        mask = mask.numpy()
        sample = {'image': image, 'mask': mask}

        if self.transform:
            sample = self.transform(sample)

        return sample

# Define few transformations for the Segmentation Dataloader

# Resize image and mask method (COMPLETE)
class Resize(object):
    """Resize image and/or masks."""

    def __init__(self, imageresize, maskresize):
        # these need to be in the form of (X, X) when passed inside the datatransform
        # function.
        self.imageresize = imageresize
        self.maskresize = maskresize

    def __call__(self, sample):
        
        image, mask = sample['image'], sample['mask']
        
        # Checks to see if the input is indeed an 3-channel image, and then converts it
        # to a BGR?
        if len(image.shape) == 3:
            image = image.transpose(1, 2, 0)
        if len(mask.shape) == 3:
            mask = mask.transpose(1, 2, 0)
            
        # When resizing masks I have seen that the best interpolation method is the 
        # NEAREST as it does not interpolate new colors, which would introduce
        # new 'classes' in our case. 
        mask = cv2.resize(mask, self.maskresize, interpolation=cv2.INTER_NEAREST)
        # mask = cv2.resize(mask, self.maskresize, interpolation=cv2.INTER_NEAREST)
        image = cv2.resize(image, self.imageresize, interpolation=cv2.INTER_NEAREST)
        
        # This code re-converts the image back to as it was in the input.
        if len(image.shape) == 3:
            image = image.transpose(2, 0, 1)
        if len(mask.shape) == 3:
            mask = mask.transpose(2, 0, 1)

        return {'image': image,
                'mask': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, maskresize=None, imageresize=None):
        image, mask = sample['image'], sample['mask']
        return {'image': torch.from_numpy(image).type(torch.FloatTensor),
                'mask':  torch.from_numpy(mask).type(torch.FloatTensor)}


# Normalize images (COMPLETE)
class Normalize(object):
    '''Normalize image'''

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        return {'image': image.type(torch.FloatTensor)/255,
                'mask': mask.type(torch.FloatTensor)/255}


def get_dataloader_sep_folder(data_dir, imageFolder='Images', maskFolder='Masks', batch_size=4):
    # Takes images and converts them to Tensors, normalizes them, and resizes them. 
    data_transforms = {
        'Train': transforms.Compose([ToTensor()]),
        'Test': transforms.Compose([ToTensor()]),
    }
    
    # The seed value is a 'random' variable. However, whenever you give it a starting point it will always return
    # the same random values given that seed value. This helps with reporducibility.
    image_datasets = {x: SegDataset(root_dir=os.path.join(data_dir, x), transform=data_transforms[x], maskFolder=maskFolder, 
                                    imageFolder=imageFolder) for x in ['Train', 'Test']}
    
    # This line remains the same since it is a built-in pytorch function   
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True) for x in ['Train', 'Test']}
    
    return dataloaders


def get_dataloader_single_folder(data_dir, imageFolder='Images', maskFolder='Masks', fraction=0.2, batch_size=4):
    """
        Create training and testing dataloaders from a single folder.
    """
    # Takes images and converts them to Tensors, normalizes them, and resizes them. 
    data_transforms = {
        'Train': transforms.Compose([ToTensor()]),
        'Test': transforms.Compose([ToTensor()]),
    }
    
    # The seed value is a 'random' variable. However, whenever you give it a starting point it will always return
    # the same random values given that seed value. This helps with reporducibility.
    image_datasets = {x: SegDataset(data_dir, imageFolder=imageFolder, maskFolder=maskFolder, seed=10, 
                                    fraction=fraction, subset=x, transform=data_transforms[x]) for x in ['Train', 'Test']}
    
    # This line remains the same since it is a built-in pytorch function
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size,shuffle=True, num_workers=8, drop_last=True) for x in ['Train', 'Test']}
    
    return dataloaders
