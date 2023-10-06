import os
import math
import cv2
import torch

import pandas as pd
import numpy as np
import torch.nn.functional as F

from torchvision.io import read_image
from torch.utils.data import Dataset

class GetPatches2D:
    def __init__(self, patch_size, image_pad):
        self.patch_size = patch_size
        self.image_pad = image_pad
    
    def __call__(self, img):

        # Assuming image_pad is already a PyTorch tensor. If not, you can convert it:
        # image_pad = torch.tensor(image_pad).to(torch.float64)

        # Using unfold to get 2D sliding windows.
        unfolded = self.image_pad.unfold(0, self.patch_size[0], 1).unfold(1, self.patch_size[1], 1)
        # The size of unfolded will be [nrows, ncols, patch_size[0], patch_size[1]]

        # Reshaping the tensor to the desired shape
        patches = unfolded.permute(2, 3, 0, 1).contiguous().view(self.patch_size[0]*self.patch_size[1], -1)

        return patches


class PatchNormalisePad:
    def __init__(self, patches):
        self.patches = patches

    
    def nanstd(self,input_tensor, dim=None, unbiased=True):
        if dim is not None:
            valid_count = torch.sum(~torch.isnan(input_tensor), dim=dim, dtype=torch.float)
            mean = torch.nansum(input_tensor, dim=dim) / valid_count
            diff = input_tensor - mean.unsqueeze(dim)
            variance = torch.nansum(diff * diff, dim=dim) / valid_count

            # Bessel's correction for unbiased estimation
            if unbiased:
                variance = variance * (valid_count / (valid_count - 1))
        else:
            valid_count = torch.sum(~torch.isnan(input_tensor), dtype=torch.float)
            mean = torch.nansum(input_tensor) / valid_count
            diff = input_tensor - mean
            variance = torch.nansum(diff * diff) / valid_count
            
            # Bessel's correction for unbiased estimation
            if unbiased:
                variance = variance * (valid_count / (valid_count - 1))

        return torch.sqrt(variance)
   
    def __call__(self, img):
        img = torch.squeeze(img,0)
        patch_size = (self.patches, self.patches)
        patch_half_size = [int((p-1)/2) for p in patch_size ]
        
        # Compute the padding. If patch_half_size is a scalar, the same value will be used for all sides.
        if isinstance(patch_half_size, int):
            pad = (patch_half_size, patch_half_size, patch_half_size, patch_half_size)  # left, right, top, bottom
        else:
            # If patch_half_size is a tuple, then we'll assume it's in the format (height, width)
            pad = (patch_half_size[1], patch_half_size[1], patch_half_size[0], patch_half_size[0])  # left, right, top, bottom

        # Apply padding
        image_pad = F.pad(img, pad, mode='constant', value=float('nan'))

        nrows = img.shape[0] 
        ncols = img.shape[1]
        patcher = GetPatches2D(patch_size,image_pad)
        patches = patcher(img)
        mus = torch.nanmean(patches, dim=0)
        stds = self.nanstd(patches, dim=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            im_norm = (img - mus.reshape(nrows, ncols)) / stds.reshape(nrows, ncols)
        
        im_norm[torch.isnan(im_norm)] = 0.0
        im_norm[im_norm < -1.0] = -1.0
        im_norm[im_norm > 1.0] = 1.0
        
        return im_norm

class SetImageAsSpikes:
    def __init__(self,intensity=255,test=True,modules=1):
        self.intensity = intensity
        self.test = test
        self.modules = modules

    def __call__(self, img_tensor):
        # Ensure the input is a 4D tensor (N x C x W x H)
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.unsqueeze(1)
        
        N, C, W, H = img_tensor.shape
        reshaped_batch = img_tensor.view(N, 1, -1)
        
        # Divide all pixel values by 255
        normalized_batch = reshaped_batch / self.intensity
        
        # If running test, repeat input over all the modules
        if self.test:
            normalized_batch = normalized_batch.repeat(self.modules, 1, 1)
           
        return normalized_batch

class ProcessImage:
    def __init__(self, dims, patches):
        self.dims = dims
        self.patches = patches
        
    def __call__(self, img):
        # Convert the image to grayscale using the standard weights for RGB channels
        if img.shape[0] == 3:
            img = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
         # Add a channel dimension to the resulting grayscale image
        img= img.unsqueeze(0)

        # gamma correction
        mid = 0.5
        mean = torch.mean(img)
        gamma = math.log(mid * 255) / math.log(mean)
        img = torch.pow(img, gamma).clip(0, 255)
        
        # resize and patch normalize        
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        img = F.interpolate(img, size=self.dims, mode='bilinear', align_corners=False)
        img = img.squeeze(0)
        patch_normaliser = PatchNormalisePad(self.patches)
        im_norm = patch_normaliser(img) 
        img = (255.0 * (1 + im_norm) / 2.0).to(dtype=torch.uint8)
        img = torch.squeeze(img,0)

        return img

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dirs, transform=None, target_transform=None, 
                 skip=1, max_samples=None, modules=1, test=True):
        self.transform = transform
        self.target_transform = target_transform
        self.skip = skip
        
        # Load image labels from each directory, apply the skip and max_samples, and concatenate
        self.img_labels = []
        for img_dir in img_dirs:
            img_labels = pd.read_csv(annotations_file)
            img_labels['file_path'] = img_labels.apply(lambda row: os.path.join(img_dir, row.iloc[0]), axis=1)
            
            # Select specific rows based on the skip parameter
            img_labels = img_labels.iloc[::skip]
            
            # Limit the number of samples to max_samples if specified
            if max_samples is not None:
                img_labels = img_labels.iloc[:max_samples]
            
            # Determine if the images being fed are training or testing
            if test:
                self.img_labels = img_labels
            else:
                # Reorder images in the DataFrame
                reordered_img_labels = self.reorder_images(img_labels, modules)
                self.img_labels.append(reordered_img_labels)
        
        if isinstance(self.img_labels,list):
            # Concatenate all the reordered DataFrames
            self.img_labels = pd.concat(self.img_labels, ignore_index=True)
        
    def reorder_images(self, img_labels, modules):
        # Calculate the number of batches
        num_batches = len(img_labels) // modules
        remainder = len(img_labels) % modules
        
        reordered_list = []
        for i in range(num_batches):
            for j in range(modules):
                idx = i + j * num_batches
                if idx < len(img_labels):
                    reordered_list.append(img_labels.iloc[idx])
        
        # If there are remaining images, append them to the reordered list
        for i in range(remainder):
            idx = num_batches * modules + i
            reordered_list.append(img_labels.iloc[idx])
        
        # Convert reordered list of Series back to DataFrame
        return pd.concat(reordered_list, axis=1).transpose().reset_index(drop=True)
    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx]['file_path']
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"No file found for index {idx} at {img_path}.")
            
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
            
        return image, label
