import os
import math
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
    def __init__(self, intensity=255, test=True):
        self.intensity = intensity
        
        # Setup QAT FakeQuantize for the activations (your spikes)
        self.fake_quantize = torch.quantization.FakeQuantize(
            observer=torch.quantization.MovingAverageMinMaxObserver, 
            quant_min=0, 
            quant_max=255, 
            dtype=torch.quint8, 
            qscheme=torch.per_tensor_affine, 
            reduce_range=False
        )
        
    def train(self):
        self.fake_quantize.train()

    def eval(self):
        self.fake_quantize.eval()    
    
    def __call__(self, img_tensor):
        N, W, H = img_tensor.shape
        reshaped_batch = img_tensor.view(N, 1, -1)
        
        # Divide all pixel values by 255
        normalized_batch = reshaped_batch / self.intensity
        normalized_batch = torch.squeeze(normalized_batch, 0)

        # Apply FakeQuantize
        spikes = self.fake_quantize(normalized_batch)
        
        if not self.fake_quantize.training:
            scale, zero_point = self.fake_quantize.calculate_qparams()
            spikes = torch.quantize_per_tensor(spikes, float(scale), int(zero_point), dtype=torch.quint8)

        return spikes

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
        mean = torch.mean(img.float())
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
        img = torch.unsqueeze(img,0)
        spike_maker = SetImageAsSpikes()
        img = spike_maker(img)
        img = torch.squeeze(img,0)

        return img

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, base_dir, img_dirs, transform=None, target_transform=None, 
                 filter=1, skip=0, max_samples=None, test=True, img_range=None):
        self.transform = transform
        self.target_transform = target_transform
        self.filter = filter
        self.img_range = img_range
        self.skip = skip
        
        # Load image labels from each directory, apply the skip and max_samples, and concatenate
        self.img_labels = []
        # if annotations_file is a single file, convert it to a list
        if not isinstance(annotations_file, list):
            annotations_file = [annotations_file]
        for idx, annotation in enumerate(annotations_file):
            img_labels = pd.read_csv(annotation)
            img_labels['file_path'] = img_labels.apply(lambda row: os.path.join(base_dir,img_dirs[idx], row.iloc[0]), axis=1)
            if self.img_range is not None:
                img_labels = img_labels.iloc[self.img_range[0]:self.img_range[1]+1]
            # Apply skip: start after the first 'skip' number of rows
            if self.skip > 0:
                img_labels = img_labels.iloc[self.skip:]
            # Select specific rows based on the skip parameter
            img_labels = img_labels.iloc[::filter]
            # Limit the number of samples to max_samples if specified
            if max_samples is not None:
                img_labels = img_labels.iloc[:max_samples]
            # Determine if the images being fed are training or testing
            if test:
                self.img_labels = img_labels
            else:
                self.img_labels.append(img_labels)
        
        if isinstance(self.img_labels,list):
            # Concatenate all the DataFrames
            self.img_labels = pd.concat(self.img_labels, ignore_index=True)
        
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