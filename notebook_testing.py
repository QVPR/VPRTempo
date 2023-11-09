import cv2
import numpy as np
import matplotlib.pyplot as plt
    
# Load network scale factor
scale_factors = np.load('./tutorials/mats/1_basicdemoquant/if_scales.npy',allow_pickle=True)
# Load the patch normalized image
patch_img = np.load('./tutorials/mats/1_basicdemoquant/summer_patchnorm.npy', allow_pickle=True)
# Divide the patch image by the QAT scale factor for input spikes
spike_scale = scale_factors[4]
patch_img_int = np.round(patch_img*spike_scale).astype(np.int32)

# Find the maximum quantized pixel intensity and print it
max_int = np.max(patch_img_int)


# Convert 2D image to a 1D-array
patch_1d = np.reshape(patch_img_int, (784,))

# Load network scale factor
scale_factors = np.load('./tutorials/mats/1_basicdemoquant/if_scales.npy',allow_pickle=True)

# Divide the patch image by the QAT scale factor for input spikes
spike_scale = scale_factors[4]
patch_img_int = np.round(patch_img*spike_scale).astype(np.int32)

# Find the maximum quantized pixel intensity and print it
max_int = np.max(patch_img_int)

# Load the input to feature excitatory and inhibitory network weights
if_exc = np.load('./tutorials/mats/1_basicdemoquant/if_exc.npy')
if_inh = np.load('./tutorials/mats/1_basicdemoquant/if_inh.npy')

if_exc = if_exc.astype(np.int32)

zeropoint_inh = 127

# Calculate feature spikes for the positive weight calculation
exc_feature_spikes = (np.matmul(if_exc,patch_1d))

# Get the required scale factors to transform the feature spikes
perslice_scale_exc = scale_factors[0]
perchannel_scale_exc = scale_factors[2]

# Transform the feature layer spikes based on the scale factors
scaled_exc_feature_spikes = np.round(exc_feature_spikes/(perslice_scale_exc*spike_scale))*perchannel_scale_exc
scaled_exc_feature_spikes = scaled_exc_feature_spikes.astype(np.int32)

# Calculate feature spikes for the negative weight calculation
inh_feature_spikes = (np.matmul(if_inh,patch_1d))

# Get the required scale factors to transform the feature spikes
perslice_scale_inh = scale_factors[1]
perchannel_scale_inh = scale_factors[3]

# Transform the feature layer spikes based on the scale factors
scaled_inh_feature_spikes = (np.round(inh_feature_spikes/(perslice_scale_inh*spike_scale))*perchannel_scale_inh) + zeropoint_inh
scaled_inh_feature_spikes = scaled_inh_feature_spikes.astype(np.int32)