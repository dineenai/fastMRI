# Adapted from fastMRI_tutorial.ipynb

# New Error - source unknown, there is no cd in this file?

# (fastmri) ainedineen@cusacklab-lamb00:~/motion_dnns/fastMRI/fastmri_examples/unet$ python fastmri_test_AD.py
# Traceback (most recent call last):
#   File "fastmri_test_AD.py", line 1, in <module>
#     cd 
# NameError: name 'cd' is not defined

import h5py
import numpy as np
from matplotlib import pyplot as plt
import os

# The fastMRI dataset is distributed as a set of HDF5 files and can be read with the h5py package.
# Here, we show how to open a file from the multi-coil dataset. Each file corresponds to one MRI
# scan and contains the k-space data, ground truth and some meta data related to the scan.

# file_name = 'multicoil_train/file1000167.h5'
# file_name = '/data2/fastMRI/fastMRI/data/brain_multicoil_test/brain_multicoil_test/file_brain_AXT2_210_6001942.h5'
# file_name = '/data2/fastMRI/fastMRI/data/multicoil_train/file_brain_AXT2_200_2000094.h5'

file_name = '/data2/fastMRI/fastMRI/data/multicoil_val/file_brain_AXT2_204_2040010.h5'

# file_name = '/data2/fastMRI/fastMRI/data/multicoil_test/file_brain_AXT2_203_2030379.h5'

# file_name = '/data2/fastMRI/fastMRI/data/multicoil_test/file_brain_AXT2_205_2050152.h5'
#file_name = /data2/fastMRI/fastMRI/data/multicoil_test_full/file_brain_AXT2_205_2050152.h5
hf = h5py.File(file_name)

base = os.path.basename(file_name)
print(base)
print(file_name)
print('Keys:', list(hf.keys()))
print('Attrs:', dict(hf.attrs))


# In multi-coil MRIs, k-space has the following shape: (number of slices, number of coils, height, width)
# For single-coil MRIs, k-space has the following shape: (number of slices, height, width)
# MRIs are acquired as 3D volumes, the first dimension is the number of 2D slices.

volume_kspace = hf['kspace'][()]
print(volume_kspace.dtype)
print(volume_kspace.shape)

# slice_kspace = volume_kspace[20] # Choosing the 20-th slice of this volume
# IndexError: index 20 is out of bounds for axis 0 with size 16

chosen_slice = 0
# slice_kspace = volume_kspace[12] # Choosing the 12-th slice of this volume
slice_kspace = volume_kspace[chosen_slice]

# Let's see what the absolute value of k-space looks like:

def show_coils(data, slice_nums, cmap=None):
    fig = plt.figure()
    for i, num in enumerate(slice_nums):
        plt.subplot(1, len(slice_nums), i + 1)
        plt.imshow(data[num], cmap=cmap)
        # plt.imsave(data[num], cmap=cmap)

        save_coil_image = '/home/ainedineen/motion_dnns/fastMRI/fastmri_test_AD_outputs/test_coil_image_prerecon_'+str(base)+'_sl_'+str(chosen_slice)+'.png'
        plt.savefig(save_coil_image)

# comment out while test below
# show_coils(np.log(np.abs(slice_kspace) + 1e-9), [0, 5, 10])  # This shows coils 0, 5 and 10
show_coils(np.log(np.abs(slice_kspace) + 1e-9), [0, 5, 10])  # This shows coils 0, 5 and 10

# /home/ainedineen/motion_dnns/fastMRI

# save_filter = '/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/visualizing_filters/conv1_filters/supRN50_conv1_21_g4_60e_e60/supRN50_conv1_21_g4_60e_e60_filter_'+str(index)+'.png'
# plt.savefig(save_filter)


# return to the above.....

import fastmri
from fastmri.data import transforms as T

slice_kspace2 = T.to_tensor(slice_kspace)      # Convert from numpy array to pytorch tensor
slice_image = fastmri.ifft2c(slice_kspace2)           # Apply Inverse Fourier Transform to get the complex image
slice_image_abs = fastmri.complex_abs(slice_image)   # Compute absolute value to get a real image

# works
# show_coils(slice_image_abs, [0, 5, 10], cmap='gray')

show_coils(slice_image_abs, [0, 1, 2, 3, 10, 15], cmap='gray') #  dimension 0 with size 20, 19 is max


slice_image_rss = fastmri.rss(slice_image_abs, dim=0)


plt.imshow(np.abs(slice_image_rss.numpy()), cmap='gray')
save_combined_coil_image = '/home/ainedineen/motion_dnns/fastMRI/fastmri_test_AD_outputs/test_combined_coil_image_'+str(base)+'_sl_'+str(chosen_slice)+'.png'
plt.savefig(save_combined_coil_image)



# for index, filter in enumerate(extractor.CNN_weights[0]):
#     plt.subplot(8, 8, index + 1) 
#     plt.imshow(filter[0, :, :].cpu().detach().numpy(), cmap='gray')
#     plt.axis('off')

# save_filter=str(args.save_filter_path)+'/RF_conv1_'+str(args.save_filter_file)+'.png'

# # plt.show()
# plt.savefig(save_filter)
#     # plt.show()




# # Part 3
# from fastmri.data.subsample import RandomMaskFunc
# mask_func = RandomMaskFunc(center_fractions=[0.04], accelerations=[8])  # Create the mask function object

# masked_kspace, mask = T.apply_mask(slice_kspace2, mask_func)   # Apply the mask to k-space

# # Let's see what the subsampled image looks like:
# sampled_image = fastmri.ifft2c(masked_kspace)           # Apply Inverse Fourier Transform to get the complex image
# sampled_image_abs = fastmri.complex_abs(sampled_image)   # Compute absolute value to get a real image
# sampled_image_rss = fastmri.rss(sampled_image_abs, dim=0)

# plt.imshow(np.abs(sampled_image_rss.numpy()), cmap='gray')
# save_undersampled_image = '/home/ainedineen/motion_dnns/fastMRI/fastmri_test_AD_outputs/test_undersampled_image1.png'
# plt.savefig(save_undersampled_image)
