# fastmri_recon_test.py

import h5py
import numpy as np
from matplotlib import pyplot as plt

import os

post_recon = '/data2/fastMRI/fastMRI/pretrained_unet_brain_mc/reconstructions/file_brain_AXT2_205_2050152.h5'
hf_post = h5py.File(post_recon, 'r')

# prerecon
pre_recon = '/data2/fastMRI/fastMRI/data/multicoil_test_1_sample/file_brain_AXT2_205_2050152.h5'

hf_pre = h5py.File(pre_recon, 'r')

base_pre = os.path.basename(pre_recon)
base_post = os.path.basename(post_recon)
print(base_pre)


print('Keys:', list(hf_post.keys())) #Keys: ['reconstruction']
print('Attrs:', dict(hf_post.attrs)) #Attrs: {}

print('Keys:', list(hf_pre.keys())) #Keys: ['ismrmrd_header', 'kspace', 'mask']
print('Attrs:', dict(hf_pre.attrs)) #Attrs: {'acceleration': 4,
                                            # 'acquisition': 'AXT2',
                                            # 'num_low_frequency': 26,
                                            # 'patient_id': '5780142de9839f41565680a2583e679a37b84c91f74bee623015486d3571155a'}

# In multi-coil MRIs, k-space has the following shape: (number of slices, number of coils, height, width)
# For single-coil MRIs, k-space has the following shape: (number of slices, height, width)
# MRIs are acquired as 3D volumes, the first dimension is the number of 2D slices.


volume_kspace = hf_pre['kspace'][()]
print(volume_kspace.dtype) #complex64
print(volume_kspace.shape) #(16, 16, 640, 320)      #(number of slices, number of coils, height, width)

chosen_slice = 8
slice_kspace = volume_kspace[chosen_slice] # Choosing the chosen_slice-th slice of this volume

print("RECON")
volume_recon = hf_post['reconstruction'][()]
print(volume_recon.dtype) #float32
print(volume_recon.shape) #(16, 1, 320, 320)
# slice_recon = volume_kspace[chosen_slice]
slice_recon = volume_recon[chosen_slice]
slice_recon0 = volume_recon[0]
slice_recon15 = volume_recon[15]
slice_recon4 = volume_recon[4]
slice_recon8 = volume_recon[8]
slice_recon12 = volume_recon[12]
# COIL IMAGE NOT WORKING????? #####- supposed to look like this?

# shows coils 


# Let's see what the absolute value of k-space looks like:

def show_coils(data, slice_nums, file="", cmap=None):
    fig = plt.figure()
    for i, num in enumerate(slice_nums):
        plt.subplot(1, len(slice_nums), i + 1)
        plt.imshow(data[num], cmap=cmap)


        # plt.imsave(data[num], cmap=cmap)

    # save_coil_image = '/home/ainedineen/motion_dnns/fastMRI/recon_test_out/'+str(file)+'_coil_image_prerecon_'+str(base_pre)+'_sl_'+str(chosen_slice)+'.png'
    # base_pre
    # save_coil_image = f'/home/ainedineen/motion_dnns/fastMRI/recon_test_out/{file}_prerecon_{base_pre}_sl_{chosen_slice}.png'
    save_coil_image = f'/home/ainedineen/motion_dnns/fastMRI/recon_test_out/{file}_postrecon_{base_post}_sl_{chosen_slice}.png'
    plt.savefig(save_coil_image)

show_coils(np.log(np.abs(slice_kspace) + 1e-9), [0, 5, 10], file="kspace_coils")  # This shows coils 0, 5 and 10

# show_coils(np.log(np.abs(slice_recon) + 1e-9), [0, 5, 10], file="coils")  # This shows coils 0, 5 and 10 #ERROR
# RECON ONLY HAS ONE COIL 8/2/22
show_coils(np.log(np.abs(slice_recon) + 1e-9), [0], file="recon_coil_gray", cmap='gray')  #
show_coils(np.log(np.abs(slice_recon0) + 1e-9), [0], file="recon_coil_sl0_gray", cmap='gray')  #
show_coils(np.log(np.abs(slice_recon15) + 1e-9), [0], file="recon_coil_sl15_gray", cmap='gray')  #
show_coils(np.log(np.abs(slice_recon4) + 1e-9), [0], file="recon_coil_sl4_gray", cmap='gray')  #
show_coils(np.log(np.abs(slice_recon8) + 1e-9), [0], file="recon_coil_sl8_gray", cmap='gray')  #
show_coils(np.log(np.abs(slice_recon12) + 1e-9), [0], file="recon_coil_sl12_gray", cmap='gray')  #
# The fastMRI repo contains some utlity functions to convert k-space into image space.
# These functions work on PyTorch Tensors.
# The to_tensor function can convert Numpy arrays to PyTorch Tensors.

# Compute absolute value to get a real image

import fastmri
from fastmri.data import transforms as T

slice_kspace2 = T.to_tensor(slice_kspace)      # Convert from numpy array to pytorch tensor
slice_image = fastmri.ifft2c(slice_kspace2)           # Apply Inverse Fourier Transform to get the complex image
slice_image_abs = fastmri.complex_abs(slice_image)   # Compute absolute value to get a real image

show_coils(slice_image_abs, [0, 5, 10], file="abs_ksp_real", cmap='gray')



slice_recon2 = T.to_tensor(slice_recon)      # Convert from numpy array to pytorch tensor
slice_image_recon = fastmri.ifft2c(slice_recon2)           # Apply Inverse Fourier Transform to get the complex image
slice_image_abs_recon = fastmri.complex_abs(slice_image_recon)   # Compute absolute value to get a real image

show_coils(slice_image_abs_recon, [0, 5, 10], file="abs_recon_real", cmap='gray')


# ADD POST VS PRE LABEL - recon vs kspace


# coils can be combined into the full image using the Root-Sum-of-Squares (RSS) transform.

slice_image_rss = fastmri.rss(slice_image_abs, dim=0)
# plt.imshow(np.abs(slice_image_rss.numpy()), cmap='gray')
# here = '/home/ainedineen/motion_dnns/fastMRI/recon_test_out'
save_combined_coil_image=f'/home/ainedineen/motion_dnns/fastMRI/recon_test_out/combined_coil_img_{base_pre}.png'

plt.imsave(save_combined_coil_image, arr=np.abs(slice_image_rss.numpy()), cmap='gray')


slice_image_recon_rss = fastmri.rss(slice_image_abs_recon, dim=0)
# plt.imshow(np.abs(slice_image_rss.numpy()), cmap='gray')
# here = '/home/ainedineen/motion_dnns/fastMRI/recon_test_out'
save_combined_coil_image_recon=f'/home/ainedineen/motion_dnns/fastMRI/recon_test_out/combined_coil_recon_img_{base_post}.png'

plt.imsave(save_combined_coil_image_recon, arr=np.abs(slice_image_recon_rss.numpy()), cmap='gray')




# So far, we have been looking at fully-sampled data.
# We can simulate under-sampled data by creating a mask and applying it to k-space.

from fastmri.data.subsample import RandomMaskFunc
mask_func = RandomMaskFunc(center_fractions=[0.04], accelerations=[8])  # Create the mask function object

# Recall: slice_kspace2 = T.to_tensor(slice_kspace) 
masked_kspace, mask = T.apply_mask(slice_kspace2, mask_func)   # Apply the mask to k-space
# ERROR
#   File "fastmri_recon_test.py", line 103, in <module>
#     masked_kspace, mask = T.apply_mask(slice_kspace2, mask_func)   # Apply the mask to k-space
# ValueError: too many values to unpack (expected 2)



# Let's see what the subsampled image looks like:

sampled_image = fastmri.ifft2c(masked_kspace)           # Apply Inverse Fourier Transform to get the complex image
sampled_image_abs = fastmri.complex_abs(sampled_image)   # Compute absolute value to get a real image
sampled_image_rss = fastmri.rss(sampled_image_abs, dim=0)

# plt.imshow(np.abs(sampled_image_rss.numpy()), cmap='gray')
undersampled_combined_coil_image=f'/home/ainedineen/motion_dnns/fastMRI/recon_test_out/undersam_combined_coil_img_{base_pre}.png'
plt.imsave(undersampled_combined_coil_image, np.abs(sampled_image_rss.numpy()), cmap='gray')