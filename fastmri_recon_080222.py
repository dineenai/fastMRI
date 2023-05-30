# fastmri_recon_080222.py
# Refined from fastmri_recon_test.py on the 8/2/22

import h5py
import numpy as np
from matplotlib import pyplot as plt
import os

# Reconstruction
recon = '/data2/fastMRI/fastMRI/pretrained_unet_brain_mc/reconstructions/file_brain_AXT2_205_2050152.h5'

hf_recon = h5py.File(recon)
base_recon = os.path.basename(recon)

print(base_recon)
print('Keys:', list(hf_recon.keys())) #Keys: ['reconstruction']
print('Attrs:', dict(hf_recon.attrs)) #Attrs: {}


# Prerecon
pre_recon = '/data2/fastMRI/fastMRI/data/multicoil_test_1_sample/file_brain_AXT2_205_2050152.h5'

base_pre = os.path.basename(pre_recon)
hf_pre = h5py.File(pre_recon)


print(base_pre)
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

volume_recon = hf_recon['reconstruction'][()]
print(volume_recon.dtype) #float32
print(volume_recon.shape) #(16, 1, 320, 320)


# chosen_slice = 8
# slice_kspace = volume_kspace[chosen_slice] # Choosing the chosen_slice-th slice of this volume
# slice_recon = volume_recon[chosen_slice]


# type="k-space/recon"
def show_coils(data, slice_nums, file="", base="", cmap=None):
    fig = plt.figure()
    for i, num in enumerate(slice_nums):
        plt.subplot(1, len(slice_nums), i + 1)
        plt.imshow(data[num], cmap=cmap)
        # plt.imsave(data[num], cmap=cmap)
    # save_coil_image = f'/home/ainedineen/motion_dnns/fastMRI/recon_test_out/{file}_prerecon_{base_pre}_sl_{chosen_slice}.png'
    # save_coil_image = f'/home/ainedineen/motion_dnns/fastMRI/recon_test_out2/{base_recon}_{file}_.png'
        save_coil_image = f'/home/ainedineen/motion_dnns/fastMRI/recon_test_out2/{base}_{file}.png'
    plt.savefig(save_coil_image)


for i in range(16):
    # Reconstruction has no coil imahe - just reconstruction
    slice_recon = volume_recon[i]
    show_coils(np.log(np.abs(slice_recon) + 1e-9), [0], file="recon_sl_"+str(i)+"_gray", base=base_recon, cmap='gray')



# # Let's see what the absolute value of k-space looks like (only applies to pre_recon)
# # same code shows value of reconstrcution!

#chose Slice from volume eg 6th slice 
slice_kspace = volume_kspace[6] 

# This shows coils 0, 5 and 10
show_coils(np.log(np.abs(slice_kspace) + 1e-9), [0, 5, 10], base=base_pre, file="kspace_coils")  



# NO LONGER APPLIES TO RECONSTRUCTION!!! - just original coils

# The fastMRI repo contains some utlity functions to convert k-space into image space.
# These functions work on PyTorch Tensors.
# The to_tensor function can convert Numpy arrays to PyTorch Tensors.

# Compute absolute value to get a real image

import fastmri
from fastmri.data import transforms as T

slice_kspace2 = T.to_tensor(slice_kspace)      # Convert from numpy array to pytorch tensor
slice_image = fastmri.ifft2c(slice_kspace2)           # Apply Inverse Fourier Transform to get the complex image
slice_image_abs = fastmri.complex_abs(slice_image)   # Compute absolute value to get a real image

show_coils(slice_image_abs, [0, 5, 10], file="abs_ksp_real_gray", base=base_pre, cmap='gray')


# Each coil in a multi-coil MRI scan focusses on a different region of the image.
# Coils can be combined into the full image using the Root-Sum-of-Squares (RSS) transform.

# TO DO set globa output path!!!

slice_image_rss = fastmri.rss(slice_image_abs, dim=0)
# plt.imshow(np.abs(slice_image_rss.numpy()), cmap='gray')
# here = '/home/ainedineen/motion_dnns/fastMRI/recon_test_out'
save_combined_coil_image=f'/home/ainedineen/motion_dnns/fastMRI/recon_test_out2/{base_pre}_combined_coil_img_gray.png'

plt.imsave(save_combined_coil_image, arr=np.abs(slice_image_rss.numpy()), cmap='gray')



# So far, we have been looking at fully-sampled data.
# We can simulate under-sampled data by creating a mask and applying it to k-space.

# NOT YET WORKING!!!!! -  look a train script and return (8/2/22)
# Temporarily commented out all of the below


# # Can I use random mask with brain!!!!!

# from fastmri.data.subsample import RandomMaskFunc
# from fastmri.data.subsample import EquiSpacedMaskFunc ##TRY

# # Create the mask function object

# # mask_func = RandomMaskFunc(center_fractions=[0.04], accelerations=[8])  # Create the mask function object

# mask_func = EquiSpacedMaskFunc(center_fractions=[0.04], accelerations=[4]) #4 above
# # mask_func = EquiSpacedMaskFunc(center_fractions=[0.04], accelerations=[8])  
# # mask_func = EquiSpacedMaskFunc()  # Create the mask function object #TypeError: __init__() missing 2 required positional arguments: 'center_fractions' and 'accelerations'


# # Recall: slice_kspace2 = T.to_tensor(slice_kspace) 
# # Recall: from fastmri.data import transforms as T
# masked_kspace, mask = T.apply_mask(slice_kspace2, mask_func)   # Apply the mask to k-space


# # ERROR
# #   File "fastmri_recon_test.py", line 103, in <module>
# #     masked_kspace, mask = T.apply_mask(slice_kspace2, mask_func)   # Apply the mask to k-space
# # ValueError: too many values to unpack (expected 2)




# # # Let's see what the subsampled image looks like:

# # sampled_image = fastmri.ifft2c(masked_kspace)           # Apply Inverse Fourier Transform to get the complex image
# # sampled_image_abs = fastmri.complex_abs(sampled_image)   # Compute absolute value to get a real image
# # sampled_image_rss = fastmri.rss(sampled_image_abs, dim=0)

# # # plt.imshow(np.abs(sampled_image_rss.numpy()), cmap='gray')
# # undersampled_combined_coil_image=f'/home/ainedineen/motion_dnns/fastMRI/recon_test_out/{base_pre}_undersam_combined_coil_img_gray.png'
# # plt.imsave(undersampled_combined_coil_image, np.abs(sampled_image_rss.numpy()), cmap='gray')