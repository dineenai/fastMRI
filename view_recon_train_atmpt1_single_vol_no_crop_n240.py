# fastmri_recon_test.py

# /data2/motion_dnns/train_atmpt1_single_vol_no_crop_n240
# /data2/motion_dnns/train_atmpt1_single_vol_no_crop_n240
#  reconstructed_test_set
#  reconstructed_train_set 

# Look at test set first

# Save outputs: /data2/motion_dnns/train_atmpt1_single_vol_no_crop_n240/reconstruction_pngs

# QUESTION
#  are these reconstructions volumewise??????????

# /home/ainedineen/motion_dnns/fastMRI

import h5py
import numpy as np
from matplotlib import pyplot as plt

import os


# prerecon
# pre_recon = "/data2/motion_dnns/motion_data/test/sub-001_ses-1_task-HH_acq-SbAxial_dir-AP_bold_vol_106.hdf5"
pre_recon = "/data2/motion_dnns/motion_data/train/sub-001_ses-1_task-HH_acq-SbAxial_dir-AP_bold_vol_99.hdf5"
# pre_recon = '/data2/fastMRI/fastMRI/data/multicoil_test_1_sample/file_brain_AXT2_205_2050152.h5'


hf_pre = h5py.File(pre_recon, 'r')

base_pre = "pre_" + os.path.basename(pre_recon)

print(base_pre)

print('Keys:', list(hf_pre.keys())) #Keys: ['motion', 'still']
print('Attrs:', dict(hf_pre.attrs)) #Attrs: {'acq': 'SbAxial', 'motion_file': 'sub-001_ses-1_task-HHmotion_acq-SbAxial_dir-AP_bold.nii.gz', 'still_file': 'sub-001_ses-1_task-HHstill_acq-SbAxial_dir-AP_bold.nii.gz', 'sub': '001', 'task': 'HH'}

still_image = hf_pre['still'][()]
print(still_image.shape) #(64, 64, 44)

# Fast pri k space is 
# print(volume_kspace.shape) #(16, 16, 640, 320)      #(number of slices, number of coils, height, width)

# post_recon =  "/data2/motion_dnns/train_atmpt1_single_vol_no_crop_n240/reconstructed_test_set/sub-001_ses-1_task-HH_acq-SbAxial_dir-AP_bold_vol_106.hdf5"
post_recon =  "/data2/motion_dnns/train_atmpt1_single_vol_no_crop_n240/reconstructed_train_set/sub-001_ses-1_task-HH_acq-SbAxial_dir-AP_bold_vol_99.hdf5"
# post_recon = '/data2/fastMRI/fastMRI/pretrained_unet_brain_mc/reconstructions/file_brain_AXT2_205_2050152.h5'
hf_post = h5py.File(post_recon, 'r')
base_post = "post_" + os.path.basename(post_recon)
print(base_post)

print('Keys:', list(hf_post.keys())) #Keys: ['reconstruction']
print('Attrs:', dict(hf_post.attrs))  #Attrs: {}

volume_recon = hf_post['reconstruction'][()]

print(volume_recon.shape) #CAUTION: only one slice - find source! (1, 1, 64, 64)
# 1 slice, 1 coil, 44x64 image

print(f"volume_recon.shape {volume_recon.shape}")  #volume_recon.shape (1, 1, 64, 64)

chosen_slice = 0

slice_recon = volume_recon[chosen_slice]
print(f"slice_recon.shape {slice_recon.shape}") #(1, 64, 64)
# slice_recon = volume_recon

vol_prerecon = hf_pre['still'][()]
print(f"vol_prerecon.shape {vol_prerecon.shape}")
slice_prerecon = vol_prerecon[:,:,chosen_slice]
print(f"slice_prerecon.shape {slice_prerecon.shape}") 

slice_prerecon = slice_prerecon.reshape((1, slice_prerecon.shape[0], slice_prerecon.shape[1]))
print(f"slice_prerecon.shape reshaped to add channel {slice_prerecon.shape}") 

def show_coils(data, slice_nums, file="", cmap=None):
    fig = plt.figure()
    for i, num in enumerate(slice_nums):
        plt.subplot(1, len(slice_nums), i + 1)
        plt.imshow(data[num], cmap=cmap)


        # plt.imsave(data[num], cmap=cmap)

    # save_coil_image = '/home/ainedineen/motion_dnns/fastMRI/recon_test_out/'+str(file)+'_coil_image_prerecon_'+str(base_pre)+'_sl_'+str(chosen_slice)+'.png'
    # base_pre
    # save_coil_image = f'/home/ainedineen/motion_dnns/fastMRI/recon_test_out/{file}_prerecon_{base_pre}_sl_{chosen_slice}.png'
    
    # /data2/motion_dnns/train_atmpt1_single_vol_no_crop_n240/reconstruction_pngs
    # save_coil_image = f'/data2/motion_dnns/train_atmpt1_single_vol_no_crop_n240/reconstruction_pngs/{file}_{base_post}_sl_{chosen_slice}.png'
    save_coil_image = f'/data2/motion_dnns/train_atmpt1_single_vol_no_crop_n240/reconstructed_train_pngs/PRE{file}_{base_post}_sl_{chosen_slice}.png'
    # save_coil_image = f'/data2/motion_dnns/train_atmpt1_single_vol_no_crop_n240/reconstruction_pngs/{file}_{base_post}.png'
    plt.savefig(save_coil_image)


# show_coils(np.log(np.abs(slice_recon) + 1e-9), [0], file="recon_coil_gray", cmap='gray')  

show_coils(np.log(np.abs(slice_prerecon) + 1e-9), [0], file="recon_coil_gray", cmap='gray')  

# volume_kspace = hf_pre['kspace'][()]
# print(volume_kspace.dtype) #complex64
# print(volume_kspace.shape) #(16, 16, 640, 320)      #(number of slices, number of coils, height, width)

# chosen_slice = 8
# slice_kspace = volume_kspace[chosen_slice] # Choosing the chosen_slice-th slice of this volume

# print("RECON")
# volume_recon = hf_post['reconstruction'][()]
# print(volume_recon.dtype) #float32
# print(volume_recon.shape) #(16, 1, 320, 320)
# # slice_recon = volume_kspace[chosen_slice]
# slice_recon = volume_recon[chosen_slice]
# slice_recon0 = volume_recon[0]
# slice_recon15 = volume_recon[15]
# slice_recon4 = volume_recon[4]
# slice_recon8 = volume_recon[8]
# slice_recon12 = volume_recon[12]
# # COIL IMAGE NOT WORKING????? #####- supposed to look like this?

# # shows coils 




# # Compute absolute value to get a real image

# import fastmri
# from fastmri.data import transforms as T




# slice_recon2 = T.to_tensor(slice_recon)      # Convert from numpy array to pytorch tensor
# slice_image_recon = fastmri.ifft2c(slice_recon2)           # Apply Inverse Fourier Transform to get the complex image
# slice_image_abs_recon = fastmri.complex_abs(slice_image_recon)   # Compute absolute value to get a real image



# # So far, we have been looking at fully-sampled data.
# # We can simulate under-sampled data by creating a mask and applying it to k-space.



# # Let's see what the subsampled image looks like:

# sampled_image = fastmri.ifft2c(masked_kspace)           # Apply Inverse Fourier Transform to get the complex image
# sampled_image_abs = fastmri.complex_abs(sampled_image)   # Compute absolute value to get a real image
# sampled_image_rss = fastmri.rss(sampled_image_abs, dim=0)

# # plt.imshow(np.abs(sampled_image_rss.numpy()), cmap='gray')
# undersampled_combined_coil_image=f'/home/ainedineen/motion_dnns/fastMRI/recon_test_out/undersam_combined_coil_img_{base_pre}.png'
# plt.imsave(undersampled_combined_coil_image, np.abs(sampled_image_rss.numpy()), cmap='gray')