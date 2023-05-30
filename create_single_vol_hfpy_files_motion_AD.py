# 03/07/22

# sub-001

import h5py
import numpy as np
import nibabel as nib 
import os

motion_001_nii = "/data2/motion_data/bids/sub-002/ses-1/func/sub-002_ses-1_task-PMmotion_acq-SbAxial_dir-AP_bold.nii.gz"
# motion_001_nii = "/data2/motion_data/bids/sub-001/ses-1/func/sub-001_ses-1_task-HHmotion_acq-SbAxial_dir-AP_bold.nii.gz"
motion_001_nib_img = nib.load(motion_001_nii)
print(f'nib_img: shape {motion_001_nib_img.shape}, type{type(motion_001_nib_img)},get_data_dtype() {motion_001_nib_img.get_data_dtype()} ') #AttributeError: 'Nifti1Image' object has no attribute 'dtype'
# nib_img: shape (64, 64, 44, 120), type<class 'nibabel.nifti1.Nifti1Image'>,get_data_dtype() int16 

# convert nib_image to np.ndarr
motion_001_nib_img_data = motion_001_nib_img.get_fdata()
print(f'motion_001_nib_img_data: shape {motion_001_nib_img_data.shape}, type{type(motion_001_nib_img_data)}, .dtype {motion_001_nib_img_data.dtype} ') #AttributeError: 'numpy.ndarray' object has no attribute 'get_data_dtype'
# motion_001_nib_img_data: shape (64, 64, 44, 120), type<class 'numpy.ndarray'>, .dtype float64 
# now a float!!!


print(len(motion_001_nib_img_data[0, 0, 0,:])) # 120
n_vols = len(motion_001_nib_img_data[0, 0, 0,:])


still_001_nii = "/data2/motion_data/bids/sub-002/ses-1/func/sub-002_ses-1_task-PMstill_acq-SbAxial_dir-AP_bold.nii.gz"
# still_001_nii = "/data2/motion_data/bids/sub-001/ses-1/func/sub-001_ses-1_task-HHstill_acq-SbAxial_dir-AP_bold.nii.gz"
still_001_nib_img = nib.load(still_001_nii)
print(f'nib_img: shape {still_001_nib_img.shape}, type{type(still_001_nib_img)},get_data_dtype() {still_001_nib_img.get_data_dtype()} ')
# convert nib_image to np.ndarr
still_001_nib_img_data = still_001_nib_img.get_fdata()
print(f'still_001_nib_img_data: shape {still_001_nib_img_data.shape}, type{type(still_001_nib_img_data)}, .dtype {still_001_nib_img_data.dtype} ') #AttributeError: 'numpy.ndarray' object has no attribute 'get_data_dtype'



# # test with tx file - works!!!!
# save_data_path = "/data2/motion_data/hdf5_single_vol_SB"
# for volume in range(n_vols): #no. from 1 --> 119
#     print(volume)
#     vol = f"vol_{volume}"
#     print(vol)

#     # where do we want this to save onsys!
#     # save here: /data2/motion_data/hdf5_single_vol_SB
#     file = f'sub-001_ses-1_task-HHmotion_acq-SbAxial_dir-AP_bold_{vol}.txt'
#     # with h5py.File(f"sub-001_ses-1_task-HHmotion_acq-SbAxial_dir-AP_bold_{vol}.hdf5", "w") as f:
#     with open(os.path.join(save_data_path, file), 'w') as f:
#         f.write('Create a new text file!')
    
#     # f.write('Create a new text file!')



save_data_path = "/data2/motion_data/hdf5_single_vol_SB"

# re-add loop and indent code when working with one file
# volume = 0
for volume in range(n_vols): #no. from 1 --> 119
    # print(volume)
    vol = f"vol_{volume}"
    # print(vol)

    still_vol = still_001_nib_img_data[:,:,:,volume]
    motion_vol = motion_001_nib_img_data[:,:,:,volume]

    file = f"sub-002_ses-1_task-PM_acq-SbAxial_dir-AP_bold_{vol}.hdf5"
    # Add in a moment

    # os.path.join(save_data_path, file)
    # with h5py.File(f"sub-001_ses-1_task-HHmotion_acq-SbAxial_dir-AP_bold_{vol}.hdf5", "w") as f:
    with h5py.File(os.path.join(save_data_path, file), "w") as f:
        # try without the dest = 
        # dset = f.create_dataset("motion", data=motion_vol)
        # dset2 = f.create_dataset("still", data=still_vol)

        f.create_dataset("motion", data=motion_vol)
        f.create_dataset("still", data=still_vol)
        
        # add attributes as header or to each dataset - both....
        # just header for now.....

        f.attrs['task'] = 'PM'
        f.attrs['sub'] = '002'
        f.attrs['acq'] = 'SbAxial'
        f.attrs['still_file'] = 'sub-002_ses-1_task-PMstill_acq-SbAxial_dir-AP_bold.nii.gz'
        f.attrs['motion_file'] = 'sub-002_ses-1_task-PMmotion_acq-SbAxial_dir-AP_bold.nii.gz'

        print(dict(f.attrs)) #{'acq': 'SbAxial', 'motion_file': 'sub-002_ses-1_task-PMmotion_acq-SbAxial_dir-AP_bold.nii.gz', 'still_file': 'ub-002_ses-1_task-PMstill_acq-SbAxial_dir-AP_bold.nii.gz', 'sub': '002', 'task': 'PM'}
        print(f'f: {list(f.keys())}') #f: ['motion', 'still']



    # /home/ainedineen/motion_dnns/fastMRI