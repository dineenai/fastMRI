# Aine Learns HDF5
import h5py
import numpy as np
from matplotlib import pyplot as plt
import os








# training file structure
train_file = "/data2/fastMRI/fastMRI/data/multicoil_train_1_sample/file_brain_AXT2_205_2050121.h5"
hf_train_file = h5py.File(train_file, 'r')


# val_sample = '/data2/fastMRI/fastMRI/data/multicoil_val/file_brain_AXT1POST_205_6000087.h5'
# hf_val_sample = h5py.File(val_sample, 'r')


# # test file structure

# post_recon = '/data2/fastMRI/fastMRI/pretrained_unet_brain_mc/reconstructions/file_brain_AXT2_205_2050152.h5'
# hf_post = h5py.File(post_recon, 'r')

# # prerecon
# pre_recon = '/data2/fastMRI/fastMRI/data/multicoil_test_1_sample/file_brain_AXT2_205_2050152.h5'

# hf_pre = h5py.File(pre_recon, 'r')

# for name in hf_pre:
#     print('test')
#     print(name)
# # same as keys!!

# base_pre = os.path.basename(pre_recon)
# base_post = os.path.basename(post_recon)
# print(base_pre)


# def printname(name):
#     print(name)
# hf_pre.visit(printname)
# # ismrmrd_header
# # kspace
# # mask

print(hf_train_file["ismrmrd_header"][()])
# print('Attrs:', dict(hf_post.attrs)) #Attrs: {}


# could use for loop......

# print('Keys:', list(hf_post.keys())) #Keys: ['reconstruction']
# print('Attrs:', dict(hf_post.attrs)) #Attrs: {}



# print('Keys:', list(hf_pre.keys())) #Keys: ['reconstruction']
# print('Attrs:', dict(hf_pre.attrs)) #Attrs: {}

# x = hf_pre['kspace']
# print(f'x.attrs {dict(x.attrs)} END') #{}
# print(f'x.attrs {x.attrs} END')  #<Attributes of HDF5 object at 140572128058352>

# print(f'Train file: Keys:, {list(hf_train_file.keys())}, Attrs:, {dict(hf_train_file.attrs)}')
# print('Keys:', list(hf_train_file.keys())) #Keys: ['reconstruction']
# print('Attrs:', dict(hf_train_file.attrs)) #Attrs: {}


# print('Keys:', list(hf_val_sample.keys())) #Keys: ['reconstruction']
# print('Attrs:', dict(hf_val_sample.attrs)) #Attrs: {}


# Create a hdf5 file to train the net: /data2/fastMRI/fastMRI/motion_data/bids

# Load our data:
# Nibabel

import os
import nibabel as nib # conda install nibabel <-- does NOT work ==>  pip install nibabel

motion_001_nii = '/data2/motion_dnns/motion_data/bids/sub-001/ses-1/func/sub-001_ses-1_task-HHmotion_acq-SbAxial_dir-AP_bold.nii.gz'

# could add header as metadate to each file dataset, for now I propose the following attributes
# eg for file: sub-001_ses-1_task-HHstill_acq-SbAxial_dir-AP_bold
# sub: sub-001
# acquisition task-HHstill_acq-SbAxial_dir-AP_bold
# is the training dependant on any attributes - presumably not.....
# master file with 
# for now create seperate hdf5 file for each pair of volumes (do i need to confrm that they are aligned as desired????? psychopy SHOULD take care of this)

# TO DO ammend the training code to enable accepting one master H5py file and chopping segments of this file as part of ataloading 




motion_001_nib_img = nib.load(motion_001_nii)
print('\n')
# print(motion_001_nib_img) #metadata
print(motion_001_nib_img.header) #metadata


# data shape (64, 64, 44, 120)
# split into 120 files?????

print(f'nib_img: shape {motion_001_nib_img.shape}, type{type(motion_001_nib_img)},get_data_dtype() {motion_001_nib_img.get_data_dtype()} ') #AttributeError: 'Nifti1Image' object has no attribute 'dtype'
# nib_img: shape (64, 64, 44, 120), type<class 'nibabel.nifti1.Nifti1Image'>,get_data_dtype() int16 



# how do i get the datatype of a nib_img ? img.get_data_dtype()

# access to the image data as a NumPy array
# .get_fdata()

motion_001_nib_img_data = motion_001_nib_img.get_fdata()
print(f'motion_001_nib_img_data: shape {motion_001_nib_img_data.shape}, type{type(motion_001_nib_img_data)}, .dtype {motion_001_nib_img_data.dtype} ') #AttributeError: 'numpy.ndarray' object has no attribute 'get_data_dtype'
# motion_001_nib_img_data: shape (64, 64, 44, 120), type<class 'numpy.ndarray'>, .dtype float64 
# now a float!!!

print(motion_001_nib_img_data[32, 32, 22, 60]) #309.0


# hdf5 file name eg sub-001_ses-1_task-HH_acq-SbAxial_dir-AP_bold_vol1

# what inputs does the training take
# what is the current loss function
# recon.....

#  print sample value

# 
# /data2/fastMRI/fastMRI/motion_data/bids/sub-001/ses-1/func/sub-001_ses-1_task-HHstill_acq-SbAxial_dir-AP_bold.nii.gz




# # TO DO create fake train file to check that I understand the h5py concpets properly
# with h5py.File("mytest_train_file.hdf5", "w") as f:
#     # dset = f.create_dataset("mydataset", (100,), dtype='i')
#     dset = f.create_dataset("reconstruction_rss", (16, 320, 320), dtype='float32')
#     # instead try
#     # dset = f.create_dataset("reconstruction_rss", data=motion_001_nib_img_data[:,:,:,0])
#     #  initialize the dataset to an existing NumPy array by providing the data parameter:
#     dset = f.create_dataset("motion_001_nib_img_data_vol_0", data=motion_001_nib_img_data[:,:,:,0])
#     print(f'f: {f}')#f: <HDF5 file "mytest_train_file.hdf5" (mode r+)>
#     print(f'f: {f.keys()}') #f: <KeysViewHDF5 ['reconstruction_rss']>
#     print(f'f: {list(f.keys())}') #f: ['reconstruction_rss']
#     print(f['reconstruction_rss'].dtype) #float32
#     print(f['reconstruction_rss'].shape) #(16, 320, 320)

#     print(f'f: {f}')#f: <HDF5 file "mytest_train_file.hdf5" (mode r+)>
#     print(f'f: {f.keys()}') #f: <KeysViewHDF5 ['motion_001_nib_img_data_vol_0', 'reconstruction_rss']>

#     print(f'f: {list(f.keys())}') #f: ['motion_001_nib_img_data_vol_0', 'reconstruction_rss']
#     print(f['motion_001_nib_img_data_vol_0'].dtype) #float64
#     print(f['motion_001_nib_img_data_vol_0'].shape) #(64, 64, 44)

#     # Take from bids
#     f.attrs['acq'] = 'HH'
#     f.attrs['sub'] = 'sub-001'
#     print(f.attrs) #<Attributes of HDF5 object at 140252331826832>
#     print(dict(f.attrs)) #{'acq': 'HH'}

    

#     # now try adding a np array

# # Try checking if motion_001_nib_img_data[:,:,:,0] == 


# print(f'Train file: Keys:, {list(hf_train_file.keys())}, Attrs:, {dict(hf_train_file.attrs)}')
# print(hf_train_file['kspace']) #<HDF5 dataset "kspace": shape (16, 20, 640, 320), type "<c8">
# print(hf_train_file['kspace'].dtype) #complex64
# print(hf_train_file['reconstruction_rss']) #<HDF5 dataset "reconstruction_rss": shape (16, 320, 320), type "<f4">
# print(hf_train_file['reconstruction_rss'].dtype) #float32

# # get type of nib img

# # print('Keys:', list(hf_train_file.keys())) #Keys: ['reconstruction']
# # print('Attrs:', dict(hf_train_file.attrs)) #Attrs: {}


# Header infor for nii files is in json sidecars!

# /data2/motion_dnns/motion_data/bids
# task-PMmotion_acq-SbAxial_bold.json
# task-PMstill_acq-SbAxial_bold.json


# task-HHmotion_acq-SbAxial_bold.json
# task-HHstill_acq-SbAxial_bold.json


# ISMRMRD HEADER from h5py file

import xml.etree.ElementTree as etree

    # def _retrieve_metadata(self, fname):
# with h5py.File(fname, "r") as hf:

from typing import Sequence

def et_query(
    root: etree.Element,
    qlist: Sequence[str],
    namespace: str = "http://www.ismrm.org/ISMRMRD",
) -> str:
    """
    ElementTree query function.

    This can be used to query an xml document via ElementTree. It uses qlist
    for nested queries.

    Args:
        root: Root of the xml to search through.
        qlist: A list of strings for nested searches, e.g. ["Encoding",
            "matrixSize"]
        namespace: Optional; xml namespace to prepend query.

    Returns:
        The retrieved data as a string.
    """
    s = "."
    prefix = "ismrmrd_namespace"

    ns = {prefix: namespace}

    for el in qlist:
        s = s + f"//{prefix}:{el}"

    value = root.find(s, ns)
    if value is None:
        raise RuntimeError("Element not found")

    return str(value.text)


hf = hf_train_file 
et_root = etree.fromstring(hf["ismrmrd_header"][()])

print(f'et_root: {et_root}') # <Element '{http://www.ismrm.org/ISMRMRD}ismrmrdHeader' at 0x7f233280e8f0>s


# in
enc = ["encoding", "encodedSpace", "matrixSize"]
enc_size = (
    int(et_query(et_root, enc + ["x"])),
    int(et_query(et_root, enc + ["y"])),
    int(et_query(et_root, enc + ["z"])),
)

print(f'enc_size: {enc_size}')

print(f'enc_size: {enc_size}') #(640, 320, 1)
rec = ["encoding", "reconSpace", "matrixSize"]
recon_size = (
    int(et_query(et_root, rec + ["x"])),
    int(et_query(et_root, rec + ["y"])),
    int(et_query(et_root, rec + ["z"])),
)
print(f'recon_size: {recon_size}') #(320, 320, 1)

lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
enc_limits_center = int(et_query(et_root, lims + ["center"]))
enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1

padding_left = enc_size[1] // 2 - enc_limits_center
padding_right = padding_left + enc_limits_max

print(f'enc_limits_center: {enc_limits_center}')
print(f'enc_limits_max: {enc_limits_max}')

num_slices = hf["kspace"].shape[0]

metadata = {
"padding_left": padding_left,
"padding_right": padding_right,
"encoding_size": enc_size,
"recon_size": recon_size,
}

print(f'recon_size: {recon_size}') #(320, 320, 1)
# return metadata, num_slices



# load one of our new files
# /home/ainedineen/motion_dnns/fastMRI

our_hf_file = '/data2/motion_dnns/motion_data/hdf5_single_vol_SB/sub-001_ses-1_task-HH_acq-SbAxial_dir-AP_bold_vol_94.hdf5'


our_hf = h5py.File(our_hf_file, 'r')

print(dict(our_hf.attrs))
print(list(our_hf.keys()))
print(our_hf["still"].shape) #(64, 64, 44)
print(our_hf["motion"].shape)

# {'acquisition': 'AXT2', 'max': 0.0008552039059386988, 'norm': 0.21647753744514137, 'patient_id': '7c1bb815070c788fd21a16005e3484b99ca5f9420ed33731933d623b839eacbc'}
# {'acq': 'SbAxial', 'motion_file': 'sub-001_ses-1_task-HHmotion_acq-SbAxial_dir-AP_bold.nii.gz', 'still_file': 'sub-001_ses-1_task-HHstill_acq-SbAxial_dir-AP_bold.nii.gz', 'sub': '001', 'task': 'HH'}