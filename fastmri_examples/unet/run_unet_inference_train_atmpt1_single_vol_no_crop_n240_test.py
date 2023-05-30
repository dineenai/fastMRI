"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

# Adapted from run_trained_unet_inference_TEST_12_7_22.py
# TO DO: add parser for state dict file!!!!!

import argparse
import time
from collections import defaultdict
from pathlib import Path

import fastmri

# Insteaf of:
# import fastmri.data.transforms as T
# Use
# from motion_data.transformsMotionAD import UnetDataTransform 
# Try:
import motion_data.transformsMotionAD as T

import numpy as np
import requests
import torch

# replace:
# from fastmri.data import SliceDataset
# with
from motion_data.motion_mri_data import SliceDataset


from fastmri.models import Unet
from tqdm import tqdm


def run_unet_model(batch, model, device):
    image, _, mean, std, fname, slice_num, _ = batch

    output = model(image.to(device).unsqueeze(1)).squeeze(1).cpu()

    mean = mean.unsqueeze(1).unsqueeze(2)
    std = std.unsqueeze(1).unsqueeze(2)
    output = (output * std + mean).cpu()

    return output, int(slice_num[0]), fname[0]


# def run_inference(challenge, state_dict_file, data_path, output_path, device):
def run_inference(state_dict_file, data_path, output_path, device):
    model = Unet(in_chans=1, out_chans=1, chans=256, num_pool_layers=4, drop_prob=0.0)
    # download the state_dict if we don't have it
    print("inference test")
    # was loading the file brain_leaderboard_state_dict.pt and NOT the actual model!!!!
    # => added direct state_dict_file path below
    # state_dict_file = "/home/ainedineen/motion_dnns/fastMRI/fastmri_examples/unet/unet/unet_demo/checkpoints/epoch=36-step=295.ckpt"
    state_dict_file =  "/data2/motion_dnns/train_atmpt1_single_vol_no_crop_n240/checkpoints/epoch=48-step=206975.ckpt"

    # NOTE THAT: .ckpt == .pt  !!!!!
    # ou can name the checkpoint file as .pt or .pth instead of .ckpt.

    # Default devics is cuda
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device is: {device}, type {type(device)}")

    # model.load_state_dict(torch.load(state_dict_file))
    # https://programmerah.com/solved-runtimeerror-errors-in-loading-state_dict-for-model-missing-keys-in-state_dict-49328/
    # Added false to account for mismatch of keys!
    # RuntimeError: Error(s) in loading state_dict for Unet:
    #   Missing key(s) in state_dict:
    
    model.load_state_dict(torch.load(state_dict_file), False)

    print("Model Loaded") #Yes!

    model = model.eval()
    print(f"model.eval() has finished") #model.eval() has finished

    # data loader setup
    data_transform = T.UnetDataTransform()

    dataset = SliceDataset(
        root=data_path,
        transform=data_transform
    )

    print("Reaching the dataloader?")
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=1, batch_size=1)
    print("data loader has run?")

    # run the model
    start_time = time.perf_counter()
    print("model about to be sent to device")
    # try re adding? 12 /7/22 - not sure what it does bu tnot causng problems at present...
    outputs = defaultdict(list) #TMP
    model = model.to(device)

    print("model sent to device")

    for batch in tqdm(dataloader, desc="Running inference"):
        outputs = defaultdict(list)
        # print(f'test outputs: {outputs}')
        with torch.no_grad():
            output, slice_num, fname = run_unet_model(batch, model, device)
        # print(f"{fname}") #file_brain_AXFLAIR_200_6002441.h5

        outputs[fname].append((slice_num, output)) 

        # add to block for now!
        # save outputs
        for fname in outputs:
            outputs[fname] = np.stack([out for _, out in sorted(outputs[fname])])

        fastmri.save_reconstructions(outputs, output_path / "reconstructed_train_set")

        end_time = time.perf_counter()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # TO DO ADD THE FOLLOWING PARSER TO REMOVE MANUAL SETTING OF VARIABLE IN THE CODE!
    # parser.add_argument(
    #     "--ckpt_path",
    #     default="unet_knee_sc",
    #     type=str,
    #     help="Path to Ckeckpoint",
    # )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="Model to run",
    )
    parser.add_argument(
        "--state_dict_file",
        default=None,
        type=Path,
        help="Path to saved state_dict (will download if not provided)",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Path to subsampled data",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Path for saving reconstructions",
    )

    args = parser.parse_args()

    run_inference(
        args.state_dict_file,
        args.data_path,
        args.output_path,
        torch.device(args.device),
    )
