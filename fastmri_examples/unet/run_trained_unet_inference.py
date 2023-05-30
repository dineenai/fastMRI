"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path

import fastmri
import fastmri.data.transforms as T
import numpy as np
import requests
import torch
from fastmri.data import SliceDataset
from fastmri.models import Unet
from tqdm import tqdm

UNET_FOLDER = "https://dl.fbaipublicfiles.com/fastMRI/trained_models/unet/"
MODEL_FNAMES = {
    "unet_knee_sc": "knee_sc_leaderboard_state_dict.pt",
    "unet_knee_mc": "knee_mc_leaderboard_state_dict.pt",
    "unet_brain_mc": "brain_leaderboard_state_dict.pt",
}


def download_model(url, fname):
    response = requests.get(url, timeout=10, stream=True)

    chunk_size = 1 * 1024 * 1024  # 1 MB chunks
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(
        desc="Downloading state_dict",
        total=total_size_in_bytes,
        unit="iB",
        unit_scale=True,
    )

    print("in_download_test")

    # with open(fname, "wb") as fh:
    # /data2/fastMRI/fastMRI/data/multicoil_test
    # with open("/data/motion-dnns/fastMRI/data/brain_multicoil_test/pretrained_unet/"+fname, "wb") as fh:
    # NOTE already downloaded in pwd 1.99 GB - now loaded at both locations but /data2 not useful unless change subsequent code
    

    # You can name the checkpoint file as .pt or .pth instead of .ckpt.

    # /home/ainedineen/motion_dnns/fastMRI/fastmri_examples/unet/unet/unet_demo/checkpoints/epoch=36-step=295.ckpt
    # /data2/fastMRI/fastMRI/pretrained_unet_brain_mc/model/brain_leaderboard_state_dict.pt
    # with open("/data2/fastMRI/fastMRI/pretrained_unet_brain_mc/model/"+fname, "wb") as fh:
    with open("/home/ainedineen/motion_dnns/fastMRI/fastmri_examples/unet/unet/unet_demo/checkpoints/epoch=36-step=295.ckpt", "wb") as fh:
        for chunk in response.iter_content(chunk_size):
            progress_bar.update(len(chunk))
            fh.write(chunk)

    progress_bar.close()


def run_unet_model(batch, model, device):
    image, _, mean, std, fname, slice_num, _ = batch

    output = model(image.to(device).unsqueeze(1)).squeeze(1).cpu()

    mean = mean.unsqueeze(1).unsqueeze(2)
    std = std.unsqueeze(1).unsqueeze(2)
    output = (output * std + mean).cpu()

    return output, int(slice_num[0]), fname[0]


def run_inference(challenge, state_dict_file, data_path, output_path, device):
    model = Unet(in_chans=1, out_chans=1, chans=256, num_pool_layers=4, drop_prob=0.0)
    # download the state_dict if we don't have it
    print("inference test")
    if state_dict_file is None:
        print("empty test")
        # comment out as model not downloaded at "/data2/fastMRI/fastMRI/pretrained_unet_brain_mc/model/"
        # SOLVED downloaded in pwd where needed instead 
        if not Path(MODEL_FNAMES[challenge]).exists():
            print("download test")
            url_root = UNET_FOLDER
            download_model(url_root + MODEL_FNAMES[challenge], MODEL_FNAMES[challenge])
        # print("download test")
        # url_root = UNET_FOLDER 
        # download_model(url_root + MODEL_FNAMES[challenge], MODEL_FNAMES[challenge])
       
        state_dict_file = MODEL_FNAMES[challenge]

    # attempted code fox 14/12 source: 
    # import os
    # CUDA_VISIBLE_DEVICES = "0"
    # # # !CUDA_VISIBLE_DEVICES=0
    # num_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',').__len__()
    # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(f'{i}' for i in range(num_gpus))

    # Device is: cuda, type <class 'torch.device'>
    print(f"Device is: {device}, type {type(device)}")

    model.load_state_dict(torch.load(state_dict_file))
    model = model.eval()

    # data loader setup
    if "_mc" in challenge:
        data_transform = T.UnetDataTransform(which_challenge="multicoil")
    else:
        data_transform = T.UnetDataTransform(which_challenge="singlecoil")

    if "_mc" in challenge:
        dataset = SliceDataset(
            root=data_path,
            transform=data_transform,
            challenge="multicoil",
        )
    else:
        dataset = SliceDataset(
            root=data_path,
            transform=data_transform,
            challenge="singlecoil",
        )
    # ERROR: RuntimeError: DataLoader worker (pid 28841) is killed by signal: Killed. 
    # https://github.com/pytorch/pytorch/issues/8976
    # dataloader = torch.utils.data.DataLoader(dataset, num_workers=4)
    # dataloader = torch.utils.data.DataLoader(dataset, num_workers=0)
    # dataloader = torch.utils.data.DataLoader(dataset, num_workers=1, batch_size=1)
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=1, batch_size=1)

    # run the model
    start_time = time.perf_counter()
    # outputs = defaultdict(list) #TMP
    model = model.to(device)

    # Possible soln - decrease batch size BUT as not specified already 1 by default!

    # for batch in tqdm(dataloader, desc="Running inference"):
    #     with torch.no_grad():
    #         output, slice_num, fname = run_unet_model(batch, model, device)
    #     # print(f"TYPE of OUTPUTf: {type(output[fname])}") #TypeError: new(): invalid data type 'str'
    #     print(f"TYPE of OUTPUTf: {type(output)}") # <class 'torch.Tensor'>
    #     print(f"{fname}") #file_brain_AXFLAIR_200_6002441.h5

    #     # MAYBE TRY TO OVERRISE INNSTEAD OF APPEND?????
    #     outputs[fname].append((slice_num, output)) #AttributeError: 'numpy.ndarray' object has no attribute 'append'
    #     print(f"TYPE of OUTPUT1: {type(output)}")
    #     # add to block for now!
    #     # save outputs
    #     for fname in outputs:
    #         outputs[fname] = np.stack([out for _, out in sorted(outputs[fname])])

    #     fastmri.save_reconstructions(outputs, output_path / "reconstructions2")

    #     end_time = time.perf_counter()
    #     print(f"TYPE of OUTPUT2: {type(output)}")

    #     print(f"Elapsed time for {len(dataloader)} slices: {end_time-start_time}")


    for batch in tqdm(dataloader, desc="Running inference"):
        outputs = defaultdict(list)
        print(f'test outputs: {outputs}')
        with torch.no_grad():
            output, slice_num, fname = run_unet_model(batch, model, device)
        # print(f"TYPE of OUTPUT[0]: {type(output[0])}") # <class 'torch.Tensor'>
        # print(f"TYPE of OUTPUTf: {type(output)}") # <class 'torch.Tensor'>
        print(f"{fname}") #file_brain_AXFLAIR_200_6002441.h5

        # MAYBE TRY TO OVERRISE INNSTEAD OF APPEND?????
        # on second ieration get #AttributeError: 'numpy.ndarray' object has no attribute 'append'
        outputs[fname].append((slice_num, output)) #AttributeError: 'numpy.ndarray' object has no attribute 'append'
        # print(f"TYPE of OUTPUT1: {type(output)}")
        # add to block for now!
        # save outputs
        for fname in outputs:
            outputs[fname] = np.stack([out for _, out in sorted(outputs[fname])])

        fastmri.save_reconstructions(outputs, output_path / "reconstructions3")

        end_time = time.perf_counter()
        # print(f"TYPE of OUTPUT2: {type(output)}")  #<class 'torch.Tensor'>

        print(f"Elapsed time for {len(dataloader)} slices: {end_time-start_time}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--challenge",
        default="unet_knee_sc",
        choices=(
            "unet_knee_sc",
            "unet_knee_mc",
            "unet_brain_mc",
        ),
        type=str,
        help="Model to run",
    )
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
        args.challenge,
        args.state_dict_file,
        args.data_path,
        args.output_path,
        torch.device(args.device),
    )
