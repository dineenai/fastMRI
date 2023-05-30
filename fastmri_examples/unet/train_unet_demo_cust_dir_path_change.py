"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import pathlib
from argparse import ArgumentParser

import pytorch_lightning as pl
from fastmri.data.mri_data import fetch_dir
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import UnetDataTransform
import torchmetrics
# https://github.com/dineenai/fastMRI/blob/main/fastmri/pl_modules/unet_module.py
# https://github.com/dineenai/fastMRI/blob/main/fastmri/pl_modules/data_module.py
from fastmri.pl_modules import FastMriDataModule, UnetModule #AttributeError: module 'pytorch_lightning' has no attribute 'metrics'
# print(pl.__version__)

def cli_main(args):
    pl.seed_everything(args.seed)

    # ------------
    # data
    # ------------
    # this creates a k-space mask for transforming input data
    mask = create_mask_for_mask_type(
        args.mask_type, args.center_fractions, args.accelerations
    )
    # use random masks for train transform, fixed masks for val transform
    train_transform = UnetDataTransform(args.challenge, mask_func=mask, use_seed=False)
    val_transform = UnetDataTransform(args.challenge, mask_func=mask)
    test_transform = UnetDataTransform(args.challenge)
    # ptl data module - this handles data loaders
    data_module = FastMriDataModule(
        data_path=args.data_path,
        challenge=args.challenge,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        test_split=args.test_split,
        test_path=args.test_path,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        # num_workers=args.num_workers,
        num_workers=12 #default is 4 - try for now.....
        distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
    )

    # print("HERE:")
    # print(args.num_workers)
    # print(args.batch_size)

    # ------------
    # model
    # ------------
    model = UnetModule(
        in_chans=args.in_chans,
        out_chans=args.out_chans,
        chans=args.chans,
        num_pool_layers=args.num_pool_layers,
        drop_prob=args.drop_prob,
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
    )

    # ------------
    # trainer
    # ------------
    trainer = pl.Trainer.from_argparse_args(args) #https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/trainer/trainer.py

    print("TRAINER")

    # ------------
    # run
    # ------------
    print(f"MODEL TYPE: {type(model)}") #<class 'fastmri.pl_modules.unet_module.UnetModule'>
    print(f"data_module TYPE: {type(data_module)}") #<class 'fastmri.pl_modules.data_module.FastMriDataModule'>
    
    # https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/utilities/apply_func.py
    #     return elem_type(OrderedDict(out))
    # TypeError: first argument must be callable or None
    
    if args.mode == "train":
        trainer.fit(model, datamodule=data_module) #TypeError: first argument must be callable or None
    elif args.mode == "test":
        trainer.test(model, datamodule=data_module)
    else:
        raise ValueError(f"unrecognized mode {args.mode}")


def build_args():
    parser = ArgumentParser()

    # basic args

    # /data/motion-dnns/fastMRI
    # 
    path_config = pathlib.Path("../../fastmri_dirs.yaml") #TRY AS IS - RELATIVE TO BRAIN PATH which is good
    # REM: data_path = "/data2/fastMRI/fastMRI/data/multicoil_test"
    
    # path_config = pathlib.Path("/home/ainedineen/motion_dnns/fastMRI/fastmri_dirs.yaml")
    # path_config = pathlib.Path("/data/motion-dnns/fastMRI/fastmri_dirs.yaml") #TRY INSTEAD
    # print(f"PATH TO CONFIG:{path_config}")
    num_gpus = 2
    backend = "ddp"
    batch_size = 1 if backend == "ddp" else num_gpus

    # DO THESE NEED TO BE UPDATED/REMOVED OR WAS SETTING IN SHELL SCRIPT SIFFICIENT?
    # set defaults based on optional directory config
    # data_path = fetch_dir("knee_path", path_config)
    data_path = fetch_dir("brain_path", path_config)
    # default_root_dir = fetch_dir("log_path", path_config) / "unet" / "unet_demo"
    default_root_dir = "/data2/fastMRI/fastMRI/train_unet_brain_mc"

    # client arguments
    parser.add_argument(
        "--mode",
        default="train",
        choices=("train", "test"),
        type=str,
        help="Operation mode",
    )

    # data transform params
    parser.add_argument(
        "--mask_type",
        choices=("random", "equispaced_fraction"),
        default="random",
        type=str,
        help="Type of k-space mask",
    )
    parser.add_argument(
        "--center_fractions",
        nargs="+",
        default=[0.08],
        type=float,
        help="Number of center lines to use in mask",
    )
    parser.add_argument(
        "--accelerations",
        nargs="+",
        default=[4],
        type=int,
        help="Acceleration rates to use for masks",
    )

    # data config with path to fastMRI data and batch size
    parser = FastMriDataModule.add_data_specific_args(parser) #https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/trainer/trainer.py
    parser.set_defaults(data_path=data_path, batch_size=batch_size, test_path=None)

    # module config
    parser = UnetModule.add_model_specific_args(parser)
    parser.set_defaults(
        in_chans=1,  # number of input channels to U-Net
        out_chans=1,  # number of output chanenls to U-Net
        chans=32,  # number of top-level U-Net channels
        num_pool_layers=4,  # number of U-Net pooling layers
        drop_prob=0.0,  # dropout probability
        lr=0.001,  # RMSProp learning rate
        lr_step_size=40,  # epoch at which to decrease learning rate
        lr_gamma=0.1,  # extent to which to decrease learning rate
        weight_decay=0.0,  # weight decay regularization strength
    )

    # trainer config
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        gpus=num_gpus,  # number of gpus to use
        replace_sampler_ddp=False,  # this is necessary for volume dispatch during val
        accelerator=backend,  # what distributed version to use
        seed=42,  # random seed
        deterministic=True,  # makes things slower, but deterministic
        default_root_dir=default_root_dir,  # directory for logs and checkpoints
        max_epochs=50,  # max number of epochs
    )

    args = parser.parse_args()

    # configure checkpointing in checkpoint_dir
    checkpoint_dir = args.default_root_dir / "checkpoints"
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    args.callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=args.default_root_dir / "checkpoints",
            save_top_k=True,
            verbose=True,
            monitor="validation_loss",
            mode="min",
        )
    ]

    # set default checkpoint if one exists in our checkpoint directory
    if args.resume_from_checkpoint is None:
        ckpt_list = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)
        if ckpt_list:
            args.resume_from_checkpoint = str(ckpt_list[-1])

    return args


def run_cli():
    args = build_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    cli_main(args)


if __name__ == "__main__":
    run_cli()
