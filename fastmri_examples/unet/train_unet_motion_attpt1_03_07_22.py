"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
# cp /home/ainedineen/motion_dnns/fastMRI/fastmri/data/transformsMotionAD.py /home/ainedineen/motion_dnns/fastMRI/fastmri_examples/unet
import os
import pathlib
from argparse import ArgumentParser

import pytorch_lightning as pl
from fastmri.data.mri_data import fetch_dir
# from fastmri.data.transforms import UnetDataTransform
# from fastmri.data.transformsMotionAD import UnetDataTransform
from motion_data.transformsMotionAD import UnetDataTransform #unclear why the above did not work...
import torchmetrics
# https://github.com/dineenai/fastMRI/blob/main/fastmri/pl_modules/unet_module.py
# https://github.com/dineenai/fastMRI/blob/main/fastmri/pl_modules/data_module.py

# from fastmri.pl_modules import FastMriDataModule, UnetModule #AttributeError: module 'pytorch_lightning' has no attribute 'metrics'
# # print(pl.__version__)
# from fastmri.pl_modules.motion_data_module import FastMriDataModule
# from fastmri.pl_modules.motion_unet_module import UnetModule

# mv motion_unet_module.py /home/ainedineen/motion_dnns/fastMRI/fastmri_examples/unet
# mv motion_data_module.py /home/ainedineen/motion_dnns/fastMRI/fastmri_examples/unet
from motion_pl_modules.motion_data_module import FastMriDataModule
from motion_pl_modules.motion_unet_module import UnetModule


# /home/ainedineen/motion_dnns/fastMRI/fastmri/data

def cli_main(args):
    pl.seed_everything(args.seed) #NB SHOULD THIS BE HERE?

    # ------------
    # data
    # ------------

    train_transform = UnetDataTransform() 
    val_transform = UnetDataTransform() 
    test_transform = UnetDataTransform()
    
    #TO DO FIND Where  these values are set - manually set changes for now!
    args.num_workers = 14

    # Double Check args before sending to dataloader:
    print(f'args.mode: {args.mode}\nargs.accelerator:{args.accelerator}')
    print(f'args.batch_size: {args.batch_size}\nargs.num_workers:{args.num_workers}')
    # args.mode: train
    # args.accelerator:ddp
    # args.batch_size: 1
    # args.num_workers:14  #defaults to 28!!
    

    # ptl data module - this handles data loaders
    data_module = FastMriDataModule(
        data_path=args.data_path,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        test_path=args.test_path,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
    )



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

    # https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/utilities/apply_func.py
    #     return elem_type(OrderedDict(out))
    # TypeError: first argument must be callable or None
    
    print(f"data_module is : {data_module}") #<motion_pl_modules.motion_data_module.FastMriDataModule object at 0x7f19340b44d0>
    if args.mode == "train":
        trainer.fit(model, datamodule=data_module) 
    elif args.mode == "test":
        trainer.test(model, datamodule=data_module)
    else:
        raise ValueError(f"unrecognized mode {args.mode}")


def build_args():
    parser = ArgumentParser()

    # basic args
    path_config = pathlib.Path("motion_dirs.yaml")

    # motion yaml
    # log_path: /data2/motion_dnns/train_atmpt1_single_vol_no_crop_n240
    # motion_data: /data2/motion_dnns/motion_data
                             
    # Original Yaml CONTENTS:
    # brain_path: /path/to/brain
    # knee_path: /path/to/knee
    # log_path: .

    # Log path needs to contain: Contains: checkpoints  lightning_logs

    # Benefits of yaml - keep paths out of code! AND use path object NOT sring!

    num_gpus = 2
    backend = "ddp"
    batch_size = 1 if backend == "ddp" else num_gpus

    print(f"BATCH SIZE = {batch_size}, should be 1 as backend = ddp, (would be 2 is gpu - this means that we are not using GPUs, should we be?")

    # set defaults based on optional directory config
    data_path = fetch_dir("motion_data", path_config)
    default_root_dir = fetch_dir("log_path", path_config)

    print(f'Paths: default_root_dir: {default_root_dir}\ndata_path: {data_path} ')

    # client arguments
    parser.add_argument(
        "--mode",
        default="train",
        choices=("train", "test"),
        type=str,
        help="Operation mode",
    )


    # data config with path to fastMRI data and batch size
    parser = FastMriDataModule.add_data_specific_args(parser) #https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/trainer/trainer.py
    # parser.set_defaults(data_path=data_path, batch_size=batch_size, test_path=None)
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

    # Should we be using a seed, is the seed relevant without the masks?
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
    # checkpoint_dir = os.path.join(args.default_root_dir, "checkpoints")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # if not checkpoint_dir.exists():
    #     checkpoint_dir.mkdir(parents=True)

    args.callbacks = [
        pl.callbacks.ModelCheckpoint(
            # dirpath=args.default_root_dir / "checkpoints",
            dirpath=os.path.join(args.default_root_dir, "checkpoints"),
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
