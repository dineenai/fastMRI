"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from argparse import ArgumentParser

import torch
from fastmri.models import Unet
from torch.nn import functional as F

# from .mri_module import MriModule
# /home/ainedineen/motion_dnns/fastMRI/fastmri_examples/unet/motion_pl_modules/volumewise
from motion_pl_modules.volumewise.motion_mri_module_volumewise import MriModule


class UnetModule(MriModule):
    """
    Unet training module.

    This can be used to train baseline U-Nets from the paper:

    J. Zbontar et al. fastMRI: An Open Dataset and Benchmarks for Accelerated
    MRI. arXiv:1811.08839. 2018.
    """

    def __init__(
        self,
        in_chans=1,
        out_chans=1,
        chans=32,
        num_pool_layers=4,
        drop_prob=0.0,
        lr=0.001,
        lr_step_size=40,
        lr_gamma=0.1,
        weight_decay=0.0,
        **kwargs,
    ):
        """
        Args:
            in_chans (int, optional): Number of channels in the input to the
                U-Net model. Defaults to 1.
            out_chans (int, optional): Number of channels in the output to the
                U-Net model. Defaults to 1.
            chans (int, optional): Number of output channels of the first
                convolution layer. Defaults to 32.
            num_pool_layers (int, optional): Number of down-sampling and
                up-sampling layers. Defaults to 4.
            drop_prob (float, optional): Dropout probability. Defaults to 0.0.
            lr (float, optional): Learning rate. Defaults to 0.001.
            lr_step_size (int, optional): Learning rate step size. Defaults to
                40.
            lr_gamma (float, optional): Learning rate gamma decay. Defaults to
                0.1.
            weight_decay (float, optional): Parameter for penalizing weights
                norm. Defaults to 0.0.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay

        self.unet = Unet(
            in_chans=self.in_chans,
            out_chans=self.out_chans,
            chans=self.chans,
            num_pool_layers=self.num_pool_layers,
            drop_prob=self.drop_prob,
        )

    # WHY   am i passing the still data to the model!!!!!!!
    # def forward(self, still):
    #     print(f"FILE: motion_unet_module.py\n  still {still.shape}, device {still.device}, dtype {still.dtype}")
    #     # still torch.Size([1, 64, 44]), device cuda:0, dtype torch.float64
    #     return self.unet(still.unsqueeze(1)).squeeze(1)
    def forward(self, motion):
        # print(f"FILE: motion_unet_module.py\n  motion {motion.shape}, device {motion.device}, dtype {motion.dtype}")
        # still torch.Size([1, 64, 44]), device cuda:0, dtype torch.float64
        # print(f'PARAMETERS TYPE: {self.parameters}') #PRINTS MODEL!
        # self.parameters.dtype() AttributeError: 'function' object has no attribute 'dtype'
        # self.parameters.dtype   AttributeError: 'function' object has no attribute 'dtype'
        
        # UNCLEAR HOW TO CHANGE DATA TYPE OF WEIGHTS FROM 32 to 64
        # RuntimeError: Input type (torch.cuda.DoubleTensor) and weight type (torch.cuda.FloatTensor) should be the same
        return self.unet(motion.unsqueeze(1)).squeeze(1)


    def training_step(self, batch, batch_idx):
        # print(f"FILE: motion_unet_module.py\n  batch.motion {batch.motion.shape}, device {batch.motion.device}, dtype {batch.motion.dtype}")
        # atch.still torch.Size([1, 64, 44]), device cuda:0, dtype torch.float64
        # PRINTS NB ALREADY CROPPED - FIND WHERE THIS training_step is called 
        # Where is the slicing happening: motion_mri_data.py def __getitem__(self, i: int):
        # batch.still torch.Size([1, 64, 44]), device cuda:0, dtype torch.float64
        output = self(batch.motion) #HALTS HERE - DOES NOT PRINT THE BELOW!
        # print(f"FILE: motion_unet_module.py\n  output: {output}")
        # loss = F.l1_loss(output, batch.still)
        loss = F.l1_loss(output, batch.target)
        # TARGET SHOULD BE STILL does it need to be renamed?
        # print(f"FILE: motion_unet_module.py\n  loss: {loss}")

        # NB the following is not printing => NOT getting this far!!
        # removing the print statment as now training :) 6/7/22
        # print(f"FILE: motion_unet_module.py\n  batch.still {batch.still.shape}, device {batch.still.device}, dtype {batch.still.dtype}")

        self.log("loss", loss.detach())

        return loss

    def validation_step(self, batch, batch_idx):
        # print(f'FILE: motion_unet_module.py\n  validation_step is running!') #NOT REACHING THIS STEP!!
        output = self(batch.motion)
        mean = batch.mean.unsqueeze(1).unsqueeze(2)
        std = batch.std.unsqueeze(1).unsqueeze(2)

        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "max_value": batch.max_value,
            "output": output * std + mean,
            "target": batch.target * std + mean,
            # "still": batch.still * std + mean, #RuntimeError: Expected key target in dict returned by validation_step.
            "val_loss": F.l1_loss(output, batch.target),
            # "val_loss": F.l1_loss(output, batch.still),
        }

    def test_step(self, batch, batch_idx):
        output = self.forward(batch.motion)
        mean = batch.mean.unsqueeze(1).unsqueeze(2)
        std = batch.std.unsqueeze(1).unsqueeze(2)

        return {
            "fname": batch.fname,
            "output": (output * std + mean).cpu().numpy(),
        }

    def configure_optimizers(self):
        optim = torch.optim.RMSprop(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )

        return [optim], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)

        # network params
        parser.add_argument(
            "--in_chans", default=1, type=int, help="Number of U-Net input channels"
        )
        parser.add_argument(
            "--out_chans", default=1, type=int, help="Number of U-Net output chanenls"
        )
        parser.add_argument(
            "--chans", default=1, type=int, help="Number of top-level U-Net filters."
        )
        parser.add_argument(
            "--num_pool_layers",
            default=4,
            type=int,
            help="Number of U-Net pooling layers.",
        )
        parser.add_argument(
            "--drop_prob", default=0.0, type=float, help="U-Net dropout probability"
        )

        # training params (opt)
        parser.add_argument(
            "--lr", default=0.001, type=float, help="RMSProp learning rate"
        )
        parser.add_argument(
            "--lr_step_size",
            default=40,
            type=int,
            help="Epoch at which to decrease step size",
        )
        parser.add_argument(
            "--lr_gamma", default=0.1, type=float, help="Amount to decrease step size"
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Strength of weight decay regularization",
        )

        return parser
