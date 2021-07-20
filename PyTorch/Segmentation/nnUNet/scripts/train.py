# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from os.path import dirname
from subprocess import call

parser = ArgumentParser(ArgumentDefaultsHelpFormatter)
parser.add_argument("--task", type=str, default="01", help="Path to data")
parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
parser.add_argument("--fold", type=int, default=0, choices=[0, 1, 2, 3, 4], help="Fold number")
parser.add_argument("--dim", type=int, default=2, choices=[2, 3], help="Dimension of UNet")

parser.add_argument("--encoder_name", type=str, default="resnet34", help="encoders are listed in https://smp.readthedocs.io/en/latest/encoders.html")
parser.add_argument("--encoder_depth", type=int, default=5, help="a number of downsampled stages used in encoder with range [3, 5]")
parser.add_argument("--encoder_pretrained_source", type=str, default=None, help="encoder weights are list in smp link.")
parser.add_argument("--decoder_channels", type=str, default="256,128,64,32,16", help="delimited list of channels in decoder")

parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision")
parser.add_argument("--tta", action="store_true", help="Enable test time augmentation")
parser.add_argument("--data", type=str, default="/data/msd/01_2d", help="Path to data directory")
parser.add_argument("--results", type=str, default="/results/industrial", help="Path to results directory")
parser.add_argument("--logname", type=str, default="log", help="Name of dlloger output")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
parser.add_argument("--learning_rate", type=float, default=0.0005, help="Learning rate")
parser.add_argument("--focal", action="store_true", help="Use focal loss instead of cross entropy")
parser.add_argument("--resume_training", action="store_true", help="resume training")


if __name__ == "__main__":
    args = parser.parse_args()
    path_to_main = os.path.join(dirname(dirname(os.path.realpath(__file__))), "main.py")
    cmd = f"python {path_to_main} --exec_mode train --task {args.task} --save_ckpt "
    
    if args.encoder_pretrained_source is not None:
        cmd += f"--encoder_name {args.encoder_name} --encoder_depth {args.encoder_depth} --encoder_pretrained_source {args.encoder_pretrained_source} "
    else:
        cmd += f"--encoder_name {args.encoder_name} --encoder_depth {args.encoder_depth} "

    cmd += f"--decoder_channels {args.decoder_channels} "
    cmd += f"--data {args.data} "
    cmd += f"--results {args.results} "
    cmd += f"--logname {args.logname} "
    cmd += f"--dim {args.dim} "
    cmd += f"--batch_size {2 if args.dim == 3 else args.batch_size} "
    cmd += f"--val_batch_size {4 if args.dim == 3 else args.batch_size} "
    cmd += f"--fold {args.fold} "
    cmd += f"--gpus {args.gpus} "
    cmd += "--amp " if args.amp else ""
    cmd += "--tta " if args.tta else ""
    cmd += f"--learning_rate {args.learning_rate} "
    
    if args.focal:
        cmd += "--focal "
    if args.resume_training:
        cmd += "--resume_training "

    call(cmd, shell=True)
