import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import yaml
import torch
import glob
import numpy as np
from segmentation_models_pytorch import Unet
from models.metrics import Dice


parser = ArgumentParser(ArgumentDefaultsHelpFormatter)
parser.add_argument("--model_path", type=str, default="/results/msd/resnet18/checkpoints/last.ckpt", help="model path with file extention []")
parser.add_argument("--hparam_path", type=str, default="/data/msd/01_2d/config.pkl", help="Path of config file")
parser.add_argument("--data_path", type=str, default="/data/msd/01_2d/val", help="Path of config file")



if __name__ == "__main__":
    
    args = parser.parse_args()


    hparams = yaml.load(open(args.model_path, "rb"))

    dice = Dice(hparams["n_class"])

    
    model = Unet(
        encoder_name=hparams["encoder_name"],
        encoder_depth=hparams["encoder_depth"],
        encoder_weights=None,
        decoder_channels=hparams["decoder_channels"],
        in_channels=hparams["in_channels"],
        classes=hparams["n_class"],
    )

    model.load_state_dict(torch.load(args.model_path))

    model.freeze()
    model.eval()


    imgs = sorted(glob(os.path.join(args.data_path, f"*x*"))) 
    lbls = sorted(glob(os.path.join(args.data_path, f"*y*"))) 
    for img, lbl in zip(imgs, lbls):
        img_t = torch.tensor(np.load(img))
        lbl_t = torch.tensor(np.load(lbl))

        pred = model(img_t)

        dice.update(pred, lbl_t)
    
    print("mean dice loss: ", dice.compute().detach().numpy())


