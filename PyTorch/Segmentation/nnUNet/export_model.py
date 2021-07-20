import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import yaml
import torch
import pickle
from models.nn_unet import NNUnet


parser = ArgumentParser(ArgumentDefaultsHelpFormatter)
parser.add_argument("--ckptpath", type=str, default="/results/resnet18/checkpoints/last.ckpt", help="Path to model checkpoint")
parser.add_argument("--configfile", type=str, default="/data/msd/01_2d/config.pkl", help="Path of config file")
parser.add_argument("--outdir", type=str, required=True, help="folder path of model out")
parser.add_argument("--prefix", type=str, default="", help="prefix of model name")
parser.add_argument("--onnx", action="store_true", help="enable output of ONNX model")
parser.add_argument("--torchckpt", action="store_true", help="enable output of nn.torch checkpoint")
parser.add_argument("--hparamfile", action="store_true", help="enable output of hyperparams into YAML")


if __name__ == "__main__":
    
    args = parser.parse_args()

    ckpt_filepath = args.ckptpath

    config = pickle.load(open(args.configfile, "rb"))

    model = NNUnet.load_from_checkpoint(args.ckptpath, data_dir=os.path.dirname(args.configfile))

    model_state_dict = model.state_dict()
    hparams = vars(model.hparams["args"])
    hparams.update(config)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    hparamfilepth = os.path.join(args.outdir, "hyperparams.yml")

    model_path = os.path.join(args.outdir, args.prefix + "_Unet_" + hparams["encoder_name"])

    if args.onnx:
        input_channels = hparams["in_channels"]
        image_size =  hparams["patch_size"]
        batch_size = hparams["val_batch_size"]
        dummy_input = torch.randn(batch_size, input_channels, image_size[0], image_size[1], device='cuda')
        
        model.to_onnx(
            model_path + ".onnx", 
            dummy_input,
            opset_version=11,
            input_names = ['input'],   # the model's input names
            output_names = ['output'], # the model's output names
            dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                          'output' : {0 : 'batch_size'}})
    
    if args.torchckpt:
        torch.save(model_state_dict, model_path + ".pth")
    
    if args.hparamfile:
        with open(hparamfilepth, "w", encoding="utf-8") as f:
            yaml.dump(hparams, f, allow_unicode=True)



