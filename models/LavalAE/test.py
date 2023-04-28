import os
import argparse
import torch
import imageio
import numpy as np

from model.Autoencoder import SkyEncoder, SkyDecoder

parser = argparse.ArgumentParser()
parser.add_argument("--encoder", type=str, default="./ckpt/enc_epoch_500")
parser.add_argument("--decoder", type=str, default="./ckpt/dec_epoch_500")
parser.add_argument("--input_dir", type=str, default="./samples")
parser.add_argument("--output_dir", type=str, default="./demo_output")
parser.add_argument("--use_cpu", action="store_true")

args = parser.parse_args()

# initialize models
device = torch.device('cuda') if not args.use_cpu else torch.device('cpu')
enc = SkyEncoder(cin=3, cout=64, activ='relu').to(device)
dec = SkyDecoder(cin=64, cout=3, activ='relu').to(device)

print("input path:", args.input_dir)
print("output path: ", args.output_dir)

# load checkpoints
print('loading encoder from ', args.encoder)
enc.load_state_dict(torch.load(args.encoder, map_location=device))
print('loading decoder from ', args.decoder)
dec.load_state_dict(torch.load(args.decoder, map_location=device))

enc.eval()
dec.eval()

with torch.no_grad():
    files = sorted([file for file in os.listdir(args.input_dir) if file.endswith('.hdr') or file.endswith('.exr')])
    for file in files:
        img_path = os.path.join(args.input_dir, file)
        print(img_path, end="")
        image = imageio.imread(img_path, format=('HDR-FI' if img_path.endswith('.hdr') else 'EXR'))
        image_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))).unsqueeze(0).to(device)
        image_tensor.clamp_min_(0.0)
        latent = enc(image_tensor)
        recon_tensor = dec(latent)
        recon_tensor.clamp_min_(0.0)
        recon_image = np.transpose(recon_tensor[0].cpu().numpy(), (1, 2, 0))
        os.makedirs(args.output_dir, exist_ok=True)
        imageio.imwrite(os.path.join(args.output_dir, 'recon_'+file), recon_image, format=('HDR-FI' if img_path.endswith('.hdr') else 'EXR'))
        print(" ->", os.path.join(args.output_dir, 'recon_'+file))

print()
print("All done. Results have been saved to %s" %(args.output_dir))
