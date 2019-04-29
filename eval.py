import torch
from torchvision.utils import make_grid, save_image
import numpy as np
import argparse
from model import IntroVAE
from main import DB, colormap

def main(args):
    print(args)

    device = torch.device('cuda')
    torch.set_grad_enabled(False)
    args.alpha, args.beta, args.margin, args.lr = 0, 0, 0, 0
    vae = IntroVAE(args).to(device)
    vae.load_state_dict(torch.load(args.load))
    print('load ckpt from:', args.load)

    args.root = '/dev/null'
    db = DB(args)
    db.images = args.input
    imgs = [img for img in db]
    x = torch.stack(imgs, dim=0).to(device)
    mu, logvar = vae.encoder(x)

    z = torch.nn.functional.interpolate(mu.permute(1,0).unsqueeze(0), \
            size=args.n_interp, mode='linear', align_corners=False \
            ).squeeze(0).permute(1,0)
    xr = vae.decoder(z)
    if args.num_classes >= 0:
        x, xr = [colormap[img.argmax(1)].permute(0,3,1,2) \
                for img in (x, xr)]
    max_len = max(x.shape[0], xr.shape[0])
    output = torch.cat( \
            [x[:]] + [torch.zeros_like(x[0:1])]*(max_len - x.shape[0]) + \
            [xr[:]] + [torch.zeros_like(xr[0:1])]*(max_len - xr.shape[0]), \
            dim=0)
    save_image(output, args.output, nrow=max_len, range=(0,1))
    print('write output to :', args.output)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--imgsz', type=int, default=128, \
            help='imgsz')
    argparser.add_argument('--z_dim', type=int, default=256, \
            help='hidden latent z dim')
    argparser.add_argument('--n_interp', type=int, default=3, \
            help='number of images to be interpolated')
    argparser.add_argument('--load', type=str, required=True, \
            help='checkpoint to load')
    argparser.add_argument('--input', type=str, required=True, nargs='*', \
            help='checkpoint to load')
    argparser.add_argument('--output', type=str, required=True, \
            help='output path')
    argparser.add_argument('--num_classes', type=int, default=-1, \
            help='set to positive value to model shapes (e.g. segmentation)')

    args = argparser.parse_args()
    main(args)

