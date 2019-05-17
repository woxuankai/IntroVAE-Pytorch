import torch
from torchvision.utils import make_grid, save_image
import numpy as np
import argparse
import skimage
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
    args.data_aug = False
    db = DB(args)
    db.images = args.input
    imgs = [img for img in db]
    x = torch.stack(imgs, dim=0).to(device)
    mu, logvar = vae.encoder(x)
    feature = mu.cpu().numpy()
    np.save(args.output, feature)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--imgsz', type=int, default=128, \
            help='imgsz')
    argparser.add_argument('--z_dim', type=int, default=256, \
            help='hidden latent z dim')
    argparser.add_argument('--load', type=str, required=True, \
            help='checkpoint to load')
    argparser.add_argument('--input', type=str, required=True, nargs='*', \
            help='input images')
    argparser.add_argument('--output', type=str, required=True, \
            help='output path')
    argparser.add_argument('--num_classes', type=int, default=-1, \
            help='set to positive value to model shapes (e.g. segmentation)')

    args = argparser.parse_args()
    main(args)

