import os, glob
import os.path
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import torch
import numpy as np
from   torch.utils.data import Dataset, DataLoader
import argparse
from   torchvision.utils import save_image
from   model import IntroVAE
import visdom
import tqdm
import skimage
import time, random, math

colormap = torch.tensor([ \
        [0  ,0  ,0  ], \
        [0  ,0  ,255], \
        [0  ,255,0  ], \
        [255,0  ,0  ], \
        [0  ,255,255], \
        [255,255,0  ], \
        [255,0  ,255]])/255.0


class DB(Dataset):
    def __init__(self, args):
        extensions = ('.jpg', '.jpeg', '.png', '.ppm', \
                '.bmp', '.pgm', '.tif', '.tiff', '.webp')
        images = []
        for root, _, fnames in sorted(os.walk(args.root)):
            for fname in sorted(fnames):
                if fname.lower().endswith(extensions):
                    path = os.path.join(root, fname)
                    images.append(path)
        self.images = images
        self.imgsz = args.imgsz
        self.num_classes = args.num_classes
        self.data_aug = args.data_aug

    def updateTransform(self):
        rotate = math.pi/15.0
        trans = self.imgsz/20.0
        t1 = skimage.transform.EuclideanTransform( \
                translation=(self.imgsz/2.0+0.5))
        r = skimage.transform.EuclideanTransform( \
                rotation=(random.random()-0.5)*2*rotate)
        t2 = skimage.transform.EuclideanTransform( \
                translation=-(self.imgsz/2.0+0.5))
        t = skimage.transform.EuclideanTransform( \
                translation=(random.random()-0.5)*2*trans)
        self.transform = t + t2 + r + t1

    def getImage(self, index):
        path = self.images[index]
        sample = skimage.io.imread(path)
        sample = skimage.transform.resize(sample, (self.imgsz, self.imgsz), \
                mode='reflect', anti_aliasing=True)
        if self.data_aug:
            sample = skimage.transform.warp(sample, self.transform, \
                    mode='reflect')
        assert len(sample.shape) == 2 or len(sample.shape) == 3
        if len(sample.shape) == 2:
            sample = np.expand_dims(sample, -1)
        assert sample.shape[-1] == 1 or sample.shape[-1] == 3
        if sample.shape[2] == 1:
            sample = sample.repeat(3, -1)
        # change H,W,C to C,H,W
        sample = sample.transpose((2, 0, 1))
        if np.issubdtype(sample.dtype, np.integer):
            sample = sample/255.
        return torch.Tensor(sample)

    def getLabel(self, index):
        path = self.images[index]
        sample = skimage.io.imread(path)
        warped = sample.astype(np.float32)
        warped = skimage.transform.resize(warped, \
                (self.imgsz, self.imgsz), order=0, \
                mode='reflect', anti_aliasing=False)
        if self.data_aug:
            warped = skimage.transform.warp(warped, self.transform, \
                    order=0, mode='reflect')
        sample = warped.astype(sample.dtype)
        assert len(sample.shape) == 2
        assert np.issubdtype(sample.dtype, np.integer)
        sample = torch.LongTensor(sample)
        sample = torch.nn.functional.one_hot(sample, self.num_classes)
        return sample.permute(2,0,1).type(torch.Tensor)

    def __getitem__(self, index):
        self.updateTransform()
        if self.num_classes < 0:
            return self.getImage(index)
        else:
            return self.getLabel(index)

    def __len__(self):
        return len(self.images)


def main(args):
    print(args)

    #torch.manual_seed(22)
    #np.random.seed(22)

    viz = visdom.Visdom(env=args.name)
    update = 'append' if args.retain_plot else None

    db = DB(args)
    db_loader = DataLoader(db, batch_size=args.batchsz, shuffle=True, \
            num_workers=8, pin_memory=True)

    device = torch.device('cuda')
    vae = IntroVAE(args).to(device)
    params = filter(lambda x: x.requires_grad, vae.parameters())
    num = sum(map(lambda x: np.prod(x.shape), params))
    print('Total trainable tensors:', num)

    for path in [args.name, args.name+'/res', args.name+'/ckpt']:
        if not os.path.exists(path):
            os.mkdir(path)
            print('mkdir:', path)

    iter_cnt = 0
    if args.resume is not None and args.resume != 'None':
        if args.resume is '': # load latest
            ckpts = glob.glob(args.name+'/ckpt/*_*.mdl')
            if not ckpts:
                print('no avaliable ckpt found.')
                raise FileNotFoundError
            ckpts = sorted(ckpts, key=os.path.getmtime)
            # print(ckpts)
            ckpt = ckpts[-1]
            iter_cnt = int(ckpt.split('.')[-2].split('_')[-1])
            vae.load_state_dict(torch.load(ckpt))
            print('load latest ckpt from:', ckpt, iter_cnt)
        else: # load specific ckpt
            if os.path.isfile(args.resume):
                vae.load_state_dict(torch.load(args.resume))
                print('load ckpt from:', args.resume, iter_cnt)
            else:
                raise FileNotFoundError
    else:
        print('training from scratch...')

    # training.
    print('>>training Intro-VAE now...')
    vae.set_alpha_beta(args.alpha, args.beta)
    last_loss, last_ckpt, last_disp = 0, 0, 0
    time_data, time_vis = 0, 0
    time_start = time.time()
    for _ in tqdm.trange(args.epoch, desc='epoch'):
        tqdm_iter = tqdm.tqdm(db_loader, desc='iter', \
                bar_format=str(args.batchsz)+': {n_fmt}/{total_fmt}'+\
                '[{elapsed}<{remaining},{rate_fmt}]'+'{postfix}')
        for x in tqdm_iter:
            time_data = time.time() - time_start

            iter_cnt += 1
            x = x.to(device, non_blocking=True)
            xr, xp, AE, E_real, E_rec, E_sam, G_rec, G_sam = vae(x)

            time_start = time.time()
            if iter_cnt % 50 == 0:
                last_loss = iter_cnt
                # display loss
                viz.line( \
                        [[args.beta*AE, E_real, E_rec, E_sam, G_rec, G_sam]], \
                        [iter_cnt], win='train', update=update, \
                        opts= None if update else dict(title='training', \
                        legend=['betaAE', 'E_real', 'E_rec', 'E_sam', \
                        'G_rec', 'G_sam']))
                update = 'append'
            if iter_cnt % 250 == 0:
                last_disp = iter_cnt
                x, xr, xp = [img[:8].cpu() for img in (x, xr, xp)]
                if args.num_classes < 0:
                    # display images
                    viz.histogram(xr[0].view(-1), win='xr_hist', \
                            opts=dict(title='xr_hist'))
                else:
                    x, xr, xp = [colormap[img.argmax(1)].permute(0,3,1,2) \
                            for img in (x, xr, xp)]
                x, xr, xp = [img.clamp(0, 1) for img in (x, xr, xp)]
                viz.images(x, nrow=4, win='x', opts=dict(title='x'))
                viz.images(xr, nrow=4, win='xr', opts=dict(title='xr'))
                viz.images(xp, nrow=4, win='xp', opts=dict(title='xp'))
                # save images
                save_image(torch.cat([x,xr,xp], 0), \
                        args.name+'/res/x_xr_xp_%010d.png' % iter_cnt, nrow=4)
                #save_image(x, args.name+'/res/x_%d.jpg' % iter_cnt, nrow=4)
                #save_image(xr, args.name+'/res/xr_%d.jpg' % iter_cnt, nrow=4)
                #save_image(xp, args.name+'/res/xp_%d.jpg' % iter_cnt, nrow=4)
            if iter_cnt % 3000 == 0:
                last_ckpt = iter_cnt
                # save checkpoint
                torch.save(vae.state_dict(), \
                        args.name+'/ckpt/vae_%010d.mdl'%iter_cnt)

            time_vis = time.time() - time_start
            time_start = time.time()
            postfix = '[%d/%d/%d/%d]'%( \
                    iter_cnt, last_loss, last_disp, last_ckpt)
            if time_data >= 0.1:
                postfix += ' data %.1f'%time_data
            if time_vis >= 0.1:
                postfix += ' vis %.1f'%time_vis
            tqdm_iter.set_postfix_str(postfix)

    torch.save(vae.state_dict(), args.name+'/ckpt/vae_%010d.mdl'%iter_cnt)
    print('saved final ckpt:', args.name+'/ckpt/vae_%010d.mdl'%iter_cnt)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--imgsz', type=int, default=128, \
            help='imgsz')
    argparser.add_argument('--batchsz', type=int, default=64, \
            help='batch size')
    argparser.add_argument('--z_dim', type=int, default=256, \
            help='hidden latent z dim')
    argparser.add_argument('--epoch', type=int, default=1000, \
            help='epochs to train')
    argparser.add_argument('--margin', type=int, default=110, help='margin')
    argparser.add_argument('--alpha', type=float, default=0.25, \
            help='alpha * loss_adv')
    argparser.add_argument('--beta', type=float, default=0.5, \
            help='beta * ae_loss')
    argparser.add_argument('--lr', type=float, default=0.0002, \
            help='learning rate')
    argparser.add_argument('--root', type=str, default='data', \
            help='root/label/*.jpg')
    argparser.add_argument('--resume', type=str, default=None, \
            help='with ckpt path, set empty str to load latest ckpt')
    argparser.add_argument('--retain_plot', action='store_true', \
            help='set this flag to ratain existing plots in visdom')
    argparser.add_argument('--name', type=str, default='IntroVAE', \
            help='name for storage and checkpoint')
    argparser.add_argument('--num_classes', type=int, default=-1, \
            help='set to positive value to model shapes (e.g. segmentation)')
    argparser.add_argument('--data_aug', action='store_true', \
            help='do data augmentation by rotation and translation')

    args = argparser.parse_args()
    main(args)

