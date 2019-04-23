import os, glob
import os.path
import torch
import numpy as np
from   torch.utils.data import Dataset, DataLoader
import argparse
from   torchvision.utils import save_image
from   torchvision import datasets, transforms
from   model import IntroVAE
import visdom
import tqdm
import time
import PIL

def load_func(path):
    img = Image.open(x)
    assert len(img.shape) == 2 or len(img.shape) == 3
    if len(img.shpae) == 2:
        img = np.expand_dims(img, -1)
    assert img.shape[-1] == 1 or img.shape[-1] == 3
    if img.shape[2] == 1:
        img = img.repeat(3, -1)
    return img

def has_file_allowed_extension(filename, extensions):
    return filename.lower().endswith(extensions)

class DB(Dataset):
    def __init__(self, root, transform=None):
        extensions = ('.jpg', '.jpeg', '.png', '.ppm', \
                '.bmp', '.pgm', '.tif', '.tiff', '.webp')
        images = []
        for root, _, fnames in sorted(os.walk(root)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    images.append(path)
        self.images = images
        self.transform = transform

    def __getitem__(self, index):
        path = self.images[index]
        sample = PIL.Image.open(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.images)


def main(args):
    print(args)

    torch.manual_seed(22)
    np.random.seed(22)

    viz = visdom.Visdom(env=args.name)
    update = 'append' if args.retain_plot else None


    transform = transforms.Compose([
        transforms.Resize([args.imgsz, args.imgsz]),
        transforms.ToTensor()])

    db = DB(args.root, transform=transform)

    db_loader = DataLoader(db, batch_size=args.batchsz, shuffle=True, \
            num_workers=4, pin_memory=True)

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
        #print('epoch '+str(epoch+1)+'/'+str(args.epoch))
        #del epoch
        #print('iter vae enc-adv dec-adv ae enc dec')
        tqdm_iter = tqdm.tqdm(db_loader, desc='iter', \
                bar_format=str(args.batchsz)+': {n_fmt}/{total_fmt}'+\
                '[{elapsed}<{remaining},{rate_fmt}]'+'{postfix}')
        for x in tqdm_iter:
            time_data = time.time() - time_start

            iter_cnt += 1
            x = x.to(device, non_blocking=True)
            if x.shape[1] == 1:
                x = x.expand(-1,3,-1,-1)
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
                '''
                viz.line([encoder_loss.item()], [iter_cnt], \
                        win='encoder_loss', update=update,
                        opts=dict(title='encoder_loss'))
                viz.line([decoder_loss.item()], [iter_cnt], \
                        win='decoder_loss', update=update,
                        opts=dict(title='decoder_loss'))
                viz.line([loss_ae.item()], [iter_cnt], \
                        win='ae_loss', update=update,
                        opts=dict(title='ae_loss'))
                viz.line([reg_ae.item()], [iter_cnt], \
                        win='reg_ae', update=update,
                        opts=dict(title='reg_ae'))
                viz.line([encoder_adv.item()], [iter_cnt], \
                        win='encoder_adv', update=update,
                        opts=dict(title='encoder_adv'))
                viz.line([decoder_adv.item()], [iter_cnt], \
                        win='decoder_adv', update=update,
                        opts=dict(title='decoder_adv'))
                '''
                update = 'append'
            if iter_cnt % 250 == 0:
                last_disp = iter_cnt
                x, xr, xp = x[:8], xr[:8], xp[:8]
                # display images
                viz.histogram(xr[0].view(-1), win='xr_hist', \
                        opts=dict(title='xr_hist'))
                x, xr, xp = [img.clamp(0, 1) for img in (x, xr, xp)]
                viz.images(x, nrow=4, win='x', opts=dict(title='x'))
                viz.images(xr, nrow=4, win='xr', opts=dict(title='xr'))
                viz.images(xp, nrow=4, win='xp', opts=dict(title='xp'))
                # save images
                save_image(x, args.name+'/res/x_%d.jpg' % iter_cnt, nrow=4)
                save_image(xr, args.name+'/res/xr_%d.jpg' % iter_cnt, nrow=4)
                save_image(xp, args.name+'/res/xp_%d.jpg' % iter_cnt, nrow=4)
            if iter_cnt % 3000 == 0:
                last_ckpt = iter_cnt
                # save checkpoint
                torch.save(vae.state_dict(), \
                        args.name+'/ckpt/introvae_%d.mdl'%iter_cnt)

            time_vis = time.time() - time_start
            time_start = time.time()
            #tqdm_iter.set_postfix_str( \
            #        'data/vis %.1f/%.1f|cnt/loss/disp/ckpt %d%d%d%d'%( \
            #        time_data, time_vis, \
            #        iter_cnt, last_loss, last_disp, last_ckpt))
            # tqdm_iter.set_postfix_str( \
            #        'data/vis %.1f/%.1f, cnts %d/%d/%d/%d'%( \
            #        time_data, time_vis, \
            #        iter_cnt, last_loss, last_disp, last_ckpt))
            postfix = '[%d/%d/%d/%d]'%( \
                    iter_cnt, last_loss, last_disp, last_ckpt)
            if time_data >= 0.1:
                postfix += ' data %.1f'%time_data
            if time_vis >= 0.1:
                postfix += ' vis %.1f'%time_vis
            tqdm_iter.set_postfix_str(postfix)

    torch.save(vae.state_dict(), args.name+'/ckpt/introvae_%d.mdl'%iter_cnt)
    print('saved final ckpt:', args.name+'/ckpt/introvae_%d.mdl'%iter_cnt)

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

    args = argparser.parse_args()
    main(args)

