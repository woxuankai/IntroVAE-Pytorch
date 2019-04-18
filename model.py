import  torch
from    torch import nn, optim
from    torch.nn import functional as F
import  math

class ResBlk(nn.Module):
    def __init__(self, kernels, chs):
        """
        :param kernels: [1, 3, 3], as [kernel_1, kernel_2, kernel_3]
        :param chs: [ch_in, 64, 64, 64], as [ch_in, ch_out1, ch_out2, ch_out3]
        :return:
        """
        assert len(chs)-1 == len(kernels), "mismatching between chs and kernels"
        assert all(map(lambda x: x%2==1, kernels)), "odd kernel size only"
        super(ResBlk, self).__init__()
        layers = []
        for idx in range(len(kernels)):
            layers += [nn.Conv2d(chs[idx], chs[idx+1], kernels[idx], \
                        padding = kernels[idx]//2), \
                        nn.LeakyReLU(0.2, True)]
        layers.pop() # remove last activation
        self.net = nn.Sequential(*layers)
        self.shortcut = nn.Sequential()
        if chs[0] != chs[-1]: # convert from ch_int to ch_out3
            self.shortcut = nn.Conv2d(chs[0], chs[-1], kernel_size=1)
        self.outAct = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.outAct(self.shortcut(x) + self.net(x))

class Encoder(nn.Module):
    def __init__(self, imgsz, ch, z_dim):
        """
        :param imgsz:
        :param ch: base channels
        :param z_dim: latent space dim
        """
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()

        self.layers.append(nn.Sequential( \
                nn.Conv2d(3, ch, kernel_size=5, stride=1, padding=2),
                nn.LeakyReLU(0.2, True), 
                nn.AvgPool2d(2, stride=None, padding=0)))

        # [b, ch_cur, imgsz, imgsz] => [b, ch_next, mapsz, mapsz]
        mapsz = imgsz // 2
        ch_cur = ch
        ch_next = ch_cur * 2
        while mapsz > 8: # util [b, ch_, 8, 8]
            # add resblk
            self.layers.append(nn.Sequential( \
                    ResBlk([1, 3, 3], [ch_cur]+[ch_next]*3), \
                    nn.AvgPool2d(kernel_size=2, stride=None)))
            mapsz = mapsz // 2
            ch_cur = ch_next
            ch_next = ch_next * 2 if ch_next < 512 else 512 # set max ch=512

        # 8*8 -> 4*4
        self.layers.append(nn.Sequential( \
                ResBlk([3, 3], [ch_cur, ch_next, ch_next]), \
                nn.AvgPool2d(kernel_size=2, stride=None)))
        mapsz = mapsz // 2

        # 4*4 -> 4*4
        self.layers.append(nn.Sequential( \
                ResBlk([3, 3], [ch_next, ch_next, ch_next])))

        # convert h_dim to 2*z_dim
        self.z_net = nn.Linear(ch_next*mapsz*mapsz, 2*z_dim)

        # just for print
        x = torch.randn(2, 3, imgsz, imgsz)
        print('Encoder:', list(x.shape), end='=>')
        with torch.no_grad():
            for layer in self.layers[:-1]:
                x = layer(x)
                print(list(x.shape), end='=>')
            x = self.layers[-1](x)
            x = x.view(x.shape[0], -1)
            print(list(x.shape), end='=>')
            x = self.z_net(x)
            print(list(x.shape), end='=>')
            x = x.chunk(2, dim=1)
            print(list(x[0].shape), list(x[1].shape))
        print(self.layers)
        print(self.z_net)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        for layer in self.layers:
            x = layer(x)
        x = x.view(x.shape[0], -1)
        mu, logvar = self.z_net(x).chunk(2, dim=1)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, imgsz, ch, z_dim):
        """
        :param imgsz:
        :param ch: base channels
        :param z_dim: latent space dim
        """
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()

        self.layers.insert(0, nn.Sequential( \
                nn.Conv2d(ch, 3, kernel_size=5, stride=1, padding=2),
                nn.LeakyReLU(0.2, True)))

        self.layers.insert(0, nn.Sequential( \
                nn.Upsample(scale_factor=2, mode='nearest'),
                ResBlk([3, 3], [ch, ch, ch])))

        mapsz = imgsz // 2
        ch_cur = ch
        ch_next = ch_cur * 2
        while mapsz > 16: # util [b, ch_, 16, 16]
            self.layers.insert(0, nn.Sequential( \
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    ResBlk([1, 3, 3], [ch_next]+[ch_cur]*3)))
            mapsz = mapsz // 2
            ch_cur = ch_next
            ch_next = ch_next * 2 if ch_next < 512 else 512 # set max ch=512

        # 16*16, 8*8
        for _ in range(2):
            self.layers.insert(0, nn.Sequential( \
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    ResBlk([3, 3], [ch_next]+[ch_cur]*2)))
            mapsz = mapsz // 2
            ch_cur = ch_next
            ch_next = ch_next * 2 if ch_next < 512 else 512 # set max ch=512

        # 4*4
        self.layers.insert(0, nn.Sequential( \
                ResBlk([3, 3], [ch_next]+[ch_cur]*2)))

        # fc
        self.z_net = nn.Sequential( \
                nn.Linear(z_dim, ch_next*mapsz*mapsz),
                nn.ReLU(True))

        # just for print
        x = torch.randn(2, z_dim)
        print('Decoder:', list(x.shape), end='=>')
        x = self.z_net(x)
        print(list(x.shape), end='=>')
        with torch.no_grad():
            x = x.view(x.shape[0], -1, 4, 4)
            x = self.layers[0](x)
            for layer in self.layers[1:]:
                print(list(x.shape), end='=>')
                x = layer(x)
            print(list(x.shape))
        print(self.z_net)
        print(self.layers)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        x = torch.randn(2, z_dim)
        x = x.view(x.shape[0], -1, 4, 4)
        for layer in self.layers:
            x = layer(x)
        return x


class IntroVAE(nn.Module):
    def __init__(self, args):
        """
        :param imgsz:
        :param z_dim: h_dim is the output dim of encoder, and we use z_net net to convert it from
        h_dim to 2*z_dim and then splitting.
        """
        super(IntroVAE, self).__init__()

        imgsz = args.imgsz
        z_dim = args.z_dim

        # set first conv channel as 16
        self.encoder = Encoder(imgsz, 16)

        # get h_dim of encoder output
        x = torch.randn(2, 3, imgsz, imgsz)
        z_ = self.encoder(x)
        h_dim = z_.size(1)


        # sample
        z, mu, log_sigma2 = self.reparameterization(z_)

        # create decoder by z_dim
        self.decoder = Decoder(imgsz, z_dim)
        out = self.decoder(z)

        # print
        print('IntroVAE x:', list(x.shape), 'z_:', list(z_.shape), 'z:', list(z.shape), 'out:', list(out.shape))


        self.alpha = args.alpha # for adversarial loss
        self.beta = args.beta # for reconstruction loss
        self.gamma = args.gamma # for variational loss
        self.margin = args.margin # margin in eq. 11
        self.z_dim = z_dim # z is the hidden vector while h is the output of encoder
        self.h_dim = h_dim

        self.optim_encoder = optim.Adam(self.encoder.parameters(), lr=args.lr)
        self.optim_decoder = optim.Adam(self.decoder.parameters(), lr=args.lr)


    def set_alph_beta_gamma(self, alpha, beta, gamma):
        """
        this func is for pre-training, to set alpha=0 to transfer to vilina vae.
        :param alpha: for adversarial loss
        :param beta: for reconstruction loss
        :param gamma: for variational loss
        :return:
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def reparam(self, mu, logvar):
        # sample from normal dist
        eps = torch.randn_like(mu)
        # reparameterization trick
        std = torch.exp(0.5*logvar)
        z = mu + std * eps
        return z

    def kld(self, mu, logvar):
        """
        compute the kl divergence between N(mu, sigma^2) and N(0, 1)
        :param mu: [b, z_dim]
        :param log_sigma2: [b, z_dim]
        :return:
        """
        batchsz = mu.size(0)
        kl = - 0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()

        return kl

    #def output_activation(self, x):
    #    """
    #    :param x:
    #    :return:
    #    """
    #    return torch.tanh(x)

    def forward(self, x):
        """
        The notation used here all come from Algorithm 1, page 6 of official paper.
        can refer to Figure7 in page 15 as well.
        :param x: [b, 3, 1024, 1024]
        :return:
        """
        batchsz = x.size(0)

        # algorithm 1 in arxiv paper
        # 4. Z <- Enc(X)
        mu, logvar = self.encoder(x)
        z = self.reparam(mu, logvar)
        # 5. Z_p <- N(0,1)
        zp = torch.randn_like(z)
        # 6. X_r <- Dec(Z), X_p <- Dec(Z_p)
        xr = self.decoder(z)
        xp = self.decoder(zp)
        # 7. L_AE <- L_AE(X_r, X)
        ae = F.mse_loss(xr, x, reduction='sum')
        # 8. Z_r <- Enc(ng(X_r)), Z_pp <- Enc(ng(X_p))
        mur, logvarr = self.encoder(xr.detach())
        mupp, logvarpp = self.encoder(xp.detach())
        # 9. L^E_adv <- L_REG(Z) + 
        #  \alpha{[m - L_REG(Z_r)]^+ + [m - L_REG(Z_pp)]^+}
        reg = self.kld(mu, logvar)
        regr = self.kld(mur, logvarr)
        regpp = self.kld(mupp, logvarpp)
        Eadv = reg + \
                self.alpha*F.relu(self.margin - regr) +
                F.relu(self.margin - regpp)
        # 10. update \phi_E with L^E_adv + \betaL_AE
        self.optim_encoder.zero_grad()
        (Eadv + self.beta*ae).backward()
        self.optim_encoder.step()
        # 11. Z_r <- Enc(X_r), Z_pp <- Enc(X_p)
        mur, logvarr = self.encoder(xr)
        mupp, logvarpp = self.encoder(xp)
        # 12. L^G_adv <- \aplha{L_REG(Z_r)+L_REG(Z_pp)}
        regr = self.kld(mur, logvarr)
        regpp = self.kld(mupp, logvarpp)
        Gadv = self.alpha*regr + regpp
        # 13 update \theta_G with L^G_adv + \betaL_AE
        self.optim_decoder.zero_grad()
        (Gadv + self.beta*ae).backward()
        self.optim_decoder.step()

        return encoder_loss, decoder_loss, reg_ae, encoder_adv, decoder_adv, loss_ae, xr, xp, \
               regr, regr_ng, regpp, regpp_ng


if __name__ == '__main__':
    Encoder(128,16,256)
    Decoder(128,16,256)
