# IntroVAE-Pytorch

Pytorch Implementation for NeuraIPS2018 paper: 
[IntroVAE: Introspective Variational Autoencoders for Photographic Image Synthesis](https://arxiv.org/abs/1807.06358).

The rep. contains a basic implementation for IntroVAE.
However, due to no official implementation released,
some hyperparameters can only be guessed and can not reach the performance
as stated in paper.

![](assets/heart.gif)

# Training
1. Download [FFHQ](https://github.com/NVlabs/ffhq-dataset)
thumbnails128x128 subset,
[CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset
will also do.

2. Start Visdom server and run
```python
python3 main.py --name FFHQt
--root /path/to/FFHQ/thumbnails128x128 --batchsz 300
```
to train from strach.

Interrupt the training process if you found the image quality
not improving any more.

3. Interpolating in latent space.
```python
python3 eval.py --load FFHQ/ckpt/introvae_xxxx.mdl
--input /path/to/image1.png /path/to/image2.png
--output rect.png --n_interp 5
```

# Results
Tested for FFHQ 128x128 thumbnails.

- Original images
![](assets/xr_750000.png)

- Reconstrucitons
![](assets/xr_750000.png)

- Samples
![](assets/xr_750000.png)
