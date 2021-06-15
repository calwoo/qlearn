import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import torchvision.transforms as transforms
from torchvision.datasets import MNIST

import pytorch_lightning as pl


class MNISTData(pl.LightningDataModule):
    def __init__(self, batch_size=256):
        super(MNISTData, self).__init__()
        self.batch_size = batch_size

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        self.dims = (1, 28, 28)
        self.n_classes = 10

    def prepare_data(self):
        MNIST(root="./data", train=True, download=True)
        MNIST(root="./data", train=False, download=True)
    
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            mnist_all = MNIST(root="./data", train=True, transform=self.transform)
            self.train_set, self.val_set = random_split(mnist_all, [55000, 5000])
        
        if stage == "test" or stage is None:
            self.test_set = MNIST(root="./data", train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=4)


### GAN
class Generator(nn.Module):
    def __init__(self, latent_dim=50):
        super(Generator, self).__init__()

        def block(in_dim, out_dim, normalize=True):
            layers = [nn.Linear(in_dim, out_dim)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_dim, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(*block(latent_dim, 128, normalize=False),
                                   *block(128, 256),
                                   *block(256, 512),
                                   *block(512, 1024),
                                   nn.Linear(1024, 28*28),
                                   nn.Tanh())

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, 28, 28)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(nn.Linear(28*28, 512),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Linear(512, 256),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Linear(256, 1),
                                   nn.Sigmoid())

    def forward(self, img):
        # flatten
        img = img.view(img.size(0), -1)
        out = self.model(img)
        return out


class GAN(pl.LightningModule):
    def __init__(self, latent_dim=100, lr=0.0002):
        super(GAN, self).__init__()
        self.save_hyperparameters()

        self.generator = Generator(latent_dim=latent_dim)
        self.discriminator = Discriminator()

        # fix latent sample for validation
        self.validation_z = torch.randn(8, self.hparams.latent_dim)
        self.example_input_array = torch.zeros(2, self.hparams.latent_dim)

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch

        # sample noise
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim)
        z = z.type_as(imgs)

        self.generated_imgs = self.generator(z)

        # generator
        if optimizer_idx == 0:
            fake_labels = torch.ones(imgs.size(0), 1)
            fake_labels = fake_labels.type_as(imgs)

            g_loss = F.binary_cross_entropy(self.discriminator(self.generated_imgs), fake_labels)
            return {"loss": g_loss}

        # discriminator
        if optimizer_idx == 1:
            real_labels = torch.ones(imgs.size(0), 1)
            real_labels = real_labels.type_as(imgs)
            real_loss = F.binary_cross_entropy(self.discriminator(imgs), real_labels)

            fake_labels = torch.zeros(imgs.size(0), 1)
            fake_labels = fake_labels.type_as(imgs)
            fake_loss = F.binary_cross_entropy(self.discriminator(self.generated_imgs), fake_labels)

            d_loss = (real_loss + fake_loss) / 2
            return {"loss": d_loss}

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999))
        return [opt_g, opt_d], []

    
# train
data = MNISTData(batch_size=256)
model = GAN(latent_dim=50, lr=0.0002)

trainer = pl.Trainer(max_epochs=5)
trainer.fit(model, data)
