from torch import nn
import numpy as np



class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        self.gen = nn.Sequential(
            nn.Linear(self.config["latent_dim"], 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, int(np.prod(config["image_dim"]))),
            nn.Tanh()
        )

    def forward(self, z):
        image = self.gen(z)
        image = image.view(image.size(0), *(self.config["image_dim"]))
        return image