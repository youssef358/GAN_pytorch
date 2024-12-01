from torch import nn
import numpy as np

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config
        self.disc = nn.Sequential(
            nn.Linear(int(np.prod(self.config["image_dim"])), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, image):
        image = image.view(image.shape[0], -1)
        return self.disc(image)
