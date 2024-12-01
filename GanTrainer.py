import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
from models.discriminator import Discriminator
from models.generator import Generator
from utils.utils import load_data
import os

class GANTrainer:

    def __init__(self, config):

        super(GANTrainer, self).__init__()
        self.config = config

        self.device = config["device"]

        self.discriminator = Discriminator(config).to(self.device)
        self.generator = Generator(self.config).to(self.device)
        self.dataloader = load_data(self.config)
        self.adversarial_loss = nn.BCELoss()

        self.optimizer_disc = optim.Adam(self.discriminator.parameters(), lr=self.config["lr"])
        self.optimizer_gen = optim.Adam(self.generator.parameters(), lr=self.config["lr"])

        self.writer_fake = SummaryWriter(log_dir=f"runs/{self.config['dataset_name']}_GAN/fake")




    def run(self):
        step = 0
        for epoch in range(self.config["num_epochs"]):
            for batch_idx, (images, _) in enumerate(self.dataloader):

                real_images = images.to(torch.float32).to(self.device)
                valid = torch.ones((images.size(0), 1), requires_grad=False).to(self.device)
                fake = torch.zeros((images.size(0), 1), requires_grad=False).to(self.device)

                self.optimizer_gen.zero_grad()
                z = torch.randn((images.size(0), self.config["latent_dim"])).to(self.device)
                gen_images = self.generator(z)
                g_loss = self.adversarial_loss(self.discriminator(gen_images), valid)
                g_loss.backward()
                self.optimizer_gen.step()

                self.optimizer_disc.zero_grad()
                real_loss = self.adversarial_loss(self.discriminator(real_images), valid)
                fake_loss = self.adversarial_loss(self.discriminator(gen_images.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                self.optimizer_disc.step()

                if batch_idx == 0:
                    with torch.no_grad():
                        img_grid_fake = make_grid(gen_images, normalize=True)

                        save_image(img_grid_fake, os.path.join(self.config['save_dir_fake'], f"fake_epoch_{epoch + 1}.png"))

                        self.writer_fake.add_image(
                            "Fake Images", img_grid_fake, global_step=step
                        )
                        step += 1
            print(f"Epoch {epoch + 1}/{self.config['num_epochs']} - D Loss: {d_loss.item()} - G Loss: {g_loss.item()}")




