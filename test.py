import torch
import os

from GanTrainer import GANTrainer
from utils.utils import create_gif


def main(dataset_name='MNIST'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lr = 3e-4
    latent_dim = 100
    data_path = f"data/{dataset_name}"

    if dataset_name == 'MNIST':
        image_dim = (1, 28, 28)
        mean = (0.5,)
        std = (0.5,)
    elif dataset_name == 'CIFAR10':
        image_dim = (3, 32, 32)
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Training configuration
    batch_size = 64
    num_epochs = 200
    save_dir_fake = f"generated_gif_grids/{dataset_name}"

    os.makedirs(save_dir_fake, exist_ok=True)

    config = {
        'save_dir_fake': save_dir_fake,
        'device': device,
        'latent_dim': latent_dim,
        'dataset_name': dataset_name,
        'image_dim': image_dim,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'mean': mean,
        'std': std,
        'data_path': data_path,
        'lr': lr
    }

    gan_trainer = GANTrainer(config)
    gan_trainer.run()

    create_gif(config)


if __name__ == "__main__":
    main()

