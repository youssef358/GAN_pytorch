from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

from PIL import Image
import glob

def load_data(config):
    transform = transforms.Compose(
        [
         transforms.ToTensor(),
         transforms.Normalize(mean=config["mean"], std=config["std"]),
         ]
    )

    if config["dataset_name"] == "MNIST":
        os.makedirs("data/MNIST", exist_ok=True)
        dataset = datasets.MNIST(config["data_path"], download=True, transform=transform, )
    elif config["dataset_name"] == "CIFAR10":
        os.makedirs("data/CIFAR10", exist_ok=True)
        dataset = datasets.CIFAR10(config["data_path"], download=True, transform=transform)

    return DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)



def create_gif(config, keep_images=False):
    save_dir_fake = config["save_dir_fake"]
    os.makedirs(save_dir_fake, exist_ok=True)

    image_paths = sorted(glob.glob(f"generated_gif_grids/{config['dataset_name']}/fake_epoch_*.png"))
    selected_image_paths = image_paths[::10]

    images = [Image.open(img_path) for img_path in selected_image_paths]
    gif_path = os.path.join(save_dir_fake, f"fake_images_{config['dataset_name']}.gif")
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=1000, loop=0)

    if not keep_images:
        for img_path in image_paths:
            try:
                os.remove(img_path)
            except Exception as e:
                print(f"Error deleting {img_path}: {e}")
