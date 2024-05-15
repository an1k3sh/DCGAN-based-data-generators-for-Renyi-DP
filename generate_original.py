import torch
import torchvision.utils as utils
import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

num_output = 64
z_dim = 100
image_size = 64
channels_img = 1
batch_size = 64

device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")

transforms = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(channels_img)], [0.5 for _ in range(channels_img)]
        ),
    ]
)

dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms, download=True)
dataset_size =  len(dataset)
loader = DataLoader(dataset, batch_size=batch_size, shuffle = True)

generated_img, _ = next(iter(loader))
plt.axis("off")
plt.imshow(np.transpose(utils.make_grid(generated_img, padding=2, normalize=True), (1,2,0)))
plt.savefig("results/generated_images_original.png", bbox_inches='tight')