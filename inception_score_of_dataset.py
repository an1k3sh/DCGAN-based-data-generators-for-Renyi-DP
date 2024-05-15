import torch
import torch.nn as nn
import torchvision.utils as utils
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

num_output = 64
z_dim = 100
image_size = 64
channels_img = 1
batch_size = 128

device = torch.device("mps" if(torch.backends.mps.is_available()) else "cpu")
print(device.type)

transforms = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(channels_img)], [0.5 for _ in range(channels_img)]
        ),
    ]
)
dataset = datasets.MNIST(root="dataset/", train=False, transform=transforms, download=True)
dataset_size =  len(dataset)
loader = DataLoader(dataset, batch_size=batch_size, shuffle = True)
test_size = len(dataset)


resnet_load_path = "resnet_mnist.pth"
resnet_mnist = torch.load(resnet_load_path, map_location=torch.device('cpu')).to(device)

preds = []
for batch_idx, (images, _) in enumerate(loader):
    images = images.to(device)
    y_pred = resnet_mnist(images).to(device)
    y_pred = nn.functional.softmax(y_pred, dim = 1)
    preds.append(y_pred)
    print(1)
    if (batch_idx == 8):
        break

preds = torch.cat(preds, 0)
print(preds.shape)
scores = []
splits = 10
for i in range(splits):
    part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
    kl = part * (torch.log(part) - torch.log(torch.unsqueeze(torch.mean(part, 0), 0)))
    kl = torch.mean(torch.sum(kl, 1))
    scores.append(torch.exp(kl))
scores = torch.stack(scores)
score_val = torch.exp(torch.mean(torch.log(scores)))
print("Inception score is: %.4f" % (score_val))