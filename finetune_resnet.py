import torch
import torch.backends
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader

device = torch.device("mps" if(torch.backends.mps.is_available()) else "cpu")
print(device.type)
learning_rate = 2e-4
batch_size = 128
image_size = 64
channels_img = 1
z_dim = 100
num_epochs = 10
features_disc = 64
features_gen = 64
noise_stddev = 0.0
noise_dim = None

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
loader = DataLoader(dataset, batch_size=batch_size)

model = models.resnet50(pretrained=True).to(device)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).to(device)
model.fc = nn.Linear(2048, 10, bias=True).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

total = 0
correct = 0

for epoch in range(num_epochs):
    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()    
        outputs = model(images).to(device)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if(batch_idx%100 == 0):
            print('Epoch: {} Batch: {}/{} loss: {}'.format(epoch, batch_idx, len(loader), loss.item()))

torch.save(model, "resnet_mnist.pth")