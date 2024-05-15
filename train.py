import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import Discriminator, Generator, initialize_weights
import pandas as pd
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device.type)
learning_rate = 2e-4
batch_size = 128
image_size = 64
channels_img = 1
z_dim = 100
num_epochs = 25
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

os.mkdir("trained_models/adam_Implementation_"+str(noise_stddev))
data_cols = {'Epoch' : [], 'Gen' : [], 'Disc' : []}
df = pd.DataFrame(data_cols)
df.to_csv("trained_models/loss_data_"+str(noise_stddev)+".csv", index=False)

dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms, download=True)
dataset_size =  len(dataset)
loader = DataLoader(dataset, batch_size=batch_size)
gen = Generator(z_dim, channels_img, features_gen).to(device)
disc = Discriminator(channels_img, features_disc).to(device)
initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.SGD(gen.parameters(), lr=learning_rate)
opt_disc = optim.SGD(disc.parameters(), lr=learning_rate)
criterion = nn.BCELoss()
fixed_noise = torch.randn(32, z_dim, 1, 1).to(device)

gen.train()
disc.train()

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        noise = torch.randn((batch_size, z_dim, 1, 1)).to(device)
        fake = gen(noise)

        # Train Discriminator max log(D(x)) + log(1 - D(G(z)))
        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).reshape(-1)
        loss_disc_fake= criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake)/2
        disc.zero_grad()
        loss_disc.backward(retain_graph = True)

        # opt_disc.step()

        with torch.no_grad():
            for param in disc.parameters():
                param -= learning_rate * param.grad
        
        with torch.no_grad():
            for param in disc.parameters():
                param.add_(- learning_rate * torch.normal(mean = torch.zeros(param.size()).to(device), std = noise_stddev * noise_stddev * torch.ones(param.size()).to(device)))

        # Train Generator min log(1 - D(G(z))) or max log(D(G(z)))
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()

        # opt_gen.step()
        
        with torch.no_grad():
            for param in gen.parameters():
                param -= learning_rate * param.grad

        if batch_idx%100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}")
            with torch.no_grad():
                fake = gen(fixed_noise)
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

    data_cols = {'Epoch' : [epoch], 'Gen' : [loss_gen.cpu().detach().numpy()], 'Disc' : [loss_disc.cpu().detach().numpy()]}
    df = pd.DataFrame(data_cols)
    df.to_csv("trained_models/loss_data_"+str(noise_stddev)+".csv", mode='a', index=False, header=False)
    torch.save(gen, "trained_models/adam_Implementation_"+str(noise_stddev)+"/gen_step"+str(epoch)+".pth")
    torch.save(disc, "trained_models/adam_Implementation_"+str(noise_stddev)+"/disc_step"+str(epoch)+".pth")

torch.save(gen, "trained_models/adam_Implementation_"+str(noise_stddev)+"/gen.pth")
torch.save(disc, "trained_models/adam_Implementation_"+str(noise_stddev)+"/disc.pth")