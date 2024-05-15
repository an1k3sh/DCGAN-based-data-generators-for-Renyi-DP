import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader

batch_size = 128
num_output = 5000
z_dim = 100

device = torch.device("mps" if(torch.backends.mps.is_available()) else "cpu")
device = torch.device("cpu")
print(device.type)

resnet_load_path = "resnet_mnist.pth"
resnet_mnist = torch.load(resnet_load_path, map_location=torch.device('cpu')).to(device)

val = 0.15
eval_model_path = "trained_models/adam_Implementation_"+str(val)+"/gen_step25.pth"
netG = torch.load(eval_model_path, map_location=torch.device('cpu')).to(device)

noise = torch.randn(int(num_output),z_dim, 1, 1, device=device)
with torch.no_grad():
    dataset = netG(noise).to(device)
loader = DataLoader(dataset, batch_size=batch_size, shuffle = True)

preds = []
for batch_idx, images in enumerate(loader):
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