import torch
import torchvision.utils as utils
import numpy as np
import matplotlib.pyplot as plt

from model import Generator

num_output = 64
z_dim = 100

device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
noise = torch.randn(int(num_output),z_dim, 1, 1, device=device)

stddev = 0.15
load_path = "trained_models/adam_Implementation_"+str(stddev)+"/gen_step25.pth"
netG = torch.load(load_path, map_location=torch.device('cpu')).to(device)
with torch.no_grad():
	generated_img = netG(noise).detach().cpu()
plt.axis("off")
plt.imshow(np.transpose(utils.make_grid(generated_img, padding=2, normalize=True), (1,2,0)))
# plt.savefig("/Users/anikeshparashar/Desktop/AI_Project/Implementation/With_25_epochs_128_batch_size/generated_images_step30/Plot_0.15.png", bbox_inches='tight')
plt.savefig("results/generated_images/.Plot"+str(stddev)+"png", bbox_inches='tight')