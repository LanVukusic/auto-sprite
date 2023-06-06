import torch
import torchvision
import torch.optim as optim
import argparse
import matplotlib
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import model
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
matplotlib.style.use('ggplot')
import glob
from PIL import Image
from PIL.Image import NEAREST
from PIL import ImageShow
# import pyscreenshot as ImageGrab
from PIL import ImageGrab
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import os
import cv2 as cv



# construct the argument parser and parser the arguments
parser = argparse.ArgumentParser()
# epochs
parser.add_argument('-e', '--epochs', default=10, type=int, 
                    help='number of epochs to train the VAE for')
# batch size
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    help='batch size for training')
args = vars(parser.parse_args())

# leanring parameters
epochs = args['epochs']
batch_size = args['batch_size']
lr = 0.00005
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# transforms
# transform = transforms.Compose([
#     transforms.ToTensor(),
# ])
to_py_tensor = transforms.ToTensor()

IMAGE_CHANNELS = 3
# reshape the data to 64x64x4 rgb alpha channel
def get_images (data_path = "../../swords"):
    # glob 
    images = []
    for file in glob.glob(data_path + "/*.png"):
        # to grayscale single channel
        image = Image.open(file).convert('RGB')
        # resize
        image =  image.resize((32,32), NEAREST)
        image = to_py_tensor(image)
        
        images.append(image)
    # convert to tensor with a batch dimension
    images = torch.stack(images)
    # plt.imshow(images[0].view(64,64,4))
    # plt.show()
    return images

class CustomImageDataset(Dataset):
    def __init__(self):
        # self.images = [TF.to_tensor(TF.rotate(i, random.random() * 360)) for i in get_images()]
        self.images = get_images()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]
    

    
train_data = CustomImageDataset()

# training and validation data loaders
train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True
)


model = model.VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss(reduction='sum') 

def final_loss(bce_loss, mu, logvar):
    """
    This function will add the reconstruction loss (BCELoss) and the 
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param bce_loss: recontruction loss
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    """
    BCE = bce_loss 
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def fit(model, dataloader):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(dataloader):
        data = data.to(device)
        # print(data.shape)
        # data = data.view(data.size(0), -1)
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(data)
        # print(reconstruction.shape, data.shape)
        bce_loss = criterion(reconstruction, data)
        loss = final_loss(bce_loss, mu, logvar)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss = running_loss/len(dataloader.dataset)
    return train_loss 

def validate(model, dataloader):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            data = data.to(device)
            # data = data.view(data.size(0), -1)
            reconstruction, mu, logvar = model(data)
            bce_loss = criterion(reconstruction, data)
            loss = final_loss(bce_loss, mu, logvar)
            running_loss += loss.item()

            # save 8 images for comparison, using inbuilt grid
            # concat data and reconstruction vertically
            concat_data_reconstruction = torch.cat([data[:8], reconstruction[:8]])
            grid = torchvision.utils.make_grid(concat_data_reconstruction.cpu(), nrow=8)
            save_image(grid, f"../outputs/reconstruction{epoch}.png")

            # generate average embedding
            embedding, mu, logvar = model.encode(data)
            first_02 = torch.mean(embedding[:2], axis=0)
            first_24 = torch.mean(embedding[2:4], axis=0)
            first_46 = torch.mean(embedding[4:6], axis=0)
            first_68 = torch.mean(embedding[6:8], axis=0)

            multiple = torch.stack([first_02,first_24,first_46,first_68])
            multiple = model.decode(multiple)
            grid = torchvision.utils.make_grid(multiple.cpu(), nrow=4)
            save_image(grid, f"../outputs/new{epoch}.png")


    val_loss = running_loss/len(dataloader.dataset)
    return val_loss

def save_checkpoint(state, filename="vae.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

if(os.path.exists("../models/vae.pth.tar")):
  model.load_state_dict(torch.load("../models/vae.pth.tar"))
  print("Loaded model from disk")

  while True:
    # read image from disk "../in_img/1.png"
    im = Image.open("../in_img/1.png")
    

    im = im.convert('RGB')
    im =  im.resize((32,32), NEAREST)

    # cv.imshow("image", im)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    plt.imshow(im)
    plt.show()

    im = to_py_tensor(im)
    im = im.unsqueeze(0).to(device)
    # print(im.shape)
    # encode image
    embedding, mu, logvar = model.encode(im)
    # generate new random image
    # embedding = torch.randn(1, 256).to(device)
    new_image = model.decode(embedding)[0].permute(1, 2, 0).detach().cpu().numpy()
    plt.imshow(new_image)
    plt.show()
    
    # cv.imshow('title', new_image.permute(1, 2, 0).detach().cpu().numpy())
    # cv.waitKey(0)
    # cv.destroyAllWindows()


print("Starting training")
train_loss = []
val_loss = []
best_loss = 1e9 #  = 10^9

i = 0
for epoch in tqdm(range(epochs)): 
    # print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = fit(model, train_loader)
    
    if(i % 20 == 0):
      train_loss.append(train_epoch_loss)
      val_epoch_loss = validate(model, train_loader)
      val_loss.append(val_epoch_loss)
      print(f"Train Loss: {train_epoch_loss:.4f}")
      print(f"Val Loss: {val_epoch_loss:.4f}")
    # if(i % 200 == 0):
    #   if val_epoch_loss < best_loss:
    #       save_checkpoint(model.state_dict(), filename='../outputs/vae.pth.tar')
    #       best_loss = val_epoch_loss
    i += 1

# save model to file
save_checkpoint(model.state_dict(), filename='../models/vae.pth.tar')