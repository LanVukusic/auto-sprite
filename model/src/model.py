import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# image
image_size = 32 # 32x32 
image_channels = 3 # color channels

# model

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), 3, 1, 1)

class VAE(nn.Module):
    def __init__(self, image_channels=image_channels, z_dim=128):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            # input: 32x32x3
            nn.Conv2d(in_channels=image_channels, out_channels=32, kernel_size=4, stride=2), # 15x15x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # 6x6x64
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(64, 64, kernel_size=4, stride=2), # 2x2x128
            nn.ReLU(), 
            nn.Dropout(0.2),
            nn.Conv2d(64, 256, kernel_size=2, stride=1),  # 1x1x256
            nn.ReLU(),
            # dropout
            nn.Dropout(0.2),

            nn.Conv2d(256, 256, kernel_size=1, stride=1),  # 1x1x256
            nn.ReLU(),
            Flatten() # h_dim
        )
        
        # mu, logvar
        self.fc1 = nn.Linear(256, z_dim) # 64
        self.fc2 = nn.Linear(256, z_dim) # 64

        # decode
        self.fc_smart_1 = nn.Linear(z_dim, 1024) # 32
        self.fc_smart_2 = nn.Linear(1024, 2048) # 32
        self.fc_smart_3 = nn.Linear(2048, z_dim) # 32
        self.fc3 = nn.Linear(z_dim, 256) # h_dim
        
        self.decoder = nn.Sequential(
            # input: -1 x 256
            UnFlatten(), # -1 x 256 x 1 x 1
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2), 
            nn.ReLU(), # 
            nn.Dropout(0.2),

            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2), 
            nn.ReLU(), # 4x4x128
            # dropout

            nn.Dropout(0.2),
            # # convnets 
            # nn.Conv2d(128, 128, kernel_size=3, stride=1),
            # nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2),
            nn.ReLU(), 
            nn.Dropout(0.2),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2), # 32,32,32
            nn.Sigmoid()

        )
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_().to(device)
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp
        return z
    

    def encode(self, x):
        h = self.encoder(x)
        # print("encode", h.shape)
        # print("ENCODED: ", h.shape)
        mu = self.fc1(h)
        logvar = self.fc2(h)
        # print("params:", mu.shape, logvar.shape)
        z = self.reparameterize(mu, logvar)
        # print("z", z.shape)
        # print("Z: ", z.shape)
        # print("MU: ", mu.shape)
        # print("LOGVAR: ", logvar.shape)
        return z, mu, logvar

    def decode(self, z):
        # print("Z: ", z.shape)
        z = self.fc_smart_1(z)
        z = self.fc_smart_2(z)
        z = self.fc_smart_3(z)

        z = self.fc3(z)
        # print("decode - fc", z.shape)
        z = self.decoder(z)
        # print("decode - decoder", z.shape)
        return z

    def forward(self, x):
        # print("in: ", x.shape)
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        # print("Z: ", z.shape)
        return z, mu, logvar