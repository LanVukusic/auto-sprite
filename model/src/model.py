# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# init_channels = 64 # initial number of filters
# image_channels = 1 # color channels
# latent_dim = 20 # number of features to consider


# # define a Conv VAE
# class ConvVAE(nn.Module):
#     def __init__(self):
#         super(ConvVAE, self).__init__()
 
#         # encoder
#         self.enc1 = nn.Conv2d(
#             in_channels=image_channels, out_channels=init_channels, 
#             kernel_size=4, stride=2, padding=2
#         )
#         self.enc2 = nn.Conv2d(
#             in_channels=init_channels, out_channels=init_channels*2, 
#             kernel_size=4, stride=2, padding=2
#         )
#         self.enc3 = nn.Conv2d(
#             in_channels=init_channels*2, out_channels=init_channels*4, 
#             kernel_size=4, stride=2, padding=2
#         )
#         self.enc4 = nn.Conv2d(
#             in_channels=init_channels*4, out_channels=init_channels*8, 
#             kernel_size=4, stride=2, padding=2
#         )
#         self.enc5 = nn.Conv2d(
#             in_channels=init_channels*8, out_channels=1024, 
#             kernel_size=4, stride=2, padding=2
#         )

#         self.fc1 = nn.Linear(1024, 2048)
#         self.fc_mu = nn.Linear(2048, latent_dim)
#         self.fc_log_var = nn.Linear(2048, latent_dim)
#         self.fc2 = nn.Linear(latent_dim, 1024)  
        
#         # decoder 
#         self.dec1 = nn.ConvTranspose2d(
#             in_channels=1024, out_channels=init_channels*8, 
#             kernel_size=3, stride=2
#         )
#         self.dec2 = nn.ConvTranspose2d(
#             in_channels=init_channels*8, out_channels=init_channels*4, 
#             kernel_size=3, stride=2
#         )
#         self.dec3 = nn.ConvTranspose2d(
#             in_channels=init_channels*4, out_channels=init_channels*2, 
#             kernel_size=3, stride=2
#         )

#         # # convolutional layer to reduce chekerboarding
#         # self.d_conv3 = nn.Conv2d(
#         #     in_channels=init_channels*2, out_channels=init_channels*2,
#         #     kernel_size=3, stride=1, padding=1
#         # )

#         self.dec4 = nn.ConvTranspose2d(
#             in_channels=init_channels*2, out_channels=init_channels, 
#             kernel_size=3, stride=2,
#         )

#         # # convolutional layer to reduce chekerboarding
#         # self.d_conv4 = nn.Conv2d(
#         #     in_channels=init_channels, out_channels=init_channels,
#         #     kernel_size=3, stride=2, padding=1
#         # )

#         self.dec5 = nn.ConvTranspose2d(
#             in_channels=init_channels, out_channels=image_channels, 
#             kernel_size=4, stride=2
#         )
        
#         # smoothing layer to reduce checkerboarding
#         self.dec6 = nn.Conv2d(
#             in_channels=image_channels, out_channels=image_channels,
#             kernel_size=3, stride=1, padding=1
#         )



#     def reparameterize(self, mu, log_var):
#         """
#         :param mu: mean from the encoder's latent space
#         :param log_var: log variance from the encoder's latent space
#         """
#         std = torch.exp(0.5*log_var) # standard deviation
#         eps = torch.randn_like(std) # `randn_like` as we need the same size
#         sample = mu + (eps * std) # sampling
#         return sample
 
#     def forward(self, x):
#         # encoding
#         # print(x.shape)
#         x = F.relu(self.enc1(x))
#         # print(x.shape)
#         x = F.relu(self.enc2(x))
#         # print(x.shape)
#         x = F.relu(self.enc3(x))
#         # print(x.shape)
#         x = F.relu(self.enc4(x))
#         # print(x.shape)
#         x = F.relu(self.enc5(x))
#         # print(x.shape)

#         # get `mu` and `log_var`
#         batch, _, _, _ = x.shape
#         x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
#         hidden = self.fc1(x)
#         # get `mu` and `log_var`
#         mu = self.fc_mu(hidden)
#         log_var = self.fc_log_var(hidden)
#         # get the latent vector through reparameterization
#         z = self.reparameterize(mu, log_var)
#         # print(z.shape)
#         z = self.fc2(z)
#         # print(z.shape)
#         z = z.view(-1, 1024, 1, 1)
#         # print(z.shape)
 
#         # decoding
#         # print("decoding")
#         # print(x.shape)
#         x = F.relu(self.dec1(z))
#         # print(x.shape)
#         x = F.relu(self.dec2(x))
#         # print(x.shape)
#         x = F.relu(self.dec3(x))
#         # x = F.relu(self.d_conv3(x))
#         # print(x.shape)
#         x = F.relu(self.dec4(x))
#         # x = F.relu(self.d_conv4(x))
#         x = F.relu(self.dec5(x))
#         # x = F.relu(self.dec6(x))
#         # print(x.shape)
#         reconstruction = torch.sigmoid(x)
#         return reconstruction, mu, log_var

import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# image
image_size = 32 # 32x32 
image_channels = 3 # color channels

# model
init_channels = 128 # initial number of filters

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), 256, 1, 1)

class VAE(nn.Module):
    def __init__(self, image_channels=image_channels, z_dim=init_channels):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            # input: 32x32x3
            nn.Conv2d(in_channels=image_channels, out_channels=32, kernel_size=4, stride=2), # 15x15x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # 6x6x64
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2), # 2x2x128
            nn.ReLU(), 
            nn.Conv2d(128, 256, kernel_size=2, stride=1),  # 1x1x256
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
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2), 
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2), 
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2),
            nn.ReLU(), 
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