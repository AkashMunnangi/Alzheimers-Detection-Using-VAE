import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================
# VARIATIONAL AUTOENCODER (VAE)
# =====================================================
class VAE(nn.Module):
    """
    VAE for MRI representation learning
    Input shape: [B, 3, 150, 150]
    Latent dimension: 128
    """

    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()

        # -------- Encoder --------
        self.enc_conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)   # 150 -> 75
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # 75 -> 37
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) # 37 -> 18

        self.fc_mu = nn.Linear(128 * 18 * 18, latent_dim)
        self.fc_logvar = nn.Linear(128 * 18 * 18, latent_dim)

        # -------- Decoder --------
        self.fc_decode = nn.Linear(latent_dim, 128 * 18 * 18)

        self.dec_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 18 -> 36
        self.dec_conv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)   # 36 -> 72
        self.dec_conv3 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)    # 72 -> 144

    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(-1, 128, 18, 18)
        x = F.relu(self.dec_conv1(x))
        x = F.relu(self.dec_conv2(x))
        x = torch.sigmoid(self.dec_conv3(x))
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# =====================================================
# LATENT SPACE CLASSIFIER
# =====================================================
class LatentClassifier(nn.Module):
    """
    Binary classifier on VAE latent space
    Output:
        0 -> Normal
        1 -> Alzheimers
    """

    def __init__(self, latent_dim=128, num_classes=2):
        super(LatentClassifier, self).__init__()

        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, z):
        z = F.relu(self.fc1(z))
        out = self.fc2(z)   # logits
        return out
