import torch
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VariationalAutoEncoder(nn.Module):
    """
    Input img -> Hidden dim -> mean, std -> Parameterization trick -> Decoder -> Output img
    """

    def __init__(self, input_dim, h_dim=200, z_dim=20):
        super().__init__()
        """Encoder"""
        self.img_2hid = nn.Linear(input_dim, h_dim)
        self.hid_2mu = nn.Linear(h_dim, z_dim)
        self.hid_2sigma = nn.Linear(h_dim, z_dim)

        """Decoder"""
        self.z_2hid = nn.Linear(z_dim, h_dim)
        self.hid_img = nn.Linear(h_dim, input_dim)

        """Common"""
        self.relu = nn.ReLU()

    def encode(self, x):
        """Encoder: Input img -> Hidden dim -> mean, std"""
        h = self.relu(self.img_2hid(x)).to(DEVICE)
        mu, sigma = self.hid_2mu(h), self.hid_2sigma(h)
        return mu, sigma

    def decode(self, z):
        """Decoder: Parameterization trick -> Decoder -> Output img"""
        h = self.relu(self.z_2hid(z)).to(DEVICE)
        return torch.sigmoid(self.hid_img(h))

    def forward(self, x):
        """Forward pass"""
        mu, sigma = self.encode(x)
        epsilon = torch.rand_like(sigma).to(DEVICE)
        """Re-parameterization trick"""
        z_re_parameterize = mu + sigma * epsilon
        x_reconstructed = self.decode(z_re_parameterize)
        return x_reconstructed, mu, sigma


if __name__ == '__main__':
    x = torch.randn(4, 784)  # 28x28 = 784
    vae = VariationalAutoEncoder(input_dim=784)
    x_reconstructed, mu, sigma = vae(x)
    print(x_reconstructed.shape)
    print(mu.shape)
    print(sigma.shape)
