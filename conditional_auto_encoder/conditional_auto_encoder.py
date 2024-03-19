import torch
from torch import nn


class ConditionalVariationalAutoEncoder(nn.Module):
    """
    Input img -> Hidden dim -> mean, std -> Parameterization trick -> Decoder -> Output img
    """

    def __init__(self, input_dim, h_dim=200, z_dim=20, no_of_classes=10):
        super().__init__()
        """Encoder"""
        self.img_2hid = nn.Linear(input_dim, h_dim)
        self.hid_2mu = nn.Linear(h_dim, z_dim)
        self.hid_2sigma = nn.Linear(h_dim, z_dim)

        """Decoder"""
        self.z_2hid = nn.Linear(z_dim + no_of_classes, h_dim)
        self.hid_img = nn.Linear(h_dim, input_dim)

        """Common"""
        self.relu = nn.ReLU()

    def encode(self, x):
        """Encoder: Input img -> Hidden dim -> mean, std"""
        h = self.relu(self.img_2hid(x))
        mu, sigma = self.hid_2mu(h), self.hid_2sigma(h)
        return mu, sigma

    def decode(self, z, d):
        """Decoder: Parameterization trick -> Decoder -> Output img"""
        digit_one_hot = torch.zeros((z.shape[0], 10))
        digit_one_hot[range(z.shape[0]), d] = 1
        z = torch.cat((z, digit_one_hot), dim=1)
        h = self.relu(self.z_2hid(z))
        return torch.sigmoid(self.hid_img(h))

    def forward(self, x, y):
        """Forward pass"""
        mu, sigma = self.encode(x)
        epsilon = torch.rand_like(sigma)
        z_re_parameterize = mu + sigma * epsilon
        x_reconstructed = self.decode(z_re_parameterize, y)
        return x_reconstructed, mu, sigma


if __name__ == '__main__':
    x = torch.randn(4, 784)  # 28x28 = 784
    y = 5
    vae = ConditionalVariationalAutoEncoder(input_dim=784)
    x_reconstructed, mu, sigma = vae(x, y)
    print(x_reconstructed.shape)
    print(mu.shape)
    print(sigma.shape)
