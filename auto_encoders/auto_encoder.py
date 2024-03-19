import torch
from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, h_dim=200, z_dim=20):
        super().__init__()
        """Encoder"""
        self.img_2hid = nn.Linear(input_dim, h_dim)

        """Decoder"""
        self.hid_img = nn.Linear(h_dim, input_dim)

        """Common"""
        self.relu = nn.ReLU()

    def encode(self, x):
        h = self.relu(self.img_2hid(x))
        return h

    def decode(self, h):
        return torch.sigmoid(self.hid_img(h))

    def forward(self, x):
        """Forward pass"""
        h = self.encode(x)
        x_bar = self.decode(h)
        return x_bar


if __name__ == '__main__':
    x = torch.randn(4, 784)  # 28x28 = 784
    vae = VariationalAutoEncoder(input_dim=784)
    x_bar = vae(x)
    print(x_bar.shape)
