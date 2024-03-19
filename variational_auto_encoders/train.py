import torch
import torchvision.datasets as datasets
from tqdm import tqdm
from torch import nn, optim
from model import VariationalAutoEncoder
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

"""Configurations"""

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 784
H_DIM = 200
Z_DIM = 20
NUM_EPOCHS = 25
BATCH_SIZE = 128
LR_RATE = 3e-4  # Karpathy constant
PATH = "saved_model/saved_model.pt"

"""dataset loading"""

dataset = datasets.MNIST(root="../dataset/", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
model = VariationalAutoEncoder(INPUT_DIM, H_DIM, Z_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR_RATE)
loss_fn = nn.BCELoss(reduction="sum")

"""start training"""
def train():
    for epoch in range(NUM_EPOCHS):
        loop = tqdm(enumerate(train_loader))
        for i, (x,_) in loop:
            """Forward pass"""
            x = x.to(DEVICE).view(x.shape[0], INPUT_DIM)
            x_reconstructed, mu, sigma = model(x)

            """Compute loss"""
            reconstruction_loss = loss_fn(x_reconstructed, x)
            kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

            """Backpropagation"""
            loss = reconstruction_loss + kl_div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(epoch=epoch+1, loss=loss.item())

    final_loss = loss.item()
    """Save model"""
    torch.save(model.state_dict(), f"saved_models/saved_model_{NUM_EPOCHS}_epochs_loss_{final_loss:.0f}.pt")


if __name__ == '__main__':
    train()