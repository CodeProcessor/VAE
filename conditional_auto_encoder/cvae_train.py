from pathlib import Path

import torch
import torchvision.datasets as datasets
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from conditional_auto_encoder import ConditionalVariationalAutoEncoder

"""Configurations"""

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
INPUT_DIM = 784
H_DIM = 200
Z_DIM = 5
NUM_EPOCHS = 50
BATCH_SIZE = 128
LR_RATE = 3e-4  # Karpathy constant
NO_OF_CLASSES = 10
MODEL_DIR = Path("saved_models")

if not MODEL_DIR.exists():
    MODEL_DIR.mkdir()

"""dataset loading"""

dataset = datasets.MNIST(root="../dataset/", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
model = ConditionalVariationalAutoEncoder(INPUT_DIM, H_DIM, Z_DIM, NO_OF_CLASSES).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR_RATE, weight_decay=1e-5)
loss_fn = nn.BCELoss(reduction="sum")


def train():
    """start training"""
    for epoch in range(NUM_EPOCHS):
        loop = tqdm(enumerate(train_loader))
        for i, (x, y) in loop:
            """Forward pass"""
            x = x.to(DEVICE).view(x.shape[0], INPUT_DIM)
            x_reconstructed, mu, sigma = model(x, y)

            """Compute loss"""
            reconstruction_loss = loss_fn(x_reconstructed, x)
            kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

            """Backpropagation"""
            loss = reconstruction_loss + kl_div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(epoch=epoch + 1, loss=loss.item())

    final_loss = loss.item()
    """Save model"""
    model_save_path = f"{MODEL_DIR}/saved_model_v2_{NUM_EPOCHS}_epochs_loss_{final_loss:.0f}.pt"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to: {model_save_path}")


if __name__ == '__main__':
    train()
