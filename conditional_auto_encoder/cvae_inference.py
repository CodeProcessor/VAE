from pathlib import Path

import torch
import torchvision.datasets as datasets
from torchvision import transforms
from torchvision.utils import save_image

from conditional_auto_encoder import ConditionalVariationalAutoEncoder

"""Configurations"""
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 784
H_DIM = 200
Z_DIM = 5
NO_OF_CLASSES = 10
MODEL_PATH = "saved_models/saved_model_v2_50_epochs_loss_10876.pt"
RESULTS_DIR = "output_v4-2"

if not Path(RESULTS_DIR).exists():
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

dataset = datasets.MNIST(root="../dataset/", train=True, transform=transforms.ToTensor(), download=True)

model = ConditionalVariationalAutoEncoder(INPUT_DIM, H_DIM, Z_DIM, NO_OF_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()


def inference(digit, no_of_examples=1):
    """
    :param digit:
    :param no_of_examples:
    :return:
    """
    # for x, y in dataset:
    #     if y == digit:
    #         mu, sigma = model.encode(x.view(1, 784))
    #         break
    # else:
    #     raise AssertionError(f"Digit {digit} not found in dataset")

    for i in range(no_of_examples):
        # epsilon = torch.rand_like(sigma)
        # z = mu + sigma * epsilon
        z = torch.randn(1, Z_DIM).to(DEVICE)
        out = model.decode(z, digit)
        output = out.view(-1, 1, 28, 28)
        image_path = Path(f"{RESULTS_DIR}/image_digit_{digit}_example_{i + 1}.jpg")
        save_image(output, image_path)
        print(f"Image saved to - {image_path}")


if __name__ == '__main__':
    for d in range(1, 10):
        inference(d, 5)
