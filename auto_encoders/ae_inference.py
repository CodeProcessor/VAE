from pathlib import Path

import torch
import torchvision.datasets as datasets
from torchvision import transforms
from torchvision.utils import save_image

from auto_encoder import AutoEncoder

"""Configurations"""
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 784
H_DIM = 200
Z_DIM = 20
MODEL_PATH = "saved_models/saved_model_25_epochs_loss_4814.pt"
RESULTS_DIR = "output"

if not Path(RESULTS_DIR).exists():
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

dataset = datasets.MNIST(root="../dataset/", train=True, transform=transforms.ToTensor(), download=True)

model = AutoEncoder(INPUT_DIM, H_DIM, Z_DIM).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

ADD_NOISE_TO_INPUT = False
ADD_NOISE_TO_HIDDEN = False


def inference(digit, no_of_examples=1):
    """
    :param digit:
    :param no_of_examples:
    :return:
    """
    for x, y in dataset:
        if y == digit:
            x = x.to(DEVICE)
            """Save original image"""
            input_image = x.resize(1, 28, 28)
            input_image_path = Path(f"{RESULTS_DIR}/input_image_digit_{digit}.jpg")
            save_image(input_image, input_image_path)
            print(f"Input image saved to - {input_image_path}")

            if ADD_NOISE_TO_INPUT:
                """Add a noise and save the image"""
                epsilon = torch.rand_like(x) * 0.5
                x = x + epsilon
                noisy_input_image = x.resize(1, 28, 28)
                noisy_input_image_path = Path(f"{RESULTS_DIR}/noisy_input_image_digit_{digit}.jpg")
                save_image(noisy_input_image, noisy_input_image_path)

            hidden = model.encode(x.view(1, 784))
            break
    else:
        raise AssertionError(f"Digit {digit} not found in dataset")

    for i in range(no_of_examples):
        """
        Add a noise to hidden layer and decode the image
        ------------------------------------------------
        Since this is an AutoEncoder
        We can't directly generate images from the hidden layer (even if we add a small noise)
        """
        epsilon = torch.rand_like(hidden) * 0.5
        out = model.decode(hidden + epsilon) if ADD_NOISE_TO_HIDDEN else model.decode(hidden)
        # out = model.decode(epsilon)
        output = out.view(-1, 1, 28, 28)
        image_path = Path(f"{RESULTS_DIR}/image_digit_{digit}_example_{i + 1}.jpg")
        save_image(output, image_path)
        print(f"Image saved to - {image_path}")


if __name__ == '__main__':
    inference(1, 1)
    # for d in range(1, 10):
    #     inference(d, 5)
