
# Variational Autoencoder

Reference implementation for a variational autoencoder in Pytorch.

Research Paper [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

## Model Types
1. Auto Encoders
2. Variational Auto Encoders
3. Conditional Variational Auto Encoders


## How to run
### Install dependencies

```bash
conda env create -n vae -f environment.yml
conda activate vae
```

### Train the model

```bash
python train.py
```
### Run the tests

```bash
python inference.py
```