import os
import traceback

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from cnn_vae import Model

BATCH_SIZE = 100
LR = 1e-3
MAX_EPOCH = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 5
LOAD_EPOCH = -1
GENERATE = True


def plot(epoch, pred, y, name='test_'):
    if not os.path.isdir('./images'):
        os.mkdir('./images')
    fig = plt.figure(figsize=(16, 16))
    for i in range(6):
        ax = fig.add_subplot(3, 2, i + 1)
        ax.imshow(pred[i, 0], cmap='gray')
        ax.axis('off')
        ax.title.set_text(str(y[i]))
    plt.savefig("./images/{}epoch_{}.jpg".format(name, epoch))
    # plt.figure(figsize=(10,10))
    # plt.imsave("./images/pred_{}.jpg".format(epoch), pred[0,0], cmap='gray')
    plt.close()


def loss_function(x, pred, mu, logvar):
    recon_loss = F.mse_loss(pred, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss, kld


def train(epoch, model, train_loader, optim):
    reconstruction_loss = 0
    kld_loss = 0
    total_loss = 0
    for i, (x, y) in enumerate(train_loader):
        try:
            label = np.zeros((x.shape[0], 10))
            label[np.arange(x.shape[0]), y] = 1
            label = torch.tensor(label)

            optim.zero_grad()
            pred, mu, logvar = model(x.to(DEVICE), label.to(DEVICE))

            recon_loss, kld = loss_function(x.to(DEVICE), pred, mu, logvar)
            loss = recon_loss + kld
            loss.backward()
            optim.step()

            total_loss += loss.cpu().data.numpy() * x.shape[0]
            reconstruction_loss += recon_loss.cpu().data.numpy() * x.shape[0]
            kld_loss += kld.cpu().data.numpy() * x.shape[0]
            if i == 0:
                print("Gradients")
                for name, param in model.named_parameters():
                    if "bias" in name:
                        print(name, param.grad[0], end=" ")
                    else:
                        print(name, param.grad[0, 0], end=" ")
                    print()
        except Exception as e:
            traceback.print_exe()
            torch.cuda.empty_cache()
            continue

    reconstruction_loss /= len(train_loader.dataset)
    kld_loss /= len(train_loader.dataset)
    total_loss /= len(train_loader.dataset)
    return total_loss, kld_loss, reconstruction_loss


def test(epoch, model, test_loader):
    reconstruction_loss = 0
    kld_loss = 0
    total_loss = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            try:
                label = np.zeros((x.shape[0], 10))
                label[np.arange(x.shape[0]), y] = 1
                label = torch.tensor(label)

                pred, mu, logvar = model(x.to(DEVICE), label.to(DEVICE))
                recon_loss, kld = loss_function(x.to(DEVICE), pred, mu, logvar)
                loss = recon_loss + kld

                total_loss += loss.cpu().data.numpy() * x.shape[0]
                reconstruction_loss += recon_loss.cpu().data.numpy() * x.shape[0]
                kld_loss += kld.cpu().data.numpy() * x.shape[0]
                if i == 0:
                    # print("gr:", x[0,0,:5,:5])
                    # print("pred:", pred[0,0,:5,:5])
                    plot(epoch, pred.cpu().data.numpy(), y.cpu().data.numpy())
            except Exception as e:
                traceback.print_exe()
                torch.cuda.empty_cache()
                continue
    reconstruction_loss /= len(test_loader.dataset)
    kld_loss /= len(test_loader.dataset)
    total_loss /= len(test_loader.dataset)
    return total_loss, kld_loss, reconstruction_loss


def generate_image(epoch, z, y, model):
    with torch.no_grad():
        label = np.zeros((y.shape[0], 10))
        label[np.arange(z.shape[0]), y] = 1
        label = torch.tensor(label)

        pred = model.decoder(torch.cat((z.to(DEVICE), label.float().to(DEVICE)), dim=1))
        plot(epoch, pred.cpu().data.numpy(), y.cpu().data.numpy(), name='Eval_')
        print("data Plotted")


def load_data():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=True, download=True,
                                                                          transform=transform), batch_size=BATCH_SIZE,
                                               num_workers=NUM_WORKERS, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=False, download=True,
                                                                         transform=transform), batch_size=BATCH_SIZE,
                                              num_workers=NUM_WORKERS, shuffle=True)

    return train_loader, test_loader


def save_model(model, epoch):
    if not os.path.isdir("./checkpoints"):
        os.mkdir("./checkpoints")
    file_name = './checkpoints/model_{}.pt'.format(epoch)
    torch.save(model.state_dict(), file_name)


if __name__ == "__main__":
    train_loader, test_loader = load_data()
    print("dataloader created")
    model = Model().to(DEVICE)
    print("model created")

    if LOAD_EPOCH > 0:
        model.load_state_dict(
            torch.load('./checkpoints/model_{}.pt'.format(LOAD_EPOCH), map_location=torch.device('cpu')))
        print("model {} loaded".format(LOAD_EPOCH))

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0.001)

    train_loss_list = []
    test_loss_list = []
    for i in range(LOAD_EPOCH + 1, MAX_EPOCH):
        model.train()
        train_total, train_kld, train_loss = train(i, model, train_loader, optimizer)
        with torch.no_grad():
            model.eval()
            test_total, test_kld, test_loss = test(i, model, test_loader)
            if GENERATE:
                z = torch.randn(6, 32).to(DEVICE)
                y = torch.tensor([1, 2, 3, 4, 5, 6]) - 1
                generate_image(i, z, y, model)

        print(
            "Epoch: {}/{} Train loss: {}, Train KLD: {}, Train Reconstruction Loss:{}".format(i, MAX_EPOCH, train_total,
                                                                                              train_kld, train_loss))
        print("Epoch: {}/{} Test loss: {}, Test KLD: {}, Test Reconstruction Loss:{}".format(i, MAX_EPOCH, test_loss,
                                                                                             test_kld, test_loss))

        save_model(model, i)
        train_loss_list.append([train_total, train_kld, train_loss])
        test_loss_list.append([test_total, test_kld, test_loss])
        np.save("train_loss", np.array(train_loss_list))
        np.save("test_loss", np.array(test_loss_list))

    # i, (example_data, exaple_target) = next(enumerate(test_loader))
    # print(example_data[0,0].shape)
    # plt.figure(figsize=(5,5), dpi=100)
    # plt.imsave("example.jpg", example_data[0,0], cmap='gray',  dpi=1000)
