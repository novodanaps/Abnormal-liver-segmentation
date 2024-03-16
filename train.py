import os
import logging
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from monai.utils import set_determinism
from monai.visualize import plot_2d_or_3d_image

from configs.get_config import read_config
from data.dataset import ALSDataset
from models.auto_encoder import Encoder, Decoder

logger = logging.getLogger("Train")


def log_results(train_name):
    """
    Log folders for each run to store the tensorboard files, saved models and the log files.
    :return: three string pointing to tensorboard, saved models and log paths respectively.
    """
    date = time.strftime("%Y%m%d", time.localtime(time.time()))
    results_path = "./results"

    folder_name = f"{date}_{train_name}"
    tensorboard_path = os.path.join(results_path, folder_name, "tensorboard")  # %tensorboard --logdir <tensorboard folder>
    saved_model_path = os.path.join(results_path, folder_name, "saved_models")
    log_path = os.path.join(results_path, folder_name, "logs")

    for folder in [tensorboard_path, saved_model_path, log_path]:
        os.makedirs(folder, exist_ok=True)

    return tensorboard_path, saved_model_path, log_path


def train(
    name,
    train_loader,
    val_loader,
    image_size,
    z_dim,
    device_ids,
    epochs,
    lr,
    weight_decay,
):
    h, w, z = image_size
    encoder = Encoder(h, w, z, z_dim=z_dim)
    decoder = Decoder(h, w, z, z_dim=z_dim)

    # if distribution
    encoder = nn.DataParallel(encoder, device_ids=device_ids).to(device)
    decoder = nn.DataParallel(decoder, device_ids=device_ids).to(device)

    ae_loss = nn.MSELoss()

    optimizer_ae = optim.Adam(
        [
            {"params": encoder.parameters()},
            {"params": decoder.parameters()}
        ],
        lr=lr,
        weight_decay=weight_decay,
    )

    tensorboard_path, saved_model_path, log_path = log_results(name)
    writer = SummaryWriter(tensorboard_path)

    step = 0
    best_loss = 100
    for epoch in range(epochs):
        logger.info("-" * 10)
        logger.info(f"epoch {epoch + 1}/{epochs}")
        encoder.train()
        decoder.train()

        autoencoder_loss_epoch = 0.0
        img = x_hat = None

        for data in train_loader:
            img = data["im"].to(device)
            # ==========forward=========
            z = encoder(img)
            x_hat = decoder(z)

            # ========== compute the loss and backpropagation =========
            encoder_decoder_loss = ae_loss(x_hat, img)

            optimizer_ae.zero_grad()
            encoder_decoder_loss.backward()
            optimizer_ae.step()

            # ======== METRICS ===========
            autoencoder_loss_epoch += encoder_decoder_loss.item()

            writer.add_scalar("step_train_loss", encoder_decoder_loss, step)

            step += 1

        train_loss = autoencoder_loss_epoch / len(train_loader)
        val_loss = val(val_loader, encoder, decoder)

        logger.info("train_loss: {:.4f}".format(train_loss))
        logger.info("val_loss: {:.4f}".format(val_loss))
        writer.add_scalars(
            "train and val loss per epoch",
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
            epoch + 1
        )

        plot_2d_or_3d_image(
            img,
            epoch + 1,
            writer,
            index=0,
            frame_dim=-1,
            tag="input_image",
        )

        plot_2d_or_3d_image(
            x_hat,
            epoch + 1,
            writer,
            index=0,
            frame_dim=-1,
            tag="reconstructed image",
        )

        if (epoch + 1) % 50 == 0 or (epoch + 1) == epochs:
            torch.save({
                "epoch": epoch + 1,
                "encoder": encoder.state_dict(),
            }, saved_model_path + f"/encoder_{epoch + 1}.pth")

            torch.save({
                "epoch": epoch + 1,
                "decoder": decoder.state_dict(),
            }, saved_model_path + f"/decoder_{epoch + 1}.pth")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                "epoch": epoch + 1,
                "encoder": encoder.state_dict(),
            }, saved_model_path + f"/encoder_best.pth")

            torch.save({
                "epoch": epoch + 1,
                "decoder": decoder.state_dict(),
            }, saved_model_path + f"/decoder_best.pth")

            logger.info(f"Saved best model in epoch: {epoch+1}")
    writer.close()


def val(dataloader, encoder, decoder):
    encoder.eval()
    decoder.eval()

    ae_loss = nn.MSELoss()
    autoencoder_loss = 0.0

    with torch.no_grad():
        for data in dataloader:
            img = data['im'].to(device)
            # ========== forward =========
            z = encoder(img)
            x_hat = decoder(z)

            # ========== compute the loss =========
            encoder_decoder_loss = ae_loss(x_hat, img)
            autoencoder_loss += encoder_decoder_loss.item()

        tol_loss = autoencoder_loss / len(dataloader)

    return tol_loss


def main(configs):
    set_determinism(seed=42)

    device_ids = configs["device_id"]
    train_img_folder = configs["data"]["train_image"]
    val_img_folder = configs["data"]["val_image"]
    image_size = configs["image_size"]
    z_dim = int(configs["z_dim"])

    lr = float(configs["lr"])
    weight_decay = float(configs["weight_decay"])
    batch_size = int(configs["batch_size"])
    num_workers = int(configs["num_workers"])
    epochs = int(configs["epochs"])

    als_dataset = ALSDataset(image_size)

    train_set = als_dataset.create_dataset(train_img_folder, num_workers, training=True)  # np.transpose(train_set[0]["im"].get_array()[:,:,:,0]*255, (1, 2, 0))
    logger.info(f"Number of images for training {train_set.__len__()}")

    val_set = als_dataset.create_dataset(val_img_folder, num_workers, training=False)
    logger.info(f"Number of images for testing {val_set.__len__()}")

    train_loader = als_dataset.create_dataloader(
        train_set,
        workers=1,
        batch_size=batch_size,
        training=True,
    )
    val_loader = als_dataset.create_dataloader(
        val_set,
        workers=1,
        batch_size=1,
        training=False,
    )

    logger.info(f"Training model {configs['name']}...\n\n")
    train(
        configs["name"],
        train_loader,
        val_loader,
        image_size,
        z_dim,
        device_ids,
        epochs,
        lr,
        weight_decay,
    )


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    configs = read_config("./configs/train_config.yml")
    main(configs)
