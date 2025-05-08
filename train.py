#!/usr/bin/env python3
import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
import torch.nn as nn
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from model import ResnetGenerator, MultiScaleDiscriminator
from dataset import PairedSeasonDataset

# Debug print to confirm script execution
print("=== train.py loaded: starting setup ===", flush=True)

def save_some_examples(gen, val_loader, epoch, folder, device):
    print(f"save_some_examples: epoch {epoch}" , flush=True)
    x, y = next(iter(val_loader))
    x, y = x.to(device), y.to(device)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # denormalize
        save_image(y_fake, os.path.join(folder, f"y_gen_{epoch}.png"))
        save_image(x * 0.5 + 0.5, os.path.join(folder, f"input_{epoch}.png"))
        if epoch == 1:
            save_image(y * 0.5 + 0.5, os.path.join(folder, f"label_{epoch}.png"))
    gen.train()


def save_checkpoint(model, optimizer, filename):
    print(f"Saving checkpoint: {filename}", flush=True)
    checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, filename)


def load_checkpoint(filepath, model, optimizer, lr, device):
    print(f"Loading checkpoint: {filepath}", flush=True)
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    print("Entering main()", flush=True)
    # -------------------- Hyperparameters --------------------
    root_dir = 'nordland-dataset'
    batch_size = 1
    image_size = 256
    num_epochs = 200
    lr = 0.0002
    beta1 = 0.5
    beta2 = 0.999
    lambda_L1 = 100.0
    num_D = 3             # number of discriminator scales
    val_pct = 0.1         # fraction for validation
    sample_interval = 10  # save images every N epochs
    checkpoint_dir = 'checkpoints'
    sample_dir = 'samples'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device: {device}", flush=True)

    # ---------------------------- Transforms ----------------------------
    transform = Compose([
        Resize(image_size, image_size),
        Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
        ToTensorV2()
    ], additional_targets={'image0':'image'})

    # ---------------------------- Dataset Split ----------------------------
    full_dataset = PairedSeasonDataset(root_dir=root_dir,
                                      src_season='fall',
                                      tgt_season='winter',
                                      transform=transform)
    total_size = len(full_dataset)
    val_size = int(total_size * val_pct)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"Dataset split: {train_size} training, {val_size} validation", flush=True)

    # ---------------------------- DataLoaders ----------------------------
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=1, shuffle=False)

    # ---------------------------- Model init ----------------------------
    G = ResnetGenerator(input_nc=3, output_nc=3, ngf=64, n_blocks=9).to(device)
    D = MultiScaleDiscriminator(input_nc=6, ndf=64, n_layers=3, num_D=num_D).to(device)

    # -------------------------- Loss & Optimizer --------------------------
    criterion_GAN = nn.MSELoss()  # LSGAN
    criterion_L1 = nn.L1Loss()
    optimizer_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2))

    # -------- create output dirs --------
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    print("Starting training...", flush=True)
    # ------------------------------ Training -----------------------------
    for epoch in range(1, num_epochs+1):
        print(f"\n--- Epoch {epoch}/{num_epochs} ---", flush=True)
        G.train(); D.train()
        epoch_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", unit='batch', ascii=True)
        for src, tgt in epoch_bar:
            src, tgt = src.to(device), tgt.to(device)
            fake = G(src)
            pred_real = D(torch.cat([src, tgt],1))
            pred_fake = D(torch.cat([src, fake.detach()],1))
            loss_D = sum((criterion_GAN(dr, torch.ones_like(dr)) + criterion_GAN(df, torch.zeros_like(df))) * 0.5
                         for dr, df in zip(pred_real, pred_fake))
            optimizer_D.zero_grad(); loss_D.backward(); optimizer_D.step()
            pred_fake_for_G = D(torch.cat([src, fake],1))
            loss_G_GAN = sum(criterion_GAN(df, torch.ones_like(df)) for df in pred_fake_for_G)
            loss_G_L1 = criterion_L1(fake, tgt) * lambda_L1
            loss_G = loss_G_GAN + loss_G_L1
            optimizer_G.zero_grad(); loss_G.backward(); optimizer_G.step()
            epoch_bar.set_postfix({'Loss_D': f"{loss_D.item():.4f}", 'Loss_G': f"{loss_G.item():.4f}"})
        epoch_bar.close()

        # Save checkpoints and samples
        save_checkpoint(G, optimizer_G, os.path.join(checkpoint_dir, f'epoch_{epoch}.pth'))
        if epoch == 1 or epoch % sample_interval == 0:
            save_some_examples(G, val_loader, epoch, sample_dir, device)

    print("Training complete.", flush=True)

# Run main unconditionally to ensure execution
main()
