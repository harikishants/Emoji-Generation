
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
import torch
from PIL import Image
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import mlflow
import mlflow.pytorch
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def rgba_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGBA')

def plot_loss(losses, save_dir="output/diffusion", filename="diffusion_losses.png"):
    # Save the losses to a JSON file
    with open(f"{save_dir}/losses.json", "w") as f:
        json.dump(losses, f)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(losses, label="Diffusion Loss (e.g., MSE)", color='purple')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Diffusion Model Training Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{filename}")
    plt.close()

def generate_images(unet, noise_scheduler, save_dir='output/diffusion_v2', num_images=50, epoch=0):
    unet.eval()
    for i in range(num_images):
        with torch.no_grad():
            pipeline = DDPMPipeline(unet=unet, scheduler=noise_scheduler)
            pipeline.to(device)
            images = pipeline(num_inference_steps=50).images
            img = images[0]
            img.save(os.path.join(save_dir, f"generated_image_epoch_{epoch}_{i+1}.png"))

if __name__ == "__main__":

    transform = transforms.Compose([
                                    transforms.ToTensor(), # converts [0,255] to [0, 1]
                                    transforms.Normalize([0.5]*4, [0.5]*4) # converts to [-1, 1] for using tanh
                                    ])

    dataset = datasets.ImageFolder(root='joypixels-7.0-free/png/labeled/32/', transform=transform, loader=rgba_loader)

    subset_size = 100
    indices = random.sample(range(len(dataset)), subset_size)
    dataset = Subset(dataset, indices)

    epochs = 5
    batch_size = 8

    save_dir = "output/diffusion_mlflow"
    os.makedirs(save_dir, exist_ok=True)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    unet = UNet2DModel(
                        sample_size=32,
                        in_channels=4,
                        out_channels=4,
                        layers_per_block=3,
                        block_out_channels=(64, 128, 256, 512),
                        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
                        up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
                        ).to(device)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    unet.load_state_dict(torch.load("output/diffusion_v2/unet_epoch_100.pt", map_location=device))
    # generate_images(unet, noise_scheduler, save_dir, num_images=50)
    # exit()

    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4)

    pipeline = DDPMPipeline(unet=unet, scheduler=noise_scheduler)

    images = pipeline().images

    diffusion_losses = []

    mlflow.set_experiment("diffusion_emoji")

    with mlflow.start_run(run_name='run-2'):
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("lr", 1e-4)
        # mlflow.log_param("architecture", unet)

        for epoch in range(epochs):
            unet.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            total_loss = 0
            batches = 0
            for batch in pbar:
                clean_images = batch[0].to(device)  # shape: (B, 4, 32, 32)
                noise = torch.randn_like(clean_images)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (clean_images.shape[0],), device=device).long()
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

                # Predict noise
                noise_pred = unet(noisy_images, timesteps).sample

                # loss = torch.nn.functional.mse_loss(noise_pred, noise)
                loss = torch.nn.functional.smooth_l1_loss(noise_pred, noise)


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix(loss=loss.item())
                total_loss += loss.item()
                batches += 1

            avg_loss = total_loss/batches
            diffusion_losses.append(avg_loss)
            plot_loss(diffusion_losses, save_dir)
            mlflow.log_metric("epoch_loss", avg_loss, step=epoch)

            if (epoch+1)%1 == 0:
                generate_images(unet, noise_scheduler, save_dir, num_images=1, epoch=epoch)
                torch.save(unet.state_dict(), os.path.join(save_dir, f"unet_epoch_{epoch+1}.pt"))
                mlflow.log_artifact(os.path.join(save_dir, f"generated_image_epoch_{epoch}_1.png"))


        print("Training complete.")
        mlflow.pytorch.log_model(unet, name="unet_model")
