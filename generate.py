from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
import torch
import os


def generate_emojis(model_path, save_dir="static/generated", num_images=5):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    unet = UNet2DModel(
                        sample_size=32,
                        in_channels=4,
                        out_channels=4,
                        layers_per_block=3,
                        block_out_channels=(64, 128, 256, 512),
                        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
                        up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
                        )

    unet.load_state_dict(torch.load(model_path, map_location=device))

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    pipeline = DDPMPipeline(unet=unet, scheduler=noise_scheduler).to(device)

    os.makedirs(save_dir, exist_ok=True)

    for i in range(num_images):
        with torch.no_grad():
            img = pipeline(num_inference_steps=50).images[0]
            img.save(os.path.join(save_dir, f'emoji_{i+1}.png'))