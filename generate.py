from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
import torch
import os
import boto3

def load_model(model_path, bucket_name=None, s3_key=None):
    print(f'[debug] {model_path}')
    if os.path.exists(model_path):
        print("[info] Loading from local")
        model = torch.load(model_path, map_location="cpu")

    elif bucket_name and s3_key:
        print("[info] Loading from AWS s3 storage")
        s3 = boto3.client("s3")
        s3.download_file(bucket_name, s3_key, model_path)
        model = torch.load(model_path, map_location="cpu")

    else:
        raise FileNotFoundError(f"Model not found locally and no S3 details provided.")

    return model

def generate_emojis(model, save_dir="static/generated", num_images=5):

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    unet = UNet2DModel(
                        sample_size=32,
                        in_channels=4,
                        out_channels=4,
                        layers_per_block=3,
                        block_out_channels=(64, 128, 256, 512),
                        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
                        up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
                        )

    unet.load_state_dict(model)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    pipeline = DDPMPipeline(unet=unet, scheduler=noise_scheduler).to(device)

    os.makedirs(save_dir, exist_ok=True)

    for i in range(num_images):
        with torch.no_grad():
            img = pipeline(num_inference_steps=50).images[0]
            img.save(os.path.join(save_dir, f'emoji_{i+1}.png'))