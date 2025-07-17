# Emoji Generation

Generates custom emojis using GANs and Diffusion Models, built with FastAPI and PyTorch. Deployed as a FastAPI-based web app, fully containerized with Docker and scalable through AWS infrastructure.

## Features

- Generate 32x32 emojis using a trained diffusion model and GANs
- Trained diffusion model using `UNet2DModel` and `DDPMScheduler` from Hugging Face `diffusers`
- Evaluated using **Frechet Inception Distance (FID)**
- REST API built using **FastAPI**
- Dockerized and deployed via **AWS EC2**
- Real-time traffic monitoring via **Prometheus**
- CI/CD integration with **AWS CodePipeline & CodeBuild**
- Model artifacts stored and accessed from **AWS S3**

## Project Structure

- `app.py` — FastAPI server for emoji generation.
- `generate.py` — Emoji generation logic using a diffusion model.
- `diffusion.py` — Training script for diffusion models.
- `GAN.py`, `simple_gan/GAN_Quadratic.py` — GAN training scripts.
- `templates/index.html` — Web UI template.
- `requirements.txt` — Python dependencies.
- `Dockerfile` — Container build instructions.

## Tech Stack

- Python, PyTorch, FastAPI
- Hugging Face `diffusers`
- Docker, AWS EC2, S3, ECR
- Prometheus (monitoring)
- GitHub Actions / AWS CodePipeline (CI/CD)
- VS Code Remote SSH


## Getting Started

### Prerequisites

- Python 3.10+
- [PyTorch](https://pytorch.org/)
- [Diffusers](https://github.com/huggingface/diffusers)
- FastAPI, Uvicorn

### Installation

```bash
git clone https://github.com/yourusername/Emoji_Generation.git
cd Emoji_Generation
pip install -r requirements.txt
```

### Running Locally

1. **Train or Download a Model**  
   Place your trained diffusion model weights at `trained_models/unet.pt`.

2. **Start the Web App**

```bash
uvicorn app:app --reload
```

Visit [http://localhost:8000](http://localhost:8000) to use the emoji generator.

### Docker

Build and run the app in a container:

```bash
docker build -t emoji-gen .
docker run -p 8000:8000 emoji-gen
```

## Usage

- Select the number of emojis to generate (1–20) and click "Generate".
- Generated emojis will appear on the page.

## Training

- Use `diffusion.py` to train a diffusion model on your emoji dataset.
- Use `GAN.py` or `simple_gan/GAN_Quadratic.py` for GAN experiments.

## File Locations

- Generated emojis: `static/generated/`
- Training outputs: `output/`

## License

MIT License

---

**Note:**  
- Emoji dataset is not included.  
- For best results, train your own models or use provided
