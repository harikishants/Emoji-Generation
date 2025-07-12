from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from generate import generate_emojis
import uuid
from pathlib import Path

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "images": []})

@app.post("/generate", response_class=HTMLResponse)
def generate(request: Request, num_images: int = Form(...)):
    session_id = str(uuid.uuid4())
    output_dir = Path("static/generated") / session_id
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_emojis(model_path = 'trained_models/unet.pt', num_images=num_images, save_dir=output_dir)

    image_paths = [f"/static/generated/{session_id}/{img.name}" for img in output_dir.iterdir() if img.suffix == ".png"]

    return templates.TemplateResponse("index.html", {"request": request, "images": image_paths})
