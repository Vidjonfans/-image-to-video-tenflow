import cv2
import numpy as np
import os
import uuid
import requests
from fastapi import FastAPI, Query, Request
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

# FastAPI app
app = FastAPI()

# Output folder
OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTDIR), name="outputs")

# Load YOLOv8 model (pretrained or custom eye detection model)
model = YOLO("yolov8n.pt")  # replace with fine-tuned weights for eyes


def download_image(image_url: str, save_path: str):
    r = requests.get(image_url, stream=True)
    if r.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(r.content)
    return save_path


def generate_blink_animation(image_path: str, output_path: str):
    # Load input image
    img = cv2.imread(image_path)
    h, w, _ = img.shape

    # Detect objects (eyes) using YOLO
    results = model(image_path)
    detections = results[0].boxes.xyxy.cpu().numpy()

    # Collect eye regions
    eyes = []
    for det in detections:
        x1, y1, x2, y2 = map(int, det[:4])
        eyes.append((x1, y1, x2, y2))

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 10
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    if not eyes:
        # fallback: agar eyes detect na ho to bhi ek valid video banao
        for _ in range(20):
            out.write(img)
    else:
        # GAN-based animation (placeholder: simple blink effect)
        for frame_id in range(20):
            frame = img.copy()
            for (x1, y1, x2, y2) in eyes:
                eye_region = frame[y1:y2, x1:x2]

                # Simple blink simulation: darken eyes gradually
                alpha = abs(np.sin(np.pi * frame_id / 20))
                overlay = np.zeros_like(eye_region, dtype=np.uint8)
                cv2.rectangle(
                    overlay,
                    (0, 0),
                    (x2 - x1, y2 - y1),
                    (0, 0, 0),
                    -1,
                )
                cv2.addWeighted(overlay, 1 - alpha, eye_region, alpha, 0, eye_region)
                frame[y1:y2, x1:x2] = eye_region

            out.write(frame)

    out.release()
    cv2.destroyAllWindows()
    return output_path


@app.get("/process")
def process(request: Request, image_url: str = Query(...)):
    # Download image
    image_id = str(uuid.uuid4())
    image_path = os.path.join(OUTDIR, f"{image_id}.jpg")
    download_image(image_url, image_path)

    # Output video path
    output_path = os.path.join(OUTDIR, f"{image_id}.mp4")

    # Generate animation
    generate_blink_animation(image_path, output_path)

    # Full public URL banake return karo
    base_url = str(request.base_url).rstrip("/")
    video_url = f"{base_url}/outputs/{os.path.basename(output_path)}"
    return {"video_url": video_url}
