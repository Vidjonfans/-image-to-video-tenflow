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

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # replace with fine-tuned weights for eyes


def download_image(image_url: str, save_path: str):
    r = requests.get(image_url, stream=True)
    if r.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(r.content)
        return save_path
    else:
        raise Exception(f"Image download failed: {r.status_code}")


def generate_blink_animation(image_path: str, output_path: str):
    # Load input image with alpha support
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise Exception("Failed to read image")

    # Convert RGBA -> BGR if needed
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

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
        for frame_id in range(20):
            frame = img.copy()
            for (x1, y1, x2, y2) in eyes:
                eye_region = frame[y1:y2, x1:x2]

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
    try:
        # Download image
        image_id = str(uuid.uuid4())
        image_path = os.path.join(OUTDIR, f"{image_id}.jpg")
        download_image(image_url, image_path)

        # Output video path
        output_path = os.path.join(OUTDIR, f"{image_id}.mp4")

        # Generate animation
        generate_blink_animation(image_path, output_path)

        # Full public URL
        base_url = str(request.base_url).rstrip("/")
        video_url = f"{base_url}/outputs/{os.path.basename(output_path)}"
        return {"video_url": video_url}

    except Exception as e:
        return {"error": str(e)}
