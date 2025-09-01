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


def generate_blink_animation(image_path: str, output_path: str, fps: int = 10, total_frames: int = 20):
    # Load input image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise Exception("Failed to read image")

    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    h, w, _ = img.shape

    results = model(image_path)
    detections = results[0].boxes.xyxy.cpu().numpy()

    eyes = [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2, *_ in detections]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    if not eyes:
        for _ in range(total_frames):
            out.write(img)
    else:
        for frame_id in range(total_frames):
            frame = img.copy()
            for (x1, y1, x2, y2) in eyes:
                eye_region = frame[y1:y2, x1:x2]
                alpha = abs(np.sin(np.pi * frame_id / total_frames))
                overlay = np.zeros_like(eye_region, dtype=np.uint8)
                cv2.rectangle(overlay, (0, 0), (x2 - x1, y2 - y1), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 1 - alpha, eye_region, alpha, 0, eye_region)
                frame[y1:y2, x1:x2] = eye_region
            out.write(frame)

    out.release()
    return {"frames": total_frames, "fps": fps, "resolution": f"{w}x{h}", "duration": total_frames / fps}


@app.get("/process")
def process(request: Request, image_url: str = Query(...)):
    try:
        image_id = str(uuid.uuid4())
        image_path = os.path.join(OUTDIR, f"{image_id}.jpg")
        download_image(image_url, image_path)

        output_path = os.path.join(OUTDIR, f"{image_id}.mp4")
        metadata = generate_blink_animation(image_path, output_path)

        base_url = str(request.base_url).rstrip("/")
        video_url = f"{base_url}/outputs/{os.path.basename(output_path)}"

        return {"video_url": video_url, **metadata}
    except Exception as e:
        return {"error": str(e)}
