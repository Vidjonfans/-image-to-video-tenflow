import os
import uuid
import requests
import subprocess
from fastapi import FastAPI, Query, Request
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

# FastAPI app
app = FastAPI()

# Output folder
OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTDIR), name="outputs")

# Load YOLOv8 model (custom eye detection model)
model = YOLO("best.pt")  # <-- apna trained model path daalna

def download_image(image_url: str, save_path: str):
    r = requests.get(image_url, stream=True)
    if r.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(r.content)
        return save_path
    else:
        raise Exception(f"Image download failed: {r.status_code}")

def convert_to_browser_friendly(input_path: str, output_path: str):
    """Convert video to H.264 + AAC mp4 using ffmpeg"""
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-movflags", "+faststart",
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

def generate_blink_animation_with_gan(image_path: str, output_path: str, detections: list, fps: int = 10, duration: int = 2):
    """
    Placeholder for GAN blink generation.
    detections: YOLOv8 detections (bounding boxes of eyes)
    """
    # Abhi ke liye ek dummy video generate karenge (static image repeat karke)
    # Baad me is jagah GAN (SadTalker / FOMM) integrate karna hoga
    temp_raw = output_path.replace(".mp4", "_raw.avi")
    from cv2 import imread, VideoWriter, VideoWriter_fourcc

    img = imread(image_path)
    if img is None:
        raise Exception("Failed to read image")
    h, w, _ = img.shape

    fourcc = VideoWriter_fourcc(*"XVID")
    out = VideoWriter(temp_raw, fourcc, fps, (w, h))

    total_frames = fps * duration
    for _ in range(total_frames):
        out.write(img)
    out.release()

    # Convert raw -> browser-friendly mp4
    convert_to_browser_friendly(temp_raw, output_path)
    if os.path.exists(temp_raw):
        os.remove(temp_raw)

    return {"video_path": output_path, "duration": duration, "fps": fps}

@app.get("/process")
def process(request: Request, image_url: str = Query(...)):
    try:
        image_id = str(uuid.uuid4())
        image_path = os.path.join(OUTDIR, f"{image_id}.jpg")
        download_image(image_url, image_path)

        # Run YOLOv8 detection
        results = model(image_path)
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "class_id": int(box.cls[0]),
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                })

        # Generate blink animation with GAN placeholder
        output_path = os.path.join(OUTDIR, f"{image_id}.mp4")
        blink_info = generate_blink_animation_with_gan(image_path, output_path, detections)

        base_url = str(request.base_url).rstrip("/")
        video_url = f"{base_url}/outputs/{os.path.basename(output_path)}"

        return {
            "video_url": video_url,
            "video_duration": blink_info["duration"],
            "fps": blink_info["fps"],
            "detections": detections
        }
    except Exception as e:
        return {"error": str(e)}
