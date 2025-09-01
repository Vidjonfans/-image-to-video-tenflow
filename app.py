import cv2
import numpy as np
import os
import uuid
import requests
import subprocess
from fastapi import FastAPI, Query, Request
from fastapi.staticfiles import StaticFiles

# FastAPI app
app = FastAPI()

# Output folder
OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTDIR), name="outputs")

# Haar Cascade for eyes
EYE_CASCADE_PATH = "cascades/haarcascade_eye.xml"
eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)


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


def detect_eyes(image):
    """Detect eyes using Haar Cascade"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return [(x, y, x + w, y + h) for (x, y, w, h) in eyes]

#------- start eyesblink----------
def simulate_eye_blink(frame, eyes, blink_factor):
    """
    Simulates natural blinking:
    - blink_factor = 1.0 (fully open)
    - blink_factor = 0.0 (fully closed)
    """
    for (x1, y1, x2, y2) in eyes:
        eye_region = frame[y1:y2, x1:x2].copy()
        h, w, _ = eye_region.shape
        if h <= 1 or w <= 1:
            continue

        # Copy original eye
        closed_eye = eye_region.copy()

        # Height of the visible area during blink
        visible_h = int(h * blink_factor)
        if visible_h < 1:
            visible_h = 1

        # Fill top and bottom with skin-tone (mean color of eye region)
        avg_color = closed_eye.mean(axis=(0, 1)).astype(np.uint8).tolist()
        closed_eye[:h//2, :] = avg_color   # upper eyelid
        closed_eye[h//2:, :] = avg_color   # lower eyelid

        # Keep only middle strip open
        start = (h - visible_h) // 2
        end = start + visible_h
        closed_eye[start:end, :] = eye_region[start:end, :]

        # Replace in frame
        frame[y1:y2, x1:x2] = closed_eye

    return frame

#------- End eyesblink----------

def generate_blink_animation(image_path: str, output_path: str, fps: int = 10, total_frames: int = 20):
    # Load input image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise Exception("Failed to read image")

    h, w, _ = img.shape

    # Detect eyes
    eyes = detect_eyes(img)

    # Temporary raw video
    temp_raw = output_path.replace(".mp4", "_raw.avi")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(temp_raw, fourcc, fps, (w, h))

    if not eyes:
        for _ in range(total_frames):
            out.write(img)
    else:
        for frame_id in range(total_frames):
            frame = img.copy()
            # Blink curve: open -> close -> open
            blink_factor = 1 - abs(np.sin(np.pi * frame_id / total_frames))
            frame = simulate_eye_blink(frame, eyes, blink_factor)
            out.write(frame)

    out.release()

    # Convert raw -> browser friendly mp4
    convert_to_browser_friendly(temp_raw, output_path)

    # Clean up temp file
    if os.path.exists(temp_raw):
        os.remove(temp_raw)

    # Verify video properties
    cap = cv2.VideoCapture(output_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_actual = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps_actual if fps_actual > 0 else 0
    cap.release()

    return {
        "frames": frame_count,
        "fps": fps_actual,
        "resolution": f"{width}x{height}",
        "duration": duration
    }


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

        return {
            "video_url": video_url,
            **metadata
        }
    except Exception as e:
        return {"error": str(e)}
