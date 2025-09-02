import cv2
import numpy as np
import mediapipe as mp
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

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)


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


def get_eye_landmarks(image):
    """Detect eye landmarks using Mediapipe FaceMesh"""
    h, w, _ = image.shape
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return []

    eyes = []
    for face_landmarks in results.multi_face_landmarks:
        # Left eye landmark indexes (Mediapipe)
        left_idx = [33, 160, 158, 133, 153, 144]
        right_idx = [362, 385, 387, 263, 373, 380]

        left_eye = [(int(face_landmarks.landmark[i].x * w),
                     int(face_landmarks.landmark[i].y * h)) for i in left_idx]
        right_eye = [(int(face_landmarks.landmark[i].x * w),
                      int(face_landmarks.landmark[i].y * h)) for i in right_idx]

        eyes.append(left_eye)
        eyes.append(right_eye)

    return eyes


def simulate_eye_blink(frame, eyes, blink_factor):
    """
    Animate eyelids closing/opening using polygon fill
    blink_factor = 1 (open) -> 0 (closed)
    """
    overlay = frame.copy()
    for eye in eyes:
        hull = cv2.convexHull(np.array(eye))
        x, y, w, h = cv2.boundingRect(hull)

        # Shrink eye vertically depending on blink_factor
        center_y = y + h // 2
        new_h = int(h * blink_factor)
        y1 = center_y - new_h // 2
        y2 = center_y + new_h // 2

        # Draw skin color over eye area
        avg_color = frame[y:y+h, x:x+w].mean(axis=(0, 1)).astype(int).tolist()
        cv2.fillConvexPoly(overlay, hull, avg_color)

        # Restore only the visible strip
        if new_h > 1:
            eye_region = frame[y1:y2, x:x+w]
            overlay[y1:y2, x:x+w] = eye_region

    return overlay


def generate_blink_animation(image_path: str, output_path: str, fps: int = 10, total_frames: int = 20):
    # Load input image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise Exception("Failed to read image")

    h, w, _ = img.shape

    # Detect eye landmarks
    eyes = get_eye_landmarks(img)

    # Temporary raw video
    temp_raw = output_path.replace(".mp4", "_raw.avi")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(temp_raw, fourcc, fps, (w, h))

    if not eyes:
        for _ in range(total_frames):
            out.write(img)
    else:
        for frame_id in range(total_frames):
            blink_factor = 1 - abs(np.sin(np.pi * frame_id / total_frames))  # open -> close -> open
            frame = simulate_eye_blink(img.copy(), eyes, blink_factor)
            out.write(frame)

    out.release()

    # Convert raw -> browser friendly mp4
    convert_to_browser_friendly(temp_raw, output_path)

    if os.path.exists(temp_raw):
        os.remove(temp_raw)

    return {"status": "done"}


@app.get("/process")
def process(request: Request, image_url: str = Query(...)):
    try:
        image_id = str(uuid.uuid4())
        image_path = os.path.join(OUTDIR, f"{image_id}.jpg")
        download_image(image_url, image_path)

        output_path = os.path.join(OUTDIR, f"{image_id}.mp4")
        generate_blink_animation(image_path, output_path)

        base_url = str(request.base_url).rstrip("/")
        video_url = f"{base_url}/outputs/{os.path.basename(output_path)}"

        return {"video_url": video_url}
    except Exception as e:
        return {"error": str(e)}
