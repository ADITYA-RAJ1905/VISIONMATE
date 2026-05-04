import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import pyttsx3
import time

# ------------------------------
# PAGE
# ------------------------------
st.title("Assistive Navigation AI")

run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

# ------------------------------
# VOICE
# ------------------------------
engine = pyttsx3.init()
engine.setProperty('rate', 160)

last_speak = 0
cooldown = 2

def speak(text):
    global last_speak
    now = time.time()
    if now - last_speak > cooldown:
        engine.say(text)
        engine.runAndWait()
        last_speak = now

# ------------------------------
# DEVICE
# ------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------
# LOAD MODELS (CACHE)
# ------------------------------
@st.cache_resource
def load_models():
    yolo = YOLO("yolov8n.pt")

    road_processor = SegformerImageProcessor.from_pretrained(
        "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    )
    road_model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    ).to(device)

    pothole_model = SegformerForSemanticSegmentation.from_pretrained(
        "pothole_model_final"   # 🔥 use your saved model
    ).to(device)

    pothole_processor = SegformerImageProcessor.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512"
    )

    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(device)
    midas.eval()
    midas_transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

    return yolo, road_model, road_processor, pothole_model, pothole_processor, midas, midas_transform


yolo, road_model, road_processor, pothole_model, pothole_processor, midas, midas_transform = load_models()

# ------------------------------
# CAMERA
# ------------------------------
cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # YOLO
    results = yolo(frame)
    annotated = results[0].plot()
    boxes = results[0].boxes

    obstacle_detected = False
    obstacle_name = ""

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2

        if 250 < cx < 390:
            obstacle_detected = True
            obstacle_name = yolo.names[int(box.cls)]

    # DEPTH
    depth_input = midas_transform(rgb).to(device)

    with torch.no_grad():
        depth = midas(depth_input)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()

    depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
    center_depth = depth_norm[200:300, 250:390].mean()

    # ROAD SEGMENTATION
    inputs = road_processor(images=rgb, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = road_model(**inputs)

    road_mask = outputs.logits.argmax(dim=1)[0].cpu().numpy()
    road_mask = cv2.resize(road_mask.astype(np.uint8), (640, 480))
    walkable = (road_mask == 0) | (road_mask == 1)

    center_walkable = walkable[200:300, 250:390]
    walkable_ratio = np.mean(center_walkable)

    # POTHOLE
    inputs = pothole_processor(images=rgb, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = pothole_model(**inputs)

    pothole_mask = outputs.logits.argmax(dim=1)[0].cpu().numpy()
    pothole_mask = pothole_mask.astype(np.uint8)
    pothole_mask = cv2.resize(pothole_mask, (640, 480))
    pothole_mask = cv2.medianBlur(pothole_mask, 5)

    center_region = pothole_mask[200:300, 250:390]
    pothole_ratio = np.sum(center_region == 1) / center_region.size

    pothole_detected = pothole_ratio > 0.08

    if pothole_detected and center_depth < 0.3:
        pothole_detected = False

    # WALL
    wall_detected = (center_depth < 0.3) and (walkable_ratio < 0.3)

    # DECISION
    message = "Path clear"

    if wall_detected:
        message = "Wall ahead. Stop"
        speak(message)

    elif obstacle_detected:
        message = f"{obstacle_name} ahead"
        speak(message)

    elif pothole_detected:
        message = "Pothole ahead. Avoid"
        speak(message)

    # VISUAL
    overlay = annotated.copy()
    overlay[pothole_mask == 1] = (255, 0, 0)
    overlay[walkable] = (0, 255, 0)

    final = cv2.addWeighted(annotated, 0.7, overlay, 0.3, 0)

    cv2.putText(final, message, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    FRAME_WINDOW.image(final)

cap.release()