#!/bin/bash
# Create checkpoints folder
mkdir -p first-order-model/checkpoints

# Install gdown if not present
pip install gdown

echo "Downloading models..."

# Vox model
gdown --id 1_v_xW1V52gZCZnXgh1Ap_gwA9YVIzUnS -O first-order-model/checkpoints/vox-cpk.pth.tar

# MGIF model
gdown --id 1L8P-hpBhZi8Q_1vP2KlQ4N6dvlzpYBvZ -O first-order-model/checkpoints/vox-adv-cpk.pth.tar

# Avatar model
gdown --id 10o7v0UdT4DVLaTIz1n6UojFbaMOGxA_y -O first-order-model/checkpoints/taichi-cpk.pth.tar

# YOLO best.pt (replace with your actual Drive file ID 👇)
gdown --id 1_v_xW1V52gZCZnXgh1Ap_gwA9YVIzUnS -O best.pt

echo "All models downloaded successfully!"
