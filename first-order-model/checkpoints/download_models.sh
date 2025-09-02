#!/bin/bash
mkdir -p first-order-model/checkpoints

# Install gdown if not present
pip install gdown

# Example models
echo "Downloading models..."

# Vox model
gdown --id 1pB5zdTqXsklgysZ425IpREQnS3-Jr48q -O first-order-model/checkpoints/vox-cpk.pth.tar

# MGIF model
gdown --id 1AbCdEfGhIjKlMnOpQrStUvWxYz123456 -O first-order-model/checkpoints/mgif.pth.tar

# Avatar model
gdown --id 1ZyXwVuTsRqPoNmLkJiHgFeDcBa987654 -O first-order-model/checkpoints/avatar.pth.tar

echo "All models downloaded successfully!"
