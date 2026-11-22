import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")

# OpenVINO Pitch Pose Model (Phase 1)
PITCH_MODEL_XML = os.path.join(MODELS_DIR, "best_pose_pro_320_fp16.xml")
OV_DEVICE = "GPU"   # GPU (Intel Arc/iGPU) | CPU (avoid NPU for Spatial Softmax)

# Detection thresholds
CONF_THRESHOLD = 0.35
EMA_ALPHA = 0.70

# FIFA Standard Pitch (meters)
PITCH_L = 105.0
PITCH_W = 68.0

for d in (MODELS_DIR, OUTPUTS_DIR, UPLOADS_DIR):
    os.makedirs(d, exist_ok=True)
