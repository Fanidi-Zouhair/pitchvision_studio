"""
Smart Model Importer & OpenVINO FP16 Converter for PitchVision Studio (V1 + V2 Model E)
Khaddmo mn pitchvision-studio/ wla pitchvision-studio/backend/:
    pip install -q timm
    python import_models_from_downloads.py

Features:
  - Auto-checks if model_X_fp16.xml + .bin ALREADY exist & are up-to-date -> SKIPS them!
  - Automatically re-converts if you download a newer .pth (or pass --force).
  - Supports V1 models (A, B, C, D @ 320x320) AND V2 Model E (512x288 RepViT-M0.9 + FPN + SafeGAU + Raw SimCC).
"""
import os
import sys
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import openvino as ov
from torchvision.models import resnet34, mobilenet_v3_large

IMG_SIZE = 320
NUM_KPTS = 33
SIMCC_BINS = IMG_SIZE * 2

# Model E (V2 Master) exact dimensions
V2_IMG_W = 512
V2_IMG_H = 288
V2_BINS_X = V2_IMG_W * 2   # 1024
V2_BINS_Y = V2_IMG_H * 2   # 576
V2_SIMCC_TAU = 0.1

# Auto-detect whether this script sits in pitchvision-studio/ or pitchvision-studio/backend/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(SCRIPT_DIR, "backend", "main.py")):
    PROJECT_ROOT = SCRIPT_DIR
    BACKEND_DIR = os.path.join(SCRIPT_DIR, "backend")
else:
    BACKEND_DIR = SCRIPT_DIR
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DOWNLOADS_DIR = os.path.join(os.path.expanduser("~"), "Downloads")
MODELS_DIR = os.path.join(BACKEND_DIR, "models")
SEARCH_DIRS = list(dict.fromkeys([PROJECT_ROOT, BACKEND_DIR, MODELS_DIR, DOWNLOADS_DIR]))
os.makedirs(MODELS_DIR, exist_ok=True)


# ==============================================================================
# 1. V1 ResNet34 U-Net Architecture (Models A, B, C)
# ==============================================================================
class PitchPoseNet33(nn.Module):
    def __init__(self, num_keypoints=NUM_KPTS, img_size=IMG_SIZE):
        super().__init__()
        hm = img_size // 4
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        backbone = resnet34(weights=None)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.enc1, self.enc2, self.enc3, self.enc4 = backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
        self.dec3 = nn.Sequential(nn.Conv2d(768, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True), nn.Dropout2d(0.10))
        self.dec2 = nn.Sequential(nn.Conv2d(384, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True), nn.Dropout2d(0.08))
        self.dec1 = nn.Sequential(nn.Conv2d(192, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True))
        self.heatmap_head = nn.Conv2d(64, num_keypoints, 1)
        self.vis_global = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, 128), nn.ReLU(True), nn.Dropout(0.25), nn.Linear(128, num_keypoints))
        self.vis_hm_scale = nn.Parameter(torch.ones(1, num_keypoints) * 0.5)
        self.vis_hm_bias = nn.Parameter(torch.zeros(1, num_keypoints))
        gy, gx = torch.meshgrid(torch.linspace(0.0, 1.0, hm), torch.linspace(0.0, 1.0, hm), indexing="ij")
        self.register_buffer("grid_x", gx.reshape(1, 1, -1))
        self.register_buffer("grid_y", gy.reshape(1, 1, -1))

    def forward(self, x):
        x = (x - self.img_mean) / self.img_std
        e1 = self.enc1(self.stem(x))
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        d3 = self.dec3(torch.cat([F.interpolate(e4, scale_factor=2, mode="bilinear", align_corners=False), e3], 1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=False), e2], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False), e1], 1))
        hm_flat = self.heatmap_head(d1).view(x.shape[0], -1, (IMG_SIZE // 4) ** 2)
        probs = F.softmax(hm_flat, dim=-1)
        coords = torch.stack([(probs * self.grid_x).sum(-1), (probs * self.grid_y).sum(-1)], -1)
        hm_peak = hm_flat.max(-1).values - hm_flat.mean(-1)
        vis_logits = self.vis_global(e4) + self.vis_hm_scale * hm_peak + self.vis_hm_bias
        return coords, vis_logits


# ==============================================================================
# 2. V1 MobileNetV3-Large + SimCC 1D Architecture (Model D)
# ==============================================================================
class PitchSimCCNet33(nn.Module):
    def __init__(self, num_keypoints=NUM_KPTS, img_size=IMG_SIZE, simcc_bins=SIMCC_BINS):
        super().__init__()
        feat_hw = img_size // 32
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.register_buffer("bin_coords", torch.linspace(0.0, 1.0, simcc_bins).view(1, 1, -1))
        self.features = mobilenet_v3_large(weights=None).features
        self.final_conv = nn.Sequential(
            nn.Conv2d(960, 256, 1, bias=False), nn.BatchNorm2d(256), nn.SiLU(True), nn.Conv2d(256, num_keypoints, 1)
        )
        self.token_mlp = nn.Sequential(nn.LayerNorm(feat_hw * feat_hw), nn.Linear(feat_hw * feat_hw, 256), nn.SiLU(True), nn.Dropout(0.10))
        self.cls_x = nn.Linear(256, simcc_bins)
        self.cls_y = nn.Linear(256, simcc_bins)
        self.vis_global = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(960, 128), nn.SiLU(True), nn.Dropout(0.20), nn.Linear(128, num_keypoints))
        self.vis_peak_scale = nn.Parameter(torch.ones(1, num_keypoints) * 0.35)
        self.vis_peak_bias = nn.Parameter(torch.zeros(1, num_keypoints))

    def forward(self, x):
        x = (x - self.img_mean) / self.img_std
        feat = self.features(x)
        emb = self.token_mlp(self.final_conv(feat).flatten(2))
        lx, ly = self.cls_x(emb), self.cls_y(emb)
        px = (F.softmax(lx, dim=-1) * self.bin_coords).sum(-1)
        py = (F.softmax(ly, dim=-1) * self.bin_coords).sum(-1)
        coords = torch.stack([px, py], -1)
        vis_logits = self.vis_global(feat) + self.vis_peak_scale * 0.5 * ((lx.max(-1).values - lx.mean(-1)) + (ly.max(-1).values - ly.mean(-1))) + self.vis_peak_bias
        return coords, vis_logits


# ==============================================================================
# 3. V2 Master Architecture (Model E: 512x288 RepViT-M0.9 + FPN + SafeGAU + Raw SimCC)
# ==============================================================================
class DWConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, k, stride=s, padding=k // 2, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn2(self.pw(self.act(self.bn1(self.dw(x))))))


class SafeGAU(nn.Module):
    def __init__(self, dim=256, expand=2, qk_dim=128, eps=1e-5):
        super().__init__()
        hidden = dim * expand
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
        self.uv = nn.Linear(dim, hidden * 2 + qk_dim, bias=False)
        self.q_scale = nn.Parameter(torch.ones(qk_dim))
        self.q_bias = nn.Parameter(torch.zeros(qk_dim))
        self.k_scale = nn.Parameter(torch.ones(qk_dim))
        self.k_bias = nn.Parameter(torch.zeros(qk_dim))
        self.out = nn.Linear(hidden, dim, bias=False)
        self.qk_dim = qk_dim
        self.hidden = hidden

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        xn = x * rms * self.scale
        u, v, z = self.uv(xn).split([self.hidden, self.hidden, self.qk_dim], dim=-1)
        u = F.silu(u)
        v = F.silu(v)
        q = z * self.q_scale + self.q_bias
        k = z * self.k_scale + self.k_bias
        attn = F.relu(torch.matmul(q, k.transpose(-1, -2)) * (self.qk_dim ** -0.5)).pow(2) / float(x.shape[1])
        return x + self.out(u * torch.matmul(attn, v))


class PitchVisionV2Net(nn.Module):
    def __init__(self, backbone_name="repvit_m0_9.dist_450e_in1k", num_kpts=NUM_KPTS):
        super().__init__()
        try:
            import timm
        except ImportError:
            raise ImportError("Model E requires 'timm'. Run: pip install -q timm")

        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        try:
            self.backbone = timm.create_model(backbone_name, pretrained=False, features_only=True)
        except Exception:
            self.backbone = timm.create_model("repvit_m0_9", pretrained=False, features_only=True)

        with torch.no_grad():
            feats = self.backbone(torch.zeros(1, 3, V2_IMG_H, V2_IMG_W))
        c3, c4, c5 = feats[-3].shape[1], feats[-2].shape[1], feats[-1].shape[1]
        h4, w4 = feats[-2].shape[2], feats[-2].shape[3]

        self.lat5 = nn.Sequential(nn.Conv2d(c5, 96, 1, bias=False), nn.BatchNorm2d(96), nn.SiLU(True))
        self.lat4 = nn.Sequential(nn.Conv2d(c4, 96, 1, bias=False), nn.BatchNorm2d(96), nn.SiLU(True))
        self.lat3 = nn.Sequential(nn.Conv2d(c3, 64, 1, bias=False), nn.BatchNorm2d(64), nn.SiLU(True))

        self.p4_refine = DWConvBNAct(96, 96, k=3, s=1)
        self.p4_to_p3 = nn.Sequential(nn.Conv2d(96, 64, 1, bias=False), nn.BatchNorm2d(64))
        self.p3_refine = DWConvBNAct(64, 64, k=3, s=1)
        self.p3_down = DWConvBNAct(64, 96, k=3, s=2)

        self.kpt_conv = nn.Sequential(
            nn.Conv2d(96, num_kpts, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(num_kpts),
            nn.SiLU(inplace=True),
        )
        self.token_proj = nn.Sequential(nn.Linear(h4 * w4, 256), nn.SiLU(inplace=True), nn.Dropout(0.10))
        self.gau = SafeGAU(dim=256, expand=2, qk_dim=128)

        self.head_x = nn.Linear(256, V2_BINS_X)
        self.head_y = nn.Linear(256, V2_BINS_Y)
        self.head_vis = nn.Linear(256, 1)

    def forward(self, x):
        x = (x - self.img_mean) / self.img_std
        feats = self.backbone(x)
        f3, f4, f5 = feats[-3], feats[-2], feats[-1]

        l5 = self.lat5(f5)
        p4 = self.p4_refine(self.lat4(f4) + F.interpolate(l5, size=f4.shape[-2:], mode="bilinear", align_corners=False))
        p3 = self.p3_refine(self.lat3(f3) + F.interpolate(self.p4_to_p3(p4), size=f3.shape[-2:], mode="bilinear", align_corners=False))
        fused = p4 + self.p3_down(p3)

        k = self.kpt_conv(fused).flatten(2)
        k = self.gau(self.token_proj(k))
        return self.head_x(k), self.head_y(k), self.head_vis(k).squeeze(-1)


# ==============================================================================
# TARGETS & SMART-SKIP CONVERSION LOGIC
# ==============================================================================
TARGETS = [
    ("model_A", "resnet34", "best_model_A_pure_vision*.pth"),
    ("model_B", "resnet34", "best_model_B_mask_soft_math*.pth"),
    ("model_C", "resnet34", "best_model_C_heatmap_heavy_math*.pth"),
    ("model_D", "simcc",    "best_model_D_simcc_mobilenet*.pth"),
    ("model_E", "v2_repvit","best_model_E_v2_repvit_simcc_512x288*.pth"),
]


def find_latest(pattern):
    matches = []
    for folder in SEARCH_DIRS:
        if os.path.exists(folder):
            matches.extend(glob.glob(os.path.join(folder, pattern)))
    if not matches:
        return None
    matches.sort(key=os.path.getmtime, reverse=True)
    return matches[0]


def is_already_converted(xml_path, pth_path=None, force=False):
    """Returns True if .xml & .bin already exist and are at least as new as .pth."""
    if force:
        return False
    bin_path = xml_path.replace(".xml", ".bin")
    if not (os.path.exists(xml_path) and os.path.exists(bin_path)):
        return False
    if pth_path is not None and os.path.getmtime(pth_path) > os.path.getmtime(xml_path):
        return False
    return True


def _decode_simcc_quick(lx, ly, tau=V2_SIMCC_TAU):
    lx = lx.astype(np.float32) / tau
    ly = ly.astype(np.float32) / tau
    px = np.exp(lx - lx.max(axis=-1, keepdims=True)); px /= (px.sum(axis=-1, keepdims=True) + 1e-8)
    py = np.exp(ly - ly.max(axis=-1, keepdims=True)); py /= (py.sum(axis=-1, keepdims=True) + 1e-8)
    gx = np.linspace(0.0, 1.0, lx.shape[-1], dtype=np.float32)
    gy = np.linspace(0.0, 1.0, ly.shape[-1], dtype=np.float32)
    return np.stack([(px * gx).sum(axis=-1), (py * gy).sum(axis=-1)], axis=-1)


def verify_v2_fp16_drift(net, xml_path, dummy_torch):
    """Quick PyTorch FP32 vs OpenVINO FP16 drift check for Model E."""
    try:
        core = ov.Core()
        dev = "GPU" if "GPU" in core.available_devices else "CPU"
        compiled = core.compile_model(xml_path, dev)
        req = compiled.create_infer_request()
        req.infer({0: dummy_torch.numpy()})
        ov_lx = req.get_tensor(compiled.output(0)).data[0]
        ov_ly = req.get_tensor(compiled.output(1)).data[0]

        with torch.no_grad():
            pt_lx, pt_ly, _ = net(dummy_torch)
        pt_xy = _decode_simcc_quick(pt_lx[0].numpy(), pt_ly[0].numpy())
        ov_xy = _decode_simcc_quick(ov_lx, ov_ly)

        dx = (ov_xy[:, 0] - pt_xy[:, 0]) * V2_IMG_W
        dy = (ov_xy[:, 1] - pt_xy[:, 1]) * V2_IMG_H
        err = np.sqrt(dx ** 2 + dy ** 2)
        print(f"  [PARITY CHECK on {dev}] Mean Drift: {err.mean():.3f} px | Max Drift: {err.max():.3f} px (@512x288)")
    except Exception as e:
        print(f"  [PARITY CHECK skipped]: {e}")


def convert_one(model_id, arch, pth_path):
    onnx_path = os.path.join(MODELS_DIR, f"{model_id}.onnx")
    xml_path = os.path.join(MODELS_DIR, f"{model_id}_fp16.xml")
    print(f"\n[{model_id}] Converting: {pth_path}")

    if arch == "v2_repvit":
        ckpt = torch.load(pth_path, map_location="cpu", weights_only=False)
        bb_name = ckpt.get("backbone_name", "repvit_m0_9.dist_450e_in1k") if isinstance(ckpt, dict) else "repvit_m0_9.dist_450e_in1k"
        state = ckpt["state_dict"] if (isinstance(ckpt, dict) and "state_dict" in ckpt) else ckpt
        net = PitchVisionV2Net(backbone_name=bb_name, num_kpts=NUM_KPTS).eval()
        net.load_state_dict(state)

        dummy = torch.rand(1, 3, V2_IMG_H, V2_IMG_W)
        torch.onnx.export(
            net, dummy, onnx_path,
            export_params=True, opset_version=18, do_constant_folding=True,
            input_names=["image"], output_names=["logits_x", "logits_y", "vis_logits"],
        )
        ov_model = ov.convert_model(onnx_path, input=[1, 3, V2_IMG_H, V2_IMG_W])
        ov.save_model(ov_model, xml_path, compress_to_fp16=True)
        size_mb = os.path.getsize(xml_path.replace(".xml", ".bin")) / (1024 * 1024)
        print(f"  [OK] OpenVINO FP16 ready -> {xml_path} ({size_mb:.2f} MB | 512x288 Raw SimCC)")
        verify_v2_fp16_drift(net, xml_path, dummy)
    else:
        net = (PitchSimCCNet33() if arch == "simcc" else PitchPoseNet33()).eval()
        net.load_state_dict(torch.load(pth_path, map_location="cpu", weights_only=True))

        dummy = torch.rand(1, 3, IMG_SIZE, IMG_SIZE)
        torch.onnx.export(
            net, dummy, onnx_path,
            export_params=True, opset_version=18, do_constant_folding=True,
            input_names=["image"], output_names=["coords", "vis_logits"],
        )
        ov_model = ov.convert_model(onnx_path, input=[1, 3, IMG_SIZE, IMG_SIZE])
        ov.save_model(ov_model, xml_path, compress_to_fp16=True)
        size_mb = os.path.getsize(xml_path.replace(".xml", ".bin")) / (1024 * 1024)
        print(f"  [OK] OpenVINO FP16 ready -> {xml_path} ({size_mb:.2f} MB | 320x320)")


def main():
    parser = argparse.ArgumentParser(description="Smart OpenVINO FP16 model importer for PitchVision Studio.")
    parser.add_argument("--force", action="store_true", help="Re-convert all models even if .xml/.bin already exist.")
    args = parser.parse_args()

    print("Scanning folders:", SEARCH_DIRS)
    converted = 0
    skipped = 0

    for model_id, arch, pattern in TARGETS:
        xml_path = os.path.join(MODELS_DIR, f"{model_id}_fp16.xml")
        pth = find_latest(pattern)

        if is_already_converted(xml_path, pth_path=pth, force=args.force):
            size_mb = os.path.getsize(xml_path.replace(".xml", ".bin")) / (1024 * 1024)
            print(f"[{model_id}] Already converted -> {os.path.basename(xml_path)} ({size_mb:.2f} MB) [SKIPPED]")
            skipped += 1
            continue

        if pth:
            convert_one(model_id, arch, pth)
            converted += 1
        else:
            print(f"[{model_id}] No .pth found (skipped).")

    print(f"\n[DONE] Converted: {converted} | Already up-to-date (skipped): {skipped} | Models dir: '{MODELS_DIR}'")


if __name__ == "__main__":
    main()