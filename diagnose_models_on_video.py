"""
DIAGNOSTIC TOOL: PyTorch (.pth) vs OpenVINO FP16 (CPU & GPU) on Real Video Frame
Khaddmo mn pitchvision-studio/ (m3a venv m-activi):
    python diagnose_models.py
Kay-qaren b d-dabtt:
  1. PyTorch FP32 (.pth) — nafs l-code dyal Colab
  2. OpenVINO FP16 (.xml) 3la CPU
  3. OpenVINO FP16 (.xml) 3la GPU
W kay-sauvegardi 'debug_video_check.jpg' bach t-chof b 3inik wach l-mouchkil mn OpenVINO wla mn pipeline!
"""
import os
import glob
import cv2
import numpy as np
import torch
import openvino as ov

from import_models_from_downloads import (
    PitchPoseNet33,
    PitchSimCCNet33,
    TARGETS,
    IMG_SIZE,
    MODELS_DIR,
    PROJECT_ROOT,
    BACKEND_DIR,
    find_latest,
)

CONF_THRESH = 0.35


def find_video():
    """Auto-find the first MP4 video in uploads/ or project root."""
    search = [
        os.path.join(BACKEND_DIR, "uploads", "*.mp4"),
        os.path.join(PROJECT_ROOT, "*.mp4"),
        os.path.join(BACKEND_DIR, "*.mp4"),
    ]
    matches = []
    for pat in search:
        matches.extend(glob.glob(pat))
    if not matches:
        return None
    matches.sort(key=os.path.getmtime, reverse=True)
    return matches[0]


def preprocess(frame_bgr):
    """Exact same preprocessing as Colab & pitch_detector.py."""
    resized = cv2.resize(frame_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    arr = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
    return np.ascontiguousarray(arr)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def draw_tile(frame_bgr, pt_c, pt_v, ov_c, ov_v, title):
    """Draw PyTorch (Green) vs OpenVINO GPU/CPU (Cyan) on a 960x540 tile."""
    img = cv2.resize(frame_bgr, (960, 540))
    h, w = img.shape[:2]

    # PyTorch predictions in GREEN + yellow IDs
    for i in range(len(pt_v)):
        if pt_v[i] >= CONF_THRESH:
            px, py = int(np.clip(pt_c[i, 0] * w, 0, w - 1)), int(np.clip(pt_c[i, 1] * h, 0, h - 1))
            cv2.circle(img, (px, py), 6, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.putText(img, f"P{i+1}", (px + 7, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, f"P{i+1}", (px + 7, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

    # OpenVINO predictions in CYAN + orange error line if different
    if ov_c is not None and ov_v is not None:
        for i in range(len(ov_v)):
            if ov_v[i] >= CONF_THRESH:
                ox, oy = int(np.clip(ov_c[i, 0] * w, 0, w - 1)), int(np.clip(ov_c[i, 1] * h, 0, h - 1))
                cv2.circle(img, (ox, oy), 3, (255, 220, 0), -1, cv2.LINE_AA)
                if pt_v[i] >= CONF_THRESH:
                    px, py = int(np.clip(pt_c[i, 0] * w, 0, w - 1)), int(np.clip(pt_c[i, 1] * h, 0, h - 1))
                    if abs(ox - px) + abs(oy - py) > 2:
                        cv2.line(img, (px, py), (ox, oy), (0, 120, 255), 2, cv2.LINE_AA)

    cv2.rectangle(img, (0, 0), (w, 34), (15, 15, 15), -1)
    cv2.putText(img, title, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def main():
    video_path = find_video()
    if not video_path:
        print("[ERROR] Ma-lqitch 7ta video .mp4 f uploads/ wla pitchvision-studio/!")
        return

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_idx = min(60, max(0, total // 3))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        print(f"[ERROR] Ma-qdartch n-qra frame {target_idx} mn '{video_path}'")
        return

    print(f"[VIDEO] Testing on '{os.path.basename(video_path)}' (Frame {target_idx}/{total}, {frame.shape[1]}x{frame.shape[0]})")
    x_np = preprocess(frame)
    x_pt = torch.from_numpy(x_np)

    core = ov.Core()
    avail_devs = core.available_devices
    print(f"[OPENVINO] Devices: {avail_devs}\n")

    print("=" * 102)
    print(f"  {'Model':10s} | {'PT Det':>6s} | {'OV-CPU Diff (Mean/Max)':>22s} | {'OV-GPU Diff (Mean/Max)':>22s} | {'GPU VisDiff':>11s} | {'Status'}")
    print("-" * 102)

    tiles = []
    for model_id, arch, pattern in TARGETS:
        pth = find_latest(pattern)
        xml = os.path.join(MODELS_DIR, f"{model_id}_fp16.xml")
        if not pth or not os.path.exists(xml):
            continue

        try:
            # 1. PyTorch FP32 reference
            net = (PitchSimCCNet33() if arch == "simcc" else PitchPoseNet33()).eval()
            net.load_state_dict(torch.load(pth, map_location="cpu", weights_only=True))
            with torch.no_grad():
                pt_c_t, pt_vl_t = net(x_pt)
            pt_c = pt_c_t[0].numpy()
            pt_v = sigmoid(pt_vl_t[0].numpy())
            n_det = int((pt_v >= CONF_THRESH).sum())

            # 2. OpenVINO CPU & GPU
            ov_res = {}
            for dev in ("CPU", "GPU"):
                if dev not in avail_devs:
                    ov_res[dev] = None
                    continue
                compiled = core.compile_model(xml, dev)
                req = compiled.create_infer_request()
                req.infer({0: x_np})
                oc = req.get_tensor(compiled.output(0)).data[0].copy()
                ovl = req.get_tensor(compiled.output(1)).data[0].copy()
                ov_p = sigmoid(ovl)
                px_diff = np.sqrt((((oc - pt_c) * IMG_SIZE) ** 2).sum(axis=-1))
                ov_res[dev] = (oc, ov_p, float(px_diff.mean()), float(px_diff.max()), float(np.max(np.abs(ov_p - pt_v))))

            cpu_str = f"{ov_res['CPU'][2]:5.2f}px / {ov_res['CPU'][3]:5.2f}px" if ov_res["CPU"] else "N/A"
            gpu_str = f"{ov_res['GPU'][2]:5.2f}px / {ov_res['GPU'][3]:5.2f}px" if ov_res["GPU"] else "N/A"
            vis_str = f"{ov_res['GPU'][4]:.4f}" if ov_res["GPU"] else "N/A"

            max_err = ov_res["GPU"][3] if ov_res["GPU"] else (ov_res["CPU"][3] if ov_res["CPU"] else 0.0)
            verdict = "OK (IDENTICAL)" if max_err < 1.5 else "BUG IN OV FP16!"
            print(f"  {model_id:10s} | {n_det:4d}/33 | {cpu_str:>22s} | {gpu_str:>22s} | {vis_str:>11s} | {verdict}")

            best_ov = ov_res["GPU"] or ov_res["CPU"]
            tiles.append(
                draw_tile(
                    frame, pt_c, pt_v,
                    best_ov[0] if best_ov else None,
                    best_ov[1] if best_ov else None,
                    f"{model_id} | PT(Green) vs OV(Cyan) | Det:{n_det} | MaxDiff:{max_err:.2f}px ({verdict})",
                )
            )
        except Exception as e:
            print(f"  {model_id:10s} | [ERROR] {e}")

    print("=" * 102)
    if tiles:
        grid = np.vstack(tiles) if len(tiles) <= 2 else np.vstack([
            np.hstack([tiles[0], tiles[1]]),
            np.hstack([tiles[2], tiles[3] if len(tiles) > 3 else np.zeros_like(tiles[0])]),
        ])
        out_path = os.path.join(PROJECT_ROOT, "debug_video_check.jpg")
        cv2.imwrite(out_path, grid)
        print(f"\n[OK] Saved visual check -> '{out_path}'")
        print("B3at li had table + s-sora 'debug_video_check.jpg' bach n-sa77o l-mouchkil f 10 tawani!")


if __name__ == "__main__":
    main()