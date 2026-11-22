import os
import json
import math
import shutil
from urllib.parse import quote
from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import MODELS_DIR, OV_DEVICE, UPLOADS_DIR, OUTPUTS_DIR
from engine.pipeline import MatchAnalysisPipeline
from analytics.metrics import compute_match_summary

app = FastAPI(title="PitchVision Studio API", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")

# Default match video loaded automatically in the studio
DEFAULT_VIDEO_PATH = r"C:\Users\hp\Videos\Video Project 2.mp4"

# Registry of all supported Pitch Pose OpenVINO FP16 models
MODEL_REGISTRY = {
    "model_E":   {"label": "Model E — V2 Master RepViT-M0.9 + FPN + SafeGAU (512×288)", "file": "model_E_fp16.xml"},
    "model_D":   {"label": "Model D — SimCC 1D (3.7M • Ultra-Fast)",                    "file": "model_D_fp16.xml"},
    "model_A":   {"label": "Model A — ResNet34 Pure Vision (23.7M)",                     "file": "model_A_fp16.xml"},
    "model_B":   {"label": "Model B — ResNet34 Mask + Soft Math (23.7M)",                "file": "model_B_fp16.xml"},
    "model_C":   {"label": "Model C — ResNet34 Heatmap + Heavy Math (23.7M)",            "file": "model_C_fp16.xml"},
    "legacy_39": {"label": "Legacy — Pro 320 (Old 39-pt Model)",                         "file": "best_pose_pro_320_fp16.xml"},
}

# Exact mapping from Native 33 IDs -> Legacy 39 IDs so Legacy 39 can be judged on the same User GT!
NATIVE33_TO_LEGACY39 = {
    1: 17, 2: 16, 3: 31, 4: 35, 5: 11, 6: 10, 7: 12, 8: 15, 9: 36, 10: 37,
    27: 32, 28: 34, 31: 33,
    11: 18, 12: 19, 13: 21, 14: 22, 15: 20, 16: 9, 33: 39,
    22: 1, 21: 2, 20: 23, 19: 27, 18: 7, 17: 8, 24: 3, 23: 28, 26: 6, 25: 29,
    30: 24, 29: 26, 32: 25,
}

_pipelines_cache = {}


def ensure_default_video():
    """Make sure DEFAULT_VIDEO_PATH is available inside UPLOADS_DIR for browser streaming."""
    if os.path.exists(DEFAULT_VIDEO_PATH):
        fname = os.path.basename(DEFAULT_VIDEO_PATH)
        dst = os.path.join(UPLOADS_DIR, fname)
        if not os.path.exists(dst):
            try:
                shutil.copy2(DEFAULT_VIDEO_PATH, dst)
            except Exception:
                return None
        return fname
    # Fallback: pick the most recent MP4 already in uploads/
    mp4s = [f for f in os.listdir(UPLOADS_DIR) if f.lower().endswith(".mp4")]
    if mp4s:
        mp4s.sort(key=lambda f: os.path.getmtime(os.path.join(UPLOADS_DIR, f)), reverse=True)
        return mp4s[0]
    return None


def get_available_models():
    avail = []
    for mid, meta in MODEL_REGISTRY.items():
        xml_path = os.path.join(MODELS_DIR, meta["file"])
        if os.path.exists(xml_path):
            avail.append({"id": mid, "label": meta["label"], "file": meta["file"]})
    return avail


def get_pipeline(model_id: str):
    if model_id not in MODEL_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Model ID unknown: {model_id}")
    xml_path = os.path.join(MODELS_DIR, MODEL_REGISTRY[model_id]["file"])
    if not os.path.exists(xml_path):
        raise HTTPException(status_code=404, detail=f"Model file not found: {xml_path}")
    if model_id not in _pipelines_cache:
        _pipelines_cache[model_id] = MatchAnalysisPipeline(xml_path, device=OV_DEVICE)
    return _pipelines_cache[model_id]


def gt_file_path(filename: str) -> str:
    stem = os.path.splitext(os.path.basename(filename))[0]
    return os.path.join(OUTPUTS_DIR, f"{stem}_user_gt.json")


def load_user_gt(filename: str) -> dict:
    path = gt_file_path(filename)
    if not os.path.exists(path):
        return {"video": filename, "frames": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "frames" not in data:
            data["frames"] = {}
        return data
    except Exception:
        return {"video": filename, "frames": {}}


def compute_temporal_jitter_px(timeline: list, ref_size: float = 320.0) -> float:
    """
    Measures frame-to-frame acceleration jitter (2nd derivative) across consecutive frames.
    Smooth camera pans have ~0 acceleration; unstable models shake back & forth (high jitter).
    """
    if len(timeline) < 3:
        return 0.0
    frame_maps = []
    for f in timeline:
        m = {}
        for item in f.get("kpts_img", []):
            kid, nx, ny = int(item[0]), float(item[1]), float(item[2])
            m[kid] = (nx * ref_size, ny * ref_size)
        frame_maps.append(m)

    accels = []
    for i in range(1, len(frame_maps) - 1):
        prev_m, curr_m, next_m = frame_maps[i - 1], frame_maps[i], frame_maps[i + 1]
        for kid, (cx, cy) in curr_m.items():
            if kid in prev_m and kid in next_m:
                px, py = prev_m[kid]
                nx, ny = next_m[kid]
                # 2nd difference: (next - curr) - (curr - prev)
                ax = (nx - cx) - (cx - px)
                ay = (ny - cy) - (cy - py)
                accels.append(math.hypot(ax, ay))
    return round(float(sum(accels) / len(accels)), 2) if accels else 0.0


def evaluate_timeline_against_gt(result: dict, user_gt: dict, ref_size: float = 320.0) -> dict:
    """
    Scores a model's timeline against the user's own annotated Ground Truth frames.
    Computes Mean Error (px @ 320), PCK@5px, PCK@10px, and Detection Recall on GT points.
    """
    gt_frames = user_gt.get("frames", {})
    if not gt_frames:
        return {
            "gt_frames_count": 0,
            "gt_pts_total": 0,
            "gt_pts_matched": 0,
            "gt_recall_pct": None,
            "gt_mean_px": None,
            "gt_pck5_pct": None,
            "gt_pck10_pct": None,
        }

    schema = result.get("schema", "native33")
    tl_by_frame = {int(f["frame"]): f for f in result.get("timeline", [])}

    errs = []
    total_gt_pts = 0
    matched_pts = 0

    for f_str, pts_dict in gt_frames.items():
        if not pts_dict:
            continue
        f_idx = int(f_str)
        frame_data = tl_by_frame.get(f_idx)
        pred_map = {}
        if frame_data:
            for item in frame_data.get("kpts_img", []):
                pred_map[int(item[0])] = (float(item[1]), float(item[2]))

        for kid_str, coords in pts_dict.items():
            if not isinstance(coords, list) or len(coords) < 2:
                continue
            total_gt_pts += 1
            gt_x, gt_y = float(coords[0]), float(coords[1])
            kid_33 = int(kid_str)
            lookup_id = NATIVE33_TO_LEGACY39.get(kid_33, kid_33) if schema == "legacy39" else kid_33

            if lookup_id in pred_map:
                px, py = pred_map[lookup_id]
                err = math.hypot((px - gt_x) * ref_size, (py - gt_y) * ref_size)
                errs.append(err)
                matched_pts += 1

    if total_gt_pts == 0:
        return {
            "gt_frames_count": 0,
            "gt_pts_total": 0,
            "gt_pts_matched": 0,
            "gt_recall_pct": None,
            "gt_mean_px": None,
            "gt_pck5_pct": None,
            "gt_pck10_pct": None,
        }

    recall = round((matched_pts / total_gt_pts) * 100.0, 1)
    if not errs:
        return {
            "gt_frames_count": len([k for k, v in gt_frames.items() if v]),
            "gt_pts_total": total_gt_pts,
            "gt_pts_matched": 0,
            "gt_recall_pct": 0.0,
            "gt_mean_px": 99.9,
            "gt_pck5_pct": 0.0,
            "gt_pck10_pct": 0.0,
        }

    return {
        "gt_frames_count": len([k for k, v in gt_frames.items() if v]),
        "gt_pts_total": total_gt_pts,
        "gt_pts_matched": matched_pts,
        "gt_recall_pct": recall,
        "gt_mean_px": round(float(sum(errs) / len(errs)), 2),
        "gt_pck5_pct": round(float(sum(1 for e in errs if e <= 5.0) / len(errs)) * 100.0, 1),
        "gt_pck10_pct": round(float(sum(1 for e in errs if e <= 10.0) / len(errs)) * 100.0, 1),
    }


def run_analysis_for_video(filename: str, model_id: str, use_cache: bool = False):
    """Run (or load cached) OpenVINO analysis for a video already inside UPLOADS_DIR."""
    avail = get_available_models()
    if not avail:
        raise HTTPException(status_code=500, detail="No OpenVINO models found in backend/models/!")
    avail_ids = [m["id"] for m in avail]
    chosen_id = model_id if model_id in avail_ids else avail_ids[0]

    video_path = os.path.join(UPLOADS_DIR, filename)
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail=f"Video not found in uploads/: {filename}")

    stem = os.path.splitext(filename)[0]
    out_name = f"{stem}_{chosen_id}_analysis.json"
    out_json = os.path.join(OUTPUTS_DIR, out_name)

    if use_cache and os.path.exists(out_json):
        with open(out_json, "r", encoding="utf-8") as f:
            result = json.load(f)
    else:
        result = get_pipeline(chosen_id).analyze_video(video_path, out_json)

    summary = compute_match_summary(result["timeline"], fps=result["fps"])
    jitter_px = compute_temporal_jitter_px(result["timeline"])
    encoded_video = quote(filename)
    encoded_json = quote(out_name)

    return {
        "model_id": chosen_id,
        "model_label": MODEL_REGISTRY[chosen_id]["label"],
        "schema": result.get("schema", "native33"),
        "filename": filename,
        "video_url": f"http://127.0.0.1:8000/uploads/{encoded_video}",
        "analysis_url": f"http://127.0.0.1:8000/outputs/{encoded_json}",
        "elapsed_sec": result["elapsed_sec"],
        "avg_fps": result["avg_fps"],
        "ms_per_frame": round(1000.0 / max(result["avg_fps"], 0.1), 1),
        "jitter_px": jitter_px,
        "summary": summary,
        "result_obj": result,
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "device": OV_DEVICE, "models": get_available_models()}


@app.get("/api/models")
def list_models():
    default_file = ensure_default_video()
    return {
        "models": get_available_models(),
        "default_video": {
            "filename": default_file,
            "video_url": f"http://127.0.0.1:8000/uploads/{quote(default_file)}" if default_file else None,
        },
    }


@app.post("/api/analyze")
async def analyze_uploaded_match(
    file: UploadFile = File(...),
    model_id: str = Query(default="model_D"),
):
    video_path = os.path.join(UPLOADS_DIR, file.filename)
    with open(video_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    res = run_analysis_for_video(file.filename, model_id, use_cache=False)
    res.pop("result_obj", None)
    return res


@app.post("/api/analyze-existing")
def analyze_existing_match(
    filename: str = Query(...),
    model_id: str = Query(default="model_D"),
):
    """Analyze an already-loaded or default video without re-uploading."""
    ensure_default_video()
    res = run_analysis_for_video(filename, model_id, use_cache=False)
    res.pop("result_obj", None)
    return res


@app.get("/api/gt")
def get_video_gt(filename: str = Query(...)):
    return load_user_gt(filename)


@app.post("/api/gt")
def save_video_gt(filename: str = Query(...), payload: dict = Body(...)):
    """Save the user's Ground Truth keypoints permanently for this video."""
    path = gt_file_path(filename)
    data = {
        "video": filename,
        "frames": payload.get("frames", {}),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return {"status": "saved", "path": path, "frames_count": len([k for k, v in data["frames"].items() if v])}


@app.post("/api/benchmark-all")
def benchmark_all_models(
    filename: str = Query(...),
    force: bool = Query(default=False),
):
    """
    Runs ALL available OpenVINO models on 'filename' and scores them on:
      1. Speed (FPS & ms/frame)
      2. Accuracy against the User's own saved Ground Truth (Mean Px, PCK@5, PCK@10, Recall)
      3. Temporal Stability (Jitter px) & Radar Lock %
    """
    ensure_default_video()
    avail = get_available_models()
    if not avail:
        raise HTTPException(status_code=500, detail="No models available.")

    user_gt = load_user_gt(filename)
    rows = []

    for m in avail:
        res = run_analysis_for_video(filename, m["id"], use_cache=(not force))
        gt_metrics = evaluate_timeline_against_gt(res["result_obj"], user_gt)
        rows.append({
            "model_id": res["model_id"],
            "model_label": res["model_label"],
            "schema": res["schema"],
            "analysis_url": res["analysis_url"],
            "elapsed_sec": res["elapsed_sec"],
            "avg_fps": res["avg_fps"],
            "ms_per_frame": res["ms_per_frame"],
            "jitter_px": res["jitter_px"],
            "radar_ok_pct": res["summary"]["radar_ok_pct"],
            **gt_metrics,
        })

    # Rank models: if user has annotated GT -> rank by lowest GT Mean Px, else by Radar OK & FPS
    has_gt = any(r["gt_mean_px"] is not None for r in rows)
    if has_gt:
        rows.sort(key=lambda r: (r["gt_mean_px"] if r["gt_mean_px"] is not None else 999.0, -r["avg_fps"]))
    else:
        rows.sort(key=lambda r: (-r["radar_ok_pct"], r["jitter_px"], -r["avg_fps"]))

    return {
        "filename": filename,
        "has_user_gt": has_gt,
        "leaderboard": rows,
    }