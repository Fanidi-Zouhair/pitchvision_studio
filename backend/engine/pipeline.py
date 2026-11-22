import os
import json
import time
import math
import cv2
import numpy as np
from engine.pitch_detector import PitchPoseDetectorOV
from engine.player_tracker import PlayerBallTracker
from engine.pitch_homography_2d import PitchHomography2D, PITCH_L, PITCH_W


FIFA_WIREFRAME_SEGMENTS = [
    ((0, 0), (105, 0)), ((105, 0), (105, 68)), ((105, 68), (0, 68)), ((0, 68), (0, 0)),
    ((52.5, 0), (52.5, 68)),
    ((0, 13.84), (16.5, 13.84)), ((16.5, 13.84), (16.5, 54.16)), ((16.5, 54.16), (0, 54.16)),
    ((0, 24.84), (5.5, 24.84)), ((5.5, 24.84), (5.5, 43.16)), ((5.5, 43.16), (0, 43.16)),
    ((105, 13.84), (88.5, 13.84)), ((88.5, 13.84), (88.5, 54.16)), ((88.5, 54.16), (105, 54.16)),
    ((105, 24.84), (99.5, 24.84)), ((99.5, 24.84), (99.5, 43.16)), ((99.5, 43.16), (105, 43.16)),
]


class MatchAnalysisPipeline:
    def __init__(self, pitch_xml: str, device: str = "GPU"):
        self.pitch_detector = PitchPoseDetectorOV(pitch_xml, device=device)
        # Auto-detect 33 vs 39 keypoints directly from OpenVINO output shape
        num_kpts = int(self.pitch_detector.compiled.output(0).shape[1])
        self.schema = "native33" if num_kpts == 33 else "legacy39"
        self.homography = PitchHomography2D(schema=self.schema, scale=5, pad=16)
        self.tracker = PlayerBallTracker()
        self._smooth_kpts = {}

    def _adaptive_smooth_kpts(self, raw_kpts: dict, frame_w: int) -> dict:
        """
        Zero-Lag Micro-Jitter Filter:
          - Stationary camera (< 2.5px move): light smoothing to kill sub-pixel jitter
          - Panning camera (>= 6px move): alpha = 1.0 (INSTANT lock, zero trailing lag)
        """
        smoothed = {}
        still_px = max(2.0, frame_w * 0.0015)   # ~2.8px on 1080p
        pan_px = max(5.5, frame_w * 0.0038)     # ~7.3px on 1080p
        active_ids = set(raw_kpts.keys())
        for old_id in list(self._smooth_kpts.keys()):
            if old_id not in active_ids:
                del self._smooth_kpts[old_id]

        for kid, (x, y, conf) in raw_kpts.items():
            if kid in self._smooth_kpts:
                px, py = self._smooth_kpts[kid]
                dist = math.hypot(x - px, y - py)
                if dist >= pan_px:
                    sx, sy = x, y
                else:
                    t = max(0.0, min(1.0, (dist - still_px) / max(pan_px - still_px, 1e-5)))
                    alpha = 0.60 + t * 0.40  # 0.60 when still -> 1.00 when panning
                    sx = alpha * x + (1.0 - alpha) * px
                    sy = alpha * y + (1.0 - alpha) * py
            else:
                sx, sy = x, y
            self._smooth_kpts[kid] = (sx, sy)
            smoothed[kid] = (sx, sy, conf)
        return smoothed

    def _dedupe_and_classify_kpts(self, kpts: dict, H_p2i, w: int, h: int) -> list:
        if not kpts:
            return []
        cmap = self.homography.coords_map
        px_scale = max(w, h) / 1920.0
        inlier_thresh = 30.0 * px_scale
        nms_dist = max(16.0, w * 0.014)

        scored = []
        for kid, (x, y, conf) in kpts.items():
            err_px, is_ok = 999.0, 1
            if H_p2i is not None and kid in cmap:
                xm, ym = cmap[kid]
                proj = cv2.perspectiveTransform(np.array([[[xm, ym]]], dtype=np.float32), H_p2i)[0, 0]
                if np.all(np.isfinite(proj)):
                    err_px = float(math.hypot(proj[0] - x, proj[1] - y))
                    is_ok = 1 if err_px <= inlier_thresh else 0
            scored.append(((is_ok, -min(err_px, 200.0), conf), kid, x, y, conf, is_ok))

        scored.sort(key=lambda item: item[0], reverse=True)
        kept = []
        for _, kid, x, y, conf, is_ok in scored:
            if not any(math.hypot(x - kx, y - ky) < nms_dist for _, kx, ky, _, _ in kept):
                kept.append((kid, x, y, conf, is_ok))

        return [[int(kid), round(x / w, 4), round(y / h, 4), round(conf, 2), int(is_ok)] for kid, x, y, conf, is_ok in kept]

    def _compute_fov_meters(self, H_p2i, w: int, h: int) -> list:
        if H_p2i is None:
            return []
        proj = self.homography._pitch_grid_h @ H_p2i.T
        valid_z = proj[:, 2] > 1e-4
        if not np.any(valid_z):
            return []
        inv_z = 1.0 / proj[valid_z, 2]
        u = proj[valid_z, 0] * inv_z
        v = proj[valid_z, 1] * inv_z
        in_frame = (u >= 0) & (u <= w) & (v >= 0) & (v <= h)
        pts_m = self.homography._pitch_grid[valid_z][in_frame]
        if len(pts_m) < 3:
            return []
        hull = cv2.convexHull(pts_m.astype(np.float32)).reshape(-1, 2)
        return [[round(float(x), 2), round(float(y), 2)] for x, y in hull]

    @staticmethod
    def _project_pitch_wireframe(H_p2i, w: int, h: int) -> list:
        if H_p2i is None:
            return []
        lines_out = []
        for (x1, y1), (x2, y2) in FIFA_WIREFRAME_SEGMENTS:
            xs = np.linspace(x1, x2, 6, dtype=np.float32)
            ys = np.linspace(y1, y2, 6, dtype=np.float32)
            pts = np.stack([xs, ys, np.ones_like(xs)], axis=1)
            proj = pts @ H_p2i.T
            valid = proj[:, 2] > 1e-3
            if valid.sum() < 2:
                continue
            u = (proj[valid, 0] / proj[valid, 2]) / w
            v = (proj[valid, 1] / proj[valid, 2]) / h
            if np.any((u >= -0.2) & (u <= 1.2) & (v >= -0.2) & (v <= 1.2)):
                lines_out.append([[round(float(ux), 4), round(float(vy), 4)] for ux, vy in zip(u, v)])
        return lines_out

    def analyze_video(self, video_path: str, out_json_path: str, progress_cb=None):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.pitch_detector.reset()
        self.homography.reset()
        self._smooth_kpts = {}

        timeline = []
        t0 = time.time()
        idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            idx += 1

            # Bypass pitch_detector's internal EMA so we never double-smooth!
            self.pitch_detector.prev_kpts = None
            raw_kpts = self.pitch_detector.detect(frame)
            kpts = self._adaptive_smooth_kpts(raw_kpts, w)
            H_i2p, H_p2i, h_info = self.homography.update(kpts, w, h)
            players, ball_xy = self.tracker.track(frame)

            kpts_img = self._dedupe_and_classify_kpts(kpts, H_p2i, w, h)

            kpts_2d, fov_m, wireframe = [], [], []
            if H_i2p is not None and H_p2i is not None:
                fov_m = self._compute_fov_meters(H_p2i, w, h)
                wireframe = self._project_pitch_wireframe(H_p2i, w, h)
                ok_kpts = [item for item in kpts_img if item[4] == 1]
                if ok_kpts:
                    pts_px = [[item[1] * w, item[2] * h] for item in ok_kpts]
                    proj_k = self.homography.image_to_pitch(pts_px)
                    for item, (xm, ym) in zip(ok_kpts, proj_k):
                        if -2.0 <= xm <= PITCH_L + 2.0 and -2.0 <= ym <= PITCH_W + 2.0:
                            kpts_2d.append([round(float(xm), 2), round(float(ym), 2), item[3]])

            players_2d = []
            if H_i2p is not None and players:
                proj = self.homography.image_to_pitch([p["foot_xy"] for p in players])
                for p, (xm, ym) in zip(players, proj):
                    players_2d.append({"id": p["id"], "team": p["team"], "x": round(float(xm), 2), "y": round(float(ym), 2)})

            ball_2d = None
            if H_i2p is not None and ball_xy is not None:
                bm = self.homography.image_to_pitch([ball_xy])[0]
                ball_2d = [round(float(bm[0]), 2), round(float(bm[1]), 2)]

            timeline.append({
                "frame": idx,
                "t": round(idx / fps, 3),
                "radar_status": h_info["status"],
                "inliers": h_info["inliers"],
                "reproj_px": round(float(h_info["reproj_px"]), 2),
                "kpts_img": kpts_img,
                "kpts_2d": kpts_2d,
                "fov": fov_m,
                "wireframe": wireframe,
                "players": players_2d,
                "ball": ball_2d,
            })

            if progress_cb and (idx % 10 == 0 or idx == total):
                progress_cb(idx, total)

        cap.release()
        elapsed = round(time.time() - t0, 2)
        result = {
            "video": os.path.basename(video_path),
            "schema": self.schema,
            "width": w,
            "height": h,
            "fps": fps,
            "total_frames": idx,
            "elapsed_sec": elapsed,
            "avg_fps": round(idx / max(elapsed, 1e-3), 1),
            "timeline": timeline,
        }
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f)
        return result