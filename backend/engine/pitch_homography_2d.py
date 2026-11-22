import cv2
import numpy as np

# ==============================================================================
# FIFA STANDARD PITCH DIMENSIONS (105m x 68m)
# ==============================================================================
PITCH_L = 105.0
PITCH_W = 68.0

# 1. VERIFIED NATIVE DATASET3 (1..33) FIFA COORDINATES (Models A, B, C, D)
PITCH_COORDS_33 = {
    # --- Left Half (13 pts) ---
    1:  (0.0, 0.0),         # Corner top-left
    2:  (0.0, 13.84),       # 18m top goal-line
    3:  (16.5, 13.84),      # 18m top inside
    4:  (16.5, 54.16),      # 18m bottom inside
    5:  (0.0, 54.16),       # 18m bottom goal-line
    6:  (0.0, 68.0),        # Corner bottom-left
    7:  (0.0, 43.16),       # 6m bottom goal-line
    8:  (0.0, 24.84),       # 6m top goal-line
    9:  (5.5, 24.84),       # 6m top inside
    10: (5.5, 43.16),       # 6m bottom inside
    27: (16.5, 26.69),      # 18m arc top left
    28: (16.5, 41.31),      # 18m arc bottom left
    31: (20.15, 34.0),      # 18m arc tip left

    # --- Center (7 pts) ---
    11: (52.5, 0.0),        # Halfway top
    12: (52.5, 24.85),      # Center circle top
    13: (43.35, 34.0),      # Center circle left
    14: (52.5, 43.15),      # Center circle bottom
    15: (61.65, 34.0),      # Center circle right
    16: (52.5, 68.0),       # Halfway bottom
    33: (52.5, 34.0),       # Center spot

    # --- Right Half (13 pts) ---
    22: (105.0, 0.0),       # Corner top-right
    21: (105.0, 13.84),     # 18m top goal-line
    20: (88.5, 13.84),      # 18m top inside
    19: (88.5, 54.16),      # 18m bottom inside
    18: (105.0, 54.16),     # 18m bottom goal-line
    17: (105.0, 68.0),      # Corner bottom-right
    24: (105.0, 24.84),     # 6m top goal-line
    23: (99.5, 24.84),      # 6m top inside
    26: (105.0, 43.16),     # 6m bottom goal-line
    25: (99.5, 43.16),      # 6m bottom inside
    30: (88.5, 26.69),      # 18m arc top right
    29: (88.5, 41.31),      # 18m arc bottom right
    32: (84.85, 34.0),      # 18m arc tip right
}

# 2. LEGACY 39-KEYPOINT FIFA COORDINATES (old best_pose_pro_320)
PITCH_COORDS_39 = {
    17: (0.0, 0.0), 16: (0.0, 13.84), 15: (0.0, 24.84), 12: (0.0, 43.16), 11: (0.0, 54.16), 10: (0.0, 68.0),
    36: (5.5, 24.84), 37: (5.5, 43.16), 31: (16.5, 13.84), 32: (16.5, 26.69), 34: (16.5, 41.31), 35: (16.5, 54.16), 33: (20.15, 34.0),
    18: (52.5, 0.0), 19: (52.5, 24.85), 21: (43.35, 34.0), 39: (52.5, 34.0), 20: (61.65, 34.0), 22: (52.5, 43.15), 9: (52.5, 68.0),
    1: (105.0, 0.0), 2: (105.0, 13.84), 3: (105.0, 24.84), 6: (105.0, 43.16), 7: (105.0, 54.16), 8: (105.0, 68.0),
    28: (99.5, 24.84), 29: (99.5, 43.16), 23: (88.5, 13.84), 24: (88.5, 26.69), 26: (88.5, 41.31), 27: (88.5, 54.16), 25: (84.85, 34.0),
}

# Backwards compatibility alias
PITCH_COORDS_M = PITCH_COORDS_33

LEGACY_REMAP_HYPOTHESES = [
    ("raw", {}),
    ("arc_left_fix", {26: 33, 25: 33}),
    ("center_from_right", {1: 18, 2: 19, 23: 21, 27: 22, 8: 9, 26: 33, 25: 33}),
    ("center_from_left", {17: 18, 16: 19, 31: 20, 35: 22, 10: 9, 34: 25, 33: 25}),
]


class PitchHomography2D:
    """
    Dual-Schema FIFA 2D Homography Engine (Zero-Lag):
      - schema='native33': Models A, B, C, D (verified 1..33 dataset3 -> direct RANSAC)
      - schema='legacy39': Old 39-pt models (multi-hypothesis fallback)
    """

    def __init__(
        self,
        schema="native33",
        min_conf=0.45,
        fallback_conf=0.35,
        ransac_px_thresh=24.0,
        max_reproj_px=26.0,
        smooth_alpha=0.65,
        max_hold_frames=12,
        scale=5,
        pad=16,
    ):
        self.schema = schema
        self.coords_map = PITCH_COORDS_33 if schema == "native33" else PITCH_COORDS_39
        self.hypotheses = [("raw", {})] if schema == "native33" else LEGACY_REMAP_HYPOTHESES

        self.min_conf = min_conf
        self.fallback_conf = fallback_conf
        self.ransac_px_thresh = ransac_px_thresh
        self.max_reproj_px = max_reproj_px
        self.smooth_alpha = smooth_alpha
        self.max_hold_frames = max_hold_frames
        self.scale = scale
        self.pad = pad

        self.H_p2i = None
        self.H_i2p = None
        self._smooth_ref_img = None
        self._ref_pitch_pts = np.array(
            [[16.5, 13.84], [88.5, 13.84], [88.5, 54.16], [16.5, 54.16]],
            dtype=np.float32,
        )
        self.hold_count = 0

        gx = np.linspace(0.0, PITCH_L, 28)
        gy = np.linspace(0.0, PITCH_W, 20)
        xx, yy = np.meshgrid(gx, gy)
        self._pitch_grid = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)
        self._pitch_grid_h = np.concatenate(
            [self._pitch_grid, np.ones((len(self._pitch_grid), 1), dtype=np.float32)], axis=1
        )
        self._grid_mm_px = np.stack([
            np.round(self._pitch_grid[:, 0] * self.scale + self.pad),
            np.round(self._pitch_grid[:, 1] * self.scale + self.pad),
        ], axis=-1).astype(np.int32)

    @staticmethod
    def _apply_remap(raw_valid, remap_dict):
        if not remap_dict:
            return list(raw_valid)
        used_ids = set()
        out = []
        for k_id, x, y, conf in raw_valid:
            if k_id not in remap_dict:
                used_ids.add(k_id)
                out.append((k_id, x, y, conf))
        for k_id, x, y, conf in raw_valid:
            if k_id in remap_dict:
                new_id = remap_dict[k_id]
                if new_id not in used_ids:
                    used_ids.add(new_id)
                    out.append((new_id, x, y, conf))
        return out

    def _filter_spatial_outliers(self, valid_list):
        cmap = self.coords_map
        pts = list(valid_list)

        left_pts = [p for p in pts if cmap[p[0]][0] < 35.0]
        right_pts = [p for p in pts if cmap[p[0]][0] > 70.0]
        if len(left_pts) >= 4 and 0 < len(right_pts) <= 2:
            pts = [p for p in pts if cmap[p[0]][0] <= 70.0]
        elif len(right_pts) >= 4 and 0 < len(left_pts) <= 2:
            pts = [p for p in pts if cmap[p[0]][0] >= 35.0]

        while len(pts) >= 5:
            n = len(pts)
            violations = np.zeros(n, dtype=np.int32)
            for i in range(n):
                ki, xi, yi, _ = pts[i]
                pxi, pyi = cmap[ki]
                for j in range(i + 1, n):
                    kj, xj, yj, _ = pts[j]
                    pxj, pyj = cmap[kj]
                    if abs(pyj - pyi) <= 12.0:
                        if (pxj - pxi) > 8.0 and (xj < xi - 15.0):
                            violations[i] += 1
                            violations[j] += 1
                        elif (pxi - pxj) > 8.0 and (xi < xj - 15.0):
                            violations[i] += 1
                            violations[j] += 1
                    if abs(pxj - pxi) <= 12.0:
                        if (pyj - pyi) > 8.0 and (yj < yi - 15.0):
                            violations[i] += 1
                            violations[j] += 1
                        elif (pyi - pyj) > 8.0 and (yi < yj - 15.0):
                            violations[i] += 1
                            violations[j] += 1
            worst_idx = int(np.argmax(violations))
            if violations[worst_idx] == 0:
                break
            pts.pop(worst_idx)
        return pts

    @staticmethod
    def _is_non_collinear(pitch_pts):
        return (
            (float(np.max(pitch_pts[:, 0]) - np.min(pitch_pts[:, 0])) >= 4.0)
            and (float(np.max(pitch_pts[:, 1]) - np.min(pitch_pts[:, 1])) >= 4.0)
        )

    def _is_valid_homography(self, H_p2i, pitch_pts_inliers):
        if H_p2i is None or not np.all(np.isfinite(H_p2i)):
            return False
        if abs(float(np.linalg.det(H_p2i))) < 1e-8:
            return False
        cx = float(np.clip(np.mean(pitch_pts_inliers[:, 0]), 5.0, PITCH_L - 5.0))
        cy = float(np.clip(np.mean(pitch_pts_inliers[:, 1]), 5.0, PITCH_W - 5.0))
        patch = np.array(
            [[[cx - 2.0, cy - 2.0]], [[cx + 2.0, cy - 2.0]],
             [[cx + 2.0, cy + 2.0]], [[cx - 2.0, cy + 2.0]]],
            dtype=np.float32,
        )
        proj = cv2.perspectiveTransform(patch, H_p2i).reshape(-1, 2)
        if not np.all(np.isfinite(proj)):
            return False
        x, y = proj[:, 0], proj[:, 1]
        area = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        if area <= 10.0:
            return False
        len_x = float(np.linalg.norm(proj[1] - proj[0]))
        len_y = float(np.linalg.norm(proj[3] - proj[0]))
        if len_x < 5.0 or len_y < 2.0 or not (0.10 <= (len_y / len_x) <= 2.20):
            return False
        return (proj[1, 0] - proj[0, 0]) > 0.0 and (proj[3, 1] - proj[0, 1]) > 0.0

    def _solve_single_hypothesis(self, pts_list, ransac_thresh, max_err_thresh, total_raw):
        valid = self._filter_spatial_outliers(pts_list)
        if len(valid) < 4:
            return None
        pitch_pts = np.array([self.coords_map[k_id] for (k_id, _, _, _) in valid], dtype=np.float32)
        img_pts = np.array([[x, y] for (_, x, y, _) in valid], dtype=np.float32)
        if not self._is_non_collinear(pitch_pts):
            return None

        H_cand, mask = cv2.findHomography(pitch_pts, img_pts, cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
        if H_cand is None or mask is None:
            return None

        inlier_mask = mask.ravel().astype(bool)
        inliers = int(inlier_mask.sum())
        min_needed = 4 if total_raw <= 5 else max(5, int(np.ceil(len(valid) * 0.55)))
        if inliers < min_needed:
            return None
        if not self._is_valid_homography(H_cand, pitch_pts[inlier_mask]):
            return None

        if inliers > 4:
            H_ref, _ = cv2.findHomography(pitch_pts[inlier_mask], img_pts[inlier_mask], 0)
            if self._is_valid_homography(H_ref, pitch_pts[inlier_mask]):
                H_cand = H_ref

        proj_img = cv2.perspectiveTransform(pitch_pts[inlier_mask].reshape(-1, 1, 2), H_cand).reshape(-1, 2)
        reproj_px = float(np.mean(np.linalg.norm(proj_img - img_pts[inlier_mask], axis=1)))
        if reproj_px > max_err_thresh:
            return None

        return {
            "H_p2i": H_cand,
            "inliers": inliers,
            "reproj_px": reproj_px,
            "inlier_pitch": pitch_pts[inlier_mask],
            "score": inliers * 100.0 - reproj_px * 1.5,
        }

    def _smooth_homography(self, H_p2i_new, pitch_pts_inliers):
        """Zero-lag adaptive smoothing: only damps micro-jitter when camera is still."""
        ref_img_new = cv2.perspectiveTransform(self._ref_pitch_pts.reshape(-1, 1, 2), H_p2i_new).reshape(-1, 2)
        if self._smooth_ref_img is None:
            self._smooth_ref_img = ref_img_new
            return H_p2i_new
        jump = float(np.mean(np.linalg.norm(ref_img_new - self._smooth_ref_img, axis=1)))
        # If camera is panning (> 8px shift), lock immediately with ZERO lag
        if jump >= 8.0:
            self._smooth_ref_img = ref_img_new
            return H_p2i_new
        # Otherwise gently damp micro-jitter (alpha scales from 0.65 -> 1.0)
        alpha = min(1.0, 0.65 + (jump / 8.0) * 0.35)
        self._smooth_ref_img = alpha * ref_img_new + (1.0 - alpha) * self._smooth_ref_img
        H_smooth = cv2.getPerspectiveTransform(self._ref_pitch_pts.astype(np.float32), self._smooth_ref_img.astype(np.float32))
        return H_smooth if self._is_valid_homography(H_smooth, pitch_pts_inliers) else H_p2i_new

    def reset(self):
        self.H_p2i = None
        self.H_i2p = None
        self._smooth_ref_img = None
        self.hold_count = 0

    def update(self, kpts_px, frame_w, frame_h):
        px_scale = max(frame_w, frame_h) / 1920.0
        ransac_thresh = self.ransac_px_thresh * px_scale
        max_err_thresh = self.max_reproj_px * px_scale

        raw_valid = [(k_id, v[0], v[1], v[2]) for k_id, v in kpts_px.items() if k_id in self.coords_map and v[2] >= self.min_conf]
        if len(raw_valid) < 4:
            raw_valid = [(k_id, v[0], v[1], v[2]) for k_id, v in kpts_px.items() if k_id in self.coords_map and v[2] >= self.fallback_conf]

        best, best_name = None, "raw"
        if len(raw_valid) >= 4:
            raw_ids = {p[0] for p in raw_valid}
            n_raw = len(raw_valid)
            for hyp_name, remap_dict in self.hypotheses:
                if remap_dict and not (raw_ids & remap_dict.keys()):
                    continue
                cand_pts = self._apply_remap(raw_valid, remap_dict)
                sol = self._solve_single_hypothesis(cand_pts, ransac_thresh, max_err_thresh, total_raw=n_raw)
                if sol is None:
                    continue
                adj_score = sol["score"] + (15.0 if hyp_name == "raw" else 0.0)
                if best is None or adj_score > best["adj_score"]:
                    sol["adj_score"] = adj_score
                    best = sol
                    best_name = hyp_name

        if best is not None:
            H_p2i_smooth = self._smooth_homography(best["H_p2i"], best["inlier_pitch"])
            try:
                H_i2p_smooth = np.linalg.inv(H_p2i_smooth)
                H_i2p_smooth = H_i2p_smooth / H_i2p_smooth[2, 2]
                self.H_p2i, self.H_i2p, self.hold_count = H_p2i_smooth, H_i2p_smooth, 0
                return self.H_i2p, self.H_p2i, {
                    "status": "OK", "inliers": best["inliers"], "total": len(raw_valid),
                    "reproj_px": best["reproj_px"], "hypothesis": best_name,
                }
            except np.linalg.LinAlgError:
                pass

        if self.H_i2p is not None and self.hold_count < self.max_hold_frames:
            self.hold_count += 1
            return self.H_i2p, self.H_p2i, {"status": "HOLD", "inliers": 0, "total": len(raw_valid), "reproj_px": 0.0, "hypothesis": "hold"}

        self.reset()
        return None, None, {"status": "N/A", "inliers": 0, "total": len(raw_valid), "reproj_px": 0.0, "hypothesis": "none"}

    def image_to_pitch(self, pts_xy):
        if self.H_i2p is None or len(pts_xy) == 0:
            return np.empty((0, 2), dtype=np.float32)
        return cv2.perspectiveTransform(np.asarray(pts_xy, dtype=np.float32).reshape(-1, 1, 2), self.H_i2p).reshape(-1, 2)

    def pitch_to_image(self, pts_m):
        if self.H_p2i is None or len(pts_m) == 0:
            return np.empty((0, 2), dtype=np.float32)
        return cv2.perspectiveTransform(np.asarray(pts_m, dtype=np.float32).reshape(-1, 1, 2), self.H_p2i).reshape(-1, 2)