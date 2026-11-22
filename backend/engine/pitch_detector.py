import cv2
import numpy as np
import openvino as ov
from config import CONF_THRESHOLD, EMA_ALPHA


class PitchPoseDetectorOV:
    """
    Universal OpenVINO FP16 Pitch Detector supporting:
      - V1 Models (A, B, C, D, Legacy): square 320x320 / 256x256, 2 outputs (coords, vis_logits)
      - V2 Model E (Master): 512x288 16:9, 3 outputs (raw logits_x, logits_y, vis_logits)
        decoded on host CPU in FP32 via Top-K Local Expectation (ZERO FP16 Softmax drift!).
    """

    def __init__(self, xml_path: str, device: str = "GPU", simcc_tau: float = 0.1, k_radius: int = 2):
        self.core = ov.Core()
        self.compiled = self.core.compile_model(xml_path, device, {"PERFORMANCE_HINT": "LATENCY"})
        self.num_outputs = len(self.compiled.outputs)

        # Auto-detect rectangular (H, W) input shape: [1, 3, H, W]
        in_shape = self.compiled.input(0).shape
        self.img_h = int(in_shape[2])
        self.img_w = int(in_shape[3])
        self.img_size = self.img_w  # backwards compatibility

        self.is_v2_raw_simcc = (self.num_outputs == 3)
        self.simcc_tau = float(simcc_tau)
        self.k_radius = int(k_radius)

        if self.is_v2_raw_simcc:
            self.out_lx = self.compiled.output(0)
            self.out_ly = self.compiled.output(1)
            self.out_vis = self.compiled.output(2)
        else:
            self.out_coords = self.compiled.output(0)
            self.out_vis = self.compiled.output(1)

        self.req = self.compiled.create_infer_request()
        self.prev_kpts = None

    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame_bgr, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = (rgb.astype(np.float32) * (1.0 / 255.0)).transpose(2, 0, 1)[None, ...]
        return np.ascontiguousarray(tensor)

    def _decode_raw_simcc_fp32(self, logits_x: np.ndarray, logits_y: np.ndarray):
        """FP32 host-side temperature-scaled Top-K Local Expectation (< 0.05 ms on CPU)."""
        lx = logits_x.astype(np.float32) / self.simcc_tau
        ly = logits_y.astype(np.float32) / self.simcc_tau

        px = np.exp(lx - lx.max(axis=-1, keepdims=True))
        px /= (px.sum(axis=-1, keepdims=True) + 1e-8)

        py = np.exp(ly - ly.max(axis=-1, keepdims=True))
        py /= (py.sum(axis=-1, keepdims=True) + 1e-8)

        def local_exp(p):
            K, n_bins = p.shape
            peak = p.argmax(axis=-1)
            out = np.empty(K, dtype=np.float32)
            r = self.k_radius
            for i in range(K):
                lo = max(0, int(peak[i]) - r)
                hi = min(n_bins, int(peak[i]) + r + 1)
                w = p[i, lo:hi]
                out[i] = float(np.sum(w * np.arange(lo, hi, dtype=np.float32)) / (w.sum() + 1e-8)) / float(n_bins - 1)
            return out

        return np.stack([local_exp(px), local_exp(py)], axis=-1)

    def detect(self, frame_bgr: np.ndarray) -> dict:
        h, w = frame_bgr.shape[:2]
        self.req.infer({0: self.preprocess(frame_bgr)})

        if self.is_v2_raw_simcc:
            lx = self.req.get_tensor(self.out_lx).data[0]
            ly = self.req.get_tensor(self.out_ly).data[0]
            vis_logits = self.req.get_tensor(self.out_vis).data[0]
            coords = self._decode_raw_simcc_fp32(lx, ly)
        else:
            coords = self.req.get_tensor(self.out_coords).data[0]
            vis_logits = self.req.get_tensor(self.out_vis).data[0]

        vis_probs = 1.0 / (1.0 + np.exp(-np.clip(vis_logits.astype(np.float32), -40.0, 40.0)))

        kpts = {}
        for i in range(coords.shape[0]):
            p = float(vis_probs[i])
            if p >= CONF_THRESHOLD:
                px = float(np.clip(coords[i, 0] * w, 0, w - 1))
                py = float(np.clip(coords[i, 1] * h, 0, h - 1))
                kid = i + 1
                if self.prev_kpts and kid in self.prev_kpts and EMA_ALPHA < 1.0:
                    ox, oy, _ = self.prev_kpts[kid]
                    px = EMA_ALPHA * px + (1.0 - EMA_ALPHA) * ox
                    py = EMA_ALPHA * py + (1.0 - EMA_ALPHA) * oy
                kpts[kid] = (px, py, p)

        self.prev_kpts = kpts
        return kpts

    def reset(self):
        self.prev_kpts = None