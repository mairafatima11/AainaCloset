"""View classification for scraped product images.

Every product (SKU) folder holds several photos: flat lays, on-model shots from
different angles, fabric close-ups, and occasionally non-apparel images. For a
try-on training pair we need (a) a flat/garment-only image and (b) a front-facing,
full-length photo of a person wearing the same garment. This module labels each
image so pairs can be picked automatically and then human-verified.

Runs on CPU. Uses only models bundled inside the mediapipe==0.10.14 wheel
(Pose, FaceDetection, SelfieSegmentation), so it needs no internet access and no GPU.

Labels
------
flat        upper garment / dress only, no person, on a plain background
flat_lower  trouser / shalwar laid flat (gap between the legs)
detail  fabric / embroidery close-up, garment fills the frame, no person
mff     model, front-facing, full length (head to ankles visible)  -> try-on target
mo      model, any other view (side, back, seated, cropped, arms raised ...)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as e:  # pragma: no cover
    raise ImportError("pip install mediapipe==0.10.14  (later versions dropped the bundled models)") from e

# landmark indices (BlazePose 33-point topology)
NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 0, 11, 12, 13, 14, 15, 16
L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 23, 24, 25, 26, 27, 28

ANALYSIS_LONG_SIDE = 640  # images are downscaled to this before analysis


@dataclass
class ViewFeatures:
    width: int = 0
    height: int = 0
    # background / layout
    border_std: float = 0.0        # colour std-dev of the outer 4% border (plain studio bg -> low)
    border_lum: float = 0.0        # mean luminance of border
    fg_frac: float = 0.0           # fraction of pixels that differ from border colour
    fg_touches_edges: int = 0      # how many image edges the foreground touches (0-4)
    fg_bbox_aspect: float = 0.0    # height/width of the foreground bounding box (trousers are tall and narrow)
    fg_bbox_fill: float = 0.0      # foreground pixels / bbox area
    leg_gap: float = 0.0           # background fraction in the centre-bottom of the garment (trousers -> high)
    # person
    face_score: float = 0.0
    face_count: int = 0
    pose_found: int = 0
    person_frac: float = 0.0       # selfie-segmentation person area
    vis_head: float = 0.0
    vis_shoulders: float = 0.0
    vis_hips: float = 0.0
    vis_knees: float = 0.0
    vis_ankles: float = 0.0
    ankles_in_frame: int = 0
    head_in_frame: int = 0
    frontal: float = 0.0           # 1 = shoulders square to camera, 0 = profile
    facing_camera: int = 0         # left shoulder appears on image right
    seated: int = 0
    arms_raised: int = 0
    arms_crossing_torso: int = 0
    body_height_frac: float = 0.0  # nose-to-ankle span / image height
    centre_offset: float = 0.0     # |hip centre x - 0.5|
    sharpness: float = 0.0         # variance of Laplacian on the person / garment region


class ViewClassifier:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1,
                                           enable_segmentation=False, min_detection_confidence=0.5)
        self.face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)

    def close(self):
        self.pose.close(); self.face.close(); self.seg.close()

    # ------------------------------------------------------------------ features
    def features(self, bgr: np.ndarray) -> ViewFeatures:
        f = ViewFeatures(width=bgr.shape[1], height=bgr.shape[0])
        s = ANALYSIS_LONG_SIDE / max(bgr.shape[:2])
        img = cv2.resize(bgr, (round(bgr.shape[1] * s), round(bgr.shape[0] * s)), interpolation=cv2.INTER_AREA) if s < 1 else bgr
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # background statistics from a thin border
        b = max(2, int(0.04 * min(h, w)))
        border = np.concatenate([img[:b].reshape(-1, 3), img[-b:].reshape(-1, 3),
                                 img[:, :b].reshape(-1, 3), img[:, -b:].reshape(-1, 3)]).astype(np.float32)
        f.border_std = float(border.std(axis=0).mean())
        f.border_lum = float(border.mean())
        bg = np.median(border, axis=0)
        diff = np.abs(img.astype(np.float32) - bg).max(axis=2)
        fg = (diff > 28).astype(np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        f.fg_frac = float(fg.mean())
        ys, xs = np.nonzero(fg)
        if len(xs):
            f.fg_bbox_aspect = float((ys.max() - ys.min() + 1) / (xs.max() - xs.min() + 1))
            f.fg_bbox_fill = float(len(xs) / ((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)))
            # trousers/shalwar laid flat show background between the legs in the lower part
            y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
            cx, bw = (x0 + x1) // 2, max(2, (x1 - x0) // 16)
            low = fg[y0 + int(0.70 * (y1 - y0)): y1, cx - bw: cx + bw]
            f.leg_gap = float(1.0 - low.mean()) if low.size else 0.0
        m = max(1, int(0.01 * min(h, w)))
        f.fg_touches_edges = int(fg[:m].mean() > 0.15) + int(fg[-m:].mean() > 0.15) + \
            int(fg[:, :m].mean() > 0.15) + int(fg[:, -m:].mean() > 0.15)

        # person segmentation
        segm = self.seg.process(rgb).segmentation_mask
        if segm is not None:
            f.person_frac = float((segm > 0.5).mean())

        # face
        fr = self.face.process(rgb)
        if fr.detections:
            f.face_count = len(fr.detections)
            f.face_score = float(max(d.score[0] for d in fr.detections))

        # pose
        pr = self.pose.process(rgb)
        region = fg
        if pr.pose_landmarks:
            f.pose_found = 1
            lm = pr.pose_landmarks.landmark
            P = np.array([[l.x, l.y] for l in lm]); V = np.array([l.visibility for l in lm])
            f.vis_head = float(V[NOSE])
            f.vis_shoulders = float(min(V[L_SH], V[R_SH]))
            f.vis_hips = float(min(V[L_HIP], V[R_HIP]))
            f.vis_knees = float(min(V[L_KN], V[R_KN]))
            f.vis_ankles = float(min(V[L_AN], V[R_AN]))
            inside = lambda i: 0.0 <= P[i, 0] <= 1.0 and 0.0 <= P[i, 1] <= 1.0
            # long kameez / gharara hems often hide the ankles, so visibility bar is low;
            # the landmark must still be inside the frame (i.e. the photo is not cropped at the knee)
            f.ankles_in_frame = int(inside(L_AN) and inside(R_AN) and P[L_AN, 1] < 0.995 and f.vis_ankles > 0.25)
            f.head_in_frame = int(inside(NOSE) and P[NOSE, 1] > 0.01 and f.vis_head > 0.5)
            sh_w = abs(P[L_SH, 0] - P[R_SH, 0]) * w
            torso_h = abs((P[L_HIP, 1] + P[R_HIP, 1]) / 2 - (P[L_SH, 1] + P[R_SH, 1]) / 2) * h
            f.frontal = float(min(1.0, sh_w / (0.75 * torso_h + 1e-6)))
            f.facing_camera = int(P[L_SH, 0] > P[R_SH, 0])
            hip_y = (P[L_HIP, 1] + P[R_HIP, 1]) / 2
            knee_y = (P[L_KN, 1] + P[R_KN, 1]) / 2
            sh_y = (P[L_SH, 1] + P[R_SH, 1]) / 2
            f.seated = int((knee_y - hip_y) < 0.45 * (hip_y - sh_y) and f.vis_knees > 0.5)
            f.arms_raised = int(min(P[L_WR, 1], P[R_WR, 1]) < sh_y and max(V[L_WR], V[R_WR]) > 0.5)
            # a wrist that crosses the body midline (hand on chest / holding dupatta across)
            mid_x = (P[L_SH, 0] + P[R_SH, 0]) / 2
            crosses = (P[L_WR, 0] < mid_x and V[L_WR] > 0.5 and f.facing_camera) or \
                      (P[R_WR, 0] > mid_x and V[R_WR] > 0.5 and f.facing_camera)
            f.arms_crossing_torso = int(bool(crosses) and sh_y < min(P[L_WR, 1], P[R_WR, 1]) < hip_y + 0.05)
            f.body_height_frac = float(((P[L_AN, 1] + P[R_AN, 1]) / 2 - P[NOSE, 1]))
            f.centre_offset = float(abs((P[L_HIP, 0] + P[R_HIP, 0]) / 2 - 0.5))
            if segm is not None:
                region = (segm > 0.5).astype(np.uint8)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        f.sharpness = float(lap[region > 0].var()) if region.any() else float(lap.var())
        return f

    # ------------------------------------------------------------------ decision
    @staticmethod
    def decide(f: ViewFeatures) -> tuple[str, float]:
        """Return (label, target_score). target_score in [0,1] ranks mff candidates."""
        # selfie-segmentation fires on garments alone, so only pose/face count as "a person"
        person = (f.pose_found and f.vis_shoulders > 0.5) or f.face_score > 0.6
        if not person:
            # no person: flat lay (whole garment on a plain margin) vs close-up / cropped shot
            if f.border_std < 12 and f.fg_frac < 0.65 and f.fg_touches_edges == 0:
                return ("flat_lower" if f.leg_gap > 0.5 else "flat"), 0.0
            return "detail", 0.0
        if not f.head_in_frame and f.face_score < 0.5:
            return "detail", 0.0   # headless crop of the garment on a body

        full = f.head_in_frame and f.ankles_in_frame and f.body_height_frac > 0.55
        front = f.facing_camera and f.frontal > 0.68 and f.face_score > 0.5
        clean = not f.seated and not f.arms_raised
        if full and front and clean:
            score = (0.35 * min(1.0, f.frontal)
                     + 0.20 * f.face_score
                     + 0.15 * (1 - min(1.0, f.centre_offset * 4))
                     + 0.15 * min(1.0, f.body_height_frac / 0.8)
                     + 0.10 * (1 - f.arms_crossing_torso)
                     + 0.05 * (f.border_std < 18))
            return "mff", round(float(score), 4)
        return "mo", 0.0

    def classify_path(self, path: str) -> dict:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            return {"path": path, "label": "unreadable", "target_score": 0.0}
        f = self.features(bgr)
        label, score = self.decide(f)
        return {"path": path, "label": label, "target_score": score, **asdict(f)}
