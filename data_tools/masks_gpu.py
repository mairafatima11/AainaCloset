"""GPU preprocessing, run ONCE after export (Colab T4 or Kaggle). Never recompute during training.

Uses CatVTON's bundled AutoMasker (SCHP human parsing + DensePose) so there is no separate
detectron2 install. For every exported pair it writes:

  <split>/agnostic-mask/<id>_mask.png   region the try-on model must repaint (white)
  <split>/image-densepose/<id>.jpg      DensePose render (only needed for a StableVITON fallback)
  <split>/image-parse/<id>.png          SCHP-LIP label map (kept for QA and the failure taxonomy)
  <split>/cloth/<id>.jpg + cloth-mask/  for 'worn' pairs: garment cut out of the cloth-source photo
  masks_qa.csv                          per-image statistics, incl. how much dupatta covers the torso

Why the mask is extended for kameez
-----------------------------------
CatVTON's 'upper' mask is built for Western tops and stops near the hip. A kameez ends at the
knee or below, so an 'upper' mask would teach the model to cut every kameez short (the
'hem truncation' failure). For upper/set garments we therefore take the union of the 'upper'
mask and the SCHP garment labels (upper-clothes / dress / coat), dilated, down to the hem.
CHECK this on the first QA sheet before running the full batch -- see notebook 03.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import pandas as pd
from PIL import Image

# SCHP label sets (CatVTON ships both parsers)
LIP = dict(bg=0, hat=1, hair=2, glove=3, sunglasses=4, upper=5, dress=6, coat=7, socks=8, pants=9,
           jumpsuit=10, scarf=11, skirt=12, face=13, l_arm=14, r_arm=15, l_leg=16, r_leg=17, l_shoe=18, r_shoe=19)
ATR = dict(bg=0, hat=1, hair=2, sunglasses=3, upper=4, skirt=5, pants=6, dress=7, belt=8, l_shoe=9, r_shoe=10,
           face=11, l_leg=12, r_leg=13, l_arm=14, r_arm=15, bag=16, scarf=17)
LIP_GARMENT_UPPER = [LIP["upper"], LIP["dress"], LIP["coat"], LIP["jumpsuit"]]
LIP_GARMENT_LOWER = [LIP["pants"], LIP["skirt"]]


def _np(x):
    if x is None:
        return None
    if isinstance(x, Image.Image):
        return np.array(x)
    return np.asarray(x)


class Masker:
    def __init__(self, catvton_repo: str, device: str = "cuda"):
        if catvton_repo not in sys.path:
            sys.path.insert(0, catvton_repo)
        from huggingface_hub import snapshot_download
        from model.cloth_masker import AutoMasker  # type: ignore  (CatVTON repo)
        ckpt = snapshot_download("zhengchong/CatVTON")
        self.am = AutoMasker(densepose_ckpt=os.path.join(ckpt, "DensePose"),
                             schp_ckpt=os.path.join(ckpt, "SCHP"), device=device)

    def run(self, img: Image.Image, mask_type: str) -> dict:
        out = self.am(img, mask_type)
        return {k: _np(v) for k, v in out.items()} if isinstance(out, dict) else {"mask": _np(out)}


def extend_to_hem(mask: np.ndarray, lip: np.ndarray | None, dilate_px: int = 21) -> np.ndarray:
    """Union of CatVTON's 'upper' mask with the parsed garment, so long kameez are fully covered."""
    m = (mask > 127).astype(np.uint8)
    if lip is None:
        return m * 255
    if lip.shape != m.shape:
        lip = cv2.resize(lip.astype(np.uint8), (m.shape[1], m.shape[0]), interpolation=cv2.INTER_NEAREST)
    g = np.isin(lip, LIP_GARMENT_UPPER).astype(np.uint8)
    g = cv2.dilate(g, np.ones((dilate_px, dilate_px), np.uint8))
    out = np.maximum(m, g)
    # fill horizontal gaps between the two halves (e.g. where an arm crosses the kameez)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    return out * 255


def garment_from_person(img_bgr: np.ndarray, lip: np.ndarray, category: str) -> np.ndarray:
    labels = LIP_GARMENT_LOWER if category == "lower" else LIP_GARMENT_UPPER
    if lip.shape[:2] != img_bgr.shape[:2]:
        lip = cv2.resize(lip.astype(np.uint8), (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    m = np.isin(lip, labels).astype(np.uint8)
    n, cc, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:  # keep the largest blob(s)
        big = stats[1:, cv2.CC_STAT_AREA].max()
        m = np.isin(cc, [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 0.1 * big]).astype(np.uint8)
    return m


def process(export_dir: str, root: str, catvton_repo: str, limit: int = 0, overwrite: bool = False):
    from .export import make_cloth, fit_pad
    pairs = pd.read_csv(os.path.join(export_dir, "pairs.csv"))
    pairs = pairs[~pairs.status.astype(str).str.startswith("error")]
    if limit:
        pairs = pairs.head(limit)
    mk = Masker(catvton_repo)
    qa = []
    for i, r in enumerate(pairs.itertuples()):
        d = os.path.join(export_dir, r.split)
        for sub in ["agnostic-mask", "image-densepose", "image-parse", "cloth", "cloth-mask"]:
            os.makedirs(os.path.join(d, sub), exist_ok=True)
        dst = os.path.join(d, "agnostic-mask", f"{r.pair_id}_mask.png")
        rec = {"pair_id": r.pair_id, "split": r.split, "pair_type": r.pair_type, "category": r.category}
        try:
            if overwrite or not os.path.exists(dst):
                person = Image.open(os.path.join(d, "image", f"{r.pair_id}.jpg")).convert("RGB")
                mtype = {"lower": "lower", "overall": "overall"}.get(r.category, "upper")
                o = mk.run(person, mtype)
                lip = o.get("schp_lip")
                mask = o["mask"]
                if mtype == "upper":
                    mask = extend_to_hem(mask, lip)
                cv2.imwrite(dst, mask.astype(np.uint8))
                if o.get("densepose") is not None:
                    dp = o["densepose"]
                    dp = dp if dp.ndim == 3 else cv2.applyColorMap((dp * (255 // max(1, dp.max()))).astype(np.uint8), cv2.COLORMAP_JET)
                    cv2.imwrite(os.path.join(d, "image-densepose", f"{r.pair_id}.jpg"), dp)
                if lip is not None:
                    cv2.imwrite(os.path.join(d, "image-parse", f"{r.pair_id}.png"), lip.astype(np.uint8))
                    torso = np.isin(lip, LIP_GARMENT_UPPER + [LIP["scarf"]])
                    rec["dupatta_frac_of_torso"] = float((lip == LIP["scarf"]).sum() / max(1, torso.sum()))
                    rec["garment_px_frac"] = float(np.isin(lip, LIP_GARMENT_UPPER).mean())
                rec["mask_frac"] = float((mask > 127).mean())
            if r.pair_type == "worn":
                dst_c = os.path.join(d, "cloth", f"{r.pair_id}.jpg")
                if overwrite or not os.path.exists(dst_c):
                    src = cv2.imread(os.path.join(root, r.cloth))
                    src_pad = fit_pad(src)
                    o2 = mk.run(Image.fromarray(cv2.cvtColor(src_pad, cv2.COLOR_BGR2RGB)), "upper")
                    gm = garment_from_person(src_pad, o2["schp_lip"], r.category)
                    if gm.sum() < 0.02 * gm.size:
                        raise ValueError("garment too small after parsing")
                    cloth, cmask = make_cloth(src_pad, gm)
                    cv2.imwrite(dst_c, cloth, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    cv2.imwrite(os.path.join(d, "cloth-mask", f"{r.pair_id}.jpg"), cmask)
            rec["status"] = "ok"
        except Exception as e:
            rec["status"] = f"error:{e!r}"[:160]
        qa.append(rec)
        if i % 25 == 0:
            print(f"{i}/{len(pairs)} {rec['status']}", flush=True)
            pd.DataFrame(qa).to_csv(os.path.join(export_dir, "masks_qa.csv"), index=False)
    qa = pd.DataFrame(qa)
    qa.to_csv(os.path.join(export_dir, "masks_qa.csv"), index=False)
    print(qa.status.value_counts().to_string())
    return qa


def qa_sheet(export_dir: str, out_png: str, n: int = 12):
    """Person | mask overlay | cloth, for the first n pairs. Look at it before the full run."""
    pairs = pd.read_csv(os.path.join(export_dir, "pairs.csv")).head(n)
    rows = []
    for r in pairs.itertuples():
        d = os.path.join(export_dir, r.split)
        im = cv2.imread(os.path.join(d, "image", f"{r.pair_id}.jpg"))
        mk = cv2.imread(os.path.join(d, "agnostic-mask", f"{r.pair_id}_mask.png"), 0)
        cl = cv2.imread(os.path.join(d, "cloth", f"{r.pair_id}.jpg"))
        if im is None or mk is None:
            continue
        ov = im.copy(); ov[mk > 127] = (0.45 * ov[mk > 127] + 0.55 * np.array([60, 60, 230])).astype(np.uint8)
        cl = cl if cl is not None else np.full_like(im, 200)
        row = np.hstack([cv2.resize(x, (288, 384)) for x in (im, ov, cl)])
        cv2.putText(row, f"{r.pair_id} {r.pair_type}/{r.category}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        rows.append(row)
    if rows:
        cv2.imwrite(out_png, np.vstack(rows))
    return out_png
