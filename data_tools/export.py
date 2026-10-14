"""Export reviewed pairs to a VITON-HD / CatVTON style directory.

    <OUT>/
      train/ image/  cloth/  cloth-mask/  cloth_full/  cloth_faceless/
             (agnostic-mask/, image-densepose/, image-parse/, cloth_cutout/ added by masks_gpu.py)
      val/   ...
      test/  ...
      train_pairs.txt   val_pairs.txt   test_pairs.txt  test_pairs_unpaired.txt   <- worn pairs (primary)
      train_flat_pairs.txt  test_flat_pairs_unpaired.txt ...                    <- flat-lay pairs, separate lists
      pairs.csv                          <- every exported pair with all metadata + split
      export_log.csv

Everything is written at 768x1024 (W x H), the resolution CatVTON and VITON-HD use.
Brand photos are 2:3 (ETHNC, Khaadi) or 5:6 (Sapphire); they are resized to fit and
padded with their own background colour rather than stretched or cropped.

Flat-lay cloth images are cut out here on CPU (plain studio background -> threshold +
largest components + hole filling). For 'worn' pairs the garment reference is another model
photo, written as cloth_full/ and cloth_faceless/ here; cloth_cutout/ (SCHP) is made on GPU by
masks_gpu.py. --worn-cloth picks which one is copied into cloth/ (the folder CatVTON reads).
cloth_person_similarity in pairs.csv flags near-identical shots (> 0.8; the one visibly near-identical Khaadi pair in the gold sample scores 0.83): keep them for
training if you like, but leave them out when reporting paired test metrics.
"""
from __future__ import annotations

import argparse
import os
import random

import cv2
import numpy as np
import pandas as pd

from .split import split_pairs, report

W, H = 768, 1024


def _border_colour(img: np.ndarray) -> np.ndarray:
    b = max(2, int(0.02 * min(img.shape[:2])))
    border = np.concatenate([img[:b].reshape(-1, 3), img[-b:].reshape(-1, 3),
                             img[:, :b].reshape(-1, 3), img[:, -b:].reshape(-1, 3)])
    return np.median(border, axis=0)


def fit_pad(img: np.ndarray, w=W, h=H, fill=None, interp=cv2.INTER_AREA) -> np.ndarray:
    """Resize to fit inside w x h keeping aspect, pad with `fill` (default: image border colour)."""
    s = min(w / img.shape[1], h / img.shape[0])
    nw, nh = round(img.shape[1] * s), round(img.shape[0] * s)
    r = cv2.resize(img, (nw, nh), interpolation=interp if s < 1 else cv2.INTER_CUBIC)
    if fill is None:
        fill = _border_colour(img) if img.ndim == 3 else 0
    out = np.empty((h, w) + img.shape[2:], img.dtype)
    out[:] = fill
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = r
    return out


def flat_cloth_mask(img: np.ndarray, thresh: int = 24) -> np.ndarray:
    """Binary mask of a garment photographed flat on a plain background."""
    bg = _border_colour(img).astype(np.float32)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_lab = cv2.cvtColor(bg.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
    d = np.linalg.norm(lab - bg_lab, axis=2)
    m = (d > thresh).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    # keep large components only (garment, maybe a separate sleeve/strap)
    n, cc, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    if n > 1:
        big = stats[1:, cv2.CC_STAT_AREA].max()
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] > 0.05 * big:
                keep[cc == i] = 1
    # fill holes (white embroidery on white-ish ground would otherwise punch holes)
    flood = keep.copy(); h, w = flood.shape
    ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff, (0, 0), 1)
    holes = (flood == 0)
    keep[holes] = 1
    # refine edges with GrabCut seeded from the threshold mask
    try:
        gc = np.where(cv2.erode(keep, np.ones((15, 15), np.uint8)) > 0, cv2.GC_FGD,
                      np.where(cv2.dilate(keep, np.ones((15, 15), np.uint8)) > 0, cv2.GC_PR_FGD, cv2.GC_BGD)).astype(np.uint8)
        small = 640 / max(img.shape[:2])
        if small < 1:
            im_s = cv2.resize(img, None, fx=small, fy=small, interpolation=cv2.INTER_AREA)
            gc_s = cv2.resize(gc, (im_s.shape[1], im_s.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            im_s, gc_s = img, gc
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(im_s, gc_s, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
        ref = np.isin(gc_s, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8)
        if small < 1:
            ref = cv2.resize(ref, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        # only accept the refinement if it did not collapse or explode
        if 0.7 < ref.sum() / max(1, keep.sum()) < 1.3:
            keep = ref
    except cv2.error:
        pass
    return keep


def make_cloth(img: np.ndarray, mask: np.ndarray, margin: float = 0.04):
    """Composite garment on white, crop to bbox, fit into W x H. Returns (cloth, mask)."""
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    my, mx = int((y1 - y0) * margin), int((x1 - x0) * margin)
    y0, y1 = max(0, y0 - my), min(img.shape[0], y1 + my)
    x0, x1 = max(0, x0 - mx), min(img.shape[1], x1 + mx)
    white = np.full_like(img, 255)
    comp = np.where(mask[..., None] > 0, img, white)[y0:y1, x0:x1]
    m = (mask[y0:y1, x0:x1] * 255).astype(np.uint8)
    return fit_pad(comp, fill=np.array([255, 255, 255], np.uint8)), fit_pad(m, fill=0, interp=cv2.INTER_NEAREST)


_POSE = None


def _pose():
    global _POSE
    if _POSE is None:
        import mediapipe as mp
        _POSE = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1)
    return _POSE


def faceless(img: np.ndarray, grey=(128, 128, 128)) -> np.ndarray:
    """Hide the head (face + hair above the shoulders) of a model photo used as a garment reference.

    Removes the identity cue the try-on model could copy, while keeping the whole garment
    (neckline sits at or below the shoulder line). Falls back to the unmodified image when
    no pose is found -- those pairs should not have passed review anyway.
    """
    res = _pose().process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.pose_landmarks:
        return img
    lm = res.pose_landmarks.landmark
    h = img.shape[0]
    sh_y = min(lm[11].y, lm[12].y) * h
    ear_y = max(lm[7].y, lm[8].y) * h          # ears: lower bound of the head
    cut = int(max(0, min(sh_y - 0.025 * h, ear_y + 0.6 * (sh_y - ear_y))))
    out = img.copy()
    out[:cut] = grey
    return out


def image_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation of tiny greyscale thumbnails: ~1.0 means the two shots are almost identical."""
    ta = cv2.resize(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), (24, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    tb = cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), (24, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    ta, tb = ta - ta.mean(), tb - tb.mean()
    return float((ta * tb).sum() / (np.sqrt((ta ** 2).sum() * (tb ** 2).sum()) + 1e-6))


def unpaired(names: list[str], cats: list[str], seed: int = 0) -> list[str]:
    """Deterministic derangement within each try-on category (cloth i -> person j != i)."""
    rng = random.Random(seed)
    out = list(names)
    for c in set(cats):
        idx = [i for i, x in enumerate(cats) if x == c]
        if len(idx) < 2:
            continue
        perm = idx[1:] + idx[:1]          # rotate -> guaranteed no fixed point
        rng.shuffle(idx)
        perm = idx[1:] + idx[:1]
        for a, b in zip(idx, perm):
            out[a] = names[b]
    return out


def export(pairs: pd.DataFrame, root: str, out: str, only_accepted: bool = True, overwrite: bool = False,
           worn_cloth: str = "faceless"):
    p = pairs.copy()
    if only_accepted and "decision" in p:
        p = p[p.decision == "accept"]
    p = split_pairs(p)
    os.makedirs(out, exist_ok=True)
    log = []
    for _, r in p.iterrows():
        d = os.path.join(out, r.split)
        for sub in ["image", "cloth", "cloth-mask"]:
            os.makedirs(os.path.join(d, sub), exist_ok=True)
        name = f"{r.pair_id}.jpg"
        dst_img = os.path.join(d, "image", name)
        status = "ok"
        person = None
        try:
            if overwrite or not os.path.exists(dst_img):
                person = fit_pad(cv2.imread(os.path.join(root, r.person)))
                cv2.imwrite(dst_img, person, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if r.pair_type == "flat":
                dst_c = os.path.join(d, "cloth", name)
                if overwrite or not os.path.exists(dst_c):
                    src = cv2.imread(os.path.join(root, r.cloth))
                    mask = flat_cloth_mask(src)
                    cloth, cmask = make_cloth(src, mask)
                    cv2.imwrite(dst_c, cloth, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    cv2.imwrite(os.path.join(d, "cloth-mask", name), cmask)
                    log_cov = float(mask.mean())
                else:
                    log_cov = np.nan
            else:
                # worn: garment reference is another model photo of the same SKU.
                #   cloth_full/      the photo as published (what a shop page shows)
                #   cloth_faceless/  same photo, head greyed out (no identity to copy)
                #   cloth_cutout/    garment only, made on GPU by masks_gpu.py (SCHP)
                #   cloth/           = the variant chosen by --worn-cloth (default faceless)
                for sub in ["cloth_full", "cloth_faceless"]:
                    os.makedirs(os.path.join(d, sub), exist_ok=True)
                f_full, f_fl = os.path.join(d, "cloth_full", name), os.path.join(d, "cloth_faceless", name)
                if overwrite or not (os.path.exists(f_full) and os.path.exists(f_fl)):
                    src = fit_pad(cv2.imread(os.path.join(root, r.cloth)))
                    fl = faceless(src)
                    cv2.imwrite(f_full, src, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    cv2.imwrite(f_fl, fl, [cv2.IMWRITE_JPEG_QUALITY, 95])
                else:
                    src, fl = cv2.imread(f_full), cv2.imread(f_fl)
                chosen = {"full": src, "faceless": fl}.get(worn_cloth)
                if chosen is not None:
                    cv2.imwrite(os.path.join(d, "cloth", name), chosen, [cv2.IMWRITE_JPEG_QUALITY, 95])
                sim = image_similarity(src, cv2.imread(dst_img))
                status = "ok" if chosen is not None else "needs_gpu_cloth"
                log_cov = np.nan
                log.append({"pair_id": r.pair_id, "split": r.split, "status": status,
                            "cloth_mask_coverage": log_cov, "cloth_person_similarity": round(sim, 4)})
                continue
        except Exception as e:  # keep going, report at the end
            status, log_cov = f"error:{e!r}"[:120], np.nan
        log.append({"pair_id": r.pair_id, "split": r.split, "status": status, "cloth_mask_coverage": log_cov})

    lg = pd.DataFrame(log)
    p = p.merge(lg, on=["pair_id", "split"])
    p.to_csv(os.path.join(out, "pairs.csv"), index=False)
    lg.to_csv(os.path.join(out, "export_log.csv"), index=False)
    for split, g in p[~p.status.str.startswith("error")].groupby("split"):
        for ptype, gg in g.groupby("pair_type"):
            names = [f"{x}.jpg" for x in gg.pair_id]
            suffix = "" if ptype == "worn" else f"_{ptype}"     # worn = primary -> CatVTON's default file names
            with open(os.path.join(out, f"{split}{suffix}_pairs.txt"), "w") as fh:
                fh.writelines(f"{n} {n}\n" for n in names)
            if split != "train":
                un = unpaired(names, gg.category.tolist(), seed=42)
                with open(os.path.join(out, f"{split}{suffix}_pairs_unpaired.txt"), "w") as fh:
                    fh.writelines(f"{a} {b}\n" for a, b in zip(names, un))
    print(report(p).to_string())
    print(lg.status.value_counts().to_string())
    return p


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="pairs_reviewed.csv (or pairs_candidates.csv with --all)")
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--all", action="store_true", help="export without review decisions (debug only)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--worn-cloth", choices=["faceless", "full", "cutout"], default="faceless",
                    help="which garment variant goes into cloth/ for worn pairs (decide with notebook 04)")
    a = ap.parse_args(argv)
    export(pd.read_csv(a.pairs), a.root, a.out, only_accepted=not a.all, overwrite=a.overwrite,
           worn_cloth=a.worn_cloth)


if __name__ == "__main__":
    main()
