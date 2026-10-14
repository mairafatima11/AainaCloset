"""Which garment reference works best with stock CatVTON? (notebook 04, Colab T4)

For the same pairs, run stock CatVTON once per garment variant and score the result
against the real photo of the target person wearing that garment (worn pairs have true
ground truth: same product, same model, different shot).

  full      the second model photo, as published on the shop page
  faceless  same photo, head greyed out
  cutout    garment only, cut out with SCHP (masks_gpu.py)
  flat      brand flat lay (only for ETHNC SKUs that also have one -> same-SKU comparison)

Metrics, all computed inside the try-on mask (the only region the model is allowed to change):
  SSIM (higher better), LPIPS (lower better), plus 'hem_gap': how far the garment's lower edge in
  the output is from the real one, in pixels at 1024 height (kameez length is the failure we care about).
Near-identical cloth/person shots (similarity > 0.8) are excluded: they would let 'full' win by copying.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import pandas as pd
from PIL import Image

VARIANTS = {"full": "cloth_full", "faceless": "cloth_faceless", "cutout": "cloth_cutout"}


def masked_ssim(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> float:
    from skimage.metrics import structural_similarity
    _, full = structural_similarity(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.cvtColor(b, cv2.COLOR_RGB2GRAY),
                                    full=True, data_range=255)
    return float(full[m > 0].mean()) if (m > 0).any() else float("nan")


def masked_lpips(fn, a: np.ndarray, b: np.ndarray, m: np.ndarray) -> float:
    import torch
    am = np.where(m[..., None] > 0, a, 0)
    bm = np.where(m[..., None] > 0, b, 0)
    t = lambda x: (torch.from_numpy(x).permute(2, 0, 1)[None].float() / 127.5 - 1).cuda()
    with torch.no_grad():
        return float(fn(t(am), t(bm)).item())


def hem_row(img: np.ndarray, m: np.ndarray) -> int:
    """Lowest row inside the mask whose colour still matches the garment's upper-middle colour."""
    ys, xs = np.nonzero(m)
    if not len(ys):
        return -1
    y0, y1 = ys.min(), ys.max()
    band = img[y0 + (y1 - y0) // 4: y0 + (y1 - y0) // 2][m[y0 + (y1 - y0) // 4: y0 + (y1 - y0) // 2] > 0]
    ref = np.median(band, axis=0) if len(band) else np.array([0, 0, 0])
    d = np.linalg.norm(img.astype(np.float32) - ref, axis=2)
    close = (d < 45) & (m > 0)
    rows = np.nonzero(close.mean(axis=1) > 0.02)[0]
    return int(rows.max()) if len(rows) else -1


class CatVTON:
    def __init__(self, repo: str, base="stable-diffusion-v1-5/stable-diffusion-inpainting", width=384, height=512):
        if repo not in sys.path:
            sys.path.insert(0, repo)
        import torch
        from diffusers.image_processor import VaeImageProcessor
        from huggingface_hub import snapshot_download
        from model.pipeline import CatVTONPipeline   # CatVTON repo
        from utils import resize_and_crop, resize_and_padding
        self.rc, self.rp, self.w, self.h = resize_and_crop, resize_and_padding, width, height
        ckpt = snapshot_download("zhengchong/CatVTON")
        self.pipe = CatVTONPipeline(base_ckpt=base, attn_ckpt=ckpt, attn_ckpt_version="mix",
                                    weight_dtype=torch.float16, use_tf32=False, device="cuda")
        self.mp = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)
        self.torch = torch

    def __call__(self, person: Image.Image, cloth: Image.Image, mask: Image.Image, seed=42, steps=50, cfg=2.5):
        person = self.rc(person, (self.w, self.h))
        cloth = self.rp(cloth, (self.w, self.h))
        mask = self.mp.blur(self.rc(mask, (self.w, self.h)), blur_factor=9)
        g = self.torch.Generator(device="cuda").manual_seed(seed)
        return self.pipe(image=person, condition_image=cloth, mask=mask, num_inference_steps=steps,
                         guidance_scale=cfg, height=self.h, width=self.w, generator=g)[0]


def run(export_dir: str, catvton_repo: str, out_dir: str, n: int = 24, splits=("test", "val"),
        width=384, height=512, steps=50):
    import lpips
    os.makedirs(out_dir, exist_ok=True)
    p = pd.read_csv(os.path.join(export_dir, "pairs.csv"))
    p = p[p.status.astype(str).str.startswith(("ok", "needs_gpu"))]
    worn = p[(p.pair_type == "worn") & p.split.isin(splits) & (p.cloth_person_similarity.fillna(0) <= 0.8)]
    flat_skus = set(p[p.pair_type == "flat"].sku)
    # put SKUs that also have a flat lay first, so the flat-vs-worn comparison has data
    worn = pd.concat([worn[worn.sku.isin(flat_skus)], worn[~worn.sku.isin(flat_skus)]]).head(n)
    model, lp = CatVTON(catvton_repo, width=width, height=height), lpips.LPIPS(net="alex").cuda()
    rows = []
    for r in worn.itertuples():
        d = os.path.join(export_dir, r.split)
        person = Image.open(os.path.join(d, "image", f"{r.pair_id}.jpg")).convert("RGB")
        mask = Image.open(os.path.join(d, "agnostic-mask", f"{r.pair_id}_mask.png")).convert("L")
        gt = np.array(model.rc(person, (width, height)))
        m = (np.array(model.rc(mask, (width, height))) > 127).astype(np.uint8)
        cand = {k: os.path.join(d, sub, f"{r.pair_id}.jpg") for k, sub in VARIANTS.items()}
        fl = p[(p.pair_type == "flat") & (p.sku == r.sku)]
        if len(fl):
            f = fl.iloc[0]
            cand["flat"] = os.path.join(export_dir, f.split, "cloth", f"{f.pair_id}.jpg")
        for variant, path in cand.items():
            if not os.path.exists(path):
                continue
            dst = os.path.join(out_dir, f"{r.pair_id}__{variant}.jpg")
            if os.path.exists(dst):
                out = np.array(Image.open(dst).convert("RGB"))
            else:
                out = np.array(model(person, Image.open(path).convert("RGB"), mask, steps=steps))
                Image.fromarray(out).save(dst, quality=95)
            hg, ho = hem_row(gt, m), hem_row(out, m)
            rows.append({"pair_id": r.pair_id, "sku": r.sku, "brand": r.brand, "category": r.category,
                         "variant": variant, "ssim_masked": masked_ssim(out, gt, m),
                         "lpips_masked": masked_lpips(lp, out, gt, m),
                         "hem_gap_px": abs(hg - ho) * (1024 / height) if hg >= 0 and ho >= 0 else np.nan})
        print(r.pair_id, "done", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "compare_modes.csv"), index=False)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out_dir, "compare_modes.csv"), index=False)
    return res


def summary(res: pd.DataFrame) -> pd.DataFrame:
    """Mean per variant over the pairs where every variant exists (fair comparison)."""
    common = res.groupby("pair_id").variant.nunique()
    full_set = res[res.pair_id.isin(common[common == common.max()].index)]
    t = res.groupby("variant")[["ssim_masked", "lpips_masked", "hem_gap_px"]].agg(["mean", "count"]).round(4)
    t2 = full_set.groupby("variant")[["ssim_masked", "lpips_masked", "hem_gap_px"]].mean().round(4)
    return t, t2


def sheet(res: pd.DataFrame, export_dir: str, out_dir: str, out_png: str, n: int = 8):
    """Row per pair: target photo | output for each variant."""
    p = pd.read_csv(os.path.join(export_dir, "pairs.csv")).set_index("pair_id")
    order = ["full", "faceless", "cutout", "flat"]
    rows = []
    for pid in res.pair_id.unique()[:n]:
        sp = p.loc[pid, "split"]
        tiles = [cv2.resize(cv2.imread(os.path.join(export_dir, sp, "image", f"{pid}.jpg")), (288, 384))]
        labels = ["target"]
        for v in order:
            f = os.path.join(out_dir, f"{pid}__{v}.jpg")
            if os.path.exists(f):
                tiles.append(cv2.resize(cv2.imread(f), (288, 384))); labels.append(v)
        row = np.hstack(tiles)
        for i, lab in enumerate(labels):
            cv2.putText(row, lab, (i * 288 + 6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        rows.append(row)
    w = max(r.shape[1] for r in rows)
    rows = [np.hstack([r, np.full((r.shape[0], w - r.shape[1], 3), 255, np.uint8)]) for r in rows]
    cv2.imwrite(out_png, np.vstack(rows))
    return out_png
