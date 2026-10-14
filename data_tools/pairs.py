"""Pick candidate try-on pairs per SKU from the classified views.

Two pair types, kept clearly separate everywhere downstream:

  worn  (primary) cloth = a DIFFERENT front-facing photo of the same SKU (ideally a full-length
        front shot, so the hem is visible), person = the best front full-length model shot. This matches what the
        app receives: Khaadi and Sapphire publish no flat lays, so the catalogue image IS a
        model photo. Never the same photo for both sides. The cloth image is exported in
        three variants (full / faceless / cutout) -- see export.py and notebook 04.

  flat  (secondary) cloth = brand flat-lay (front). ETHNC only, ~40% of its SKUs. Kept as
        the VITON-HD-style setting and for a same-SKU comparison against 'worn'.

A flat_lower (shalwar / trouser flat) gives an extra 'lower' pair when present.

Selection is deliberately conservative; every candidate still goes through human review
(review.py) before it enters the dataset.
"""
from __future__ import annotations

import argparse

import networkx as nx
import pandas as pd

TARGET_MIN_SCORE = 0.55
MIN_SHORT_SIDE = 700          # px; below this the 1024x768 export would be upscaled too much


def eligible_skus(cat: pd.DataFrame) -> pd.Series:
    sku = cat.drop_duplicates("sku").set_index("sku")
    ok = sku.is_apparel.astype(bool) & sku.is_ethnic.astype(bool) & (sku.audience == "women")
    if "khaadi_relisted" in sku:
        ok &= ~sku.khaadi_relisted.fillna(False).astype(bool)
    return ok


def leakage_groups(df: pd.DataFrame) -> pd.Series:
    """Merge design groups that share any identical image file (MD5) into one split group."""
    g = nx.Graph()
    g.add_nodes_from(df.design_group.unique())
    for _, grp in df.dropna(subset=["md5"]).groupby("md5"):
        dg = grp.design_group.unique()
        for a, b in zip(dg[:-1], dg[1:]):
            g.add_edge(a, b)
    comp = {}
    for i, c in enumerate(nx.connected_components(g)):
        name = sorted(c)[0]
        for n in c:
            comp[n] = name
    return df.design_group.map(comp)


def tryon_category(garment_type: str) -> str:
    # CatVTON mask types: upper | lower | overall.
    # A kameez is upper-body but long; 'set'/'upper' both start from 'upper' and the GPU step
    # extends the mask down to the hem (see masks_gpu.py). Dresses/kaftans use 'overall'.
    return {"lower": "lower", "overall": "overall"}.get(garment_type, "upper")


def cloth_source_score(r) -> float:
    """How good is this model photo as a GARMENT reference (not as a target)?

    The garment reference must show the whole front of the garment, including the hem:
    kameez length is exactly what Western-trained models get wrong. So:
    front full-length shot > front shot cropped below the knee > headless torso crop
    (sharp print, but the hem is cut off). Side/back/seated/arms-up views are never used.
    """
    if not (r.pose_found and r.facing_camera and r.frontal > 0.68) or r.arms_raised or r.seated:
        return 0.0
    if r.label == "mff":
        return 0.8 + 0.2 * r.target_score
    if r.label == "mo" and r.head_in_frame and r.vis_knees > 0.5:
        return 0.6 + 0.1 * min(1.0, r.frontal)
    if r.label == "mo" and r.head_in_frame:      # 3/4 crop, hem may be cut: reviewer decides
        return 0.4 + 0.1 * min(1.0, r.frontal)
    if r.label == "detail":
        return 0.3 + 0.1 * min(1.0, r.frontal)
    return 0.0


def _pair(base, pair_type, category, cloth, person, alt_cloth="", alt_person=""):
    return {**base, "pair_type": pair_type, "category": category, "cloth": cloth.rel_path,
            "person": person.rel_path, "target_score": person.target_score,
            "cloth_score": getattr(cloth, "cloth_score", float("nan")),
            "alt_cloth": alt_cloth, "alt_person": alt_person, "bg_plain": int(person.border_std < 18)}


def build_pairs(cat: pd.DataFrame, views: pd.DataFrame) -> pd.DataFrame:
    """One 'worn' pair for every SKU that has a usable target + a different front view,
    plus a 'flat' pair (and a shalwar 'flat' pair) wherever a flat lay exists.

    'worn' is the primary type: it is what the app actually receives for brands that publish
    no flat lays (Khaadi, Sapphire, most of ETHNC). 'flat' pairs on the same ETHNC SKUs give a
    controlled comparison of the two conditioning sources.
    """
    df = cat.merge(views, on="rel_path", how="inner", suffixes=("", "_v"))
    df = df[~df.filename_dup.astype(bool)]
    df["split_group"] = leakage_groups(df)
    df["cloth_score"] = df.apply(cloth_source_score, axis=1)
    ok = eligible_skus(cat)
    rows = []
    for sku, g in df.groupby("sku"):
        if not ok.get(sku, False):
            continue
        g = g.sort_values("img_index")
        meta = g.iloc[0]
        base = dict(sku=sku, brand=meta.brand, collection=meta.collection, design_group=meta.design_group,
                    split_group=meta.split_group, garment_type=meta.garment_type, item_type=meta.item_type,
                    n_images=len(g))
        targets = g[(g.label == "mff") & (g.target_score >= TARGET_MIN_SCORE)
                    & (g[["width", "height"]].min(axis=1) >= MIN_SHORT_SIDE)]
        targets = targets.sort_values("target_score", ascending=False)
        if targets.empty:
            continue
        t = targets.iloc[0]
        alt_t = "|".join(targets.rel_path.iloc[1:4])
        cat_ = tryon_category(meta.garment_type)
        flats, lowers = g[g.label == "flat"], g[g.label == "flat_lower"]

        # --- worn (primary): garment from a different front-facing photo of the same SKU
        if meta.garment_type != "lower":
            src = g[(g.rel_path != t.rel_path) & (g.cloth_score > 0)].sort_values("cloth_score", ascending=False)
            if len(src):
                rows.append(_pair(base, "worn", cat_, src.iloc[0], t,
                                  alt_cloth="|".join(src.rel_path.iloc[1:4]), alt_person=alt_t))
        # --- flat (secondary, ETHNC): brand flat lay
        if len(flats) and meta.garment_type != "lower":
            rows.append(_pair(base, "flat", cat_, flats.iloc[0], t,
                              alt_cloth="|".join(flats.rel_path.iloc[1:]), alt_person=alt_t))
        if len(lowers) and meta.garment_type in {"set", "lower"}:
            rows.append(_pair(base, "flat", "lower", lowers.iloc[0], t, alt_person=alt_t))
    out = pd.DataFrame(rows)
    if len(out):
        out.insert(0, "pair_id", [f"p{i:05d}" for i in range(len(out))])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--views", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    p = build_pairs(pd.read_csv(a.catalog), pd.read_csv(a.views))
    p.to_csv(a.out, index=False)
    print(p.groupby(["brand", "pair_type", "category"]).size().to_string())
    print(f"{len(p)} candidate pairs, {p.sku.nunique()} SKUs -> {a.out}")


if __name__ == "__main__":
    main()
