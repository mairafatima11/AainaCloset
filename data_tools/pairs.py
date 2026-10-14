"""Pick candidate try-on pairs per SKU from the classified views.

Two pair types, kept clearly separate everywhere downstream:

  flat  cloth = brand flat-lay (front), person = best front full-length model shot.
        This is the standard VITON-HD-style pair and the core of the dataset.
        (ETHNC is the only brand in the current scrape that publishes flat lays.)

  worn  cloth = garment segmented out of one model photo (done later, on GPU, with SCHP),
        person = a *different* model photo of the same SKU. Weaker supervision: the cloth
        is already shaped by a body. Used for Khaadi / Sapphire, for the benchmark and
        as an optional extra training slice -- never mixed into 'flat' metrics.

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


def build_pairs(cat: pd.DataFrame, views: pd.DataFrame) -> pd.DataFrame:
    df = cat.merge(views, on="rel_path", how="inner", suffixes=("", "_v"))
    df = df[~df.filename_dup.astype(bool)]
    df["split_group"] = leakage_groups(df)
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
        flats = g[g.label == "flat"]
        lowers = g[g.label == "flat_lower"]
        alt_targets = "|".join(targets.rel_path.iloc[1:4])

        if len(targets) and len(flats):
            t = targets.iloc[0]
            rows.append({**base, "pair_type": "flat", "category": tryon_category(meta.garment_type),
                         "cloth": flats.iloc[0].rel_path, "person": t.rel_path,
                         "target_score": t.target_score, "alt_cloth": "|".join(flats.rel_path.iloc[1:]),
                         "alt_person": alt_targets, "bg_plain": int(t.border_std < 18)})
            if len(lowers) and meta.garment_type in {"set", "lower"}:
                rows.append({**base, "pair_type": "flat", "category": "lower",
                             "cloth": lowers.iloc[0].rel_path, "person": t.rel_path,
                             "target_score": t.target_score, "alt_cloth": "", "alt_person": alt_targets,
                             "bg_plain": int(t.border_std < 18)})
        elif len(lowers) and len(targets) and meta.garment_type == "lower":
            t = targets.iloc[0]
            rows.append({**base, "pair_type": "flat", "category": "lower", "cloth": lowers.iloc[0].rel_path,
                         "person": t.rel_path, "target_score": t.target_score, "alt_cloth": "",
                         "alt_person": alt_targets, "bg_plain": int(t.border_std < 18)})
        elif len(targets) >= 2:
            # garment source = 2nd best (often a slightly different pose), person = best
            t, src = targets.iloc[0], targets.iloc[1]
            rows.append({**base, "pair_type": "worn", "category": tryon_category(meta.garment_type),
                         "cloth": src.rel_path, "person": t.rel_path, "target_score": t.target_score,
                         "alt_cloth": "", "alt_person": "|".join(targets.rel_path.iloc[2:5]),
                         "bg_plain": int(t.border_std < 18)})
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
