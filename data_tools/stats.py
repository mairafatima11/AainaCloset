"""Numbers for the datasheet and the dataset chapter, straight from the build files.

    python -m data_tools.stats --build _build --export DesiFit-VTON-v1 > docs/dataset_stats.md
"""
from __future__ import annotations

import argparse
import os

import pandas as pd


def md(df: pd.DataFrame) -> str:
    return df.to_markdown() if hasattr(df, "to_markdown") else df.to_string()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True, help="folder with catalog.csv, views.csv, pairs_*.csv")
    ap.add_argument("--export", default=None, help="exported dataset folder (pairs.csv, masks_qa.csv)")
    a = ap.parse_args(argv)
    cat = pd.read_csv(os.path.join(a.build, "catalog.csv"))
    out = ["# Dataset statistics", "", f"Generated from `{a.build}`.", ""]

    sku = cat.drop_duplicates("sku")
    out += ["## Raw scrape", "", f"{len(cat):,} images, {len(sku):,} SKUs, {sku.design_group.nunique():,} design groups.", ""]
    out += [md(sku.groupby(["brand", "audience", "is_apparel", "is_ethnic"]).size().rename("SKUs").to_frame()), ""]

    vp = os.path.join(a.build, "views.csv")
    if os.path.exists(vp):
        v = pd.read_csv(vp).merge(cat[["rel_path", "brand", "sku"]], on="rel_path")
        out += ["## Image views (automatic classifier)", "", md(pd.crosstab(v.brand, v.label, margins=True)), ""]
        has = v.groupby("sku").label.agg(lambda s: set(s))
        tab = pd.DataFrame({"brand": cat.drop_duplicates("sku").set_index("sku").brand.reindex(has.index),
                            "has_flat": has.map(lambda s: "flat" in s),
                            "has_target": has.map(lambda s: "mff" in s)})
        out += ["SKUs with a flat lay / with a front full-length model shot:", "",
                md(tab.groupby("brand")[["has_flat", "has_target"]].mean().round(3)), ""]
        dup = v.md5.duplicated(keep=False).sum() if "md5" in v else 0
        out += [f"Byte-identical images appearing more than once: {dup}", ""]

    for name in ["pairs_candidates.csv", "pairs_reviewed.csv"]:
        p = os.path.join(a.build, name)
        if os.path.exists(p):
            d = pd.read_csv(p)
            out += [f"## {name}", "", md(d.groupby(["brand", "pair_type", "category"]).size().rename("pairs").to_frame()), ""]
            if "decision" in d:
                out += ["Review outcome:", "", md(d.decision.value_counts(dropna=False).to_frame()), ""]
                if "reason" in d:
                    out += ["Reject reasons:", "", md(d.reason.value_counts().to_frame()), ""]
                if "edited" in d:
                    out += [f"Pairs where the reviewer changed the automatic image choice: {int(d.edited.sum())}", ""]

    if a.export:
        ep = os.path.join(a.export, "pairs.csv")
        if os.path.exists(ep):
            e = pd.read_csv(ep)
            out += ["## Exported dataset", "", md(e.pivot_table(index=["brand", "pair_type", "category"], columns="split",
                                                           values="pair_id", aggfunc="count", fill_value=0, margins=True)), ""]
        qp = os.path.join(a.export, "masks_qa.csv")
        if os.path.exists(qp):
            q = pd.read_csv(qp)
            out += ["## Mask QA", "", md(q.status.str.split(":").str[0].value_counts().to_frame()), ""]
            if "dupatta_frac_of_torso" in q:
                out += ["Dupatta coverage of the torso (fraction of garment+dupatta pixels labelled 'scarf' by SCHP):", "",
                        md(q.groupby("pair_type").dupatta_frac_of_torso.describe().round(3)), ""]
    print("\n".join(out))


if __name__ == "__main__":
    main()
