"""Leak-free train / val / test split.

The unit of splitting is `split_group`, not the image and not even the SKU:
  * same design in a different colour shares an ETHNC style code  (e0775-...-001 vs -901)
  * Khaadi 'T-' re-listings and article families share a code prefix
  * any two groups that contain a byte-identical image are merged (pairs.leakage_groups)
so one outfit can never appear on both sides.

Assignment is a hash of the group name -> stable: adding new SKUs later never moves an
existing group to a different split, so earlier results stay comparable.
"""
from __future__ import annotations

import hashlib

import pandas as pd

DEFAULT = {"train": 0.75, "val": 0.10, "test": 0.15}


def assign(group: str, fractions=DEFAULT, salt: str = "desifit-v1") -> str:
    h = int(hashlib.sha1(f"{salt}:{group}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    acc = 0.0
    for name, frac in fractions.items():
        acc += frac
        if h < acc:
            return name
    return list(fractions)[-1]


def split_pairs(pairs: pd.DataFrame, fractions=DEFAULT) -> pd.DataFrame:
    p = pairs.copy()
    p["split"] = p.split_group.map(lambda g: assign(g, fractions))
    check_no_leak(p)
    return p


def check_no_leak(p: pd.DataFrame):
    for key in ["split_group", "design_group", "sku"]:
        n = p.groupby(key).split.nunique()
        bad = n[n > 1]
        assert bad.empty, f"leak: {len(bad)} {key}s appear in >1 split, e.g. {bad.index[:3].tolist()}"
    for col in ["cloth", "person"]:
        n = p.groupby(col).split.nunique()
        assert (n <= 1).all(), f"leak: same {col} image in several splits"


def report(p: pd.DataFrame) -> pd.DataFrame:
    return p.pivot_table(index=["brand", "pair_type", "category"], columns="split", values="pair_id",
                         aggfunc="count", fill_value=0, margins=True)
