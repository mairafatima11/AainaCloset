"""Build the image catalog from the scraped folder tree and attach product metadata
that can be inferred from each brand's SKU / handle naming.

Folder layout produced by the existing download scripts:

    <ROOT>/ETHNC_Images/<shopify-handle>_<product-id>/image_NN.jpg
    <ROOT>/Khaadi_Images/<product-code>/image_NN.jpg
    <ROOT>/Sapphire_Images/<collection>/<product-code>.html/image_NN.jpg

Every rule below was derived from inspecting the real data (Oct 2026). Each one is a
heuristic: re-check it whenever a brand is re-scraped, and prefer the scraped
product_type / tags once scraper/metadata_*.py has been run (see merge_metadata()).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}

# ---------------------------------------------------------------- ETHNC (Shopify handle)
# handle example: casual-suit-e1598-402-416-416-casual-suit-e1598-302-416-416
#   product line  : casual / rozana / boutique / boho / timelesz ...
#   item type     : suit / kurti / trouser / dress / kaftan ...
#   style code    : e1598          (same design in other colours shares this)
#   segment       : 1xx women, 2xx unstitched, 4xx (and 3xx) girls  -> verified visually on sample
#   colour code   : 416
ETHNC_RE = re.compile(r"^(?P<words>[a-z]+(?:-[a-z]+)*?)-(?:s\d)?(?P<style>e\d{3,5})-(?P<seg>\d)(?P<cls>\d\d)-(?P<colour>\d{3})")
ETHNC_WESTERN_LINES = {"timelesz", "timeless"}
ETHNC_NON_APPAREL_ITEMS = {"heels", "shoes", "puri", "sandals", "mules", "pumps"}
# segment x09 = footwear (109 women, 309/409 kids); 120 = western 'timelesz' line
ETHNC_LOWER_ITEMS = {"trouser", "pants", "jeans", "skirt", "shalwar", "pant"}
ETHNC_UPPER_ITEMS = {"kurti", "shirt", "top", "waistcoat", "coat", "jacket", "kimono"}
ETHNC_OVERALL_ITEMS = {"dress", "kaftan"}
ETHNC_SET_ITEMS = {"suit", "ensemble", "piece"}   # 'three-piece' = unstitched 3pc shown stitched

# ---------------------------------------------------------------- Khaadi
# codes: 2-26-403-C-C1-VG_MULTI, A1D1-26-403FE2-E-VG_MULTI, T-A33-26-205FB1-VG_MULTI, EDP19-VG_MULTI ...
KHAADI_NON_APPAREL_PREFIX = ("EDP", "C44", "C4", "HOME")   # EDP = perfume, C44 = cushions/home
KHAADI_KIDS_TOKENS = ("CHD", "KIDS")

# ---------------------------------------------------------------- Sapphire
# codes: 2PCSTCRC063S_999, U3PEDY26V968, 1PCCCASV912S_999 ...
#   leading U = unstitched fabric (shown stitched on a model), digit = number of pieces
SAPPHIRE_RE = re.compile(r"^(?P<u>U|0U)?(?P<pieces>\d)")


def _ethnc(handle: str) -> dict:
    m = ETHNC_RE.match(handle)
    if not m:
        return {"design_group": "ethnc:" + handle, "audience": "unknown", "item_type": "unknown",
                "garment_type": "unknown", "is_apparel": True, "is_ethnic": True}
    words = m["words"].split("-")
    line, item, seg, cls = words[0], words[-1], m["seg"], m["cls"]
    is_apparel = item not in ETHNC_NON_APPAREL_ITEMS and cls != "09"
    if item in ETHNC_LOWER_ITEMS:
        gtype = "lower"
    elif item in ETHNC_UPPER_ITEMS or (words[-2:] == ["t", "shirt"]):
        gtype = "upper"
    elif item in ETHNC_OVERALL_ITEMS:
        gtype = "overall"
    elif item in ETHNC_SET_ITEMS:
        gtype = "set"            # kameez + shalwar (and sometimes dupatta)
    else:
        gtype = "unknown"
    return {
        "design_group": f"ethnc:{m['style']}",
        "audience": "kids" if seg in {"3", "4"} else "women",
        "item_type": "-".join(words[1:]) or line,
        "garment_type": gtype if is_apparel else "non_apparel",
        "is_apparel": is_apparel,
        "is_ethnic": line not in ETHNC_WESTERN_LINES,
        "unstitched": seg == "2" or "unstitched" in words,
        "colour_code": m["colour"],
    }


def _khaadi(code: str) -> dict:
    base = re.sub(r"-VG_MULTI$", "", code)
    relisted = base.startswith("T-")
    core = re.sub(r"^T-", "", base)
    core = re.sub(r"-E$", "", core)
    toks = core.split("-")
    group = "-".join(toks[:3]) if len(toks) >= 3 else core
    return {
        "design_group": f"khaadi:{group}",
        "audience": "kids" if any(t in core.upper() for t in KHAADI_KIDS_TOKENS) else "women",
        "item_type": "unknown",
        "garment_type": "unknown",
        "is_apparel": not core.upper().startswith(KHAADI_NON_APPAREL_PREFIX),
        "is_ethnic": True,
        "khaadi_relisted": relisted,
    }


def _sapphire(code: str) -> dict:
    core = re.sub(r"\.html$", "", code)
    core = re.sub(r"_\d+$", "", core)
    m = SAPPHIRE_RE.match(core)
    pieces = int(m["pieces"]) if m else None
    return {
        "design_group": f"sapphire:{core}",
        "audience": "women",
        "item_type": "unstitched" if (m and m["u"]) else (f"{pieces}pc" if pieces else "unknown"),
        "garment_type": "set" if (pieces or 0) >= 2 else ("upper" if pieces == 1 else "unknown"),
        "is_apparel": True,
        "is_ethnic": True,
        "pieces": pieces,
    }


def parse_product(brand: str, product_folder: str) -> dict:
    if brand.startswith("ETHNC"):
        handle = re.sub(r"_\d{8,}$", "", product_folder)   # strip shopify product id
        return _ethnc(handle)
    if brand.startswith("Khaadi"):
        return _khaadi(product_folder)
    if brand.startswith("Sapphire"):
        return _sapphire(product_folder)
    return {"design_group": f"{brand}:{product_folder}", "audience": "unknown", "item_type": "unknown",
            "garment_type": "unknown", "is_apparel": True, "is_ethnic": True}


def build_catalog(root: str | os.PathLike) -> pd.DataFrame:
    """Walk ROOT and return one row per image with brand / product metadata."""
    root = Path(root)
    rows = []
    for brand_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.endswith("_Images")):
        brand = brand_dir.name.replace("_Images", "")
        for img in sorted(brand_dir.rglob("*")):
            if img.suffix.lower() not in IMAGE_EXT or not img.is_file():
                continue
            rel = img.relative_to(root).as_posix()
            parts = img.relative_to(brand_dir).parts
            product_folder = parts[-2]
            collection = parts[0] if len(parts) == 3 else ""
            idx = re.search(r"image_(\d+)", img.stem)
            rows.append({
                "rel_path": rel,
                "brand": brand,
                "collection": collection,
                "product_folder": product_folder,
                "sku": f"{brand.lower()}:{re.sub(r'.html$', '', product_folder)}",
                "img_index": int(idx.group(1)) if idx else -1,
                "filename_dup": "(" in img.stem,      # 'image_05 (1).jpg' = Drive/OneDrive duplicate copy
                **parse_product(brand, product_folder),
            })
    df = pd.DataFrame(rows)
    return df


def merge_metadata(catalog: pd.DataFrame, products_jsonl: str) -> pd.DataFrame:
    """Join scraped product metadata (title, product_type, tags, image URL) onto the catalog.

    products_jsonl rows: {"sku": ..., "img_index": n, "url": ..., "title": ..., "product_type": ..., "tags": ...}
    produced by scraper/metadata_*.py. Rows the scraper cannot match keep NaN.
    """
    meta = pd.read_json(products_jsonl, lines=True)
    return catalog.merge(meta, on=["sku", "img_index"], how="left", suffixes=("", "_meta"))


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    sku = df.drop_duplicates("sku")
    return (sku.groupby(["brand", "audience", "is_apparel", "is_ethnic", "garment_type"])
               .size().rename("skus").reset_index())
