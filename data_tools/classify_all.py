"""Run the view classifier over every image in the catalog.

Resumable: results are appended to OUT_CSV in chunks; rows already present are skipped,
so a Colab disconnect costs at most one chunk. Also records an MD5 of each file so exact
duplicates across SKUs (e.g. Khaadi 'T-' re-listings) can be found later.

usage (Colab):
    python -m data_tools.classify_all --root /content/drive/MyDrive/AainaClosetDataset \
        --out /content/drive/MyDrive/AainaClosetDataset/_build/views.csv --workers 2
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from multiprocessing import Pool

import pandas as pd

from .catalog import build_catalog

_vc = None


def _init():
    global _vc
    import warnings; warnings.filterwarnings("ignore")
    os.environ.setdefault("GLOG_minloglevel", "2")
    from .views import ViewClassifier
    _vc = ViewClassifier()


def _work(args):
    root, rel = args
    path = os.path.join(root, rel)
    try:
        with open(path, "rb") as fh:
            md5 = hashlib.md5(fh.read()).hexdigest()
        r = _vc.classify_path(path)
    except Exception as e:  # keep going; record the failure
        return {"rel_path": rel, "label": "error", "error": repr(e)[:200]}
    r.pop("path", None)
    r["rel_path"] = rel
    r["md5"] = md5
    return r


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--catalog-out", default=None, help="default: <out dir>/catalog.csv")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0, help="only first N images (smoke test)")
    a = ap.parse_args(argv)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    cat_path = a.catalog_out or os.path.join(os.path.dirname(a.out), "catalog.csv")
    print("building catalog ...", flush=True)
    cat = build_catalog(a.root)
    cat.to_csv(cat_path, index=False)
    print(f"catalog: {len(cat)} images, {cat.sku.nunique()} SKUs -> {cat_path}")

    todo = cat.rel_path.tolist()
    if a.limit:
        todo = todo[: a.limit]
    done = set()
    if os.path.exists(a.out):
        done = set(pd.read_csv(a.out, usecols=["rel_path"]).rel_path)
    todo = [r for r in todo if r not in done]
    print(f"{len(done)} already classified, {len(todo)} to go", flush=True)

    t0 = time.time()
    with Pool(a.workers, initializer=_init) as pool:
        for i in range(0, len(todo), a.chunk):
            batch = todo[i: i + a.chunk]
            res = pool.map(_work, [(a.root, r) for r in batch])
            pd.DataFrame(res).to_csv(a.out, mode="a", header=not os.path.exists(a.out), index=False)
            n = i + len(batch)
            rate = n / (time.time() - t0)
            print(f"  {n}/{len(todo)}  {rate:.1f} img/s  eta {(len(todo) - n) / rate / 60:.0f} min", flush=True)
    print("done.")


if __name__ == "__main__":
    sys.exit(main())
