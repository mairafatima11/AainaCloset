"""Re-collect product METADATA (no image downloads) for the images already on disk.

The original download scripts saved images but not the product URL, title, type or the
image URL behind each image_NN.jpg. Those are required to publish the dataset the
licence-safe way (release URLs + annotations, never the pixels) and to fill the datasheet.

Each brand function reproduces the exact image ordering of the matching download script,
so image_NN maps back to the right URL:
  ETHNC     download_ethnc.py      Shopify /products.json, images in listed order
  Sapphire  download_sapphire.py / download_new_in.py  product page <img data-high-src>, sorted(set())
  Khaadi    download_khaadi.py     product page '/hi-res/' URLs from src/data-*/srcset, sorted(set())

Output: one JSON line per image
  {"sku", "img_index", "url", "page_url", "title", "product_type", "tags", "price", "scraped_at"}
which data_tools.catalog.merge_metadata() joins onto the catalog.

Run from the folder that contains ETHNC_Images/ etc. (it only reads folder names):
    python scraper/metadata.py --root . --brand ethnc --out _build/products_ethnc.jsonl
Be polite: 1 request/second by default, identifying user agent, resumable (skips SKUs already written).
NOTE: written without network access to test against; run --limit 5 first and check the
'matched' count it prints before a full run.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse, urlunparse

import requests

UA = "DesiFitAI-research/1.0 (ITU final-year project; non-commercial; contact bscs23149@itu.edu.pk)"
S = requests.Session()
S.headers.update({"User-Agent": UA})
DELAY = 1.0


def _get(url, **kw):
    for attempt in range(3):
        r = S.get(url, timeout=40, **kw)
        if r.status_code == 429:
            time.sleep(int(r.headers.get("Retry-After", "60") or 60)); continue
        r.raise_for_status()
        time.sleep(DELAY)
        return r
    r.raise_for_status()


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_name(value):  # identical to download_ethnc.py
    value = re.sub(r'[<>:"/\\|?*]', "_", str(value))
    return value.strip(" ._")[:100] or "product"


# ------------------------------------------------------------------ ETHNC (Shopify)
def ethnc(root, done):
    base = "https://pk.ethnc.com"
    local = set(os.listdir(os.path.join(root, "ETHNC_Images")))
    page = 1
    while True:
        items = _get(f"{base}/products.json", params={"limit": 250, "page": page}).json().get("products", [])
        if not items:
            break
        for p in items:
            folder = _safe_name(f"{p.get('handle')}_{p.get('id')}")
            sku = f"ethnc:{folder}"
            if folder not in local or sku in done:
                continue
            tags = p.get("tags", [])
            tags = tags if isinstance(tags, list) else [t.strip() for t in str(tags).split(",")]
            price = (p.get("variants") or [{}])[0].get("price")
            n = 0
            for img in p.get("images", []):
                src = img.get("src", "") if isinstance(img, dict) else str(img)
                if not src:
                    continue
                n += 1
                yield {"sku": sku, "img_index": n, "url": src, "page_url": f"{base}/products/{p.get('handle')}",
                       "title": p.get("title"), "product_type": p.get("product_type"), "tags": tags,
                       "price": price, "scraped_at": _now()}
        page += 1


# ------------------------------------------------------------------ Sapphire (product pages)
def _sapphire_images(html):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    urls = set()
    for img in soup.find_all("img"):
        u = img.get("data-high-src")
        if u and os.path.splitext(urlparse(u).path)[1].lower() in (".jpg", ".jpeg", ".png", ".webp"):
            urls.add(u)
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    return urls, title


def sapphire(root, done):
    base = "https://pk.sapphireonline.pk"
    sroot = os.path.join(root, "Sapphire_Images")
    for coll in sorted(os.listdir(sroot)):
        for folder in sorted(os.listdir(os.path.join(sroot, coll))):
            code = re.sub(r"\.html$", "", folder)
            sku = f"sapphire:{code}"
            if sku in done:
                continue
            page_url = f"{base}/products/{folder}"
            try:
                r = _get(page_url)
            except requests.HTTPError:
                page_url = f"{base}/products/{code}"
                r = _get(page_url)
            urls, title = _sapphire_images(r.text)
            for n, u in enumerate(sorted(urljoin(base, x) for x in urls), 1):
                yield {"sku": sku, "img_index": n, "url": u, "page_url": page_url, "title": title,
                       "product_type": coll, "tags": [], "price": None, "scraped_at": _now()}


# ------------------------------------------------------------------ Khaadi (product pages)
KHAADI_START = ["/ready-to-wear/", "/fabrics/", "/new-in/", "/sale/", "/winter-collection/"]


def _khaadi_product_urls():
    """Same discovery logic as download_khaadi.py, returns {product_id: url}."""
    from bs4 import BeautifulSoup
    base = "https://pk.khaadi.com"
    pending, seen, products = [urljoin(base, p) for p in KHAADI_START], set(), {}
    while pending:
        cu = pending.pop(0)
        if cu in seen:
            continue
        seen.add(cu)
        for page in range(1, 41):
            try:
                soup = BeautifulSoup(_get(cu + (f"?page={page}" if page > 1 else "")).text, "html.parser")
            except requests.HTTPError:
                break
            found = 0
            for a in soup.find_all("a", href=True):
                u = urlunparse(urlparse(urljoin(base, a["href"]))._replace(query="", fragment=""))
                path = urlparse(u).path.lower()
                if not urlparse(u).netloc.endswith("khaadi.com"):
                    continue
                if path.endswith(".html"):
                    pid = os.path.splitext(os.path.basename(urlparse(u).path))[0]
                    if pid not in products:
                        products[pid] = u; found += 1
                elif path.endswith("/") and any(w in path for w in ["ready-to-wear", "fabrics", "new-in", "sale",
                                                                     "winter-collection", "chapter", "signature",
                                                                     "essentials", "khaas", "casuals"]):
                    if u not in seen:
                        pending.append(u)
            if page > 1 and not found:
                break
    return products


def _khaadi_images(html):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    images = set()
    for img in soup.find_all("img"):
        for attr in ["src", "data-src", "data-high-src", "data-zoom-image"]:
            v = img.get(attr, "")
            if "/hi-res/" in v:
                images.add(urljoin("https://pk.khaadi.com", v))
        for attr in ["srcset", "data-srcset"]:
            for item in img.get(attr, "").split(","):
                c = item.strip().split(" ")[0]
                if "/hi-res/" in c:
                    images.add(urljoin("https://pk.khaadi.com", c))
    images = {u.replace("&amp;", "&") for u in images}
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    crumbs = [a.get_text(" ", strip=True) for a in soup.select("nav.breadcrumb a, .breadcrumb a")]
    return images, title, crumbs


def khaadi(root, done):
    local = set(os.listdir(os.path.join(root, "Khaadi_Images")))
    for pid, url in sorted(_khaadi_product_urls().items()):
        folder = _safe_name(pid)
        sku = f"khaadi:{folder}"
        if folder not in local or sku in done:
            continue
        imgs, title, crumbs = _khaadi_images(_get(url).text)
        for n, u in enumerate(sorted(imgs), 1):
            yield {"sku": sku, "img_index": n, "url": u, "page_url": url, "title": title,
                   "product_type": " > ".join(crumbs), "tags": [], "price": None, "scraped_at": _now()}


BRANDS = {"ethnc": ethnc, "sapphire": sapphire, "khaadi": khaadi}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder containing ETHNC_Images/, Khaadi_Images/, Sapphire_Images/")
    ap.add_argument("--brand", choices=list(BRANDS), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="stop after N products (test run)")
    a = ap.parse_args(argv)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        with open(a.out) as fh:
            done = {json.loads(l)["sku"] for l in fh if l.strip()}
    skus = set()
    with open(a.out, "a") as fh:
        for row in BRANDS[a.brand](a.root, done):
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            skus.add(row["sku"])
            if a.limit and len(skus) > a.limit:
                break
    print(f"matched {len(skus)} new SKUs ({len(done)} already done) -> {a.out}")


if __name__ == "__main__":
    main()
