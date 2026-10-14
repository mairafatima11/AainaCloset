# Dataset audit — AainaClosetDataset, 14 Oct 2026

What was on Drive before the pipeline was built, what is wrong with it, and what each finding changed.
Evidence: `manifest.csv`, `verified_pairs.csv`, the three download scripts, and a visual inspection of
137 images from 22 SKUs sampled across all brands and product types (the *gold set*, `data_tools/gold.py`).

## Inventory

| Brand | SKUs | Images | Img/SKU | Resolution | Source format |
|---|---:|---:|---:|---|---|
| ETHNC | 1,334 | 10,218 | 7.7 | 1333×2000 (2:3) | Shopify `products.json` |
| Khaadi | 702 | 4,425 | 6.3 | 800×1200 (2:3) | HTML product pages |
| Sapphire | 477 | 2,147 | 4.5 | 2000×2400 (5:6) | HTML product pages |
| **Total** | **2,513** | **16,790** | | | |

The laptop copy (`OneDrive\Documents\AainaClosetDataset`) has ~60 more Sapphire SKUs (537) than Drive (477): the
Drive upload is incomplete. Files on the laptop are OneDrive *online-only*, so they must be synced before use.

## Findings

**F1 — `verified_pairs.csv` (488 pairs) is not usable as training data.**
It was built by an image-text similarity score, and the scores are low (median model confidence 0.65). On inspection:
- *Khaadi (193 pairs) and Sapphire (0 pairs)*: Khaadi and Sapphire publish **no flat lays**. Every Khaadi "flat_cloth_path" is a photo of a model. Example `1-26-140-C-H`: "model" = seated pose, "flat" = another model shot.
- *ETHNC*: pairs pick the **back** of the garment as the cloth, or a pose with arms raised / garment occluded as the person. Example `boho-dress-e0788`: chosen 09 (arms raised, bicycle in front) + 13 (garment back); correct pair is 02 + 01.
→ Replaced by `data_tools/views.py` + `pairs.py` + human review. The old file is kept, not used.

**F2 — 207 ETHNC SKUs are children's wear photographed on children.**
ETHNC's handle encodes the department in the segment after the style code: `1xx` women, `2xx` unstitched, `3xx/4xx` girls,
`x09` footwear (verified visually: `boutique-suit-e0911-404`, `casual-suit-e1598-402`, `casual-suit-e1835-402` are girls).
→ Excluded (`audience = kids`). Outside the project's scope (women's wear) and the images are of minors.

**F3 — Non-apparel and western products are mixed in.**
ETHNC: 24 footwear SKUs (`heels`, `khola-puri`, `shoes`), 74 western `timelesz` SKUs (jeans, t-shirts, waistcoats).
Khaadi: perfumes (`EDP…`), cushions/home (`C44…`), and one infographic image inside a product gallery.
→ `is_apparel` / `is_ethnic` flags in `catalog.py`; excluded from pairs.

**F4 — Duplicate listings would leak between train and test.**
Khaadi `T-…` codes are re-listings of existing designs (97 SKUs, e.g. `T-A1D1-26-403FE2` = `A1D1-26-403FE2-E`).
ETHNC colourways share a style code (`e0775-…-001` / `e0775-…-901`). One Sapphire code appears under two collections.
→ Split unit is a *design group*, merged further whenever two groups share a byte-identical image (MD5).

**F5 — Flat lays exist only for ETHNC, for roughly 40% of its SKUs, and they are at the END of the gallery.**
In the gold set, 4 of 10 ETHNC SKUs had flats (front, back, trouser, detail — in that order). The brief's assumption
that "all brands publish a flat + model shot per SKU" holds for ETHNC only.
→ Two pair types: `flat` (ETHNC, the core) and `worn` (cloth cut from another model photo of the same SKU, Khaadi/Sapphire).

**F6 — Dupattas frequently cover most of the kameez in Khaadi / Sapphire model shots.**
→ Reject reason `dupatta_occludes` in review; `dupatta_frac_of_torso` measured by SCHP in `masks_gpu.py`. This is itself
evidence for the failure taxonomy: the parser labels the dupatta as `scarf` and the try-on mask ignores it.

**F7 — The scrapers saved images but no metadata.**
No product URL, title, product type or image URL was stored. Publishing the dataset the licence-safe way (URLs +
annotations, not pixels) needs them. → `scraper/metadata.py` re-collects metadata only and reproduces each script's
image ordering so `image_NN` maps back to its URL.

**F8 — `manifest.csv` has an empty `category` for ETHNC/Khaadi and contains OneDrive duplicate files (`image_05 (1).jpg`).**
→ Catalog is rebuilt from the folder tree; `(n)` copies are flagged `filename_dup` and skipped.

## What this means for the numbers
After exclusions: **1,022 ETHNC**, ~560 Khaadi (non-relisted apparel) and 477 Sapphire SKUs are eligible.
Expected candidate pairs ≈ 300–400 `flat` (ETHNC) + several hundred `worn`. The Gate 3 target (≥250 clean pairs) is
reachable with `flat` pairs alone, provided review acceptance stays above ~70%. These are estimates from a 22-SKU
sample. Notebook 02 prints the real counts.
