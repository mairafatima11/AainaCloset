# desi-fit-ai — dataset pipeline

Builds **DesiFit-VTON**, a paired virtual try-on dataset of Pakistani women's ethnic wear, from
brand product images already scraped to Google Drive (`AainaClosetDataset/`).

```
scraped images (Drive) ─► 02_build_dataset.ipynb  [Colab, CPU]
                             catalog → view classification → candidate pairs → review pages
                                                     ▲                            │
                                                     │ you review in the browser  ▼
                             merge reviews → leak-free split → export (VITON-HD layout)
                                                                                  │
                          ─► 03_gpu_masks.ipynb     [Colab T4 / Kaggle, once]     ▼
                             agnostic masks · DensePose · SCHP parse · worn-pair garments
                                                                                  │
                                                     DesiFit-VTON-v1/  ◄──────────┘
```

## Layout
| Path | What |
|---|---|
| `data_tools/catalog.py` | Folder tree → one row per image; brand rules (kids, non-apparel, design groups) |
| `data_tools/views.py` | MediaPipe view classifier (flat / flat_lower / detail / mff / mo) |
| `data_tools/classify_all.py` | Batch classification, resumable, MD5 per file |
| `data_tools/gold.py` | 137 hand-labelled images + evaluation |
| `data_tools/pairs.py` | Candidate `flat` and `worn` pairs per SKU, leakage groups |
| `data_tools/review.py` | Self-contained HTML review pages, merge, Cohen's κ |
| `data_tools/split.py` | Hash-based split by design group with leak assertions |
| `data_tools/export.py` | 768×1024 export, flat-lay garment cut-out, pairs files |
| `data_tools/masks_gpu.py` | CatVTON AutoMasker wrapper, kameez hem extension, worn-pair garment extraction |
| `data_tools/stats.py` | Numbers for the datasheet |
| `scraper/metadata.py` | Re-collects product URLs/titles for the images on disk (no image downloads) |
| `notebooks/` | `02_build_dataset`, `03_gpu_masks` |
| `docs/` | audit, decisions log, review guide, datasheet draft |

## Output format (`DesiFit-VTON-v1/`)
```
train/ val/ test/
   image/<id>.jpg   cloth/<id>.jpg   cloth-mask/<id>.jpg
   agnostic-mask/<id>_mask.png   image-densepose/<id>.jpg   image-parse/<id>.png
train_pairs.txt  val_pairs.txt  test_pairs.txt  test_pairs_unpaired.txt
*_worn_pairs.txt                  # worn pairs listed separately
pairs.csv  masks_qa.csv  dataset_stats.md
```
These names match CatVTON's `VITONHDTestDataset` (checked against `inference.py`, Oct 2026): it reads
`test_pairs_unpaired.txt` and `test/{image,cloth,agnostic-mask}` with masks named `<id>_mask.png`, so
`python inference.py --dataset vitonhd --data_root_path DesiFit-VTON-v1 ...` runs on the test split as-is
(add `--eval_pair` for the paired setting). The worn pairs are kept in separate lists. To benchmark them,
point a copy of `test_pairs_unpaired.txt` at `test_worn_pairs_unpaired.txt`.

## Where each step runs
| Step | Where | Why |
|---|---|---|
| View classification, pairs, review pages, export | Colab **CPU** runtime (~1.5 h total) | MediaPipe on CPU is fast enough, so no GPU quota is spent |
| Reviewing pairs | your laptop browser | HTML pages work offline |
| Masks / DensePose / worn garments | Colab **T4**, once (~1–2 h) | SCHP + DensePose need a GPU |
| LoRA fine-tuning (Phase 3) | **Kaggle** (T4/P100, 30 h/week per account, "Save & Run All") | 12 h headless sessions; Colab free disconnects |

## Local quick test
```
python3.11 -m venv .venv && . .venv/bin/activate && pip install -r requirements-data.txt
python -m data_tools.classify_all --root /path/to/AainaClosetDataset --out _build/views.csv --limit 100
python -m data_tools.gold --views _build/views.csv
```
