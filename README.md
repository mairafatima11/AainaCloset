# desi-fit-ai — dataset pipeline

Builds **DesiFit-VTON**, a paired virtual try-on dataset of Pakistani women's ethnic wear, from
brand product images already scraped to Google Drive (`AainaClosetDataset/`, the source of truth;
the laptop copy is a backup).

```
scraped images (Drive) ─► 02_build_dataset.ipynb  [Colab, CPU]
                             catalog → view classification → candidate pairs → review pages
                                                     ▲                            │
                                                     │ you review in the browser  ▼
                             merge reviews → leak-free split → export (VITON-HD layout)
                                                                                  │
                          ─► 03_gpu_masks.ipynb     [Colab T4, once]              ▼
                             agnostic masks · DensePose · SCHP parse · garment cut-outs
                          ─► 04_compare_garment_inputs.ipynb  [Colab T4, ~1 h]
                             stock CatVTON with full / faceless / cutout / flat garment input
                             → choose --worn-cloth, re-export
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
| `data_tools/masks_gpu.py` | CatVTON AutoMasker wrapper, kameez hem extension, SCHP garment cut-outs |
| `data_tools/compare_modes.py` | Stock-CatVTON comparison of garment inputs (SSIM / LPIPS / hem gap inside the mask) |
| `data_tools/stats.py` | Numbers for the datasheet |
| `scraper/metadata.py` | Re-collects product URLs/titles for the images on disk (no image downloads) |
| `notebooks/` | `02_build_dataset` (CPU), `03_gpu_masks` (T4), `04_compare_garment_inputs` (T4) |
| `docs/` | audit, decisions log, review guide, datasheet draft |

## Output format (`DesiFit-VTON-v1/`)
```
train/ val/ test/
   image/<id>.jpg   cloth/<id>.jpg   cloth-mask/<id>.jpg
   cloth_full/  cloth_faceless/  cloth_cutout/      # worn-pair garment variants; cloth/ = the chosen one
   agnostic-mask/<id>_mask.png   image-densepose/<id>.jpg   image-parse/<id>.png
train_pairs.txt  val_pairs.txt  test_pairs.txt  test_pairs_unpaired.txt      # worn pairs (primary)
*_flat_pairs.txt  *_flat_pairs_unpaired.txt                                  # flat-lay pairs
pairs.csv  masks_qa.csv  dataset_stats.md
```
These names match CatVTON's `VITONHDTestDataset` (checked against `inference.py`, Oct 2026): it reads
`test_pairs_unpaired.txt` and `test/{image,cloth,agnostic-mask}` with masks named `<id>_mask.png`, so
`python inference.py --dataset vitonhd --data_root_path DesiFit-VTON-v1 ...` runs on the test split as-is
(add `--eval_pair` for the paired setting). These default lists are the worn pairs. To run the flat-lay
pairs instead, copy `test_flat_pairs_unpaired.txt` over `test_pairs_unpaired.txt` in a separate copy of the folder.

## Where each step runs
| Step | Where | Why |
|---|---|---|
| View classification, pairs, review pages, export | Colab **CPU** runtime (~1.5 h total) | MediaPipe on CPU is fast enough, so no GPU quota is spent |
| Reviewing pairs | your laptop browser | HTML pages work offline |
| Masks / DensePose / garment cut-outs | Colab **T4**, once (~1–2 h) | SCHP + DensePose need a GPU |
| Garment-input comparison | Colab **T4** (~1 h) | 24 pairs × 4 variants of stock CatVTON at 384×512 |
| LoRA fine-tuning (Phase 3) | Colab **T4** (checkpoint every 250 steps, resume after disconnects); Kaggle optional as a second GPU | free T4 is enough if every run can resume |

## Local quick test
```
python3.11 -m venv .venv && . .venv/bin/activate && pip install -r requirements-data.txt
python -m data_tools.classify_all --root /path/to/AainaClosetDataset --out _build/views.csv --limit 100
python -m data_tools.gold --views _build/views.csv
```
