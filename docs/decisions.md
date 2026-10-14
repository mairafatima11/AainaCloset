# Decisions log

One line per choice: date · decision · reason. Add to it every time you choose something.

| Date | Decision | Reason |
|---|---|---|
| 2026-10-14 | Do not use `verified_pairs.csv` | Khaadi "flats" are model photos; ETHNC pairs pick garment backs and occluded poses (audit F1). |
| 2026-10-14 | View classification with MediaPipe Pose + FaceDetection (mediapipe 0.10.14), not CLIP | Models are bundled in the wheel: runs on CPU, no downloads, no GPU quota. Evaluated on 137 hand-labelled images: 87.6% image accuracy, flats 10/10 with 0 false positives, correct target in 21/22 SKUs. |
| 2026-10-14 | Selfie segmentation is not used to decide "person present" | It fires on garments alone (flat lays scored up to 0.3 person area). |
| 2026-10-14 | "Full length" allows low ankle visibility (>0.25) | Long kameez / gharara hems cover the ankles; the landmark only needs to be inside the frame. |
| 2026-10-14 | Front flat = first flat in gallery order | ETHNC orders flats front → back → trouser → detail (4/4 SKUs in the gold set). Reviewer can override. |
| 2026-10-14 | Trouser flats detected by background showing between the legs (`leg_gap > 0.5`) | Bounding-box shape did not separate trousers from long dresses; leg gap separated them perfectly (0.92–0.96 vs ≤0.03). |
| 2026-10-14 | Exclude ETHNC department 3xx/4xx (girls) and x09 (footwear) | Out of scope (women's wear); images of minors (audit F2/F3). |
| 2026-10-14 | Exclude Khaadi `T-` re-listings, `EDP`, `C44` | Duplicates / perfume / home textiles (audit F3/F4). |
| 2026-10-14 | Two pair types, `flat` and `worn`, never mixed in one metric | Only ETHNC has flat lays. `worn` cloth is already shaped by a body, an easier and different task. |
| 2026-10-14 | Split by `split_group` (design group + MD5-linked groups), hash-assigned 75/10/15 | Prevents colourways / re-listings leaking across splits; hash assignment keeps splits stable as data grows. |
| 2026-10-14 | Export at 768×1024, fit + pad with the photo's own background colour | Matches CatVTON/VITON-HD; no stretching, no cropping of hems or heads. |
| 2026-10-14 | Kameez agnostic mask = CatVTON 'upper' ∪ dilated SCHP garment labels | 'upper' stops at the hip → would teach hem truncation. **To verify on the first QA sheet in notebook 03.** |
| 2026-10-14 | Every pair goes through human review | Automatic selection is ~95% right per SKU on the gold set; 5% bad pairs is still ~20 poisoned training samples. |
| 2026-10-14 | Preprocessing (masks, DensePose, garment cut-outs) runs once, stored on Drive | Deterministic and expensive; never recompute during training. |
