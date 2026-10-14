# Datasheet — DesiFit-VTON v1 (draft)

Format: *Datasheets for Datasets* (Gebru et al., 2021). Fields marked **[fill]** are produced by
`python -m data_tools.stats` after the build, or need a decision from the team.

## Motivation
- **Purpose.** A paired image dataset for training and evaluating 2D virtual try-on of Pakistani women's ethnic wear (kameez, shalwar/trouser, sets with dupatta). Public try-on datasets (VITON-HD, DressCode) contain fitted Western garments almost exclusively. Loose, long and layered South Asian garments are not represented.
- **Creators.** Maira Fatima and Mihab Khan, Information Technology University, Lahore. Final-year project supervised by Dr. Mehwish Ghafoor.
- **Funding.** None.

## Composition
- **Instances.** Each instance is a *pair*: a person image and a garment image of the same product, plus masks. Two types:
  - `worn` (primary, all brands): the garment image is a *different* model photo of the same product. It ships in three versions: as published, with the head greyed out, or with the garment cut out. This is the input a real shop page gives you.
  - `flat` (secondary, ETHNC only): the garment image is the brand's flat-lay photo. It is reported separately and also used for a same-product flat vs. worn comparison.
- **Counts.** **[fill from stats: pairs per brand × type × category × split]**
- **Source pool.** 16,790 product images of 2,513 products from three Pakistani brands' public online stores (ETHNC, Khaadi, Sapphire), collected October 2026.
- **Per instance.** `image/` person 768×1024; `cloth/` garment image 768×1024 (worn: the chosen variant; `cloth_full/`, `cloth_faceless/`, `cloth_cutout/` all kept); `cloth-mask/`; `agnostic-mask/`; `image-densepose/`; `image-parse/` (SCHP-LIP labels); a row in `pairs.csv` with brand, product code, design group, garment category, pair type, review decision, split and image URLs.
- **Labels.** View labels (automatic, MediaPipe), pair selection (automatic, then accepted / corrected / rejected by a human reviewer with a coded reason), masks (automatic: SCHP + DensePose via CatVTON AutoMasker, plus a kameez hem extension).
- **Splits.** Train / val / test ≈ 75 / 10 / 15 by *design group*. Colourways and re-listings of one design never cross splits.
- **Excluded on purpose.** Children's wear and every image of a minor; footwear, perfume, home textiles; western-line products; Khaadi re-listings; byte-identical duplicates.
- **Known gaps and biases.**
  - Professional models only: young, slim, light-to-medium skin tones, studio or editorial lighting.
  - ETHNC is the only source of flat lays, so the `flat` set reflects one brand's aesthetic.
  - Dupattas are rarely visible in `flat` pairs. In `worn` pairs they often occlude the kameez.
  - Front-facing standing poses only.
- **Confidential / sensitive data.** The images are public commercial product photos of professional models. No private individuals appear. The self-shot test set, if added, is covered separately by signed consent and is not released.

## Collection process
- **How.** Public product listing endpoints (`/products.json` for ETHNC) and product pages (Khaadi, Sapphire), using scripts in `scraper/`. Rate-limited at 0.5–2 s between requests.
- **Who / when.** The authors, 9–10 October 2026. Metadata re-collected with `scraper/metadata.py` on **[fill date]**.
- **Ethics review.** **[fill: ITU response on whether ethics review is required]**

## Preprocessing
1. **View classification.** Each image is labelled flat / flat_lower / detail / model-front-full / model-other. The classifier was evaluated on 137 hand-labelled images: 87.6% accuracy, flat lays 10/10 with no false positives, correct try-on target in 21/22 products.
2. **Pairing.** Candidate pairs are selected automatically, then reviewed by two people. Agreement is measured on a shared page: κ = **[fill]**.
3. **Export.** Images are resized to 768×1024 and padded with the photo's own background colour, never stretched. The flat-lay garment is cut out with a background threshold plus GrabCut.
4. **Masks.** CatVTON AutoMasker (SCHP + DensePose). For kameez, the 'upper' mask is extended to the hem using SCHP garment labels.
- **Raw data.** The original scraped images are kept privately on the authors' Drive. They are not redistributed.

## Uses
- **Intended.** Fine-tuning and evaluating virtual try-on models, and studying how Western-trained models fail on South Asian garments.
- **Not intended.** Commercial deployment, identifying or profiling the models, or generating images of real people without consent.
- **Licence constraints of the tooling.** CatVTON, IDM-VTON, StableVITON and VITON-HD are research or non-commercial only (CC BY-NC-SA 4.0 / research-only).

## Distribution
- **What is released.** Code, `pairs.csv`, image URLs, review decisions, masks and annotations under **[fill: CC BY-NC 4.0 suggested]**.
- **What is not released.** Brand photographs. They remain the brands' copyright, and users re-download them from the listed URLs for non-commercial research.
- **Link rot.** Product pages disappear when stock sells out. The release records the scrape date, and the authors can provide the exact file list for verification on request.

## Maintenance
- **Maintainers.** The authors, until project submission (May 2027). Contact: **[fill]**.
- **Takedown.** Any brand or model may request the removal of URLs and annotations. Requests are handled within 14 days, and every change is versioned.
