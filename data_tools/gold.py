"""Hand-labelled gold set for the view classifier (137 images, 22 SKUs, 3 brands).

Labelled by eye from contact sheets on 14 Oct 2026 (Claude, to be re-checked by Maira).
Paths are relative to the AainaClosetDataset root, so the evaluation can be re-run on Drive:

    python -m data_tools.gold --views _build/views.csv

Coarse classes: flat (incl. trouser flats), detail (close-up / headless crop), mff (model,
front-facing, full length), mo (model, other view), non (not apparel / infographic).
Result at time of writing: 87.6% image accuracy, 10/10 flats found with 0 false flats,
correct try-on target picked for 21/22 SKUs.

This set was ALSO used to tune the rules, so it is optimistic. Maira: label a fresh
~150 images the same way (different SKUs) and report that number in the methods chapter.
"""
import argparse

import pandas as pd

G = {
'ETHNC_Images/boho-dress-e0788-119-902-902_7949818495078': 'flat mff mo mo mo mo mo mo mo mo mo mo flat detail',
'ETHNC_Images/boutique-suit-e0911-404-313-313-boutique-suit-e0911-304-313-313_7859919781990': 'mff mo mo mff mo detail detail',
'ETHNC_Images/boutique-suit-e0977-404-328-328-boutique-suit-e0977-304-328-328_7950159872102': 'mo mff mo mo mo mo',
'ETHNC_Images/casual-kurti-e9619-102-616-616_8099982377062': 'mo mff mo mo mo mo flat detail flat',
'ETHNC_Images/casual-suit-e1598-402-416-416-casual-suit-e1598-302-416-416_8005384175718': 'mff mo mo mo mo mo flat flat detail flat',
'ETHNC_Images/casual-suit-e1835-402-405-405-casual-suit-e1835-302-405-405_8172136333414': 'mo mo mff mo mo mo mo mo flat flat flat detail',
'ETHNC_Images/casual-suit-e9649-102-523-523_8099984310374': 'mff mo mo mo detail',
'ETHNC_Images/casual-trouser-e9735-102-002-002_8134925353062': 'mo mo mo',
'ETHNC_Images/rozana-suit-e4734-103-901-901_8099969335398': 'mff mo mo mo detail',
'ETHNC_Images/rozana-suit-e4808-103-218-218_8099972776038': 'mff mo mo mo detail',
'Khaadi_Images/1-26-140-C-H-VG_MULTI': 'mff mo mo mo mo mff',
'Khaadi_Images/2-26-403-C-C1-VG_MULTI': 'mff mo mff mff mff mo mo mo',
'Khaadi_Images/26-01-2E1-09BA-VG_MULTI': 'mff mo',
'Khaadi_Images/A1D1-26-403FE2-E-VG_MULTI': 'mo mff mff mo mo mo mo mo',
'Khaadi_Images/A33-26-210FJ1-E-VG_MULTI': 'mo mo non mo mo mff',
'Khaadi_Images/C44-26-203AAC-VG_MULTI': 'non non non non non',
'Sapphire_Images/new-in/U3PEDY26V968.html': 'mff mff mo detail mo',
'Sapphire_Images/ready-to-wear/1PCCCASV912S_999.html': 'detail mff mo mo',
'Sapphire_Images/ready-to-wear/2PCSTCRC063S_999.html': 'mo mo mo mff',
'Sapphire_Images/ready-to-wear/2SFS26FMV616_999.html': 'detail mo mo mff',
'Sapphire_Images/unstitched/U3PDYS26V946.html': 'mo mo mo mo mff',
'Sapphire_Images/unstitched/U3PESG26V941.html': 'mff detail mo mo',
}
def labels():
    out={}
    for k,v in G.items():
        for i,l in enumerate(v.split(),1): out[f'{k}/image_{i:02d}.jpg']=l
    return out


def evaluate(views: pd.DataFrame) -> dict:
    L = labels()
    v = views[views.rel_path.isin(L)].copy()
    v["gold"] = v.rel_path.map(L).replace({"non": "detail"})
    v["pred"] = v.label.replace({"flat_lower": "flat"})
    v["sku"] = v.rel_path.str.rsplit("/", n=1).str[0]
    ok = 0
    for _, g in v.groupby("sku"):
        m = g[g.pred == "mff"].sort_values("target_score", ascending=False)
        has = (g.gold == "mff").any()
        ok += (m.empty and not has) or (not m.empty and m.iloc[0].gold == "mff")
    flat_tp = ((v.pred == "flat") & (v.gold == "flat")).sum()
    return {"n_images": len(v), "accuracy": round((v.pred == v.gold).mean(), 3),
            "flat_recall": round(flat_tp / max(1, (v.gold == "flat").sum()), 3),
            "flat_precision": round(flat_tp / max(1, (v.pred == "flat").sum()), 3),
            "target_pick_correct": f"{ok}/{v.sku.nunique()}",
            "confusion": pd.crosstab(v.gold, v.pred)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--views", required=True)
    r = evaluate(pd.read_csv(ap.parse_args().views))
    print(r.pop("confusion").to_string()); print(r)
