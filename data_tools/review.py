"""Generate self-contained HTML pages for human review of candidate pairs.

Each page holds up to --per-page pairs with every image of the SKU as a thumbnail
strip, so the reviewer can fix a wrong automatic choice with one click instead of
rejecting the whole pair. Decisions are kept in the browser while working and exported
as a CSV ("Download decisions") which is merged back with --merge.

Keys:  A accept   R reject   J / K next / previous   U undo decision
Click a thumbnail -> make it the PERSON image; Shift+click -> make it the CLOTH image.
Reject reasons are one key each (1-6) so they become a usable statistic in the report.
"""
from __future__ import annotations

import argparse
import base64
import glob
import html
import io
import json
import os

import pandas as pd
from PIL import Image

REJECT_REASONS = {
    "1": "cloth_not_matching",     # flat lay is a different item / colour than worn
    "2": "dupatta_occludes",       # dupatta covers most of the kameez in the person image
    "3": "bad_pose",               # side/seated/arms raised slipped through
    "4": "cloth_back_or_detail",   # chosen cloth is the back or a close-up
    "5": "low_quality",            # blur, heavy crop, watermark, collage
    "6": "other",
}


def thumb_b64(path: str, h: int) -> str:
    try:
        im = Image.open(path).convert("RGB")
        im.thumbnail((int(h * 0.8), h))
        buf = io.BytesIO(); im.save(buf, "JPEG", quality=72)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pair review __PAGE__</title><style>
:root{--bg:#f6f5f2;--card:#fff;--ink:#1d1d1f;--mut:#6b6b70;--ok:#1f7a45;--bad:#b3261e;--cl:#2a5bd7;--pe:#b06000}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--card:#1e1e22;--ink:#ececf0;--mut:#9a9aa3}}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif}
header{position:sticky;top:0;background:var(--card);padding:10px 16px;border-bottom:1px solid #8884;z-index:2;display:flex;gap:16px;flex-wrap:wrap;align-items:center}
.pair{background:var(--card);margin:14px 16px;padding:12px;border-radius:8px;border:2px solid transparent}
.pair.cur{border-color:#888}.pair.accept{border-color:var(--ok)}.pair.reject{border-color:var(--bad);opacity:.6}
.main{display:flex;gap:10px;flex-wrap:wrap}.main figure{margin:0;text-align:center}.main img{height:340px;border-radius:4px}
figcaption{font-size:12px;color:var(--mut)}.strip{display:flex;gap:4px;overflow-x:auto;margin-top:8px}
.strip img{height:110px;cursor:pointer;border:3px solid transparent;border-radius:3px}
.strip img.c{border-color:var(--cl)}.strip img.p{border-color:var(--pe)}
.meta{color:var(--mut);font-size:12px;margin-bottom:6px;word-break:break-all}
button{font:inherit;padding:4px 10px;border-radius:5px;border:1px solid #8886;background:var(--card);color:var(--ink);cursor:pointer}
kbd{border:1px solid #8886;border-radius:3px;padding:0 4px;font-size:12px}
</style></head><body>
<header><b>Page __PAGE__</b><span id="stat"></span>
<span><kbd>A</kbd> accept <kbd>R</kbd>+<kbd>1-6</kbd> reject <kbd>J</kbd>/<kbd>K</kbd> move · click thumb = person · shift+click = cloth</span>
<span style="color:var(--mut)">reasons: 1 cloth≠worn · 2 dupatta covers · 3 bad pose · 4 back/detail · 5 low quality · 6 other</span>
<button onclick="dl()">Download decisions</button></header>
<div id="list"></div>
<script>
const PAIRS=__PAIRS__, REASONS=__REASONS__, KEY='review___PAGE__';
let st={}; try{st=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
let cur=0, waitReason=false;
function save(){try{localStorage.setItem(KEY,JSON.stringify(st))}catch(e){} stat()}
function stat(){const v=Object.values(st);document.getElementById('stat').textContent=
 `${v.filter(x=>x.decision=='accept').length} accepted · ${v.filter(x=>x.decision=='reject').length} rejected · ${PAIRS.length} total`}
function render(){const L=document.getElementById('list');L.innerHTML='';
 PAIRS.forEach((p,i)=>{const s=st[p.pair_id]||{};const cl=s.cloth||p.cloth, pe=s.person||p.person;
  const d=document.createElement('div');d.className='pair'+(i==cur?' cur':'')+(s.decision?' '+s.decision:'');d.id='p'+i;
  const img=r=>p.imgs[r]?`data:image/jpeg;base64,${p.imgs[r]}`:'';
  d.innerHTML=`<div class="meta">${p.pair_id} · ${p.brand} · ${p.pair_type}/${p.category} · ${p.sku} · score ${p.target_score}${s.decision?' · <b>'+s.decision+(s.reason?' ('+s.reason+')':'')+'</b>':''}</div>
  <div class="main"><figure><img src="${img(cl)}"><figcaption style="color:var(--cl)">CLOTH ${cl.split('/').pop()}</figcaption></figure>
  <figure><img src="${img(pe)}"><figcaption style="color:var(--pe)">PERSON ${pe.split('/').pop()}</figcaption></figure></div>
  <div class="strip">${p.all.map(r=>`<img data-r="${r}" class="${r==cl?'c':''}${r==pe?' p':''}" src="${img(r)}" title="${r.split('/').pop()}">`).join('')}</div>`;
  d.querySelectorAll('.strip img').forEach(t=>t.onclick=e=>{cur=i;const s2=st[p.pair_id]=st[p.pair_id]||{};
   if(e.shiftKey)s2.cloth=t.dataset.r;else s2.person=t.dataset.r;save();render()});
  d.onclick=()=>{if(cur!=i){cur=i;render()}};L.appendChild(d)});stat()}
function go(i){cur=Math.max(0,Math.min(PAIRS.length-1,i));render();document.getElementById('p'+cur).scrollIntoView({block:'center'})}
document.onkeydown=e=>{const p=PAIRS[cur];if(!p)return;const s=st[p.pair_id]=st[p.pair_id]||{};
 if(waitReason&&REASONS[e.key]){s.decision='reject';s.reason=REASONS[e.key];waitReason=false;save();go(cur+1);return}
 waitReason=false;
 if(e.key=='a'){s.decision='accept';delete s.reason;save();go(cur+1)}
 else if(e.key=='r'){waitReason=true}
 else if(e.key=='u'){delete s.decision;delete s.reason;save();render()}
 else if(e.key=='j')go(cur+1);else if(e.key=='k')go(cur-1)}
function dl(){const rows=[['pair_id','decision','reason','cloth','person','reviewer','page']];
 const who=prompt('Reviewer name (maira / mihab)')||'';
 PAIRS.forEach(p=>{const s=st[p.pair_id]||{};rows.push([p.pair_id,s.decision||'',s.reason||'',s.cloth||p.cloth,s.person||p.person,who,'__PAGE__'])});
 const b=new Blob([rows.map(r=>r.map(x=>'"'+String(x).replace(/"/g,'""')+'"').join(',')).join('\\n')],{type:'text/csv'});
 const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='review___PAGE__.csv';a.click()}
render();
</script></body></html>"""


def make_pages(pairs: pd.DataFrame, cat: pd.DataFrame, root: str, out_dir: str, per_page: int = 60,
               thumb_h: int = 420):
    os.makedirs(out_dir, exist_ok=True)
    by_sku = cat.sort_values("img_index").groupby("sku").rel_path.apply(list)
    pages = []
    for k in range(0, len(pairs), per_page):
        chunk = pairs.iloc[k: k + per_page]
        page = f"{k // per_page + 1:03d}"
        items = []
        for _, p in chunk.iterrows():
            allp = [r for r in by_sku.get(p.sku, []) if "(" not in os.path.basename(r)]
            imgs = {r: thumb_b64(os.path.join(root, r), thumb_h) for r in allp}
            items.append({c: (None if pd.isna(p[c]) else (p[c].item() if hasattr(p[c], "item") else p[c]))
                          for c in ["pair_id", "brand", "sku", "pair_type", "category", "cloth", "person", "target_score"]}
                         | {"all": allp, "imgs": imgs})
        htmlp = (PAGE.replace("__PAIRS__", json.dumps(items)).replace("__REASONS__", json.dumps(REJECT_REASONS))
                 .replace("__PAGE__", page))
        fn = os.path.join(out_dir, f"review_{page}.html")
        with open(fn, "w", encoding="utf-8") as fh:
            fh.write(htmlp)
        pages.append(fn)
        print(f"{fn}  ({len(chunk)} pairs, {os.path.getsize(fn) / 1e6:.1f} MB)", flush=True)
    return pages


def merge_reviews(pairs: pd.DataFrame, review_glob: str) -> pd.DataFrame:
    """Apply review CSVs (downloaded from the pages) onto the candidate pairs."""
    files = sorted(glob.glob(review_glob))
    if not files:
        raise FileNotFoundError(review_glob)
    rv = pd.concat([pd.read_csv(f) for f in files]).drop_duplicates("pair_id", keep="last")
    m = pairs.merge(rv, on="pair_id", how="left", suffixes=("", "_rev"))
    m["cloth"] = m.cloth_rev.fillna(m.cloth)
    m["person"] = m.person_rev.fillna(m.person)
    m["edited"] = (m.cloth_rev.notna() & (m.cloth_rev != pairs.set_index("pair_id").cloth.reindex(m.pair_id).values)) | \
                  (m.person_rev.notna() & (m.person_rev != pairs.set_index("pair_id").person.reindex(m.pair_id).values))
    return m.drop(columns=["cloth_rev", "person_rev"])


def kappa(csv_a: str, csv_b: str) -> dict:
    """Cohen's kappa between two reviewers on the same page (accept vs reject)."""
    a = pd.read_csv(csv_a).set_index("pair_id").decision
    b = pd.read_csv(csv_b).set_index("pair_id").decision
    both = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    if both.empty:
        return {"n": 0}
    po = (both.a == both.b).mean()
    pe = sum((both.a == c).mean() * (both.b == c).mean() for c in set(both.a) | set(both.b))
    k = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return {"n": len(both), "observed_agreement": round(po, 3), "kappa": round(k, 3),
            "disagreements": both[both.a != both.b].index.tolist()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--per-page", type=int, default=60)
    ap.add_argument("--merge", default=None, help="glob of downloaded review CSVs -> writes pairs_reviewed.csv")
    ap.add_argument("--kappa", nargs=2, metavar=("CSV_A", "CSV_B"), help="agreement between two reviewers")
    a = ap.parse_args(argv)
    if a.kappa:
        print(kappa(*a.kappa)); return
    pairs = pd.read_csv(a.pairs)
    if a.merge:
        m = merge_reviews(pairs, a.merge)
        out = os.path.join(os.path.dirname(a.pairs), "pairs_reviewed.csv")
        m.to_csv(out, index=False)
        print(m.decision.value_counts(dropna=False).to_string()); print("->", out)
        return
    make_pages(pairs, pd.read_csv(a.catalog), a.root, a.out_dir, a.per_page)


if __name__ == "__main__":
    main()
