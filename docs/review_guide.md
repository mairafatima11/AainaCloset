# Pair review guide

Every candidate pair is checked by a person before it enters the dataset. Budget ~4 seconds per pair
once you are used to it: 60 pairs/page ≈ 5 minutes, ~700 pairs ≈ 1 hour each if split between two people.

## Open a page
Download `_build/review/review_NNN.html` from Drive and open it in Chrome or Edge. It works offline;
your decisions are remembered in that browser while you work, but **only the downloaded CSV counts**.

| Key | Action |
|---|---|
| `A` | accept, go to next |
| `R` then `1`–`6` | reject with a reason, go to next |
| `J` / `K` | next / previous pair |
| `U` | undo the decision on the current pair |
| click thumbnail | make it the **PERSON** image (orange) |
| Shift + click thumbnail | make it the **CLOTH** image (blue) |

When the page is done: **Download decisions**, type your name, upload the CSV to `_build/review_csv/`.
Name it `review_NNN_<name>.csv`.

## Accept only if all of these hold
1. **Same garment.** CLOTH and PERSON show the same item, in the same colour and print.
2. **Cloth is the front.** For `flat` pairs, the cloth is the *front* of the garment (neckline / placket visible), not the back, not a close-up.
3. **Person is usable.** Facing the camera, head to (roughly) ankles in frame, not seated, arms not raised above the shoulders, garment not hidden by a bag or a hand.
4. **Garment is visible.** For kameez pairs, at least ~2/3 of the kameez front is visible. A dupatta over one shoulder is fine; a dupatta wrapped over the chest and stomach is not.
5. **Image quality.** Not blurred, not a collage or infographic, no text over the garment.

If the automatic choice is wrong but a better image exists in the strip, **fix it with a click** instead of rejecting.

## Reject reasons
| Key | Reason | Typical case |
|---|---|---|
| 1 | `cloth_not_matching` | flat lay is the dupatta/trouser only, or a different colourway |
| 2 | `dupatta_occludes` | dupatta covers most of the kameez in the person image |
| 3 | `bad_pose` | side view, seated, arms up, walking stride that hides the hem |
| 4 | `cloth_back_or_detail` | only the back or a fabric close-up exists as cloth |
| 5 | `low_quality` | blur, heavy crop, watermark, collage |
| 6 | `other` | anything else, then write it in `docs/decisions.md` |

The reject reasons are data: their counts go into the dataset chapter (e.g. "38% of Khaadi candidates rejected
because the dupatta occluded the kameez").

## Agreement
Both reviewers do page 001 independently first. Notebook 02 prints Cohen's κ. If κ < 0.6, talk through the
disagreements it lists, tighten the rules above, and redo page 001 before splitting the rest.
