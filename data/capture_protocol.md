# Capture protocol v1.0

Status: active. Frozen as of 2026-09-16.

Do not change this document mid collection. If something must change, bump the version, write a new section, and record the date the change took effect. Every image carries the protocol version it was shot under.

## What we are capturing

For each physical garment we capture two sets:

- Packing shots: what a seller would photograph when packing the order. Clean reference.
- Rider shots: what a delivery rider would photograph at the customer's doorstep. Messy query.

The value of the dataset is the gap between the two. Rider shots that look like packing shots are worthless to us.

## Garment selection

- Target 200 to 400 garments overall. First batch is 30.
- Categories: kurtas, tees, shirts, jeans, dresses. Roughly balanced, no category under 10 percent.
- Deliberate hard negatives, at least a third of the set:
  - 2 to 3 identical units of the same design (same design_id, different unit_id)
  - the same design in a different color or size
  - lookalikes from different brands (same plain black tee, same generic blue kurta)
- Record where each garment came from (own, family, shop, seller) in the notes column.

## Packing shots: 3 to 4 per garment

Shot on any phone, good conditions. These stand in for seller catalog photos.

1. Full front, flat lay, garment smoothed out, plain light surface, even daylight
2. Full back, flat lay
3. Close up of the brand tag or care label
4. Close up of the main print, logo, or embroidery (skip if the garment is plain)

Rules: whole garment in frame for shots 1 and 2, no hands, no shadow across the garment, phone roughly parallel to the surface.

## Rider shots: 3 to 6 per garment

Shot on a cheap Android phone, held in the hand, in realistic doorstep conditions. At least 2 phones across the dataset, 3 is better. Do not clean up the garment first.

Every garment gets at least one shot from each of these three:

1. Folded as a customer would hand it back
2. Crumpled or bunched, held up in one hand
3. Inside a transparent polybag, with visible glare from a light source

Then add 1 to 3 more sampled from:

4. Dim indoor light (corridor, evening room, no flash)
5. Motion blur (take the shot while moving the phone slightly)
6. Partial view (garment fills the frame, edges cut off)
7. Tag visible close up, handheld
8. Busy background (floor, doorstep, sofa, other objects in frame)

Across the whole dataset aim for roughly: 40 percent in polybag, 35 percent low light, 50 percent crumpled, 30 percent with tag visible. Track these as you go, do not batch all of one condition at the end.

## Phones

Register every phone in `data/phones.csv` before shooting with it. At least one phone must be a budget Android in the Rs 8k to 12k range. Never shoot rider photos on a flagship phone only, that quietly deletes the domain gap we are trying to study.

## Naming and storage

Images never enter git. They live in the shared image store, one folder per garment:

```
images/
  g0001/
    g0001_pack_01.jpg
    g0001_pack_02.jpg
    g0001_rider_01.jpg
    g0001_rider_02.jpg
```

- `garment_id`: `g` plus 4 digits, zero padded, unique per physical item, never reused.
- Filename: `{garment_id}_{pack|rider}_{NN}.jpg`, NN starting at 01 per shot type.
- Shoot at the phone's default resolution. Do not crop, rotate, filter, or "fix" images. Do not let a cloud service re compress them, check that uploads are originals.

## Logging

One row per image in `data/metadata.csv`, filled in the same session as the shoot, not from memory later. See `data/metadata_schema.md` for the columns and allowed values. Rows with missing condition flags are unusable for the per condition breakdown tables, which is our key result.

## Definition of done for one garment

- 3 to 4 packing shots, 3 to 6 rider shots
- all three mandatory rider conditions covered
- every image has a row in metadata.csv with all required fields
- design_id and unit_id assigned, lookalike_group set if it applies
