# CLAUDE.md

Context for Claude Code. Read this before doing any work in this repo.

## 1. Project in one paragraph

We are building a computer vision system that verifies ecommerce returns in India. When a garment is packed, the seller photographs it. When the customer returns it, the delivery rider photographs it at the doorstep during the existing reverse pickup QC step. Our model compares the two sets of photos and says whether the returned item is the same product (and ideally the same physical unit) that was shipped, a different item, or uncertain. The goal is to catch swap fraud (customer returns a different, cheaper, used, or fake item, or an empty polybag) before the rider accepts the pickup, and to give sellers visual evidence for marketplace claims.

This is a two person student team project. It is both a CV research project (a real domain gap problem plus our own dataset) and an app we intend to publish.

## 2. Problem and market context

- Swap fraud is a known pain for Indian fashion sellers on Myntra, AJIO, Meesho, Flipkart, and Amazon. Sellers ship the right garment and get back a different or worn item, an empty polybag, or junk. Claims are often rejected for lack of proof. Order values are low (a Rs 499 kurta is typical on Meesho), so any per check cost must be tiny.
- Fraud runs both ways: there are documented cases of fake sellers filing false wrong item claims against platforms.
- In India, return QC mostly happens at the customer's doorstep by the courier rider, not in a warehouse. This shapes the product.
- Logistics landscape: Delhivery (acquired Ecom Express in 2025), Shadowfax, Xpressbees, and Blue Dart carry most third party ecommerce parcels. Amazon and Flipkart/Myntra (Ekart) largely use in house networks. Meesho routes through its Valmo platform. D2C brands often ship through aggregators like Shiprocket and ClickPost. Porter is intra city on demand transport, not a core ecommerce parcel network.
- Delhivery's reverse QC is a rider checklist (SKU, brand, color, size, item images, used/damaged/price tag checks). Riders tap through it and upload photos. We found no evidence that anyone automatically checks rider photos against what the seller shipped. That is our opening.
- Shadowfax has already built its own AI image QC, so it is a competitor, not a customer.
- US competitors (for reference, not our market): Two Boxes (AI returns processing for 3PLs, enterprise and mid market, ships hardware), Happy Returns Return Vision (UPS), ReturnPro plus Clarity (X-ray plus CV). Catalog level matching alone is not a differentiator anymore.
- Indian seller side competitor: TrackVid and similar tools record packing videos linked to order IDs as claim evidence. As far as we found, they store video but do not do visual matching.

### Target customers (in order)

1. Mid size Indian fashion D2C brands and marketplace sellers (they feel the loss directly and already pay for claim protection tools).
2. Shipping aggregators (Shiprocket, ClickPost) via API, plugging into the existing rider QC photo flow.
3. Carriers without their own AI QC (Delhivery, Xpressbees) later.

Not first customers: Myntra, Meesho, Amazon, Flipkart (they build in house), Zara and H&M India (global HQ run systems).

### Product shape

1. Seller photographs item at packing (linked to order ID).
2. Rider takes the doorstep QC photos they already take.
3. Our API compares them and returns: MATCH, DIFFERENT_PRODUCT, SUSPICIOUS, or RETAKE/UNSURE.
4. Seller gets a claim ready evidence packet.

Optional physical anchor: seller attaches a cheap tag with a unique printed QR and random pattern at packing. Rider photographs it. Check that the code matches the order AND that the tag is on a garment matching the product.

## 3. The core CV problem

Reference photos are clean packing shots. Query photos are taken by riders on cheap Android phones, in bad lighting, with the garment folded or crumpled, often inside a glossy polybag, with blur and partial views. Matching across that domain gap is the research problem. The project must not be a thin wrapper around a pretrained model.

Split the task into three questions of increasing difficulty and report each separately:

1. Garment present and right category? (not a brick, empty bag, phone) Easy.
2. Same product? (same design, color, print) Moderate. Main domain gap problem.
3. Same physical unit? Very hard for mass produced clothing. Identical units from the same batch may be indistinguishable. Unit cues: print placement relative to seams, tag position, thread tails, defects, wear. Be honest in results about where this fails. The QR tag anchor is the practical fallback.

## 4. Technical approach (build in layers, measure each)

0. Capture quality gate. Blur (Laplacian variance), brightness, garment detected. Fail means retake. Cheap and high impact.
1. Segmentation. Isolate garment from background, hands, polybag. Garment segmentation model or SAM prompted with a detector box. Track polybag glare as a failure case.
2. Zero shot baselines. DINOv2 and CLIP embeddings, cosine similarity. Must be reported before any claims of improvement.
3. Fine tuned metric learning (global match). Pretrain/fine tune DINOv2 on DeepFashion2 consumer to shop pairs (also DeepFashion Consumer to Shop, Street2Shop), then on our data. Contrastive loss (InfoNCE or ArcFace) with hard negative mining (identical designs, lookalikes, same design different color).
4. Local feature matching (fine detail). SuperPoint plus LightGlue or LoFTR on crops of tags, logos, prints, embroidery. Small patches stay roughly rigid under crumpling. Best shot at unit level cues.
5. Multi view set matching. Rider has about 3 photos, seller about 4. Compare sets: max pairwise score, or a small attention head over all pairs.
6. Fusion and calibration. Combine global score, local match count, QR/tag check, category check. Temperature scaling. Thresholds set for a target false positive rate. Explicit uncertain band that routes to retake or human review.

## 5. Dataset protocol

Our own dataset is a core contribution. Nobody else has rider quality paired garment photos. The operational detail lives in `data/capture_protocol.md`. Summary:

- Garments: 200 to 400 items: kurtas, tees, shirts, jeans, dresses. Sourced from family, local shops, friendly sellers.
- Hard negatives on purpose: 2 to 3 identical units of the same design; same design in different color/size; lookalikes from different brands.
- Packing shots: 3 to 4 per item, flat lay, plain surface, good light.
- Rider shots: 3 to 6 per item on at least 2 to 3 cheap Android phones (Rs 8k to 12k range). Vary: folded, crumpled, inside polybag with glare, dim indoor light, motion blur, partial view, tag visible vs not.
- Metadata per image: garment_id, design_id, unit_id, shot_type (packing/rider), phone_model, lighting, in_polybag, crumpled, tag_visible, split.
- Splits are by garment (and by design for hard negatives), never by image. Leakage across splits makes results fake.
- The capture protocol is written as a one page doc in /data before any shooting and does not change mid collection without versioning.

## 6. Evaluation

- Verification: ROC, TPR at 1% FPR. Plain accuracy is not the headline metric. Flagging honest customers is the costly error.
- Retrieval: Recall@1 and Recall@5 against a single seller's full catalog.
- Breakdowns by condition: polybag vs not, low vs good light, crumpled vs folded, easy negatives vs identical unit negatives. This table is the key result.
- Ablations: zero shot vs fine tuned; with vs without segmentation; global only vs global plus local; single view vs multi view.
- Calibration: reliability diagram / ECE after temperature scaling.
- Deployment: latency and model size. Server side inference is fine for V1 (riders already upload asynchronously). On device distillation is a stretch goal.
- Every experiment gets one row in docs/results.md: date, commit hash, data version, model, config, metrics, notes.

## 7. Stack

- ML: Python, PyTorch, notebooks for exploration, scripts for anything reported.
- Backend: FastAPI inference server.
- App: cross platform. Recommended: React Native with Expo (EAS builds iOS without needing Xcode locally). Flutter is the alternative if a team member knows Dart. Final choice not yet made; ask before scaffolding the app.
- Do not start with Xcode/iOS only. Target users are riders and small sellers on cheap Android phones.
- Publishing: Google Play first ($25 one time; new personal developer accounts have closed testing requirements before public release, so check current rules early). App Store later ($99/year).
- Storage: images in Google Drive or S3. Never in git.

## 8. Repo layout

```
/ml        notebooks, training, evaluation (Python)
/data      capture protocol doc, metadata schema, metadata.csv (no images)
/backend   FastAPI inference server
/app       mobile app (Expo or Flutter, TBD)
/docs      decisions, results.md, weekly notes
```

## 9. Build order and milestones

Model risk comes before app work. The app is a camera, upload, and result screen; the uncertain part is whether matching works on rider photos.

1. Week 1: GitHub org plus monorepo, capture protocol doc, metadata schema, photograph first 30 garments.
2. Weeks 1 to 3: zero shot DINOv2/CLIP baseline notebook, Recall@1 on first 30 garments. First real milestone.
3. Weeks 3 to 5: internal data capture app (guided angles, auto tag garment ID, phone, lighting, polybag). A data tool, not the product.
4. Weeks 4 to 6: segmentation, quality gate, reach 200 plus garments.
5. Weeks 7 to 9: metric learning fine tune with hard negatives.
6. Weeks 10 to 11: local matching, multi view fusion, calibration.
7. Weeks 12 to 14: full evaluation tables, demo app (seller packing capture plus rider return capture), writeup.
8. After: production backend, real seller/rider app, Play Store closed test, then public release.

Parallel track: talk to 10 to 15 Indian fashion sellers on Myntra/AJIO/Meesho. Ask: wrong item returns per month, share of claims rejected, what one caught case is worth, whether they would switch from packing videos.

## 10. Team workflow

- GitHub organization (not a personal account); both members are owners.
- Protected main branch. All changes via pull request, reviewed by the other person.
- GitHub Issues as the task board.
- Role split: one person owns data plus evaluation, the other owns models. The person reporting results is not the one tuning the model.

## 11. Rules for Claude Code in this repo

- Never commit images, model weights, or datasets. Update .gitignore if needed. Only metadata.csv and small config files go in git.
- Never create train/val/test splits at the image level. Split by garment_id (and keep design groups together for hard negatives).
- Any reported number must come from a script with a fixed seed and a recorded commit hash, not an ad hoc notebook cell. Add a row to docs/results.md.
- Always run and report zero shot baselines alongside any fine tuned model.
- Headline metrics are TPR at 1% FPR and Recall@1, with per condition breakdowns. Do not report accuracy alone.
- Keep thresholds and calibration separate from training code, fit on validation only.
- Prefer small, reviewable PRs. Explain design decisions in docs/ when they are non obvious.
- Do not scaffold /app until the framework (Expo vs Flutter) is confirmed.
- Do not put secrets or API keys in code. Use environment variables.
- Writing style for any docs or copy: no em dashes.

## 12. Open decisions

Tracked in `docs/decisions.md`.

- App framework: Expo (recommended) vs Flutter.
- Role assignment: who owns data/eval vs models.
- Which Android phones serve as "rider phones" (need at least one cheap Android if the team uses iPhones).
- Image storage: Google Drive vs S3.
- Whether V1 includes the physical QR tag anchor or is vision only.
