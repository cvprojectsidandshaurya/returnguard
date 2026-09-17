# Metadata schema

One row per image in `data/metadata.csv`. This file is the only dataset artifact that lives in git.

| column | type | required | allowed values | notes |
| --- | --- | --- | --- | --- |
| image_id | string | yes | filename without extension | primary key, e.g. `g0001_rider_02` |
| garment_id | string | yes | `g` plus 4 digits | one physical item |
| design_id | string | yes | `d` plus 4 digits | shared by identical designs |
| unit_id | int | yes | 1, 2, 3 | which unit of that design |
| lookalike_group | string | no | `lg` plus 3 digits, or empty | different designs that look confusingly similar |
| category | string | yes | kurta, tee, shirt, jeans, dress | |
| color | string | yes | free text, lowercase | dominant color |
| shot_type | string | yes | packing, rider | |
| relative_path | string | yes | `g0001/g0001_rider_02.jpg` | path inside the image store |
| phone_id | string | yes | key into phones.csv | |
| lighting | string | yes | good, dim, mixed | |
| in_polybag | bool | yes | true, false | |
| crumpled | string | yes | flat, folded, crumpled | |
| tag_visible | bool | yes | true, false | |
| blurry | bool | yes | true, false | subjective call at capture time |
| partial_view | bool | yes | true, false | garment edges cut off |
| split | string | yes | train, val, test | assigned by script, never by hand |
| protocol_version | string | yes | 1.0 | protocol the shot was taken under |
| source | string | no | own, family, shop, seller | provenance |
| notes | string | no | free text | |

## Split rules

Splits are assigned by `ml/scripts/make_splits.py` with a fixed seed, never edited by hand.

The unit of splitting is not the image and not the garment. It is the **split group**: the connected component formed by joining garments that share a `design_id` and garments that share a `lookalike_group`. Every image of every garment in a group lands in the same split.

If identical units or lookalikes straddle train and test, the model can memorise the design instead of learning to match, and the reported numbers become fiction.

`ml/scripts/validate_metadata.py` checks this and fails loudly. Run it before any evaluation.
