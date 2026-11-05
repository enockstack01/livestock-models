---
title: Livestock Health Detector
emoji: 🐄
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
license: agpl-3.0
short_description: Cattle health / lumpy skin / FMD detector (demo model)
---

Livestock health detector demo running on ZeroGPU ([code](https://github.com/enockstack01/livestock-models)).
Weak whole-animal labels: it does not find lesions and is not for field or veterinary decisions.

## JSON API (for servers)

`api_name="detect"`: one photo and an optional confidence threshold in, plain JSON out:

```json
{"model_version": "livestock-v1", "image": {"width": 1568, "height": 1176},
 "detections": [{"label": "lumpy_skin_disease", "confidence": 0.893,
                 "box": {"x1": 0, "y1": 0, "x2": 1568, "y2": 1176}, "reportable": true}]}
```

Box values are pixels of the photo as uploaded; an empty `detections` list is valid. Over HTTP (all three need
`Authorization: Bearer <HF_TOKEN>` while the Space is private):

1. `POST /gradio_api/upload` (multipart, field `files`) returns `["<server path>"]`
2. `POST /gradio_api/call/detect` with `{"data": [{"path": "<server path>", "meta": {"_type": "gradio.FileData"}}, 0.25]}` returns an `event_id`
3. `GET /gradio_api/call/detect/<event_id>` streams the result; the `complete` event carries it

LivestockPro's server calls it this way (`server/lib/vision.js`); phones and browsers never hold the token.
