# Model assets

ProctorVision loads both models only from this directory and never downloads a
model silently. Exact filenames and tested checksums are recorded in
[`manifest.json`](manifest.json).

## Face Landmarker

Download the official MediaPipe Face Landmarker task bundle to
`model/face_landmarker.task`:

```powershell
Invoke-WebRequest `
  -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" `
  -OutFile "model/face_landmarker.task"
```

The MediaPipe source repository is Apache-2.0 licensed. Model assets can have
separate terms, so verify the upstream asset terms before redistributing the
`.task` file.

## Phone detector

Place the tested fine-tuned checkpoint at `model/best_yolov12.pt`. It must expose
class index `0` as `phone`; startup rejects incompatible or corrupt checkpoints.

The tested checkpoint embeds Ultralytics AGPL-3.0 metadata, but its original
training-dataset provenance and redistribution permission are not recorded.
Consequently, the binary is ignored by Git and should not be published until
those rights are verified. For a clean clone, obtain the authorized checkpoint
from the project owner and compare it with the manifest checksum.

The repository's MIT license applies to its source code only; it does not grant
rights to either local model binary.

## Verify local files

```powershell
Get-FileHash model/face_landmarker.task -Algorithm SHA256
Get-FileHash model/best_yolov12.pt -Algorithm SHA256
```

Different hashes are not automatically corrupt, but they are not the exact
artifacts used for the documented validation results.
