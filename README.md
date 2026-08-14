# ProctorVision

**Author:** Ayush Goel  
**License:** MIT

ProctorVision is a local Python application for reviewing online-proctoring
observations from a webcam or recorded video. It monitors gaze, head movement,
face presence, multiple faces, and visible phones, then stores sustained review
flags with screenshots for human inspection.

Flags are observations, not proof of misconduct.

## What you get

- Live webcam monitoring or local-video replay
- Gaze and relative head-pose monitoring using MediaPipe landmarks
- Primary-candidate tracking that avoids silently switching to another face
- Phone detection with a local YOLO checkpoint
- SQLite session and event history with evidence screenshots
- FastAPI REST API and a browser dashboard for reviewing sessions

## Requirements

- Python 3.11
- A webcam for live monitoring, or a local video file
- The two local model files described in [model/README.md](model/README.md)

## Quick start (Windows)

Open PowerShell in the project directory:

```powershell
cd "C:\path\to\Cheating-Surveillance-System-main"
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `py -3.11` is not available, install Python 3.11 from
[python.org](https://www.python.org/downloads/) and run `py --list` to confirm it
is detected.

## Add the models

Before running the monitor, place these files in `model/`:

```text
model/face_landmarker.task
model/best_yolov12.pt
```

Follow [model/README.md](model/README.md) for the download/checksum instructions.
The application never downloads models automatically and will show a clear error
if a model is missing, corrupted, or uses the wrong phone class mapping.

## Run monitoring

Use the default webcam:

```powershell
python main.py
```

Use another camera:

```powershell
python main.py --source 1
```

Use a recorded video:

```powershell
python main.py --source Demo_vid/controlled-exam.mp4
```

You may also supply an absolute video path:

```powershell
python main.py --source "C:\videos\controlled-exam.mp4"
```

At startup, keep one face visible and look naturally toward the camera until
calibration finishes. Press `q` to stop the monitor cleanly. Add `--debug` only
when investigating detector behavior.

## View saved sessions and evidence

Start the web application in a second PowerShell window, using the same virtual
environment:

```powershell
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Then open:

- Dashboard: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- API documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

The monitor and dashboard use the same local SQLite database. A running session
appears automatically in the dashboard and refreshes periodically.

## Screenshots

**Live Monitoring**

![Live Monitoring](data/images/live-monitoring.jpg)

The real-time monitor view, showing face tracking and gaze/head-pose signals as they are captured.

**Session Dashboard**

![Session Dashboard](data/images/session-dashboard.png)

The browser dashboard, used to review sessions, confirmed events, and evidence after a run.

## Where results are saved

```text
data/
├── surveillance.db
├── evidence/
│   └── <session-id>/
│       └── <event-id>.jpg
└── images/
    ├── live-monitoring.jpg
    └── session-dashboard.png
```

The database stores event details and the relative path to each screenshot. The
image files themselves stay on disk in `data/evidence/`. `data/images/` holds
non-sensitive screenshots used for documentation only.

## How it works

```text
webcam/video frame
        ↓
shared face landmarks + primary-face tracking
        ↓
gaze, head-pose, phone, and face-presence observations
        ↓
temporal AlertEngine confirms sustained review events
        ↓
SQLite session/event records + evidence screenshots
        ↓
REST API and dashboard
```

Short or uncertain observations do not immediately become events. A confirmed
event is created only after the configured sustained-duration rule is met.

## Run tests

```powershell
python -m unittest discover -s tests -v
```

Optional code-quality check:

```powershell
python -m ruff check .
```

## Common setup issues

**`No suitable Python runtime found`**

Install Python 3.11, run `py --list`, then create the virtual environment again.

**PowerShell cannot run `Activate.ps1`**

Run this once in the current terminal, then activate the environment again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

**Calibration times out**

Keep one well-lit face visible, look forward, and restart the application. Avoid
large head movement during the initial calibration period.

**Camera cannot open**

Close other applications using the camera, check camera permissions, or try a
different index such as `--source 1`.

**Dashboard has no sessions**

Run the monitor first, then start Uvicorn from the same project directory. Both
processes must use the same `data/surveillance.db` file.

## Project structure

```text
main.py                 # monitor entry point
surveillance_engine.py  # per-frame orchestration
detectors.py            # common detector result model
alert_engine.py         # sustained-event rules
session_service.py      # session lifecycle and event persistence
persistence.py          # SQLite repository
api.py                  # REST API
dashboard.py            # dashboard routes
model/                  # local model instructions and manifest
tests/                  # automated tests
```

## License

Copyright (c) 2026 Ayush Goel. This project source code is licensed under the
[MIT License](LICENSE).

The local model binaries are not included in this license or repository upload;
they remain subject to their own terms and the provenance guidance in
[model/README.md](model/README.md).
