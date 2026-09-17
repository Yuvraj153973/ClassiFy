# ClassiFy
 
**AI-powered classroom occupancy detection for smart, energy-efficient HVAC control.**
 
By Yuvraj Singh
 
---
 
## The Problem
 
Classrooms routinely waste enormous amounts of energy cooling empty or near-empty rooms:
 
- **Energy waste in empty spaces** — air conditioning frequently runs at full capacity during free periods, breaks, or schedule changes, cooling rooms with no one in them.
- **No real-time adaptation** — standard HVAC systems rely on fixed schedules or basic thermostats rather than responding dynamically to how many students are actually in the room.
- **High operational costs** — continuous, unoptimized cooling drives up utility bills and places a real financial burden on schools.
- **Environmental impact** — unnecessary energy consumption increases electricity demand and directly contributes to higher carbon emissions.
## The Solution
 
ClassiFy is a smart-campus system that uses computer vision to count students in a classroom in real time, and uses that occupancy count to make informed decisions about HVAC/AC control — instead of cooling a room on a fixed schedule regardless of whether anyone is in it.
 
**Core components of the vision:**
- **AI-powered detection** — a custom object detection model (trained on YOLOv8, using a curated classroom/student dataset built from Roboflow) counts the number of students visible in a live camera feed.
- **Smart AC control** — the occupancy count drives automatic decisions about whether cooling is needed at all, and at what intensity.
- **Edge processing** — designed to run entirely on local hardware (originally scoped as a Raspberry Pi 4), so no video is sent off-device.
- **Manual override** — an accessible override lets a teacher or admin correct the system's decision if it gets something wrong (e.g. a false positive/negative), so comfort is never sacrificed for the sake of energy savings.
- **Privacy by design** — camera footage is processed locally and is never stored or transmitted anywhere. Only the resulting occupancy count/signal leaves the detection unit.
## Current Implementation Status
 
This repository currently implements and demonstrates the **core AI detection engine and hardware signaling pipeline** — the heart of the system. It is a working prototype that proves out the full pipeline end-to-end: camera → trained model → live occupancy count → hardware signal.
 
**What's built and working:**
- A custom-trained YOLOv8 object detection model, trained specifically to detect students/people in classroom-style photos
- A Flask web application that runs the model live on a webcam feed, displaying bounding boxes and a running headcount on a browser page
- A hardware signaling demo: the detected occupancy count is sent over a USB serial connection to a microcontroller (ESP32-S3), which lights up a color-coded LED representing three occupancy tiers
**What's planned but not yet implemented in this repo** (see [Future Roadmap](#future-roadmap)):
- Actual AC/HVAC control via an IR transmitter module (the LED demo stands in for this — swapping the LED logic for IR signal codes is the natural next step)
- Deployment on a Raspberry Pi 4 rather than a laptop
- The manual override interface
- Multi-room support and the other enhancements listed below
This staged approach lets the detection accuracy and signaling logic be validated independently before wiring up to real HVAC hardware.
 
---
 
## How It Works
 
### 1. Dataset
 
The model was built and trained using a single Roboflow dataset:
 
**[student-counting-v2](https://app.roboflow.com/yuvraj-singh-dijfm/student-counting-v2-cpwv7/train)**
 
- Single class: `person`/`student` — bounding boxes around each individual in classroom-style photos
- ~5,400+ images
- Exported in **YOLOv8 format** (images + YOLO `.txt` bounding-box annotations + `data.yaml`)
### 2. Model Training
 
- **Architecture:** YOLOv8n (nano) — chosen for a good balance of accuracy and inference speed
- **Training environment:** Google Colab, using the `ultralytics` Python package
- **Training setup:**
  - Pretrained COCO weights (`yolov8n.pt`) as the starting checkpoint, fine-tuned on the classroom dataset above
  - 100 epochs, checkpoints saved to Google Drive every epoch (`save_period=1`) to survive Colab disconnects
  - Early stopping patience of 20 epochs
- **Final validation results:**
  | Metric | Score |
  |---|---|
  | Precision | 93.8% |
  | Recall | 94.1% |
  | mAP50 | 97.2% |
  | mAP50-95 | 68.5% |
  These numbers mean the model correctly identifies the vast majority of students in a frame (94% recall) while rarely raising false alarms (93.8% precision) — solid performance for occupancy counting.
### 3. Live Detection (`app.py`)
 
A Flask web application that:
1. Opens a webcam feed via OpenCV
2. Runs the trained YOLOv8 model (`best.pt`) on frames in a background thread
3. Draws bounding boxes and a live headcount overlay on each frame
4. Streams the annotated video to a browser page (`index.html`) using MJPEG streaming
5. Exposes a `/count` JSON endpoint reporting the current occupancy number
6. Classifies the count into one of three occupancy tiers and sends a signal to the connected microcontroller only when the tier changes (not on every single frame)
**Occupancy tiers:**
 
| Student count | Tier | LED color (current demo) |
|---|---|---|
| 0 | Empty | Blue |
| 1–5 | Low occupancy | Yellow |
| 6+ | Full occupancy | Red |
 
### 4. Hardware Signal (ESP32-S3)
 
The Flask app communicates with an ESP32-S3 over a **wired USB serial connection** (no WiFi required for this stage):
- Python sends a plain-text command (`"blue"`, `"yellow"`, or `"red"`) down the USB cable whenever the occupancy tier changes
- The ESP32-S3 firmware (`esp32_firmware.ino`) continuously listens on serial and switches on the corresponding LED, switching the others off
This is a stand-in for the eventual IR-transmitter-to-AC-unit signal described in the original project concept — the logic (decide a tier, send a signal) is identical; only the output device differs.
 
---
 
## Repository Structure
 
```
ClassiFy/
├── app.py                  # Flask app: webcam capture, YOLO inference, MJPEG streaming, serial signaling
├── best.pt                 # Trained YOLOv8 model weights (not committed — see Setup)
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html          # Web page displaying the live detection feed + count
├── esp32_firmware/
│   └── esp32_firmware.ino  # Arduino sketch: reads serial commands, drives LEDs
└── README.md
```
 
---
 
## Hardware (Target Deployment)
 
The cost breakdown below reflects the **originally scoped deployment target** (Raspberry Pi + IR control). The current repo's prototype substitutes a laptop webcam and an ESP32-S3 + LEDs for development/testing purposes.
 
| Component | Function | Estimated Cost (AED) |
|---|---|---|
| Raspberry Pi 4 (4GB) | Core processing and YOLOv8 inference unit | ~280 AED |
| Raspberry Pi Camera Module | Live video feed input for occupant detection | ~60 AED |
| KY-005 IR Transmitter Module | Sends temperature/power signals to the AC unit | ~10 AED |
| MicroSD Card (32GB) + Power Adapter | System OS storage and power supply | ~45 AED |
| **Total estimated cost** | Complete hardware setup per classroom | **~395 AED** |
 
**Current prototype hardware used instead (for development):**
- A laptop with a webcam (any resolution; 640×480 recommended for inference speed)
- An ESP32-S3 development board
- 3 LEDs (blue, yellow, red) + current-limiting resistors (~220Ω each)
- A USB cable connecting the ESP32-S3 to the laptop
---
 
## Setup & Build Instructions
 
### Prerequisites
- Python 3.10+
- Node.js (only if you plan to re-upload/re-annotate data via the Edge Impulse CLI — not required for running the core app)
- Arduino IDE (for flashing the ESP32-S3)
- A webcam
- An ESP32-S3 development board + 3 LEDs + resistors (optional — the web app runs without it, just skip the `--esp32-port` argument handling)
### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/ClassiFy.git
cd ClassiFy
```
 
### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```
`requirements.txt` includes: `flask`, `opencv-python`, `ultralytics`, `pyserial`
 
### 3. Get the trained model
The trained `best.pt` weights are not committed to this repo (binary model files don't belong in git without Git LFS). Either:
- Train your own following the [Dataset](#1-dataset) and [Model Training](#2-model-training) steps above, or
- Download the pre-trained weights from [link to your release/Drive here] and place `best.pt` in the project root
### 4. Wire up and flash the ESP32-S3 (optional, for hardware signaling)
1. Connect an LED + resistor from each of GPIO 25 (blue), 26 (yellow), and 27 (red) to ground
2. Open `esp32_firmware/esp32_firmware.ino` in the Arduino IDE
3. Select your ESP32-S3 board and the correct COM port
4. Upload the sketch
5. Note the COM port shown in Device Manager (Windows) or `/dev/tty.*` (Mac/Linux) — you'll need it in the next step
### 5. Run the app
With the ESP32 connected:
```bash
python app.py --esp32-port COM5
```
(replace `COM5` with your actual port)
 
You'll be prompted to choose a camera index (usually `0` for a built-in/default webcam).
 
### 6. View it
Open your browser to:
```
http://localhost:5000
```
You should see your live webcam feed with bounding boxes drawn around detected students and a running headcount. If an ESP32-S3 is connected, its LED will update automatically as the occupancy tier changes.
 
### Command-line options
| Flag | Default | Description |
|---|---|---|
| `--model` | `best.pt` | Path to the trained model weights |
| `--camera` | `0` | Webcam index |
| `--conf` | `0.25` | Detection confidence threshold |
| `--imgsz` | `320` | Inference image size (lower = faster, less accurate) |
| `--port` | `5000` | Port the Flask web server runs on |
| `--esp32-port` | *(required)* | COM port the ESP32-S3 is connected to |
 
---
 
## Environmental Impact & Measurable Benefits
 
By cooling only occupied classrooms rather than running on fixed schedules, ClassiFy targets meaningful reductions in:
- Wasted HVAC energy consumption during empty periods
- School utility costs
- Associated carbon emissions from unnecessary electricity demand
*Sources referenced in project research: Pang et al. (2020), Applied Energy; US EPA GHG Calculator; Raspberry Pi specifications & benchmarks.*
 
---
 
## Privacy
 
Camera footage is processed entirely locally and is **never stored or transmitted** off the detection device. Only the derived occupancy count (and the resulting tier signal) is used downstream — the actual video never leaves the unit.
 
---
 
## Future Roadmap
 
- **IR/AC integration** — replace the LED demo output with real IR transmitter codes to directly control classroom AC units
- **Manual override interface** — an accessible control allowing teachers/admins to override automatic decisions in case of false positives
- **Raspberry Pi deployment** — move from a laptop prototype to the target Raspberry Pi 4 + camera module setup for actual classroom installation
- **CO2 & air quality sensors** — regulate ventilation alongside cooling based on air quality, not just occupancy
- **Central facility dashboard** — track live occupancy and energy savings across an entire campus from a web app
- **Timetable pre-cooling** — sync with school schedules to begin cooling a few minutes before class starts, rather than reacting purely to occupancy
- **Multi-room AI processing** — add an AI accelerator (e.g. Coral Edge TPU) to allow one Pi to monitor multiple rooms simultaneously
- **Model optimization** — convert the YOLOv8 model to TFLite or TensorRT to further reduce power consumption on edge hardware
