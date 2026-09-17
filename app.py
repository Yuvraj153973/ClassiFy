import argparse
import threading
import time

import cv2
import serial
from flask import Flask, Response, render_template
from ultralytics import YOLO

app = Flask(__name__)

latest_frame = None
frame_lock = threading.Lock()
current_count = 0
camera = int(input("Choose the Camera: "))

esp32_serial = None
last_led_state = None


def get_led_color(count: int) -> str:
    if count == 0:
        return "blue"
    elif 1 <= count <= 5:
        return "yellow"
    else:
        return "red"


def send_led_command(color: str):
    global last_led_state
    if color == last_led_state:
        return

    try:
        esp32_serial.write((color + "\n").encode()) # type: ignore
        last_led_state = color
        print(f"Sent LED command: {color}")
    except Exception as e:
        print(f"Could not send to ESP32: {e}")


def capture_and_detect(model_path: str, camera_index: int, conf: float, imgsz: int):
    global latest_frame, current_count

    model = YOLO(model_path)
    cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print(f"ERROR: could not open camera index {camera}")
        return

    print("Camera opened successfully, starting detection loop...")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Warning: failed to read frame")
            time.sleep(0.1)
            continue

        results = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)
        result = results[0] # type: ignore
        annotated = result.plot() # type: ignore
        count = len(result.boxes) # type: ignore

        color = get_led_color(count)
        send_led_command(color)

        cv2.rectangle(annotated, (0, 0), (280, 40), (0, 0, 0), -1)
        cv2.putText(
            annotated, f"Students: {count} ({color})", (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )

        ok, buffer = cv2.imencode(".jpg", annotated)
        if not ok:
            continue

        with frame_lock:
            latest_frame = buffer.tobytes()
            current_count = count


def generate_mjpeg():
    while True:
        with frame_lock:
            frame = latest_frame
        if frame is None:
            time.sleep(0.05)
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/count")
def count():
    return {"count": current_count}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="best.pt")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--esp32-port", required=True, help="e.g. COM5")
    args = parser.parse_args()

    esp32_serial = serial.Serial(args.esp32_port, 115200, timeout=1)
    time.sleep(2)  # give the ESP32 a moment to reset after the serial connection opens

    t = threading.Thread(
        target=capture_and_detect,
        args=(args.model, args.camera, args.conf, args.imgsz),
        daemon=True,
    )
    t.start()
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)