#!/usr/bin/env python3
"""Simple MJPEG streaming server for PTZ camera."""

import os

import cv2
import time
from flask import Flask, Response

app = Flask(__name__)

_user = os.environ.get("CAMERA_USER", "admin")
_pass = os.environ.get("CAMERA_PASS", "")
_ip = os.environ.get("CAMERA_IP", "192.168.1.108")
CAMERA_URL = f'rtsp://{_user}:{_pass}@{_ip}/cam/realmonitor?channel=1&subtype=0'


def generate_frames():
    """Generate MJPEG frames from camera."""
    cap = cv2.VideoCapture(CAMERA_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Failed to open camera")
        return

    print("Camera connected, streaming...")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        # Resize for web streaming (optional, comment out for full res)
        frame = cv2.resize(frame, (1280, 720))

        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return '''
    <html>
    <head><title>PTZ Camera Stream</title></head>
    <body style="margin:0; background:#000;">
        <img src="/stream" style="width:100%; height:100vh; object-fit:contain;">
    </body>
    </html>
    '''


@app.route('/stream')
def stream():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    print("Starting PTZ camera stream server...")
    print("Open http://192.168.86.42:5000 in your browser")
    app.run(host='0.0.0.0', port=5000, threaded=True)
