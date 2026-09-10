import cv2
import time
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import os
import sys
import io
import csv
from datetime import date
from pathlib import Path
from contextlib import asynccontextmanager

# Setup paths to ensure we can import from project root
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import CAMERA_SOURCE, SIMILARITY_THRESHOLD, SNAPSHOTS_DIR, FACES_DIR
from database.db import init_db, get_today_attendance_summary, get_all_users
from modules.detector import FaceDetector
from modules.recognizer import FaceRecognizer
from modules.attendance_engine import AttendanceEngine
from main_camera import draw_hud

# Ensure static directories exist
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
FACES_DIR.mkdir(parents=True, exist_ok=True)

# Global variables for AI & Camera
engine = None
detector = None
recognizer = None
cap = None
current_mode = "AUTO"  # "AUTO", "IN", "OUT"


def calculate_stats():
    """Helper to compute aggregate metrics for dashboard and API."""
    today_records = get_today_attendance_summary()
    all_users = get_all_users()
    total_users = len(all_users)
    total_attendance = len(today_records)

    in_count = sum(1 for r in today_records if r.get("status") == "IN")
    out_count = sum(1 for r in today_records if r.get("status") == "OUT")

    unique_user_ids = {r.get("user_id") for r in today_records if r.get("user_id")}
    unique_today = len(unique_user_ids)

    attendance_rate = round((unique_today / total_users * 100), 1) if total_users > 0 else 0

    return {
        "total_users": total_users,
        "total_attendance": total_attendance,
        "unique_today": unique_today,
        "in_count": in_count,
        "out_count": out_count,
        "attendance_rate": attendance_rate,
        "current_mode": current_mode,
        "records": today_records
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, detector, recognizer, cap
    print("[INFO] Starting FastAPI server...")
    init_db()

    # Initialize Core Components
    engine = AttendanceEngine()
    detector = FaceDetector()
    recognizer = FaceRecognizer()

    # Initialize Camera
    cam_id = int(CAMERA_SOURCE) if str(CAMERA_SOURCE).isdigit() else CAMERA_SOURCE
    if isinstance(cam_id, int):
        if sys.platform.startswith('linux'):
            cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
        elif sys.platform.startswith('win'):
            cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        else:
            cap = cv2.VideoCapture(cam_id)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        cap = cv2.VideoCapture(cam_id)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {cam_id}")

    yield

    print("[INFO] Shutting down...")
    if cap and cap.isOpened():
        cap.release()


app = FastAPI(title="AbsensiYolo Web Dashboard", lifespan=lifespan)
templates = Jinja2Templates(directory=str(ROOT_DIR / "web" / "templates"))

# Mount static files
app.mount("/snapshots", StaticFiles(directory=str(SNAPSHOTS_DIR)), name="snapshots")
app.mount("/faces", StaticFiles(directory=str(FACES_DIR)), name="faces")


def generate_frames():
    """
    Background generator that captures frames from webcam,
    runs YuNet face detection & SFace recognition, processes attendance,
    and yields the encoded JPEG for the web browser.
    """
    global engine, detector, recognizer, cap, current_mode

    prev_time = time.time()
    fps = 0.0
    active_notification = None

    today_count = len(get_today_attendance_summary())

    while True:
        if not cap or not cap.isOpened():
            time.sleep(1)
            continue

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        cur_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / (cur_time - prev_time + 1e-6))
        prev_time = cur_time

        # 1. Face Detection
        faces = detector.detect(frame)

        # 2. Face Recognition & Business Logic
        for face_info in faces:
            bbox = face_info["box"]
            face_raw = face_info["raw"]
            x, y, w, h = bbox

            feat = recognizer.extract_feature(frame, face_raw)
            user, score = recognizer.identify(feat, engine.known_users, threshold=SIMILARITY_THRESHOLD)

            if user:
                att_res = engine.process_face(user, score, frame, manual_mode=current_mode)

                if att_res["is_recorded"]:
                    today_count += 1
                    status_lbl = "MASUK" if att_res["status"] == "IN" else "PULANG"
                    color = (0, 255, 0) if att_res["status"] == "IN" else (0, 140, 255)
                    name_label = f"[✓ ABSEN {status_lbl}] {user['name']} ({score*100:.0f}%)"
                    thickness = 3

                    active_notification = {
                        "name": user["name"],
                        "user_code": user["user_code"],
                        "status": att_res["status"],
                        "timestamp": att_res["event_data"]["timestamp"],
                        "time": time.time()
                    }
                elif att_res["reason"] == "COOLDOWN":
                    last_lbl = "MASUK" if att_res["last_status"] == "IN" else ("PULANG" if att_res["last_status"] == "OUT" else "SUDAH ABSEN")
                    color = (255, 200, 0)
                    name_label = f"{user['name']} | [SUDAH ABSEN {last_lbl}] (Jeda {att_res['cooldown_str']})"
                    thickness = 2
                else:
                    color = (0, 255, 0)
                    name_label = f"{user['name']} ({score*100:.0f}%)"
                    thickness = 2
            else:
                color = (0, 0, 255)
                name_label = f"[UNKNOWN] Belum Terdaftar ({score*100:.0f}%)"
                thickness = 2

            # 3. Draw Bounding Box & Label
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
            label_size, _ = cv2.getTextSize(name_label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
            label_w, label_h = label_size
            cv2.rectangle(frame, (x, max(0, y - label_h - 12)), (x + label_w + 10, y), color, -1)
            cv2.putText(frame, name_label, (x + 5, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2)

        # 4. Draw HUD (Heads Up Display)
        draw_hud(frame, fps, len(engine.known_users), today_count, active_notification, current_mode)

        # 5. Encode to JPEG format for web streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        # Yield as multipart
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


class ModeRequest(BaseModel):
    mode: str


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    stats = calculate_stats()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "today_records": stats["records"],
            "total_users": stats["total_users"],
            "total_attendance": stats["total_attendance"],
            "unique_today": stats["unique_today"],
            "in_count": stats["in_count"],
            "out_count": stats["out_count"],
            "attendance_rate": stats["attendance_rate"],
            "current_mode": current_mode
        }
    )


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/logs")
async def api_logs():
    return get_today_attendance_summary()


@app.get("/api/stats")
async def api_stats():
    stats = calculate_stats()
    stats.pop("records", None)
    return stats


@app.get("/api/mode")
async def get_mode():
    return {"mode": current_mode}


@app.post("/api/mode")
async def set_mode(payload: ModeRequest):
    global current_mode
    target_mode = payload.mode.upper()
    if target_mode not in ["AUTO", "IN", "OUT"]:
        raise HTTPException(status_code=400, detail="Invalid mode. Must be AUTO, IN, or OUT.")
    current_mode = target_mode
    return {"status": "ok", "mode": current_mode}


@app.get("/api/users")
async def api_users():
    users = get_all_users()
    # Exclude heavy embedding vectors from API payload
    clean_users = []
    for u in users:
        clean_users.append({
            "id": u.get("id"),
            "user_code": u.get("user_code"),
            "name": u.get("name"),
            "department": u.get("department", "Umum"),
            "photo_path": u.get("photo_path", "")
        })
    return clean_users


@app.get("/api/export")
async def export_today_csv():
    records = get_today_attendance_summary()
    today_str = date.today().strftime("%Y-%m-%d")

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Kode Pengguna", "Nama", "Divisi", "Waktu", "Status", "Akurasi (%)", "File Snapshot"])

    for r in records:
        conf_pct = f"{round(r.get('confidence', 0) * 100, 1)}%" if r.get('confidence') else "-"
        writer.writerow([
            r.get("id"),
            r.get("user_code"),
            r.get("name"),
            r.get("department"),
            r.get("timestamp"),
            r.get("status"),
            conf_pct,
            r.get("snapshot_path", "")
        ])

    csv_data = output.getvalue().encode("utf-8-sig")
    filename = f"rekap_absensi_{today_str}.csv"

    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    )


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
