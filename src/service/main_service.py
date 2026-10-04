"""
سرویس اصلی ANPR - با بهترین فریم و رأی‌گیری وزنی
- خوندن آدرس RTSP از cameras.json
- تشخیص خودکار whitelist/blacklist
"""

import os
import sys
import cv2
import time
import json
import torch
import numpy as np
from datetime import datetime
from persiantools.jdatetime import JalaliDate
from torchvision import transforms
from collections import defaultdict

from src.config import (
    CHAR_LIST, IDX2CHAR,
    IMAGE_WIDTH, IMAGE_HEIGHT,
    POSE_IMGSZ, POSE_CONF,
)
from src.pipeline.full_pipeline import (
    load_pose_model, load_ocr_model,
    perspective_transform, ctc_decode_single,
)
from src.service.direction import process_plate


# ==================== تنظیمات ====================
CAMERAS_FILE = "data/cameras.json"
FALLBACK_RTSP = "rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101"
CAMERA_ID = "cam-1"
SNAPSHOT_DIR = "data/snapshots"
DEDUPE_SECONDS = 30
PROCESS_EVERY_N = 2
MIN_VOTES = 3
VOTE_WINDOW = 60
CHECK_INTERVAL = 30
VALID_LETTERS = set('ابپتثجدرزژسشصطعفقکگلمنوهی')
# ==================================================


def get_jalali_date():
    now = datetime.now()
    jalali = JalaliDate(now)
    return jalali.strftime("%Y-%m-%d")


def save_snapshot(frame, plate, direction):
    date_dir = os.path.join(SNAPSHOT_DIR, get_jalali_date())
    os.makedirs(date_dir, exist_ok=True)

    now = datetime.now()
    time_str = now.strftime("%H%M%S")
    filename = f"{time_str}_{plate}_{direction}.jpg"
    filepath = os.path.join(date_dir, filename)

    cv2.imwrite(filepath, frame)
    return filepath


def load_camera_from_json(camera_id=None):
    """
    خوندن آدرس RTSP از cameras.json
    اگه camera_id داده بشه، همون رو برمی‌گردونه
    وگرنه اولین دوربین فعال رو
    """
    if not os.path.exists(CAMERAS_FILE):
        print(f"⚠️ {CAMERAS_FILE} پیدا نشد")
        return None, None

    try:
        with open(CAMERAS_FILE, "r", encoding="utf-8") as f:
            cameras = json.load(f)

        # اگه camera_id مشخص شده
        if camera_id:
            for cam in cameras:
                if cam.get("id") == camera_id:
                    if cam.get("enabled") and cam.get("rtsp"):
                        return cam["id"], cam["rtsp"]
                    else:
                        print(f"⚠️ دوربین {camera_id} غیرفعاله یا RTSP نداره")
                        return None, None

        # اولین دوربین فعال
        for cam in cameras:
            if cam.get("enabled") and cam.get("rtsp"):
                return cam["id"], cam["rtsp"]

        return None, None

    except Exception as e:
        print(f"⚠️ خطا در خواندن {CAMERAS_FILE}: {e}")
        return None, None


def calculate_plate_score(best_pose, frame_shape):
    """محاسبه امتیاز پلاک"""
    pts = best_pose["pts"]
    conf = best_pose["conf"]

    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    width = x_max - x_min
    height = y_max - y_min
    area = width * height

    aspect = width / max(height, 1)
    aspect_score = 1.0 - min(abs(aspect - 5.0) / 5.0, 1.0)

    rect = pts.reshape(4, 2)
    angles = []
    for i in range(4):
        p1 = rect[i]
        p2 = rect[(i + 1) % 4]
        p3 = rect[(i + 2) % 4]
        v1 = p2 - p1
        v2 = p3 - p2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        angles.append(angle)
    angle_score = 1.0 - min(abs(np.mean(angles) - 90) / 90, 1.0)

    score = (
        (area / 10000) * 0.4 +
        conf * 0.3 +
        aspect_score * 0.2 +
        angle_score * 0.1
    )

    return score, area, conf, aspect


def process_frame_advanced(frame, pose_model, ocr_model, device,
                           plate_data, frame_count):
    """پردازش فریم با بهترین فریم و رأی‌گیری وزنی"""
    results = pose_model.predict(frame, imgsz=POSE_IMGSZ, conf=POSE_CONF, verbose=False)

    all_kpts = []
    for r in results:
        if r.keypoints is None or len(r.keypoints) == 0:
            continue
        kpts = r.keypoints.xy.cpu().numpy()
        confs = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None
        for i, kp in enumerate(kpts):
            pts = kp.astype(np.float32)
            conf = confs[i].mean() if confs is not None else 0
            x_min, y_min = pts.min(axis=0)
            x_max, y_max = pts.max(axis=0)
            area = (x_max - x_min) * (y_max - y_min)
            all_kpts.append({"pts": pts, "conf": conf, "area": area})

    if not all_kpts:
        return

    all_kpts.sort(key=lambda x: x["area"], reverse=True)
    best = all_kpts[0]

    try:
        warped = perspective_transform(frame, best["pts"])
    except Exception:
        return

    if warped.size == 0 or warped.shape[0] < 5 or warped.shape[1] < 5:
        return

    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warped_resized = cv2.resize(warped_gray, (IMAGE_WIDTH, IMAGE_HEIGHT))

    img_normalized = warped_resized.astype(np.float32) / 255.0
    img_normalized = (img_normalized - 0.5) / 0.5

    img_tensor = torch.FloatTensor(img_normalized).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = ocr_model(img_tensor)
        log_probs = torch.nn.functional.log_softmax(output, dim=2)
        pred = ctc_decode_single(log_probs)

    if len(pred) != 8:
        return

    if pred[2] not in VALID_LETTERS:
        return

    score, area, conf, aspect = calculate_plate_score(best, frame.shape)

    if pred not in plate_data:
        plate_data[pred] = {
            "votes": [],
            "weighted_votes": 0,
            "raw_votes": 0,
            "best_score": 0,
            "best_frame": None,
            "best_warped": None,
            "best_pose": None,
            "first_seen": frame_count,
            "last_seen": frame_count,
        }

    data = plate_data[pred]
    data["votes"].append({
        "frame_count": frame_count,
        "score": score,
        "area": area,
        "conf": conf,
    })
    data["weighted_votes"] += score
    data["raw_votes"] += 1
    data["last_seen"] = frame_count

    if score > data["best_score"]:
        data["best_score"] = score
        data["best_frame"] = frame.copy()
        data["best_warped"] = warped.copy()
        data["best_pose"] = best

    data["votes"] = [v for v in data["votes"] if frame_count - v["frame_count"] < VOTE_WINDOW]
    data["weighted_votes"] = sum(v["score"] for v in data["votes"])
    data["raw_votes"] = len(data["votes"])


def should_register(plate, data, seen_plates):
    """بررسی اینکه آیا پلاک باید ثبت بشه"""
    if data["raw_votes"] < MIN_VOTES:
        return False, f"رأی کم ({data['raw_votes']})"

    if data["weighted_votes"] < MIN_VOTES * 0.5:
        return False, f"وزن کم ({data['weighted_votes']:.2f})"

    now = time.time()
    if plate in seen_plates:
        if now - seen_plates[plate] < DEDUPE_SECONDS:
            return False, "تکراری"

    return True, "ok"


def try_register_plates(plate_data, seen_plates, camera_id, label=""):
    """تلاش برای ثبت پلاک‌ها"""
    registered = 0

    for plate, data in list(plate_data.items()):
        if plate in seen_plates:
            continue

        ok, reason = should_register(plate, data, seen_plates)

        if ok:
            snapshot = save_snapshot(
                data["best_frame"], plate, "detected"
            )

            result = process_plate(
                plate=plate,
                plate_fa=plate,
                confidence=data["best_pose"]["conf"],
                snapshot_entry=snapshot,
                snapshot_exit=snapshot,
                camera_id=camera_id,
            )

            direction_fa = "ورود" if result["direction"] == "entry" else "خروج"
            duration = f" | مدت: {result['duration_min']:.1f} دقیقه" if result["duration_min"] else ""

            print(f"  ✅ {direction_fa} | {plate} | "
                  f"{result['owner_name'] or 'متفرقه'}{duration} "
                  f"(رأی: {data['raw_votes']}, وزن: {data['weighted_votes']:.2f}, "
                  f"امتیاز: {data['best_score']:.2f})")

            seen_plates[plate] = time.time()
            registered += 1

    return registered


def main():
    print("=" * 60)
    print("سرویس ANPR - نسخه ۲.۰ (بهترین فریم + رأی‌گیری وزنی)")
    print("=" * 60)

    # ==================== بررسی آرگومان‌ها ====================
    camera_id_arg = None
    video_path_arg = None

    if len(sys.argv) > 1:
        if sys.argv[1] == "--video" and len(sys.argv) > 2:
            video_path_arg = sys.argv[2]
        elif sys.argv[1] == "--camera-id" and len(sys.argv) > 2:
            camera_id_arg = sys.argv[2]

    # ==================== تعیین منبع ویدیو ====================
    video_path = None
    camera_id = CAMERA_ID

    if video_path_arg:
        # از آرگومان --video
        video_path = video_path_arg
        print(f"\n📁 منبع: فایل ویدیو")

    else:
        # از cameras.json
        print(f"\n📡 خوندن تنظیمات از {CAMERAS_FILE}...")
        cam_id, cam_rtsp = load_camera_from_json(camera_id_arg)

        if cam_id and cam_rtsp:
            video_path = cam_rtsp
            camera_id = cam_id
            print(f"✅ دوربین انتخابی: {camera_id}")
            print(f"   آدرس: {video_path}")
        else:
            print("⚠️ هیچ دوربین فعالی در cameras.json نیست")
            print(f"   استفاده از آدرس پیش‌فرض: {FALLBACK_RTSP}")
            video_path = FALLBACK_RTSP

    # ==================== لود مدل‌ها ====================
    print("\n🤖 لود مدل‌ها...")
    pose_model = load_pose_model()
    ocr_model, device = load_ocr_model()

    # ==================== اتصال به دوربین ====================
    print(f"\n📹 اتصال به: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ اتصال برقرار نشد!")
        print("   بررسی کن:")
        print("   ۱. آدرس RTSP در cameras.json درست باشه")
        print("   ۲. MediaMTX و FFmpeg در حال اجرا باشن")
        return

    print("✅ اتصال برقرار شد")

    # ==================== متغیرها ====================
    frame_count = 0
    seen_plates = {}
    plate_data = {}

    print(f"\n{'=' * 60}")
    print("🚀 شروع پردازش...")
    print(f"   هر {PROCESS_EVERY_N} فریم | حداقل {MIN_VOTES} رأی | پنجره {VOTE_WINDOW} فریم")
    print(f"   دوربین: {camera_id}")
    print(f"{'=' * 60}\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ فریم دریافت نشد. اتصال قطع شد؟")
                # تلاش برای اتصال مجدد
                print("⏳ تلاش مجدد در ۵ ثانیه...")
                time.sleep(5)
                cap.release()
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print("❌ اتصال مجدد ناموفق")
                    break
                continue

            frame_count += 1

            if frame_count % PROCESS_EVERY_N != 0:
                continue

            process_frame_advanced(
                frame, pose_model, ocr_model, device,
                plate_data, frame_count
            )

            if frame_count % CHECK_INTERVAL == 0:
                print(f"\n--- بررسی فریم {frame_count} ---")
                try_register_plates(plate_data, seen_plates, camera_id)

    except KeyboardInterrupt:
        print("\n\n⏹️ توقف توسط کاربر")

    finally:
        cap.release()

        # بررسی نهایی
        print(f"\n--- بررسی نهایی ---")
        try_register_plates(plate_data, seen_plates, camera_id)

        print(f"\n{'=' * 60}")
        print(f"✅ مجموع فریم‌ها: {frame_count}")
        print(f"✅ پلاک‌های یکتا: {len(seen_plates)}")
        print(f"{'=' * 60}")

        if seen_plates:
            print("\n📋 پلاک‌های ثبت‌شده:")
            for plate in seen_plates.keys():
                data = plate_data.get(plate)
                if data:
                    print(f"  {plate} | رأی: {data['raw_votes']} | امتیاز: {data['best_score']:.2f}")


if __name__ == "__main__":
    main()