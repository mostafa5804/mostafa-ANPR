"""
Pipeline کامل روی ویدیو: Pose + Perspective + OCR + رأی‌گیری
"""

import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from collections import Counter

from src.config import (
    CHAR_LIST, IDX2CHAR,
    IMAGE_WIDTH, IMAGE_HEIGHT,
    POSE_IMGSZ, POSE_CONF,
)
from src.pipeline.full_pipeline import (
    load_pose_model, load_ocr_model,
    perspective_transform, ctc_decode_single,
)


# ==================== فیلتر اعتبار ====================
VALID_LETTERS = set('ابپتثجدرزژسشصطعفقکگلمنوهی')
VALID_REGIONS = set(range(10, 100))  # ۱۰ تا ۹۹
MIN_VOTES = 2  # حداقل ۲ بار دیده بشه
DEDUPE_FRAMES = 60  # ضدتکرار
PROCESS_EVERY_N = 5  # هر ۵ فریم


def is_valid_plate(pred):
    """بررسی اعتبار پلاک"""
    if len(pred) != 8:
        return False

    # حرف باید در لیست حروف فارسی باشه
    if pred[2] not in VALID_LETTERS:
        return False

    # کد منطقه (۲ رقم آخر) باید معتبر باشه
    try:
        region = int(pred[6:8])
        if region not in VALID_REGIONS:
            return False
    except ValueError:
        return False

    # ۶ رقم اول باید عدد باشه
    if not all(c.isdigit() for c in pred[:2] + pred[3:6] + pred[6:8]):
        return False

    return True


def process_video(video_path, output_path="output_video.mp4"):
    """پردازش ویدیو با pipeline کامل"""
    print("=" * 60)
    print("Pipeline کامل روی ویدیو (با فیلتر اعتبار)")
    print("=" * 60)

    # لود مدل‌ها
    print("\n🤖 لود مدل‌ها...")
    pose_model = load_pose_model()
    ocr_model, device = load_ocr_model()

    # باز کردن ویدیو
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ ویدیو باز نشد: {video_path}")
        return

    # اطلاعات ویدیو
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n📹 ویدیو: {w}x{h} @ {fps:.1f}fps | {total_frames} فریم")

    # ذخیره ویدیوی خروجی
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w, h)
    )

    # رأی‌گیری
    vote_buffer = {}  # {plate: count}
    plate_votes = {}  # {plate: frame_count}

    frame_count = 0
    plates_detected = 0
    rejected_invalid = 0
    rejected_low_votes = 0

    print(f"\n{'=' * 60}")
    print("🚀 شروع پردازش...")
    print(f"{'=' * 60}\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # هر N فریم پردازش کن
        if frame_count % PROCESS_EVERY_N != 0:
            out.write(frame)
            continue

        # Pose
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
            out.write(frame)
            continue

        # بهترین پلاک
        all_kpts.sort(key=lambda x: x["area"], reverse=True)
        best = all_kpts[0]

        # Perspective
        try:
            warped = perspective_transform(frame, best["pts"])
        except Exception:
            out.write(frame)
            continue

        if warped.size == 0 or warped.shape[0] < 5 or warped.shape[1] < 5:
            out.write(frame)
            continue

        # OCR
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        warped_resized = cv2.resize(warped_gray, (IMAGE_WIDTH, IMAGE_HEIGHT))

        img_normalized = warped_resized.astype(np.float32) / 255.0
        img_normalized = (img_normalized - 0.5) / 0.5

        img_tensor = torch.FloatTensor(img_normalized).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            output = ocr_model(img_tensor)
            log_probs = torch.nn.functional.log_softmax(output, dim=2)
            pred = ctc_decode_single(log_probs)

        # فیلتر اعتبار
        if not is_valid_plate(pred):
            rejected_invalid += 1
            out.write(frame)
            continue

        # رأی‌گیری
        vote_buffer[pred] = vote_buffer.get(pred, 0) + 1

        if vote_buffer[pred] < MIN_VOTES:
            rejected_low_votes += 1
            out.write(frame)
            continue

        # ضدتکرار
        if pred in plate_votes:
            if frame_count - plate_votes[pred] < DEDUPE_FRAMES:
                out.write(frame)
                continue

        plate_votes[pred] = frame_count
        plates_detected += 1

        print(f"[فریم {frame_count}] پلاک: {pred} (رأی: {vote_buffer[pred]})")

        # رسم
        pts_int = best["pts"].astype(int)
        cv2.polylines(frame, [pts_int], True, (0, 255, 0), 3)
        cv2.putText(frame, pred, (pts_int[0][0], pts_int[0][1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        out.write(frame)

        if frame_count % 100 == 0:
            print(f"  ... فریم {frame_count}/{total_frames}")

    cap.release()
    out.release()

    print(f"\n{'=' * 60}")
    print(f"✅ پردازش تمام شد")
    print(f"   فریم‌ها: {frame_count}")
    print(f"   پلاک‌های یکتا: {len(plate_votes)}")
    print(f"   رد شده (غیرمجاز): {rejected_invalid}")
    print(f"   رد شده (رأی کم): {rejected_low_votes}")
    print(f"   ویدیوی خروجی: {output_path}")
    print(f"{'=' * 60}")

    if plate_votes:
        print("\n📋 پلاک‌های ثبت‌شده:")
        for plate in plate_votes.keys():
            print(f"  {plate}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = "samples/video.mp4"

    process_video(video_path)