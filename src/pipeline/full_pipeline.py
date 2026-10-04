"""
Pipeline کامل: Pose + Perspective + OCR
"""

import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from ultralytics import YOLO

from src.config import (
    CHAR_LIST, IDX2CHAR,
    IMAGE_WIDTH, IMAGE_HEIGHT,
    POSE_CHECKPOINT_DIR, OCR_CHECKPOINT_DIR,
    POSE_IMGSZ, POSE_CONF,
)
from src.ocr.model import CRNN
from src.pose.predict import load_model as load_pose_model


def load_ocr_model(checkpoint_path=None):
    """لود مدل OCR"""
    if checkpoint_path is None:
        checkpoint_path = os.path.join(OCR_CHECKPOINT_DIR, "best.pt")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CRNN(
        input_channel=1,
        num_classes=len(CHAR_LIST),
        hidden_size=256,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✅ OCR لود شد (epoch {checkpoint['epoch']}, acc: {checkpoint['best_acc']*100:.1f}%)")
    return model, device


def ctc_decode_single(log_probs):
    """Decode CTC"""
    _, max_indices = log_probs.max(2)
    max_indices = max_indices.squeeze(1)

    text = ""
    prev = -1
    for idx in max_indices:
        idx = idx.item()
        if idx != prev and idx != 0:
            text += IDX2CHAR.get(idx, "")
        prev = idx
    return text


def perspective_transform(img, pts):
    """Perspective Transform با ۴ گوشه"""
    rect = pts.reshape(4, 2).astype("float32")
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1],
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return warped


def process_image(image_path, pose_model, ocr_model, device, save_debug=True):
    """پردازش یک تصویر: Pose → Perspective → OCR"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ تصویر پیدا نشد: {image_path}")
        return None

    print(f"📷 تصویر: {image_path} | ابعاد: {img.shape}")

    # ۱. Pose
    results = pose_model.predict(img, imgsz=POSE_IMGSZ, conf=POSE_CONF, verbose=False)

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
        print("❌ هیچ پلاکی تشخیص داده نشد")
        return None

    # بهترین پلاک (بزرگ‌ترین)
    all_kpts.sort(key=lambda x: x["area"], reverse=True)
    best = all_kpts[0]

    print(f"✅ Pose: {len(all_kpts)} پلاک | بهترین: مساحت={best['area']:.0f}, اطمینان={best['conf']:.2f}")

    # ۲. Perspective
    warped = perspective_transform(img, best["pts"])

    if save_debug:
        cv2.imwrite("debug_plate_warped.jpg", warped)
        print(f"💾 ذخیره شد: debug_plate_warped.jpg")

    # ۳. OCR
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warped_resized = cv2.resize(warped_gray, (IMAGE_WIDTH, IMAGE_HEIGHT))

    if save_debug:
        cv2.imwrite("debug_plate_ocr_input.jpg", warped_resized)
        print(f"💾 ذخیره شد: debug_plate_ocr_input.jpg")

    # Normalize
    img_normalized = warped_resized.astype(np.float32) / 255.0
    img_normalized = (img_normalized - 0.5) / 0.5

    img_tensor = torch.FloatTensor(img_normalized).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = ocr_model(img_tensor)
        log_probs = torch.nn.functional.log_softmax(output, dim=2)
        pred = ctc_decode_single(log_probs)

    print(f"🎯 نتیجه OCR: {pred}")

    # رسم
    result_img = img.copy()
    pts_int = best["pts"].astype(int)
    cv2.polylines(result_img, [pts_int], True, (0, 255, 0), 3)

    cv2.putText(result_img, pred, (pts_int[0][0], pts_int[0][1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

    cv2.imwrite("pipeline_result.jpg", result_img)
    print(f"💾 ذخیره شد: pipeline_result.jpg")

    return pred


def main():
    import sys

    # لود مدل‌ها
    print("=" * 60)
    print("Pipeline کامل: Pose + Perspective + OCR")
    print("=" * 60)

    print("\n🤖 لود مدل‌ها...")
    pose_model = load_pose_model()
    ocr_model, device = load_ocr_model()

    # تست
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "data/pose/train/images/day_00011_jpg.rf.042e1a363c8957331f852bf4d91bcf05.jpg"

    print(f"\n{'=' * 60}")
    result = process_image(image_path, pose_model, ocr_model, device)

    if result:
        print(f"\n{'=' * 60}")
        print(f"🎯 پلاک نهایی: {result}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()