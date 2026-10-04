"""
تست مدل Pose آموزش‌دیده
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO

from src.config import POSE_CHECKPOINT_DIR, POSE_IMGSZ, POSE_CONF


def load_model(checkpoint_path=None):
    """لود مدل Pose"""
    if checkpoint_path is None:
        checkpoint_path = os.path.join(POSE_CHECKPOINT_DIR, "best.pt")

    if not os.path.exists(checkpoint_path):
        print(f"❌ مدل پیدا نشد: {checkpoint_path}")
        return None

    model = YOLO(checkpoint_path)
    print(f"✅ مدل Pose لود شد: {checkpoint_path}")
    return model


def predict_pose(model, image_path, save_output=True):
    """تشخیص ۴ گوشه پلاک در تصویر"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ تصویر پیدا نشد: {image_path}")
        return None

    # پیش‌بینی
    results = model.predict(
        img,
        imgsz=POSE_IMGSZ,
        conf=POSE_CONF,
        verbose=False,
    )

    detections = []
    for r in results:
        if r.keypoints is None or len(r.keypoints) == 0:
            continue

        kpts = r.keypoints.xy.cpu().numpy()
        confs = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None

        for i, kp in enumerate(kpts):
            pts = kp.astype(np.float32)
            conf = confs[i].mean() if confs is not None else 0

            # محاسبه مساحت
            x_min, y_min = pts.min(axis=0)
            x_max, y_max = pts.max(axis=0)
            area = (x_max - x_min) * (y_max - y_min)

            detections.append({
                "pts": pts,
                "conf": conf,
                "area": area,
            })

    if not detections:
        print("❌ هیچ پلاکی تشخیص داده نشد")
        return None

    # بهترین = بزرگ‌ترین
    detections.sort(key=lambda x: x["area"], reverse=True)
    best = detections[0]

    # رسم
    result_img = img.copy()
    pts_int = best["pts"].astype(int)
    cv2.polylines(result_img, [pts_int], True, (0, 255, 0), 3)

    # شماره‌گذاری
    names = ["TL", "TR", "BR", "BL"]
    for j, (x, y) in enumerate(pts_int):
        cv2.circle(result_img, (x, y), 8, (0, 0, 255), -1)
        cv2.putText(result_img, names[j], (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    if save_output:
        output_path = "pose_result.jpg"
        cv2.imwrite(output_path, result_img)
        print(f"✅ نتیجه ذخیره شد: {output_path}")

    print(f"\n--- پلاک 1 ---")
    print(f"  اطمینان: {best['conf']:.2f}")
    print(f"  مساحت: {best['area']:.0f}")
    print(f"  گوشه‌ها:")
    for j, (x, y) in enumerate(pts_int):
        print(f"    {names[j]}: ({x}, {y})")

    return best


def test_on_file(image_path):
    """تست روی یک فایل"""
    model = load_model()
    if model is None:
        return

    print(f"\n🖼️ تست: {image_path}")
    result = predict_pose(model, image_path)

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test_on_file(sys.argv[1])
    else:
        print("استفاده: python -m src.pose.predict <image_path>")
        print("مثال: python -m src.pose.predict data/pose/train/images/xxx.jpg")