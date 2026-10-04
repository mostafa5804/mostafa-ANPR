"""
آموزش YOLOv8-Pose برای تشخیص ۴ گوشه پلاک
"""

import os
from ultralytics import YOLO

from src.config import (
    POSE_DATA_YAML, POSE_CHECKPOINT_DIR,
    POSE_EPOCHS, POSE_BATCH, POSE_IMGSZ,
)


def train_pose():
    print("=" * 60)
    print("آموزش YOLOv8-Pose برای تشخیص ۴ گوشه پلاک")
    print("=" * 60)

    # بررسی GPU
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🔧 Device: {device}")
    if device == "cuda":
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")

    # ساخت پوشه خروجی
    os.makedirs(POSE_CHECKPOINT_DIR, exist_ok=True)

    # لود مدل پایه
    print("\n🤖 لود مدل پایه yolov8n-pose.pt...")
    model = YOLO("yolov8n-pose.pt")

    # آموزش
    print(f"\n{'=' * 60}")
    print(f"🚀 شروع آموزش ({POSE_EPOCHS} epoch)...")
    print(f"{'=' * 60}\n")

    results = model.train(
        data=POSE_DATA_YAML,
        epochs=POSE_EPOCHS,
        batch=POSE_BATCH,
        imgsz=POSE_IMGSZ,
        lr0=0.001,
        optimizer="AdamW",
        patience=15,
        project=".",
        name="train",
        exist_ok=True,
        workers=0,
        device=device,
        verbose=True,
    )

    print(f"\n{'=' * 60}")
    print(f"✅ آموزش تمام شد!")
    print(f"   بهترین مدل: {POSE_CHECKPOINT_DIR}/train/weights/best.pt")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    train_pose()