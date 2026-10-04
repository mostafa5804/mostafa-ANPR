"""
تست مدل OCR آموزش‌دیده
"""

import os
import cv2
import torch
import numpy as np

from src.config import (
    CHAR_LIST, IDX2CHAR, IMAGE_WIDTH, IMAGE_HEIGHT,
    OCR_CHECKPOINT_DIR, OCR_IMAGES_DIR,
)
from src.ocr.model import CRNN


def ctc_decode_single(log_probs):
    """Decode خروجی CTC برای یک نمونه"""
    _, max_indices = log_probs.max(2)  # (T, 1)
    max_indices = max_indices.squeeze(1)  # (T,)

    text = ""
    prev = -1
    for idx in max_indices:
        idx = idx.item()
        if idx != prev and idx != 0:
            text += IDX2CHAR.get(idx, "")
        prev = idx
    return text


def load_model(checkpoint_path=None):
    """لود مدل آموزش‌دیده"""
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

    print(f"✅ مدل لود شد")
    print(f"   Epoch: {checkpoint['epoch']}")
    print(f"   Best Seq Acc: {checkpoint['best_acc'] * 100:.2f}%")
    print(f"   Best CER: {checkpoint['best_cer'] * 100:.2f}%")
    print(f"   Device: {device}")

    return model, device


def clean_image_path(img_path):
    """حذف 'images/' اضافی از مسیر"""
    # حذف 'images/' از ابتدای مسیر
    clean = img_path.replace("images/", "").replace("images\\", "")
    return clean


def predict_plate(model, device, image_path):
    """پیش‌بینی یک پلاک"""
    # خواندن تصویر
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None

    # Resize
    img_resized = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))

    # Normalize
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_normalized = (img_normalized - 0.5) / 0.5

    # به tensor
    img_tensor = torch.FloatTensor(img_normalized).unsqueeze(0).unsqueeze(0).to(device)

    # پیش‌بینی
    with torch.no_grad():
        output = model(img_tensor)
        log_probs = torch.nn.functional.log_softmax(output, dim=2)
        pred = ctc_decode_single(log_probs)

    return pred


def test_on_val():
    """تست روی ۱۰ نمونه val"""
    from src.config import OCR_VAL_LABELS

    model, device = load_model()

    print(f"\n{'=' * 60}")
    print("تست روی ۱۰ نمونه val")
    print(f"{'=' * 60}\n")

    # خواندن نمونه‌ها
    samples = []
    with open(OCR_VAL_LABELS, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) == 2 and len(parts[1]) == 8:
                samples.append((parts[0], parts[1]))

    samples = samples[:10]

    correct = 0
    for i, (img_path, gt) in enumerate(samples, 1):
        # ✅ اصلاح مسیر
        clean_path = clean_image_path(img_path)
        full_path = os.path.join(OCR_IMAGES_DIR, clean_path)

        if not os.path.exists(full_path):
            print(f"❌ {i}. فایل پیدا نشد: {full_path}")
            continue

        pred = predict_plate(model, device, full_path)

        if pred is None:
            print(f"❌ {i}. تصویر خوانده نشد: {full_path}")
            continue

        status = "✅" if pred == gt else "❌"
        if pred == gt:
            correct += 1

        print(f"{status} {i}. Pred: {pred:<12} | GT: {gt}")

    print(f"\n{'=' * 60}")
    print(f"نتیجه: {correct}/{len(samples)} = {correct / len(samples) * 100:.1f}%")
    print(f"{'=' * 60}")


def test_on_file(image_path):
    """تست روی یک فایل"""
    model, device = load_model()
    print(f"\n🖼️ تست: {image_path}")

    if not os.path.exists(image_path):
        print(f"❌ فایل پیدا نشد: {image_path}")
        return

    pred = predict_plate(model, device, image_path)

    if pred:
        print(f"\n📝 نتیجه: {pred}")
        print(f"   طول: {len(pred)} کاراکتر")
    else:
        print("❌ پیش‌بینی ناموفق")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # تست روی فایل
        test_on_file(sys.argv[1])
    else:
        # تست روی val
        test_on_val()