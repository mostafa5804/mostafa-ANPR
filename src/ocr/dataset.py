"""
Dataset برای OCR پلاک
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from src.config import (
    CHAR2IDX, IDX2CHAR,
    IMAGE_WIDTH, IMAGE_HEIGHT,
    OCR_IMAGES_DIR,
    BATCH_SIZE,
)


class PlateDataset(Dataset):
    """
    دیتاست پلاک‌های کراپ‌شده
    فرمت فایل برچسب: نام_تصویر متن_پلاک
    """

    def __init__(self, labels_file, images_dir=OCR_IMAGES_DIR, augment=False):
        self.images_dir = images_dir
        self.augment = augment
        self.samples = []

        # خواندن برچسب‌ها
        with open(labels_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(" ", 1)
                if len(parts) != 2:
                    continue

                img_path, label = parts

                # فیلتر: فقط ۸ کاراکتر
                if len(label) != 8:
                    continue

                # بررسی کاراکترهای مجاز
                if not all(c in CHAR2IDX for c in label):
                    continue

                self.samples.append((img_path, label))

        print(f"📊 {len(self.samples)} نمونه لود شد از {labels_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # خواندن تصویر
        full_path = os.path.join(self.images_dir, img_path)
        if not os.path.exists(full_path):
            # fallback: مسیر نسبی
            full_path = os.path.join(os.path.dirname(self.images_dir), img_path)

        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # تصویر خراب → تصویر خالی
            img = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)

        # Resize
        img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))

        # Augmentation
        if self.augment:
            img = self._augment(img)

        # Normalize
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5  # [-1, 1]

        # به tensor
        img_tensor = torch.FloatTensor(img).unsqueeze(0)  # (1, H, W)

        # برچسب
        label_tensor = torch.LongTensor([CHAR2IDX[c] for c in label])
        label_len = len(label)

        return img_tensor, label_tensor, label_len

    def _augment(self, img):
        """Augmentation سبک"""
        # چرخش تصادفی
        if np.random.random() < 0.3:
            angle = np.random.uniform(-3, 3)
            h, w = img.shape
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # تغییر روشنایی
        if np.random.random() < 0.3:
            alpha = np.random.uniform(0.8, 1.2)
            beta = np.random.uniform(-20, 20)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        return img


def collate_fn(batch):
    """
    چون طول برچسب‌ها یکسانه (۸)، می‌تونیم از default collate استفاده کنیم
    """
    imgs, labels, label_lens = zip(*batch)
    imgs = torch.stack(imgs, 0)
    labels = torch.stack(labels, 0)
    label_lens = torch.LongTensor(label_lens)
    return imgs, labels, label_lens


def get_dataloaders(train_labels, val_labels, batch_size=BATCH_SIZE):
    """ساخت DataLoader برای train و val"""
    train_dataset = PlateDataset(train_labels, augment=True)
    val_dataset = PlateDataset(val_labels, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader


def test_dataset():
    """تست سریع"""
    from src.config import OCR_TRAIN_LABELS

    print("=" * 60)
    print("تست Dataset")
    print("=" * 60)

    dataset = PlateDataset(OCR_TRAIN_LABELS)
    print(f"\nتعداد نمونه: {len(dataset)}")

    if len(dataset) > 0:
        img, label, label_len = dataset[0]
        print(f"\nنمونه اول:")
        print(f"  شکل تصویر: {img.shape}")
        print(f"  برچسب: {label}")
        print(f"  طول: {label_len}")

        # تبدیل به متن
        text = "".join(IDX2CHAR[idx.item()] for idx in label)
        print(f"  متن: {text}")

    print("\n✅ تست موفق!")


if __name__ == "__main__":
    test_dataset()