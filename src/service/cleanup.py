"""
حذف تصاویر قدیمی‌تر از ۱ هفته
"""

import os
import shutil
from datetime import datetime, timedelta

SNAPSHOT_DIR = "data/snapshots"
MAX_AGE_DAYS = 14


def cleanup_old_snapshots(days=MAX_AGE_DAYS):
    """حذف تصاویر قدیمی‌تر از N روز"""
    if not os.path.exists(SNAPSHOT_DIR):
        print("⚠️ پوشه snapshots وجود ندارد")
        return 0

    cutoff = datetime.now() - timedelta(days=days)
    deleted = 0
    freed_size = 0

    for item in os.listdir(SNAPSHOT_DIR):
        item_path = os.path.join(SNAPSHOT_DIR, item)

        if not os.path.isdir(item_path):
            continue

        # تاریخ پوشه رو از نامش بخون (مثلاً 1405-07-12)
        try:
            folder_date = datetime.strptime(item, "%Y-%m-%d")
        except ValueError:
            continue

        if folder_date < cutoff:
            # محاسبه حجم
            for root, _, files in os.walk(item_path):
                for f in files:
                    try:
                        freed_size += os.path.getsize(os.path.join(root, f))
                    except:
                        pass

            # حذف
            try:
                shutil.rmtree(item_path)
                print(f"  🗑️ حذف شد: {item}")
                deleted += 1
            except Exception as e:
                print(f"  ❌ خطا در حذف {item}: {e}")

    print(f"\n✅ {deleted} پوشه حذف شد")
    print(f"✅ {freed_size / (1024*1024):.1f} MB آزاد شد")

    return deleted


def get_snapshots_size():
    """حجم کل تصاویر"""
    if not os.path.exists(SNAPSHOT_DIR):
        return 0

    total = 0
    for root, _, files in os.walk(SNAPSHOT_DIR):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except:
                pass
    return total


def get_snapshots_count():
    """تعداد کل تصاویر"""
    if not os.path.exists(SNAPSHOT_DIR):
        return 0

    count = 0
    for root, _, files in os.walk(SNAPSHOT_DIR):
        count += len(files)
    return count


if __name__ == "__main__":
    print(f"📊 حجم فعلی: {get_snapshots_size() / (1024*1024):.1f} MB")
    print(f"📊 تعداد تصاویر: {get_snapshots_count()}")
    print()
    cleanup_old_snapshots()