"""
بکاپ خودکار دیتابیس ANPR
"""

import os
import shutil
import schedule  # pip install schedule
import time
from datetime import datetime

from src.database.models import backup_db, get_backups, get_db_size


def daily_backup():
    """بکاپ روزانه"""
    path = backup_db()
    if path:
        print(f"✅ بکاپ روزانه: {path}")
    else:
        print("❌ بکاپ ناموفق")


def start_scheduler():
    """شروع زمان‌بند"""
    schedule.every().day.at("23:59").do(daily_backup)
    schedule.every(6).hours.do(daily_backup)

    print("⏰ زمان‌بند بکاپ فعال شد")
    print("   - روزانه ساعت ۲۳:۵۹")
    print("   - هر ۶ ساعت")

    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    start_scheduler()