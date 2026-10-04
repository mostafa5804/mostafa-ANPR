"""
مدل‌های دیتابیس ANPR - نسخه ۲.۰
- جداول: vehicles, events, whitelist, blacklist
- مایگریشن خودکار
- بکاپ خودکار
"""

import sqlite3
import os
import shutil
from datetime import datetime


DB_PATH = "data/anpr.db"
BACKUP_DIR = "data/backups"


def get_connection():
    """اتصال به دیتابیس"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """ساخت جداول + مایگریشن"""
    conn = get_connection()
    cursor = conn.cursor()

    # ==================== vehicles ====================
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vehicles (
            plate TEXT PRIMARY KEY,
            plate_fa TEXT,
            owner_name TEXT,
            vehicle_type TEXT,
            is_registered INTEGER DEFAULT 0,
            status TEXT DEFAULT 'outside',
            last_entry TEXT,
            last_exit TEXT,
            total_visits INTEGER DEFAULT 0,
            last_duration REAL DEFAULT 0
        )
    """)

    # ==================== events ====================
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate TEXT NOT NULL,
            plate_fa TEXT,
            direction TEXT NOT NULL,
            event_time TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            duration_min REAL,
            entry_time TEXT,
            exit_time TEXT,
            is_registered INTEGER DEFAULT 0,
            owner_name TEXT,
            vehicle_type TEXT,
            snapshot_entry TEXT,
            snapshot_exit TEXT,
            confidence REAL,
            camera_id TEXT,
            created_at TEXT
        )
    """)

    # ==================== whitelist ====================
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS whitelist (
            plate TEXT PRIMARY KEY,
            plate_fa TEXT,
            owner_name TEXT,
            vehicle_type TEXT,
            department TEXT,
            phone TEXT,
            is_active INTEGER DEFAULT 1,
            created_at TEXT
        )
    """)

    # ==================== blacklist ====================
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS blacklist (
            plate TEXT PRIMARY KEY,
            plate_fa TEXT,
            reason TEXT,
            created_at TEXT
        )
    """)

    # ==================== indices ====================
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_plate ON events(plate)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_date ON events(date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_direction ON events(direction)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_vehicles_status ON vehicles(status)")

    conn.commit()

    # ==================== مایگریشن ====================
    _migrate(conn)

    conn.close()
    print("✅ دیتابیس آماده شد")


def _migrate(conn):
    """مایگریشن خودکار - اضافه کردن ستون‌های جدید"""
    cursor = conn.cursor()

    # بررسی ستون‌های vehicles
    cursor.execute("PRAGMA table_info(vehicles)")
    vehicles_cols = {row[1] for row in cursor.fetchall()}

    # بررسی ستون‌های events
    cursor.execute("PRAGMA table_info(events)")
    events_cols = {row[1] for row in cursor.fetchall()}

    # اضافه کردن ستون‌های جدید
    if "blacklist_checked" not in vehicles_cols:
        cursor.execute("ALTER TABLE vehicles ADD COLUMN blacklist_checked INTEGER DEFAULT 0")
        print("  + vehicles.blacklist_checked")

    if "whitelist_checked" not in vehicles_cols:
        cursor.execute("ALTER TABLE vehicles ADD COLUMN whitelist_checked INTEGER DEFAULT 0")
        print("  + vehicles.whitelist_checked")

    conn.commit()


def backup_db():
    """بکاپ از دیتابیس"""
    if not os.path.exists(DB_PATH):
        return None

    os.makedirs(BACKUP_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(BACKUP_DIR, f"anpr_{timestamp}.db")

    shutil.copy2(DB_PATH, backup_path)

    # حذف بکاپ‌های قدیمی‌تر از ۳۰ روز
    _cleanup_old_backups()

    return backup_path


def _cleanup_old_backups(days=30):
    """حذف بکاپ‌های قدیمی"""
    if not os.path.exists(BACKUP_DIR):
        return

    now = datetime.now()
    for filename in os.listdir(BACKUP_DIR):
        filepath = os.path.join(BACKUP_DIR, filename)
        if not os.path.isfile(filepath):
            continue

        try:
            file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
            if (now - file_time).days > days:
                os.remove(filepath)
                print(f"  🗑️ بکاپ قدیمی حذف شد: {filename}")
        except Exception:
            pass


def restore_db(backup_path):
    """بازگردانی از بکاپ"""
    if not os.path.exists(backup_path):
        return False

    # بکاپ از دیتابیس فعلی
    if os.path.exists(DB_PATH):
        backup_db()

    shutil.copy2(backup_path, DB_PATH)
    print(f"✅ بازگردانی از: {backup_path}")
    return True


def get_db_size():
    """حجم دیتابیس"""
    if not os.path.exists(DB_PATH):
        return 0
    return os.path.getsize(DB_PATH)


def get_backups():
    """لیست بکاپ‌ها"""
    if not os.path.exists(BACKUP_DIR):
        return []

    backups = []
    for filename in sorted(os.listdir(BACKUP_DIR), reverse=True):
        filepath = os.path.join(BACKUP_DIR, filename)
        if os.path.isfile(filepath):
            backups.append({
                "filename": filename,
                "path": filepath,
                "size": os.path.getsize(filepath),
                "modified": datetime.fromtimestamp(os.path.getmtime(filepath)).strftime("%Y-%m-%d %H:%M:%S"),
            })
    return backups


if __name__ == "__main__":
    init_db()
    backup_db()
    print(f"✅ بکاپ ساخته شد. حجم دیتابیس: {get_db_size() / 1024:.1f} KB")