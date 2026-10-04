"""
توابع CRUD برای دیتابیس ANPR - نسخه ۲.۰
"""

from datetime import datetime
from persiantools.jdatetime import JalaliDate

from src.database.models import get_connection


# ==================== تاریخ شمسی ====================
def get_now():
    now = datetime.now()
    jalali = JalaliDate(now)
    return {
        "datetime": now,
        "date": jalali.strftime("%Y/%m/%d"),
        "time": now.strftime("%H:%M:%S"),
        "datetime_str": f"{jalali.strftime('%Y/%m/%d')} {now.strftime('%H:%M:%S')}",
    }


def _to_str(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.decode("utf-8", errors="replace")
    return str(value)


def _row_to_dict(row):
    if row is None:
        return None
    item = dict(row)
    for key, value in item.items():
        if isinstance(value, bytes):
            item[key] = _to_str(value)
    return item


# ==================== vehicles ====================
def get_vehicle(plate):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM vehicles WHERE plate = ?", (_to_str(plate),))
    row = cursor.fetchone()
    conn.close()
    return _row_to_dict(row)


def upsert_vehicle(plate, plate_fa="", owner_name="", vehicle_type="",
                   is_registered=0, status="outside",
                   last_entry=None, last_exit=None,
                   total_visits=0, last_duration=0):
    conn = get_connection()
    cursor = conn.cursor()

    plate = _to_str(plate)
    plate_fa = _to_str(plate_fa)
    owner_name = _to_str(owner_name)
    vehicle_type = _to_str(vehicle_type)
    last_entry = _to_str(last_entry)
    last_exit = _to_str(last_exit)

    existing = get_vehicle(plate)

    if existing:
        cursor.execute("""
            UPDATE vehicles SET
                plate_fa = ?, owner_name = ?, vehicle_type = ?,
                is_registered = ?, status = ?,
                last_entry = ?, last_exit = ?,
                total_visits = ?, last_duration = ?
            WHERE plate = ?
        """, (plate_fa, owner_name, vehicle_type, is_registered,
              status, last_entry, last_exit, total_visits, last_duration, plate))
    else:
        cursor.execute("""
            INSERT INTO vehicles
            (plate, plate_fa, owner_name, vehicle_type, is_registered,
             status, last_entry, last_exit, total_visits, last_duration)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (plate, plate_fa, owner_name, vehicle_type, is_registered,
              status, last_entry, last_exit, total_visits, last_duration))

    conn.commit()
    conn.close()


def get_all_vehicles(status=None):
    conn = get_connection()
    cursor = conn.cursor()

    if status:
        cursor.execute("SELECT * FROM vehicles WHERE status = ? ORDER BY last_entry DESC", (status,))
    else:
        cursor.execute("SELECT * FROM vehicles ORDER BY last_entry DESC")

    rows = cursor.fetchall()
    conn.close()
    return [_row_to_dict(r) for r in rows]


# ==================== events ====================
def insert_event(plate, plate_fa, direction, duration_min=None,
                 entry_time=None, exit_time=None,
                 is_registered=0, owner_name="", vehicle_type="",
                 snapshot_entry=None, snapshot_exit=None,
                 confidence=0.0, camera_id="cam-1"):
    now = get_now()
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO events
        (plate, plate_fa, direction, event_time, date, time,
         duration_min, entry_time, exit_time,
         is_registered, owner_name, vehicle_type,
         snapshot_entry, snapshot_exit, confidence, camera_id, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        _to_str(plate), _to_str(plate_fa), _to_str(direction), now["datetime_str"],
        now["date"], now["time"],
        duration_min, _to_str(entry_time), _to_str(exit_time),
        is_registered, _to_str(owner_name), _to_str(vehicle_type),
        _to_str(snapshot_entry), _to_str(snapshot_exit), confidence, _to_str(camera_id),
        now["datetime_str"]
    ))

    event_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return event_id


def get_events(plate=None, date=None, direction=None, limit=100, offset=0):
    conn = get_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM events WHERE 1=1"
    params = []

    if plate:
        query += " AND plate LIKE ?"
        params.append(f"%{_to_str(plate)}%")

    if date:
        query += " AND date LIKE ?"
        params.append(f"%{_to_str(date)}%")

    if direction:
        query += " AND direction = ?"
        params.append(_to_str(direction))

    query += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [_row_to_dict(r) for r in rows]


def count_events(plate=None, date=None, direction=None):
    conn = get_connection()
    cursor = conn.cursor()

    query = "SELECT COUNT(*) FROM events WHERE 1=1"
    params = []

    if plate:
        query += " AND plate LIKE ?"
        params.append(f"%{_to_str(plate)}%")
    if date:
        query += " AND date LIKE ?"
        params.append(f"%{_to_str(date)}%")
    if direction:
        query += " AND direction = ?"
        params.append(_to_str(direction))

    cursor.execute(query, params)
    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_last_entry(plate):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM events
        WHERE plate = ? AND direction = 'entry'
        ORDER BY id DESC LIMIT 1
    """, (_to_str(plate),))
    row = cursor.fetchone()
    conn.close()
    return _row_to_dict(row)


def delete_event(event_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM events WHERE id = ?", (event_id,))
    conn.commit()
    conn.close()


def update_event(event_id, plate=None, owner_name=None, note=None):
    conn = get_connection()
    cursor = conn.cursor()

    updates = []
    params = []

    if plate:
        updates.append("plate = ?")
        params.append(_to_str(plate))
    if owner_name:
        updates.append("owner_name = ?")
        params.append(_to_str(owner_name))

    if not updates:
        conn.close()
        return

    params.append(event_id)
    query = f"UPDATE events SET {', '.join(updates)} WHERE id = ?"
    cursor.execute(query, params)
    conn.commit()
    conn.close()


# ==================== whitelist ====================
def add_to_whitelist(plate, plate_fa="", owner_name="", vehicle_type="",
                     department="", phone=""):
    now = get_now()
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO whitelist
        (plate, plate_fa, owner_name, vehicle_type, department, phone, is_active, created_at)
        VALUES (?, ?, ?, ?, ?, ?, 1, ?)
    """, (_to_str(plate), _to_str(plate_fa), _to_str(owner_name),
          _to_str(vehicle_type), _to_str(department), _to_str(phone),
          now["datetime_str"]))

    conn.commit()
    conn.close()


def get_whitelist():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM whitelist WHERE is_active = 1 ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [_row_to_dict(r) for r in rows]


def remove_from_whitelist(plate):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM whitelist WHERE plate = ?", (_to_str(plate),))
    conn.commit()
    conn.close()


def is_in_whitelist(plate):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM whitelist WHERE plate = ? AND is_active = 1", (_to_str(plate),))
    row = cursor.fetchone()
    conn.close()
    return _row_to_dict(row)


# ==================== blacklist ====================
def add_to_blacklist(plate, plate_fa="", reason=""):
    now = get_now()
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO blacklist
        (plate, plate_fa, reason, created_at)
        VALUES (?, ?, ?, ?)
    """, (_to_str(plate), _to_str(plate_fa), _to_str(reason), now["datetime_str"]))

    conn.commit()
    conn.close()


def get_blacklist():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM blacklist ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [_row_to_dict(r) for r in rows]


def remove_from_blacklist(plate):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM blacklist WHERE plate = ?", (_to_str(plate),))
    conn.commit()
    conn.close()


def is_in_blacklist(plate):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM blacklist WHERE plate = ?", (_to_str(plate),))
    row = cursor.fetchone()
    conn.close()
    return _row_to_dict(row)


# ==================== آمار ====================
def get_hourly_stats(date=None):
    """آمار ساعتی"""
    if date is None:
        date = get_now()["date"]

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT time, direction FROM events WHERE date = ?
    """, (date,))
    rows = cursor.fetchall()
    conn.close()

    hourly = {f"{h:02d}": {"entry": 0, "exit": 0} for h in range(24)}
    for row in rows:
        hour = row["time"][:2]
        if hour in hourly:
            hourly[hour][row["direction"]] += 1

    return hourly


def get_daily_stats(days=7):
    """آمار روزانه"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT date, direction, COUNT(*) as count
        FROM events
        GROUP BY date, direction
        ORDER BY date DESC
        LIMIT ?
    """, (days * 2,))
    rows = cursor.fetchall()
    conn.close()

    daily = {}
    for row in rows:
        date = row["date"]
        if date not in daily:
            daily[date] = {"entry": 0, "exit": 0}
        daily[date][row["direction"]] = row["count"]

    return daily


def get_top_plates(limit=10, days=30):
    """پرتکرارترین پلاک‌ها"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT plate, COUNT(*) as count
        FROM events
        GROUP BY plate
        ORDER BY count DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [{"plate": r["plate"], "count": r["count"]} for r in rows]


def get_stats_summary():
    """خلاصه آمار"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM events")
    total_events = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM events WHERE direction = 'entry'")
    total_entries = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM events WHERE direction = 'exit'")
    total_exits = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT plate) FROM events")
    unique_plates = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM vehicles WHERE status = 'inside'")
    inside_now = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM whitelist WHERE is_active = 1")
    whitelist_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM blacklist")
    blacklist_count = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(duration_min) FROM events WHERE duration_min IS NOT NULL")
    avg_duration = cursor.fetchone()[0] or 0

    conn.close()

    return {
        "total_events": total_events,
        "total_entries": total_entries,
        "total_exits": total_exits,
        "unique_plates": unique_plates,
        "inside_now": inside_now,
        "whitelist_count": whitelist_count,
        "blacklist_count": blacklist_count,
        "avg_duration": round(avg_duration, 1),
    }


if __name__ == "__main__":
    from src.database.models import init_db
    init_db()
    print("✅ CRUD آماده است")