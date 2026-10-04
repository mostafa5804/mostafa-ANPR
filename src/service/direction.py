"""
منطق ورود/خروج هوشمند
"""

from datetime import datetime
from persiantools.jdatetime import JalaliDate

from src.database import crud


def parse_jalali_datetime(datetime_str):
    """
    تبدیل رشته تاریخ شمسی به datetime میلادی
    فرمت ورودی: 1405/07/12 11:51:50
    """
    try:
        parts = datetime_str.split(" ")
        date_parts = parts[0].split("/")
        time_parts = parts[1].split(":")

        # تبدیل شمسی به میلادی
        jalali_date = JalaliDate(
            int(date_parts[0]),
            int(date_parts[1]),
            int(date_parts[2])
        )
        gregorian_date = jalali_date.to_gregorian()

        # ساخت datetime میلادی
        return datetime(
            gregorian_date.year,
            gregorian_date.month,
            gregorian_date.day,
            int(time_parts[0]),
            int(time_parts[1]),
            int(time_parts[2])
        )
    except Exception as e:
        print(f"⚠️ خطا در پارس تاریخ: {e}")
        return None


def calculate_duration_minutes(entry_time_str):
    """
    محاسبه مدت توقف از زمان ورود تا الان
    """
    entry_datetime = parse_jalali_datetime(entry_time_str)

    if entry_datetime is None:
        return None

    now = datetime.now()
    duration = now - entry_datetime
    return duration.total_seconds() / 60


def decide_direction(plate, confidence=0.0, plate_fa=""):
    """
    تصمیم‌گیری ورود/خروج

    منطق:
    - بار اول → ورود
    - بار دوم → خروج
    - بار سوم → ورود
    - ...

    Returns:
        dict: {
            "direction": "entry" یا "exit",
            "status": "inside" یا "outside",
            "duration_min": مدت توقف (برای خروج),
            "vehicle": اطلاعات خودرو
        }
    """
    # بررسی خودرو در دیتابیس
    vehicle = crud.get_vehicle(plate)

    if vehicle is None:
        # اولین بار → ورود
        return {
            "direction": "entry",
            "status": "inside",
            "duration_min": None,
            "vehicle": None,
            "is_new": True,
        }

    if vehicle["status"] == "outside":
        # داخل نیست → ورود
        return {
            "direction": "entry",
            "status": "inside",
            "duration_min": None,
            "vehicle": vehicle,
            "is_new": False,
        }
    else:
        # داخل است → خروج
        last_entry = crud.get_last_entry(plate)
        duration_min = None

        if last_entry:
            duration_min = calculate_duration_minutes(last_entry["event_time"])
            if duration_min is not None:
                print(f"   ⏱️ مدت توقف: {duration_min:.2f} دقیقه")

        return {
            "direction": "exit",
            "status": "outside",
            "duration_min": duration_min,
            "vehicle": vehicle,
            "is_new": False,
        }


def process_plate(plate, plate_fa="", confidence=0.0,
                  snapshot_entry=None, snapshot_exit=None,
                  camera_id="cam-1"):
    """
    پردازش کامل یک پلاک:
    ۱. بررسی whitelist
    ۲. تصمیم ورود/خروج
    ۳. ثبت در events
    ۴. آپدیت vehicles

    Returns:
        dict: اطلاعات کامل پردازش
    """
    # بررسی whitelist
    wl = crud.is_in_whitelist(plate)
    is_registered = 1 if wl else 0
    owner_name = wl["owner_name"] if wl else ""
    vehicle_type = wl["vehicle_type"] if wl else ""

    # تصمیم ورود/خروج
    decision = decide_direction(plate, confidence, plate_fa)
    direction = decision["direction"]
    status = decision["status"]
    duration_min = decision["duration_min"]
    vehicle = decision["vehicle"]

    # ثبت رویداد
    event_id = crud.insert_event(
        plate=plate,
        plate_fa=plate_fa,
        direction=direction,
        duration_min=duration_min,
        entry_time=vehicle["last_entry"] if vehicle and direction == "exit" else None,
        exit_time=None,
        is_registered=is_registered,
        owner_name=owner_name,
        vehicle_type=vehicle_type,
        snapshot_entry=snapshot_entry if direction == "entry" else None,
        snapshot_exit=snapshot_exit if direction == "exit" else None,
        confidence=confidence,
        camera_id=camera_id,
    )

    # آپدیت vehicles
    now_str = crud.get_now()["datetime_str"]

    if direction == "entry":
        crud.upsert_vehicle(
            plate=plate,
            plate_fa=plate_fa,
            owner_name=owner_name,
            vehicle_type=vehicle_type,
            is_registered=is_registered,
            status=status,
            last_entry=now_str,
            last_exit=vehicle["last_exit"] if vehicle else None,
            total_visits=(vehicle["total_visits"] + 1) if vehicle else 1,
            last_duration=duration_min if duration_min else 0,
        )
    else:  # exit
        crud.upsert_vehicle(
            plate=plate,
            plate_fa=plate_fa,
            owner_name=owner_name,
            vehicle_type=vehicle_type,
            is_registered=is_registered,
            status=status,
            last_entry=vehicle["last_entry"] if vehicle else None,
            last_exit=now_str,
            total_visits=vehicle["total_visits"] if vehicle else 1,
            last_duration=duration_min if duration_min else 0,
        )

    return {
        "direction": direction,
        "status": status,
        "duration_min": duration_min,
        "is_registered": is_registered,
        "owner_name": owner_name,
        "vehicle_type": vehicle_type,
        "event_id": event_id,
        "plate": plate,
        "plate_fa": plate_fa,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("تست منطق ورود/خروج")
    print("=" * 60)

    # پاک کردن دیتابیس برای تست
    import os
    if os.path.exists("data/anpr.db"):
        os.remove("data/anpr.db")
        print("🗑️ دیتابیس پاک شد")

    from src.database.models import init_db
    init_db()

    # افزودن به whitelist
    crud.add_to_whitelist(
        "51ق81660", "۶۰ - ۸۱۶ ق ۵۱",
        "علی احمدی", "سواری", "مدیریت", "09131234567"
    )
    print("✅ افزودن به whitelist")

    # تست ۱: ورود
    print("\n--- تست ۱: ورود اول ---")
    result = process_plate("51ق81660", "۶۰ - ۸۱۶ ق ۵۱", confidence=0.95)
    print(f"  جهت: {result['direction']}")
    print(f"  وضعیت: {result['status']}")
    print(f"  ثبت‌شده: {result['is_registered']}")
    print(f"  مالک: {result['owner_name']}")

    # تست ۲: خروج
    print("\n--- تست ۲: خروج ---")
    result = process_plate("51ق81660", "۶۰ - ۸۱۶ ق ۵۱", confidence=0.95)
    print(f"  جهت: {result['direction']}")
    print(f"  وضعیت: {result['status']}")
    if result['duration_min']:
        print(f"  مدت توقف: {result['duration_min']:.2f} دقیقه")

    # تست ۳: ورود دوباره
    print("\n--- تست ۳: ورود دوباره ---")
    result = process_plate("51ق81660", "۶۰ - ۸۱۶ ق ۵۱", confidence=0.95)
    print(f"  جهت: {result['direction']}")
    print(f"  وضعیت: {result['status']}")

    # تست ۴: پلاک متفرقه
    print("\n--- تست ۴: پلاک متفرقه ---")
    result = process_plate("99ب99999", "۹۹ - ۹۹۹ ب ۹۹", confidence=0.85)
    print(f"  جهت: {result['direction']}")
    print(f"  ثبت‌شده: {result['is_registered']}")

    # نمایش رویدادها
    print("\n--- رویدادها ---")
    events = crud.get_events()
    for e in events:
        duration = f"{e['duration_min']:.1f} دقیقه" if e['duration_min'] else "-"
        print(f"  {e['date']} {e['time']} | {e['plate']} | {e['direction']} | {duration} | {e['owner_name']}")

    # نمایش خودروها
    print("\n--- خودروها ---")
    vehicles = crud.get_all_vehicles()
    for v in vehicles:
        print(f"  {v['plate']} | {v['status']} | بازدید: {v['total_visits']} | {v['owner_name']}")

    print("\n✅ تست موفق!")