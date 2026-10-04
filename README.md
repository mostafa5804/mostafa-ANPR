# 🚘 پلاک‌خوان هوشمند خودروهای ایرانی | ANPR

<div align="center">

### سامانه تشخیص و خواندن پلاک خودروهای ایرانی

سامانه‌ای برای تشخیص چهار گوشه پلاک، خواندن متن پلاک و ثبت رویدادهای ورود و خروج خودروها با **YOLOv8-Pose** و **CRNN**.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/Vision-OpenCV-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[📦 مشاهده مخزن](https://github.com/mostafa5804/mostafa-ANPR) · [👤 پروفایل سازنده](https://github.com/mostafa5804)

</div>

---

## ✨ قابلیت‌ها

- 🔎 تشخیص چهار گوشه پلاک با YOLOv8-Pose
- 🔤 خواندن متن پلاک با مدل CRNN و CTC
- 🚦 تشخیص و ثبت رویدادهای ورود و خروج
- 📹 پشتیبانی از چند دوربین و استریم RTSP
- ✅ مدیریت فهرست سفید و ⛔ فهرست سیاه
- 📊 داشبورد وب برای مشاهده رویدادها و آمار
- 📤 خروجی CSV و Excel
- 💾 پشتیبان‌گیری از پایگاه داده و پاک‌سازی تصاویر قدیمی

## 🧠 خط پردازش

```text
دوربین / ویدئو
      ↓
YOLOv8-Pose — تشخیص چهار گوشه پلاک
      ↓
تبدیل پرسپکتیو — صاف‌سازی تصویر پلاک
      ↓
CRNN + CTC — خواندن متن پلاک
      ↓
رأی‌گیری وزنی — تجمیع نتایج
      ↓
SQLite — ثبت رویدادهای ورود و خروج
```

## 📈 معیارهای گزارش‌شده

| مدل | معیار | مقدار |
|---|---|---:|
| YOLOv8-Pose | mAP@50 | ۹۵٫۸٪ |
| YOLOv8-Pose | Precision | ۹۷٫۲٪ |
| YOLOv8-Pose | Recall | ۹۳٫۵٪ |
| CRNN | Sequence Accuracy | ۸۷٫۷٪ |
| CRNN | CER | ۲٫۶٪ |

> این اعداد مطابق اطلاعات فعلی پروژه درج شده‌اند؛ نتیجهٔ واقعی با داده، کیفیت تصویر و شرایط دوربین تغییر می‌کند.

## 🧰 پیش‌نیازها

| مورد | نیازمندی |
|---|---|
| Python | نسخهٔ 3.10 یا بالاتر |
| حافظه | حداقل 8 گیگابایت RAM |
| فضای دیسک | حدود 5 گیگابایت |
| GPU | کارت NVIDIA با CUDA توصیه می‌شود؛ اجرای CPU به مدل و سرعت موردنیاز بستگی دارد |
| Git LFS | برای دریافت فایل‌های مدلِ ذخیره‌شده با LFS |

## 🚀 نصب و اجرا

### ۱. دریافت پروژه

```bash
git lfs install
git clone https://github.com/mostafa5804/mostafa-ANPR.git
cd mostafa-ANPR
```

### ۲. ساخت محیط مجازی

**Windows PowerShell**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

### ۳. نصب وابستگی‌ها

برای نصب وابستگی‌های ثبت‌شده در پروژه:

```bash
pip install -r requirements.txt
```

اگر به اجرای CUDA نیاز دارید، نسخهٔ PyTorch را متناسب با نسخهٔ CUDA و سیستم خود از راهنمای رسمی PyTorch نصب کنید.

### ۴. اجرای داشبورد و API

```bash
python -m src.api.app
```

سپس در مرورگر باز کنید:

- داشبورد: [http://localhost:8000](http://localhost:8000)
- مستندات API: [http://localhost:8000/docs](http://localhost:8000/docs)

> برای استفاده از دوربین، آدرس RTSP را در بخش مدیریت دوربین‌ها یا در فایل تنظیمات دوربین وارد کنید. آدرس نمونه زیر را با مشخصات دوربین خود جایگزین کنید.

نمونهٔ ساختار `data/cameras.json`:

```json
[
  {
    "id": "cam-1",
    "name": "دوربین ورودی",
    "rtsp": "rtsp://USERNAME:PASSWORD@CAMERA_IP:554/STREAM_PATH",
    "enabled": true
  }
]
```

### ۵. اجرای سرویس پلاک‌خوان

دوربین پیش‌فرض:

```bash
python -m src.service.main_service
```

دوربین مشخص:

```bash
python -m src.service.main_service --camera-id cam-2
```

پردازش فایل ویدئویی:

```bash
python -m src.service.main_service --video samples/video.mp4
```

## 🔌 API

| روش | مسیر | کاربرد |
|---|---|---|
| GET | `/api/stats` | آمار کلی |
| GET | `/api/events` | فهرست رویدادها |
| GET | `/api/events/export/csv` | دریافت خروجی CSV |
| GET | `/api/events/export/excel` | دریافت خروجی Excel |
| GET | `/api/vehicles` | فهرست خودروها |
| GET / POST | `/api/whitelist` | مشاهده یا افزودن به فهرست سفید |
| GET / POST | `/api/blacklist` | مشاهده یا افزودن به فهرست سیاه |
| GET | `/api/cameras` | فهرست دوربین‌ها |
| GET | `/api/camera/{id}/stream` | دریافت استریم دوربین |
| GET | `/api/backup/list` | فهرست پشتیبان‌ها |
| POST | `/api/backup/create` | ایجاد پشتیبان |

## 🗂️ ساختار اصلی پروژه

```text
mostafa-ANPR/
├── src/
│   ├── api/          # API و داشبورد
│   ├── database/     # مدل‌ها، عملیات پایگاه داده و پشتیبان‌گیری
│   ├── ocr/          # مدل و آموزش OCR
│   ├── pose/         # مدل و آموزش تشخیص نقاط پلاک
│   ├── pipeline/     # خط پردازش تصویر و ویدئو
│   ├── service/      # سرویس دوربین و تشخیص جهت
│   └── config.py
├── data/             # تنظیمات دوربین، پایگاه داده و تصاویر
├── outputs/          # وزن‌های مدل
├── requirements.txt
├── LICENSE
└── README.md
```

## 🏋️ آموزش مدل‌ها

آموزش OCR:

```bash
python -m src.ocr.train
```

آموزش مدل Pose:

```bash
python -m src.pose.train
```

داده‌های آموزشی موردنیاز را در مسیرهای مربوط به هر مدل قرار دهید. تنظیمات آموزش در `src/config.py` تعریف شده‌اند.

## 🛠️ رفع اشکال

<details>
<summary>خطای کمبود حافظهٔ GPU</summary>

اندازهٔ batch را در تنظیمات آموزش کاهش دهید؛ برای نمونه از 32 به 8.

</details>

<details>
<summary>خطای پیدا نشدن PyTorch</summary>

محیط مجازی را فعال کنید و وابستگی‌ها را در همان محیط نصب کنید. نصب CUDA باید با نسخهٔ درایور و CUDA سیستم سازگار باشد.

</details>

<details>
<summary>عدم اتصال دوربین</summary>

- آدرس RTSP را بررسی کنید.
- دسترسی شبکه به دوربین و پورت RTSP را بررسی کنید.
- نام کاربری و گذرواژه را کنترل کنید.
- اتصال را با VLC یا FFmpeg آزمایش کنید.
- دسترسی فایروال را بررسی کنید.

</details>

## 🌐 دربارهٔ GitHub Pages

داشبورد فعلی با FastAPI و سرویس پردازش دوربین اجرا می‌شود؛ بنابراین به یک محیط Python در دسترس نیاز دارد و به‌تنهایی روی GitHub Pages (میزبانی ایستای HTML) اجرا نمی‌شود. برای استفاده، پروژه را طبق دستورهای بالا روی رایانه یا سرور اجرا کنید.

## 📄 مجوز

این پروژه تحت مجوز [MIT](LICENSE) منتشر شده است.

## 👨‍💻 سازنده

**Mostafa Erfani** · [GitHub: @mostafa5804](https://github.com/mostafa5804)

## 🙏 سپاس‌گزاری

- [Ultralytics](https://github.com/ultralytics/ultralytics) — ابزارهای YOLO
- [Hezar](https://github.com/hezarai/hezar) — منابع مرتبط با پلاک فارسی
- [Roboflow](https://roboflow.com/) — ابزارهای داده و بینایی ماشین
- [PersianTools](https://github.com/majiidd/persiantools) — ابزارهای زبان فارسی

---

<div align="center">

اگر این پروژه برایتان مفید بود، با ⭐ دادن به مخزن از آن حمایت کنید.

</div>
