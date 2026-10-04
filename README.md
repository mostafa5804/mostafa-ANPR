\# ANPR - سیستم پلاک خوان خودروهای ایرانی



سیستم هوشمند تشخیص و خواندن پلاک خودروهای ایرانی با استفاده از YOLOv8-Pose و CRNN.



\## قابلیت ها



\- تشخیص ۴ گوشه پلاک با YOLOv8-Pose

\- خواندن متن پلاک با CRNN

\- تشخیص هوشمند ورود و خروج

\- پشتیبانی از ۴ دوربین همزمان

\- لیست سفید و لیست سیاه

\- داشبورد وب کامل

\- خروجی CSV و Excel

\- بکاپ خودکار دیتابیس

\- حذف خودکار تصاویر قدیمی



\## عملکرد مدل ها



| مدل | متریک | مقدار |

| :--- | :--- | :--- |

| Pose (YOLOv8) | mAP@50 | ۹۵.۸ درصد |

| Pose (YOLOv8) | Precision | ۹۷.۲ درصد |

| Pose (YOLOv8) | Recall | ۹۳.۵ درصد |

| OCR (CRNN) | Sequence Accuracy | ۸۷.۷ درصد |

| OCR (CRNN) | CER | ۲.۶ درصد |



\## معماری



دوربین

&#x20;  |

&#x20;  v

YOLOv8-Pose  (تشخیص ۴ گوشه پلاک)

&#x20;  |

&#x20;  v

Perspective Transform  (صاف کردن پلاک)

&#x20;  |

&#x20;  v

CRNN + CTC  (خواندن متن پلاک)

&#x20;  |

&#x20;  v

رأی گیری وزنی  (افزایش دقت)

&#x20;  |

&#x20;  v

SQLite  (ذخیره ورود و خروج)



\## پیش نیازها



| مورد | حداقل |

| :--- | :--- |

| Python | 3.10 به بالا |

| GPU | NVIDIA با CUDA (توصیه GTX 1650 به بالا) |

| RAM | 8 گیگابایت |

| فضا | 5 گیگابایت |

| Git LFS | برای دانلود مدل ها |



\## نصب



مرحله ۱ - نصب Git LFS



git lfs install



مرحله ۲ - کلون پروژه



git clone https://github.com/YOUR\_USERNAME/mostafa-ANPR.git

cd mostafa-ANPR



مرحله ۳ - محیط مجازی



python -m venv venv



در ویندوز:

venv\\Scripts\\activate



در لینوکس و مک:

source venv/bin/activate



مرحله ۴ - نصب PyTorch CUDA



pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124



مرحله ۵ - نصب بقیه بسته ها



pip install -r requirements.txt



\## تست CUDA



python -c "import torch; print('CUDA:', torch.cuda.is\_available())"



خروجی مورد انتظار:

CUDA: True



\## استفاده



\### ۱. تنظیم دوربین



از داشبورد:

python -m src.api.app



برو به http://localhost:8000 و بخش مدیریت دوربین ها



یا از فایل data/cameras.json:



\[

&#x20; {

&#x20;   "id": "cam-1",

&#x20;   "name": "دوربین ورودی",

&#x20;   "rtsp": "rtsp://admin:pass@192.168.1.100:554/Streaming/Channels/101",

&#x20;   "enabled": true

&#x20; }

]



\### ۲. اجرای API



python -m src.api.app



آدرس API: http://localhost:8000

مستندات: http://localhost:8000/docs



\### ۳. اجرای سرویس پلاک خوان



دوربین پیش فرض:

python -m src.service.main\_service



دوربین مشخص:

python -m src.service.main\_service --camera-id cam-2



فایل ویدیو:

python -m src.service.main\_service --video samples/video.mp4



\### ۴. داشبورد



http://localhost:8000



\## API Endpoints



| متد | Endpoint | توضیح |

| :--- | :--- | :--- |

| GET | /api/stats | آمار کلی |

| GET | /api/events | رویدادها |

| GET | /api/events/export/csv | خروجی CSV |

| GET | /api/events/export/excel | خروجی Excel |

| GET | /api/vehicles | خودروها |

| GET | /api/whitelist | لیست سفید |

| POST | /api/whitelist | افزودن به لیست سفید |

| GET | /api/blacklist | لیست سیاه |

| POST | /api/blacklist | افزودن به لیست سیاه |

| GET | /api/cameras | دوربین ها |

| GET | /api/camera/{id}/stream | استریم زنده |

| GET | /api/backup/list | لیست بکاپ ها |

| POST | /api/backup/create | ایجاد بکاپ |



\## ساختار پروژه



mostafa-ANPR/

&#x20; src/

&#x20;   api/

&#x20;     app.py

&#x20;     dashboard.html

&#x20;   service/

&#x20;     main\_service.py

&#x20;     direction.py

&#x20;     cleanup.py

&#x20;   database/

&#x20;     models.py

&#x20;     crud.py

&#x20;     backup.py

&#x20;   ocr/

&#x20;     model.py

&#x20;     dataset.py

&#x20;     train.py

&#x20;     predict.py

&#x20;   pose/

&#x20;     train.py

&#x20;     predict.py

&#x20;   pipeline/

&#x20;     full\_pipeline.py

&#x20;     video\_pipeline.py

&#x20;   config.py

&#x20; outputs/

&#x20;   ocr/

&#x20;     best.pt

&#x20;   pose/

&#x20;     best.pt

&#x20; data/

&#x20;   cameras.json

&#x20;   pose/

&#x20;     data.yaml

&#x20;   anpr.db

&#x20;   snapshots/

&#x20;   backups/

&#x20; requirements.txt

&#x20; .gitignore

&#x20; .gitattributes

&#x20; LICENSE

&#x20; README.md



\## آموزش مدل ها



آموزش OCR:

python -m src.ocr.train



دیتاست: data/ocr/

خروجی: outputs/ocr/best.pt



آموزش Pose:

python -m src.pose.train



دیتاست: data/pose/

خروجی: outputs/pose/best.pt



\## تنظیمات



فایل src/config.py:



OCR:

IMAGE\_WIDTH = 160

IMAGE\_HEIGHT = 32

BATCH\_SIZE = 32

LEARNING\_RATE = 0.001

EPOCHS = 50



Pose:

POSE\_IMGSZ = 960

POSE\_CONF = 0.10

POSE\_EPOCHS = 20



\## تکنولوژی ها



| بخش | تکنولوژی |

| :--- | :--- |

| Deep Learning | PyTorch 2.6 |

| تشخیص پلاک | Ultralytics YOLOv8-Pose |

| خواندن متن | CRNN + CTC Loss |

| API | FastAPI + Uvicorn |

| دیتابیس | SQLite |

| داشبورد | HTML + Tailwind + Chart.js |

| پردازش تصویر | OpenCV |

| تاریخ شمسی | persiantools |



\## رفع مشکلات



خطای CUDA out of memory:

در config.py مقدار BATCH\_SIZE را به 8 کاهش بده



خطای No module named torch:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124



خطای اتصال به دوربین:

آدرس RTSP را در VLC تست کن

مطمئن شو دوربین روی همان شبکه است

فایروال ویندوز را چک کن



\## لایسنس



این پروژه تحت لایسنس MIT منتشر شده است. فایل LICENSE را ببین.



\## سازنده



نام شما



GitHub: https://github.com/YOUR\_USERNAME

Email: your.email@example.com



\## تشکر



Ultralytics - YOLOv8

Hezar - دیتاست پلاک ایران

Roboflow - دیتاست Pose

PersianTools - تاریخ شمسی



اگر این پروژه برایت مفید بود، یک ستاره بده

