"""
تنظیمات پروژه ANPR
"""

# ==================== OCR ====================
CHAR_LIST = [
    '-',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'ا', 'ب', 'پ', 'ت', 'ث', 'ج', 'د', 'ز', 'ژ', 'س', 'ش',
    'ص', 'ط', 'ع', 'ف', 'ق', 'ک', 'گ', 'ل', 'م', 'ن', 'و',
    'ه', 'ی',
]

CHAR2IDX = {char: idx for idx, char in enumerate(CHAR_LIST)}
IDX2CHAR = {idx: char for idx, char in enumerate(CHAR_LIST)}

IMAGE_WIDTH = 160      # ← تغییر از ۱۲۸ به ۱۶۰ (نسبت ۵.۰)
IMAGE_HEIGHT = 32
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50
NUM_CLASSES = len(CHAR_LIST)

# مسیرها
OCR_TRAIN_LABELS = "data/ocr/train_labels.txt"
OCR_VAL_LABELS = "data/ocr/val_labels.txt"
OCR_IMAGES_DIR = "data/ocr/images"
OCR_CHECKPOINT_DIR = "outputs/ocr"

# ==================== Pose ====================
POSE_DATA_YAML = "data/pose/data.yaml"
POSE_CHECKPOINT_DIR = "outputs/pose"
POSE_EPOCHS = 20
POSE_BATCH = 16
POSE_IMGSZ = 640
POSE_CONF = 0.10
POSE_KPT_SHAPE = [4, 3]  # 4 keypoint, هر کدوم 3 مقدار (x, y, visibility)