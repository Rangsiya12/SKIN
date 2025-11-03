import os
import io
import sys
import logging
from flask import Flask, request, abort, send_from_directory
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, ImageMessage, TextSendMessage,
    ImageSendMessage, QuickReply, QuickReplyButton, MessageAction
)
import tempfile
import base64
import time
import random

# ตั้งค่า logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ตรวจสอบและ import โมดูลที่จำเป็น
try:
    import numpy as np
    test_array = np.array([1, 2, 3])
    NUMPY_AVAILABLE = True
    logger.info(f"NumPy imported successfully - version: {np.__version__}")
except Exception as e:
    logger.error(f"NumPy not available or not working: {e}")
    NUMPY_AVAILABLE = False

try:
    import torch
    if NUMPY_AVAILABLE:
        test_tensor = torch.tensor([1, 2, 3])
        test_numpy = test_tensor.cpu().numpy()
        logger.info(f"PyTorch-NumPy integration working")
    TORCH_AVAILABLE = True
    logger.info(f"PyTorch imported successfully - version: {torch.__version__}")
except Exception as e:
    logger.error(f"PyTorch not available or NumPy integration failed: {e}")
    TORCH_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
    logger.info("OpenCV imported successfully")
except ImportError as e:
    logger.error(f"OpenCV not available: {e}")
    CV2_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
    logger.info("PIL imported successfully")
except ImportError as e:
    logger.error(f"PIL not available: {e}")
    PIL_AVAILABLE = False

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
    logger.info("Ultralytics imported successfully")
except ImportError as e:
    logger.error(f"Ultralytics not available: {e}")
    ULTRALYTICS_AVAILABLE = False

app = Flask(__name__)

# ตั้งค่า LINE Bot จาก Railway Environment Variables
LINE_CHANNEL_ACCESS_TOKEN = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.getenv('LINE_CHANNEL_SECRET')

# ตั้งค่า BASE_URL อัตโนมัติสำหรับ Railway - แก้ไขใหม่
RAILWAY_PUBLIC_DOMAIN = os.getenv('RAILWAY_PUBLIC_DOMAIN')
if RAILWAY_PUBLIC_DOMAIN:
    BASE_URL = f"https://{RAILWAY_PUBLIC_DOMAIN}"
else:
    # ลองใช้ static url
    RAILWAY_STATIC_URL = os.getenv('RAILWAY_STATIC_URL')
    if RAILWAY_STATIC_URL:
        BASE_URL = RAILWAY_STATIC_URL
    else:
        # สร้าง URL จากชื่อ project
        project_name = os.getenv('RAILWAY_PROJECT_NAME', 'skin-cancer-linebot-v8')
        BASE_URL = f"https://{project_name}.up.railway.app"

logger.info(f"BASE_URL set to: {BASE_URL}")

if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_CHANNEL_SECRET:
    logger.error("LINE credentials not found in environment variables")
    raise ValueError("LINE credentials required")

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# โหลด YOLO model
MODEL_PATH = 'models/best.pt'
model = None

if ULTRALYTICS_AVAILABLE and TORCH_AVAILABLE and NUMPY_AVAILABLE:
    try:
        # ตั้งค่า device เป็น CPU เพื่อหลีกเลี่ยงปัญหา
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        
        if os.path.exists(MODEL_PATH):
            model = YOLO(MODEL_PATH)
            model.to('cpu')
            logger.info("Custom model loaded successfully on CPU")
        else:
            logger.warning(f"Model file not found at {MODEL_PATH}, using YOLOv8n")
            model = YOLO('yolov8n.pt')
            model.to('cpu')
            logger.info("Fallback model loaded successfully on CPU")
            
        # ทดสอบโมเดล
        try:
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            test_results = model(test_img, device='cpu', verbose=False)
            logger.info("Model test prediction successful")
        except Exception as test_error:
            logger.warning(f"Model test failed: {test_error}")
            
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model = None
else:
    missing_modules = []
    if not ULTRALYTICS_AVAILABLE:
        missing_modules.append("ultralytics")
    if not TORCH_AVAILABLE:
        missing_modules.append("torch")
    if not NUMPY_AVAILABLE:
        missing_modules.append("numpy")
    logger.warning(f"Required dependencies not available: {missing_modules}. Model not loaded.")

# คลาสโรคผิวหนัง
SKIN_CANCER_CLASSES = {
    0: "Melanoma",
    1: "Nevus", 
    2: "Seborrheic Keratosis"
}

SKIN_CANCER_CLASSES_TH = {
    0: "เมลาโนมา (Melanoma)",
    1: "เนวัส (Nevus)", 
    2: "เซบอร์รีอิก เคราโทซิส (Seborrheic Keratosis)"
}

RISK_LEVELS = {
    0: "ความเสี่ยงสูง - ควรปรึกษาแพทย์",
    1: "ความเสี่ยงต่ำ",
    2: "ความเสี่ยงปานกลาง"
}

CLASS_COLORS = {
    0: (255, 0, 0),    # แดง
    1: (0, 255, 0),    # เขียว
    2: (255, 165, 0)   # ส้ม
}

def save_image_temporarily(image, filename):
    """บันทึกรูปภาพชั่วคราวสำหรับ Railway - แก้ไขปัญหา bounding box"""
    try:
        # สร้างโฟลเดอร์ static สำหรับ Railway
        static_dir = "static"
        if not os.path.exists(static_dir):
            os.makedirs(static_dir)
        
        # สร้างโฟลเดอร์ย่อย images
        images_dir = os.path.join(static_dir, "images")
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
        
        # บันทึกรูปภาพ
        file_path = os.path.join(images_dir, filename)
        
        # ตรวจสอบว่ารูปภาพเป็น PIL Image object
        if not isinstance(image, Image.Image):
            logger.error(f"Invalid image type: {type(image)}")
            return None, None
        
        # แปลงเป็น RGB ก่อนบันทึกเป็น JPEG (แก้ไขปัญหา bounding box หาย)
        if image.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            background.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # บันทึกด้วยคุณภาพสูงเพื่อไม่ให้ bounding box เสีย
        image.save(file_path, 'JPEG', quality=95, optimize=False)
        
        # ตรวจสอบว่าไฟล์ถูกสร้างแล้ว
        if not os.path.exists(file_path):
            raise Exception("ไม่สามารถสร้างไฟล์รูปภาพได้")
        
        # ตรวจสอบขนาดไฟล์
        file_size = os.path.getsize(file_path)
        logger.info(f"Image saved: {file_path}, Size: {file_size} bytes")
        
        # สร้าง URL หลายรูปแบบ
        image_urls = [
            f"{BASE_URL}/static/images/{filename}",
            f"{BASE_URL}/images/{filename}",
            f"{BASE_URL}/serve_image/{filename}"
        ]
        
        logger.info(f"Image URLs: {image_urls}")
        
        return image_urls, file_path
        
    except Exception as e:
        logger.error(f"Error saving image temporarily: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return None, None

def cleanup_old_images():
    """ลบไฟล์รูปภาพเก่า"""
    try:
        for dir_name in ["static/images", "temp_images"]:
            if not os.path.exists(dir_name):
                continue
            
            current_time = time.time()
            max_age = 3600  # 1 hour
            
            for filename in os.listdir(dir_name):
                file_path = os.path.join(dir_name, filename)
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getctime(file_path)
                    if file_age > max_age:
                        try:
                            os.remove(file_path)
                            logger.info(f"Cleaned up old file: {filename}")
                        except Exception as e:
                            logger.error(f"Error removing file {filename}: {e}")
                            
    except Exception as e:
        logger.error(f"Error in cleanup_old_images: {e}")

def download_image_from_line(message_id):
    """ดาวน์โหลดรูปภาพจาก LINE"""
    if not PIL_AVAILABLE:
        logger.error("PIL not available for image processing")
        return None
        
    try:
        message_content = line_bot_api.get_message_content(message_id)
        image_data = io.BytesIO()
        for chunk in message_content.iter_content():
            image_data.write(chunk)
        image_data.seek(0)
        
        # เปิดรูปภาพและตรวจสอบ
        image = Image.open(image_data)
        logger.info(f"Downloaded image: {image.size}, mode: {image.mode}")
        return image
        
    except Exception as e:
        logger.error(f"Error downloading image: {e}")
        return None

def draw_bounding_boxes(image, results):
    """วาด bounding boxes บนรูปภาพ - ปรับปรุงขนาด font"""
    try:
        # ตรวจสอบว่าเป็น PIL Image
        if not isinstance(image, Image.Image):
            logger.error(f"Invalid image type for drawing: {type(image)}")
            return image
        
        # แปลงเป็น RGB ถ้าจำเป็น
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # สร้างสำเนาของรูปภาพเพื่อวาด bounding box
        img_with_boxes = image.copy()
        draw = ImageDraw.Draw(img_with_boxes)
        
        # คำนวณขนาด font ตามขนาดรูปภาพ
        img_width, img_height = img_with_boxes.size
        
        # คำนวณขนาด font ที่เหมาะสม (สัดส่วนกับขนาดรูป)
        base_font_size = max(16, min(img_width, img_height) // 25)  # ขั้นต่ำ 16px
        
        # จำกัดขนาดสูงสุดเพื่อไม่ให้ใหญ่เกินไป
        font_size = min(base_font_size, 48)
        
        logger.info(f"Image size: {img_width}x{img_height}, calculated font size: {font_size}")
        
        # ลองใช้ font ต่างๆ ตามลำดับความสำคัญ
        font = None
        font_paths = [
            "arial.ttf",