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
            "Arial.ttf", 
            "arialbd.ttf",  # Arial Bold
            "calibri.ttf",
            "Calibri.ttf",
            "DejaVuSans.ttf",
            "DejaVuSans-Bold.ttf"
        ]
        
        # ลองโหลด TrueType font
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, size=font_size)
                logger.info(f"Successfully loaded font: {font_path}, size: {font_size}")
                break
            except (IOError, OSError):
                continue
        
        # ถ้าโหลด TrueType font ไม่ได้ ใช้ default font
        if font is None:
            try:
                # ใช้ default font และปรับขนาดถ้าเป็นไปได้
                font = ImageFont.load_default()
                logger.info(f"Using default font, target size: {font_size}")
                
                # ลองสร้าง default font ขนาดใหญ่ขึ้น (สำหรับ Pillow version ใหม่)
                try:
                    font = ImageFont.load_default(size=font_size)
                    logger.info(f"Default font loaded with size: {font_size}")
                except TypeError:
                    # Pillow version เก่าไม่รองรับ size parameter
                    font = ImageFont.load_default()
                    logger.info("Using basic default font (no size parameter)")
                    
            except Exception as font_error:
                logger.error(f"Cannot load any font: {font_error}")
                font = None
        
        logger.info(f"Drawing on image size: {img_with_boxes.size}")
        
        if len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            logger.info(f"Found {len(boxes)} boxes to draw")
            
            for i, box in enumerate(boxes):
                try:
                    # ดึงข้อมูล bounding box
                    if hasattr(box, 'xyxy') and len(box.xyxy) > 0:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        class_id = int(box.cls.item()) if hasattr(box.cls, 'item') else int(box.cls)
                        confidence = float(box.conf.item()) if hasattr(box.conf, 'item') else float(box.conf)
                        
                        logger.info(f"Box {i}: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}), class: {class_id}, conf: {confidence:.3f}")
                        
                        # ตรวจสอบว่า coordinates อยู่ในขอบเขตรูปภาพ
                        x1 = max(0, min(x1, img_width))
                        y1 = max(0, min(y1, img_height))
                        x2 = max(0, min(x2, img_width))
                        y2 = max(0, min(y2, img_height))
                        
                        # ตรวจสอบว่า bounding box มีขนาดที่เหมาะสม
                        if x2 <= x1 or y2 <= y1:
                            logger.warning(f"Invalid box dimensions: ({x1}, {y1}, {x2}, {y2})")
                            continue
                        
                        color = CLASS_COLORS.get(class_id, (255, 255, 0))
                        
                        # คำนวณความหนาของเส้นตามขนาดรูปและ bounding box
                        box_width = x2 - x1
                        box_height = y2 - y1
                        box_area = box_width * box_height
                        
                        # ความหนาของเส้นขึ้นอยู่กับขนาดของ bounding box
                        box_thickness = max(2, min(8, int((box_area / (img_width * img_height)) * 100)))
                        
                        # วาด bounding box หนาขึ้น
                        for thickness in range(box_thickness):
                            draw.rectangle([x1+thickness, y1+thickness, x2-thickness, y2-thickness], 
                                         outline=color, width=1)
                        
                        class_name = SKIN_CANCER_CLASSES.get(class_id, "Unknown")
                        
                        # สร้าง label text
                        main_label = f"{class_name}"
                        confidence_label = f"{confidence:.1%}"
                        
                        # วาด text labels ถ้ามี font
                        if font:
                            try:
                                # คำนวณขนาด text สำหรับแต่ละบรรทัด
                                main_bbox = draw.textbbox((0, 0), main_label, font=font)
                                conf_bbox = draw.textbbox((0, 0), confidence_label, font=font)
                                
                                main_width = main_bbox[2] - main_bbox[0]
                                main_height = main_bbox[3] - main_bbox[1]
                                conf_width = conf_bbox[2] - conf_bbox[0]
                                conf_height = conf_bbox[3] - conf_bbox[1]
                                
                                # หาความกว้างสูงสุด
                                max_width = max(main_width, conf_width)
                                line_spacing = max(4, font_size // 8)  # ระยะห่างระหว่างบรรทัด
                                total_height = main_height + conf_height + line_spacing
                                
                                # คำนวณตำแหน่ง text
                                text_x = x1
                                text_y = max(5, y1 - total_height - 10)
                                
                                # ถ้า text อยู่เหนือขอบบน ให้ย้ายลงมาใต้ bounding box
                                if text_y < 5:
                                    text_y = y2 + 5
                                
                                # คำนวณ padding สำหรับ background
                                padding = max(4, font_size // 8)
                                
                                # วาด background สำหรับ text
                                bg_x1 = text_x - padding
                                bg_y1 = text_y - padding
                                bg_x2 = text_x + max_width + padding
                                bg_y2 = text_y + total_height + padding
                                
                                # ตรวจสอบไม่ให้ background เกินขอบรูป
                                bg_x1 = max(0, bg_x1)
                                bg_y1 = max(0, bg_y1)
                                bg_x2 = min(img_width, bg_x2)
                                bg_y2 = min(img_height, bg_y2)
                                
                                # วาด background
                                draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=color)
                                
                                # วาดขอบสีขาวรอบ background
                                draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], 
                                             outline=(255, 255, 255), width=1)
                                
                                # กำหนดสีของ text ตาม class (เพื่อความเด่นชัด)
                                if class_id == 0:  # Melanoma (ความเสี่ยงสูง)
                                    #main_text_color = (255, 255, 255)  # ขาว
                                    main_text_color = (0, 0, 0)  # ดำ
                                    conf_text_color = (0, 0, 0)  # แดงอ่อน
                                elif class_id == 1:  # Nevus (ความเสี่ยงต่ำ)
                                    #main_text_color = (255, 255, 255)  # ขาว
                                    main_text_color = (0, 0, 0)  # ดำ
                                    conf_text_color = (0, 0, 0)  # เขียวอ่อน
                                else:  # Seborrheic Keratosis (ความเสี่ยงปานกลาง)
                                    #main_text_color = (255, 255, 255)  # ขาว
                                    main_text_color = (0, 0, 0)  # ดำ
                                    conf_text_color = (0, 0, 0)  # ส้มอ่อน
                                
                                # วาด text แต่ละบรรทัดพร้อม text shadow เพื่อความชัดเจน
                                current_y = text_y
                                
                                # เพิ่ม text shadow (เงา) เพื่อให้อ่านง่าย
                                shadow_offset = max(1, font_size // 16)
                                shadow_color = (0, 0, 0)  # เงาสีดำ
                                
                                # บรรทัดที่ 1: ชื่อโรค
                                # วาดเงาก่อน
                                draw.text((text_x + shadow_offset, current_y + shadow_offset), 
                                         main_label, fill=shadow_color, font=font)
                                # วาด text หลัก
                                draw.text((text_x, current_y), main_label, fill=main_text_color, font=font)
                                current_y += main_height + line_spacing
                                
                                # บรรทัดที่ 2: ความแม่นยำ
                                # วาดเงาก่อน
                                draw.text((text_x + shadow_offset, current_y + shadow_offset), 
                                         confidence_label, fill=shadow_color, font=font)
                                # วาด text หลัก (ใช้สีตาม class)
                                draw.text((text_x, current_y), confidence_label, fill=conf_text_color, font=font)
                                
                                logger.info(f"Drew text: {main_label} | {confidence_label} at ({text_x}, {text_y}) with font size {font_size}")
                                
                            except Exception as text_error:
                                logger.error(f"Error drawing text: {text_error}")
                                
                                # Fallback: วาดข้อความแบบง่ายพร้อมสีที่เหมาะสม
                                try:
                                    simple_label = f"{class_name} {confidence:.1%}"
                                    
                                    # กำหนดสี text สำหรับ fallback
                                    if class_id == 0:  # Melanoma
                                        text_color = (255, 200, 200)  # แดงอ่อน
                                    elif class_id == 1:  # Nevus
                                        text_color = (200, 255, 200)  # เขียวอ่อน
                                    else:  # Seborrheic Keratosis
                                        text_color = (255, 220, 150)  # ส้มอ่อน
                                    
                                    if font:
                                        bbox = draw.textbbox((0, 0), simple_label, font=font)
                                        text_width = bbox[2] - bbox[0]
                                        text_height = bbox[3] - bbox[1]
                                    else:
                                        # ประมาณขนาด text ถ้าไม่มี font
                                        text_width = len(simple_label) * (font_size // 2)
                                        text_height = font_size
                                    
                                    text_x = x1
                                    text_y = max(0, y1 - text_height - 10)
                                    
                                    # วาด background มีความโปร่งใส
                                    padding = 4
                                    bg_color = tuple(int(c * 0.8) for c in color)  # สีเข้มกว่าเดิมเล็กน้อย
                                    
                                    draw.rectangle([text_x-padding, text_y-padding, 
                                                  text_x+text_width+padding, text_y+text_height+padding], 
                                                 fill=bg_color)
                                    
                                    # วาดขอบขาว
                                    draw.rectangle([text_x-padding, text_y-padding, 
                                                  text_x+text_width+padding, text_y+text_height+padding], 
                                                 outline=(255, 255, 255), width=1)
                                    
                                    # วาด text shadow
                                    shadow_offset = 1
                                    if font:
                                        draw.text((text_x + shadow_offset, text_y + shadow_offset), 
                                                simple_label, fill=(0, 0, 0), font=font)
                                        draw.text((text_x, text_y), simple_label, fill=text_color, font=font)
                                    else:
                                        # ใช้ default font ถ้าไม่มี font
                                        draw.text((text_x + shadow_offset, text_y + shadow_offset), 
                                                simple_label, fill=(0, 0, 0))
                                        draw.text((text_x, text_y), simple_label, fill=text_color)
                                    
                                    logger.info(f"Drew fallback text: {simple_label} with color {text_color}")
                                    
                                except Exception as fallback_error:
                                    logger.error(f"Fallback text drawing failed: {fallback_error}")
                        else:
                            # ไม่มี font ใช้ได้ - วาดแค่ bounding box
                            logger.warning("No font available, drawing bounding box only")
                    
                except Exception as box_error:
                    logger.error(f"Error processing box {i}: {box_error}")
                    continue
        else:
            logger.warning("No valid boxes found in results")
        
        logger.info("Bounding box drawing completed")
        return img_with_boxes
        
    except Exception as e:
        logger.error(f"Error drawing bounding boxes: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return image
def predict_skin_cancer(image):
    """ทำนายโรคผิวหนังจากรูปภาพ - แก้ไขปัญหา bounding box"""
    if model is None:
        return None, None, "โมเดลไม่พร้อมใช้งาน"
    
    try:
        # ตรวจสอบ image input
        if not isinstance(image, Image.Image):
            logger.error(f"Invalid image input type: {type(image)}")
            return None, None, "รูปภาพไม่ถูกต้อง"
        
        logger.info(f"Input image: size={image.size}, mode={image.mode}")
        
        # ทดสอบ NumPy
        try:
            test_array = np.array([1, 2, 3])
            logger.info("NumPy test passed")
        except Exception as np_error:
            logger.error(f"NumPy test failed: {np_error}")
            return None, None, f"NumPy ไม่ทำงานอย่างถูกต้อง: {str(np_error)}"
        
        # แปลงรูปภาพเป็น numpy array
        try:
            # แปลงเป็น RGB ก่อน
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            img_array = np.array(image)
            logger.info(f"Image converted to array successfully - shape: {img_array.shape}")
        except Exception as img_error:
            logger.error(f"Failed to convert image to array: {img_error}")
            return None, None, f"ไม่สามารถแปลงรูปภาพ: {str(img_error)}"
        
        # ทำการทำนาย
        try:
            if hasattr(model, 'to'):
                model.to('cpu')
            
            # ใช้ numpy array โดยตรง
            results = model(img_array, device='cpu', verbose=False, conf=0.1)  # ลด threshold
            logger.info(f"Model prediction completed, results count: {len(results)}")
            
            # ตรวจสอบผลลัพธ์
            if len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and result.boxes is not None:
                    logger.info(f"Found {len(result.boxes)} detections")
                else:
                    logger.info("No boxes in result")
            
        except Exception as model_error:
            logger.error(f"Model prediction failed: {model_error}")
            import traceback
            logger.error(f"Model prediction traceback: {traceback.format_exc()}")
            return None, None, f"การทำนายล้มเหลว: {str(model_error)}"
        
        # วาด bounding boxes (ทำก่อน analyze results)
        img_with_boxes = draw_bounding_boxes(image, results)
        
        # วิเคราะห์ผลลัพธ์
        if len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            best_idx = 0
            best_conf = 0
            
            # หา detection ที่มี confidence สูงสุด
            for i, box in enumerate(boxes):
                conf = float(box.conf.item()) if hasattr(box.conf, 'item') else float(box.conf)
                if conf > best_conf:
                    best_conf = conf
                    best_idx = i
            
            best_detection = boxes[best_idx]
            class_id = int(best_detection.cls.item()) if hasattr(best_detection.cls, 'item') else int(best_detection.cls)
            confidence = float(best_detection.conf.item()) if hasattr(best_detection.conf, 'item') else float(best_detection.conf)
            
            prediction_result = {
                'class_id': class_id,
                'class_name': SKIN_CANCER_CLASSES_TH.get(class_id, "Unknown"),
                'confidence': confidence,
                'risk_level': RISK_LEVELS.get(class_id, "ไม่ทราบ"),
                'total_detections': len(boxes)
            }
            
            logger.info(f"Best prediction: {prediction_result}")
            return prediction_result, img_with_boxes, None
        else:
            logger.info("No detections found")
            return None, img_with_boxes, "ไม่พบรอยโรคผิวหนังในรูปภาพ"
            
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        import traceback
        logger.error(f"Full prediction traceback: {traceback.format_exc()}")
        return None, None, f"เกิดข้อผิดพลาดในการวิเคราะห์: {str(e)}"

def create_result_message(prediction_result):
    """สร้างข้อความผลลัพธ์"""
    if prediction_result is None:
        return "ไม่สามารถวิเคราะห์รูปภาพได้"
    
    message = f"""🏥 ผลการวิเคราะห์ภาพผิวหนัง

🔍 ผลการตรวจพบ: {prediction_result['class_name']}
📊 ความแม่นยำ: {prediction_result['confidence']:.2%}
⚠️ ระดับความเสี่ยง: {prediction_result['risk_level']}
📍 จำนวนจุดที่ตรวจพบ: {prediction_result.get('total_detections', 1)} จุด

⚕️ คำแนะนำ:"""
    
    if prediction_result['class_id'] == 0:  # เมลาโนมา
        message += "\n• ควรปรึกษาแพทย์ผิวหนังโดยเร็ว\n• อาจต้องการการตรวจเพิ่มเติม"
    elif prediction_result['class_id'] == 2:  # เซบอร์รีอิก เคราโทซิส
        message += "\n• ควรติดตามอาการ\n• หากมีการเปลี่ยนแปลง ควรพบแพทย์"
    else:  # เนวัส
        message += "\n• ดูแลสุขภาพผิวหนังอย่างสม่ำเสมอ\n• หลีกเลี่ยงแสงแดดจัด"
    
    message += "\n\n🎯 กรอบสีในรูปภาพ:"
    message += "\n🔴 แดง = ความเสี่ยงสูง (เมลาโนมา)"
    message += "\n🟠 ส้ม = ความเสี่ยงปานกลาง (เซบอร์รีอิก เคราโทซิส)"
    message += "\n🟢 เขียว = ความเสี่ยงต่ำ (เนวัส)"
    
    message += "\n\n⚠️ หมายเหตุ: ผลนี้เป็นเพียงการประเมินเบื้องต้น ควรปรึกษาแพทย์เพื่อการวินิจฉัยที่แม่นยำ"
    
    return message

# Routes
@app.route("/")
def home():
    return """
    <h1>LINE Bot Skin Cancer Detection - Fixed Bounding Box</h1>
    <p>Status: Active</p>
    <p>Model: """ + ("Loaded" if model is not None else "Not Loaded") + """</p>
    <p>BASE_URL: """ + BASE_URL + """</p>
    <p>Webhook URL: """ + BASE_URL + """/webhook</p>
    <p>Bounding Box Fix: Applied</p>
    """

# เพิ่ม routes หลายรูปแบบสำหรับการเสิร์ฟรูปภาพ
@app.route("/static/images/<filename>")
def serve_static_image(filename):
    """ให้บริการรูปภาพแบบ static"""
    try:
        return send_from_directory('static/images', filename)
    except Exception as e:
        logger.error(f"Error serving static image: {e}")
        abort(404)

@app.route("/images/<filename>")
def serve_image_alt(filename):
    """ให้บริการรูปภาพแบบทางเลือก"""
    try:
        # ลองหาใน static/images ก่อน
        if os.path.exists(os.path.join('static/images', filename)):
            return send_from_directory('static/images', filename)
        # ลองหาใน temp_images
        elif os.path.exists(os.path.join('temp_images', filename)):
            return send_from_directory('temp_images', filename)
        else:
            abort(404)
    except Exception as e:
        logger.error(f"Error serving image alt: {e}")
        abort(404)

@app.route("/serve_image/<filename>")
def serve_image_custom(filename):
    """ให้บริการรูปภาพแบบกำหนดเอง"""
    try:
        # ลองหาใน static/images ก่อน
        if os.path.exists(os.path.join('static/images', filename)):
            return send_from_directory('static/images', filename)
        # ลองหาใน temp_images
        elif os.path.exists(os.path.join('temp_images', filename)):
            return send_from_directory('temp_images', filename)
        else:
            abort(404)
    except Exception as e:
        logger.error(f"Error serving custom image: {e}")
        abort(404)

@app.route("/temp_images/<filename>")
def serve_temp_image(filename):
    """ให้บริการรูปภาพชั่วคราว"""
    try:
        return send_from_directory('temp_images', filename)
    except Exception as e:
        logger.error(f"Error serving temp image: {e}")
        abort(404)

@app.route("/health")
def health_check():
    """ตรวจสอบสถานะระบบ"""
    try:
        status = {
            "status": "ok",
            "model_loaded": model is not None,
            "numpy_available": NUMPY_AVAILABLE,
            "torch_available": TORCH_AVAILABLE,
            "cv2_available": CV2_AVAILABLE,
            "pil_available": PIL_AVAILABLE,
            "ultralytics_available": ULTRALYTICS_AVAILABLE,
            "base_url": BASE_URL,
            "directories": {
                "static_images": os.path.exists('static/images'),
                "temp_images": os.path.exists('temp_images')
            },
            "bounding_box_fix": "applied"
        }
        return status, 200
    except Exception as e:
        return {"status": "error", "message": str(e)}, 500

@app.before_request
def before_request():
    """ทำความสะอาดไฟล์เก่าก่อนประมวลผล request"""
    if random.randint(1, 10) == 1:
        cleanup_old_images()

@app.route("/webhook", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        logger.error("Invalid signature")
        abort(400)

    return 'OK', 200

@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    """จัดการข้อความข้อความ"""
    text = event.message.text.lower()
    
    if 'สวัสดี' in text or 'hello' in text.lower():
        reply_text = """สวัสดีครับ! 👋

ผมเป็นบอทช่วยตรวจโรคผิวหนังเบื้องต้น

📸 วิธีใช้งาน:
1. ส่งรูปภาพผิวหนังที่ต้องการตรวจ
2. รอผลการวิเคราะห์
3. ได้รับรูปภาพพร้อม bounding box สีใส
4. ได้รับคำแนะนำเบื้องต้น

🎯 สีของกรอบ:
🔴 แดง = ความเสี่ยงสูง
🟠 ส้ม = ความเสี่ยงปานกลาง
🟢 เขียว = ความเสี่ยงต่ำ 

⚠️ สำคัญ: ผลการตรวจเป็นเพียงข้อมูลเบื้องต้น ควรปรึกษาแพทย์เพื่อการวินิจฉัยที่แม่นยำ"""
        
    elif 'สถานะ' in text or 'status' in text.lower():
        reply_text = f"""✅ สถานะระบบ: พร้อมใช้งาน

🤖 โมเดล: {'✅ พร้อมใช้งาน' if model is not None else '❌ ไม่พร้อม'}
📦 NumPy: {'✅' if NUMPY_AVAILABLE else '❌'}
🔥 PyTorch: {'✅' if TORCH_AVAILABLE else '❌'}
🖼️ OpenCV: {'✅' if CV2_AVAILABLE else '❌'}
🎨 PIL: {'✅' if PIL_AVAILABLE else '❌'}
🚀 Ultralytics: {'✅' if ULTRALYTICS_AVAILABLE else '❌'}
🌐 Base URL: {BASE_URL}
📁 Static Dir: {'✅' if os.path.exists('static/images') else '❌'}
📁 Temp Dir: {'✅' if os.path.exists('temp_images') else '❌'}

🎯 ฟีเจอร์ Bounding Box: ✅ แก้ไขแล้ว

ระบบพร้อมรับรูปภาพเพื่อวิเคราะห์และแสดงผลด้วย bounding box ที่ชัดเจน"""
        
    else:
        reply_text = """กรุณาส่งรูปภาพผิวหนังที่ต้องการตรวจ 📸

คำสั่งที่ใช้ได้:
• "สถานะ" - ตรวจสอบสถานะระบบ

🎯 ระบบจะส่งรูปภาพกลับพร้อมกรอบสีแสดงผลการตรวจที่ชัดเจน"""
    
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply_text)
    )

@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    """จัดการรูปภาพ - แก้ไขปัญหา bounding box"""
    try:
        # ส่งข้อความแจ้งว่ากำลังประมวลผล
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="🔍 กำลังวิเคราะห์รูปภาพ กรุณารอสักครู่...")
        )
        
        # ดาวน์โหลดรูปภาพ
        image = download_image_from_line(event.message.id)
        if image is None:
            line_bot_api.push_message(
                event.source.user_id,
                TextSendMessage(text="ไม่สามารถดาวน์โหลดรูปภาพได้ กรุณาลองใหม่")
            )
            return
        
        logger.info(f"Processing image: {image.size}, mode: {image.mode}")
        
        # ทำการทำนาย
        prediction, img_with_boxes, error = predict_skin_cancer(image)
        
        if error:
            line_bot_api.push_message(
                event.source.user_id,
                TextSendMessage(text=f"เกิดข้อผิดพลาด: {error}")
            )
            return
        
        # สร้างข้อความผลลัพธ์
        result_message = create_result_message(prediction)
        
        # บันทึกและส่งรูปภาพที่มี bounding box
        if img_with_boxes is not None:
            try:
                # สร้างชื่อไฟล์ unique
                timestamp = int(time.time())
                random_num = random.randint(1000, 9999)
                filename = f"result_{timestamp}_{random_num}.jpg"
                
                logger.info(f"Saving processed image: {filename}")
                
                # บันทึกรูปภาพชั่วคราว
                image_urls, file_path = save_image_temporarily(img_with_boxes, filename)
                
                success_sent = False
                
                if image_urls and file_path:
                    # ตรวจสอบว่าไฟล์ถูกบันทึกจริง
                    if os.path.exists(file_path):
                        file_size = os.path.getsize(file_path)
                        logger.info(f"Image file saved successfully: {file_path}, size: {file_size} bytes")
                        
                        # ลองส่งรูปภาพด้วย URL แต่ละตัว
                        for i, image_url in enumerate(image_urls):
                            try:
                                logger.info(f"Attempting to send image with URL {i+1}: {image_url}")
                                
                                messages = [
                                    ImageSendMessage(
                                        original_content_url=image_url,
                                        preview_image_url=image_url
                                    ),
                                    TextSendMessage(text=result_message)
                                ]
                                
                                line_bot_api.push_message(event.source.user_id, messages)
                                logger.info(f"Image sent successfully with URL: {image_url}")
                                success_sent = True
                                break
                                
                            except Exception as url_error:
                                logger.warning(f"Failed to send image with URL {image_url}: {url_error}")
                                continue
                    else:
                        logger.error(f"Image file was not saved: {file_path}")
                
                if not success_sent:
                    # ถ้าส่งรูปภาพไม่ได้ทุก URL ส่งแค่ข้อความ
                    logger.error("All image URLs failed, sending text only")
                    line_bot_api.push_message(
                        event.source.user_id,
                        TextSendMessage(text=f"{result_message}\n\n⚠️ ไม่สามารถส่งรูปภาพผลลัพธ์ได้ แต่การวิเคราะห์เสร็จสมบูรณ์แล้ว")
                    )
                else:
                    logger.info("Image with bounding boxes sent successfully")
                    
            except Exception as img_error:
                logger.error(f"Error in image processing: {img_error}")
                import traceback
                logger.error(f"Image processing traceback: {traceback.format_exc()}")
                
                # ส่งแค่ข้อความผลลัพธ์
                line_bot_api.push_message(
                    event.source.user_id,
                    TextSendMessage(text=f"{result_message}\n\n⚠️ เกิดข้อผิดพลาดในการส่งรูปภาพผลลัพธ์: {str(img_error)}")
                )
        else:
            # ไม่มีรูปภาพ ส่งแค่ข้อความ
            line_bot_api.push_message(
                event.source.user_id,
                TextSendMessage(text=result_message)
            )
            
    except Exception as e:
        logger.error(f"Error in handle_image_message: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        line_bot_api.push_message(
            event.source.user_id,
            TextSendMessage(text=f"เกิดข้อผิดพลาดในการประมวลผล: {str(e)}")
        )

if __name__ == "__main__":
    print("🚀 Starting LINE Bot Server on Railway...")
    print(f"📡 BASE_URL: {BASE_URL}")
    print(f"🤖 Model Status: {'✅ Loaded' if model is not None else '❌ Not Loaded'}")
    print("🎯 Bounding Box Fix: Applied ✅")
    
    # สร้างโฟลเดอร์ที่จำเป็น
    directories = ["temp_images", "static", "static/images"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Created directory: {directory}")
    
    # รันแอป
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
