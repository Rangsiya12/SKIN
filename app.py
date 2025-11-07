import os
import io
import sys
import logging
import tempfile
import base64
import time
import random

from flask import Flask, request, abort, send_from_directory
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, ImageMessage,
    TextSendMessage, ImageSendMessage,
    QuickReply, QuickReplyButton, MessageAction
)

# --- Optional Libraries ---
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except:
    NUMPY_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except:
    TORCH_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except:
    CV2_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except:
    PIL_AVAILABLE = False

try:
    from ultralytics import YOLO
    import ultralytics
    ULTRALYTICS_AVAILABLE = True
except:
    ULTRALYTICS_AVAILABLE = False

# ================================================
# 🚀 App Initialization / การเริ่มต้นแอป Flask
# ================================================
app = Flask(__name__)

LINE_CHANNEL_ACCESS_TOKEN = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.getenv('LINE_CHANNEL_SECRET')

RAILWAY_PUBLIC_DOMAIN = os.getenv('RAILWAY_PUBLIC_DOMAIN')
if RAILWAY_PUBLIC_DOMAIN:
    BASE_URL = f"https://{RAILWAY_PUBLIC_DOMAIN}"
else:
    RAILWAY_STATIC_URL = os.getenv('RAILWAY_STATIC_URL')
    if RAILWAY_STATIC_URL:
        BASE_URL = RAILWAY_STATIC_URL
    else:
        project_name = os.getenv('RAILWAY_PROJECT_NAME', 'skin-cancer-linebot-v8')
        BASE_URL = f"https://{project_name}.up.railway.app"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"🌐 BASE_URL set to: {BASE_URL}")

if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_CHANNEL_SECRET:
    raise ValueError("❌ LINE credentials required / ต้องตั้งค่า LINE Access Token และ Secret ก่อน")

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# ================================================
# 🧩 Load YOLO Model / โหลดโมเดล YOLO
# ================================================
MODEL_PATH = 'models/best.pt'
model = None
MODEL_TYPE = "Unknown"

if ULTRALYTICS_AVAILABLE and TORCH_AVAILABLE and NUMPY_AVAILABLE:
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)

        if os.path.exists(MODEL_PATH):
            model = YOLO(MODEL_PATH)
            model.to('cpu')
            MODEL_TYPE = "Custom YOLOv8/v11"
            logger.info("✅ Custom model loaded successfully on CPU / โหลดโมเดลที่ฝึกเองสำเร็จ")
        else:
            model = YOLO('yolo11n.pt')
            model.to('cpu')
            MODEL_TYPE = "YOLOv11n (Default)"
            logger.info("✅ Fallback model (YOLOv11n) loaded successfully / ใช้โมเดล YOLOv11n แทน")

    except Exception as e:
        logger.error(f"❌ Error loading model / โหลดโมเดลล้มเหลว: {e}")
        model = None
        MODEL_TYPE = "Not Loaded"
else:
    logger.warning("⚠️ Some dependencies missing / ขาดโมดูลบางส่วน โมเดลไม่ถูกโหลด")
    MODEL_TYPE = "Dependencies Missing"

# ================================================
# 🩺 Define Skin Disease Classes / นิยามคลาสโรคผิวหนัง
# ================================================
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
    0: "High Risk / ความเสี่ยงสูง",
    1: "Low Risk / ความเสี่ยงต่ำ",
    2: "Medium Risk / ความเสี่ยงปานกลาง"
}

CLASS_COLORS = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (255, 165, 0)
}
# ================================================
# 🧩 Helper Functions / ฟังก์ชันช่วยเหลือ
# ================================================

def save_image_temporarily(image, filename):
    """บันทึกรูปภาพชั่วคราว / Save image temporarily"""
    try:
        static_dir = "static"
        if not os.path.exists(static_dir):
            os.makedirs(static_dir)
        images_dir = os.path.join(static_dir, "images")
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)

        file_path = os.path.join(images_dir, filename)

        if not isinstance(image, Image.Image):
            logger.error("❌ Invalid image type / ประเภทของรูปภาพไม่ถูกต้อง")
            return None, None

        if image.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            background.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')

        image.save(file_path, 'JPEG', quality=95)
        if not os.path.exists(file_path):
            raise Exception("❌ ไม่สามารถสร้างไฟล์รูปภาพได้ / Unable to create image file.")

        file_size = os.path.getsize(file_path)
        logger.info(f"✅ Image saved: {file_path} ({file_size} bytes)")

        image_urls = [
            f"{BASE_URL}/static/images/{filename}",
            f"{BASE_URL}/images/{filename}",
            f"{BASE_URL}/serve_image/{filename}"
        ]
        return image_urls, file_path

    except Exception as e:
        logger.error(f"❌ Error saving image temporarily / เกิดข้อผิดพลาดในการบันทึกรูปภาพ: {e}")
        return None, None


def cleanup_old_images():
    """ลบไฟล์รูปภาพเก่า / Remove old temporary images"""
    try:
        for dir_name in ["static/images", "temp_images"]:
            if not os.path.exists(dir_name):
                continue
            current_time = time.time()
            max_age = 3600
            for filename in os.listdir(dir_name):
                file_path = os.path.join(dir_name, filename)
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getctime(file_path)
                    if file_age > max_age:
                        os.remove(file_path)
                        logger.info(f"🧹 Cleaned up old file: {filename}")
    except Exception as e:
        logger.error(f"❌ Error cleaning images / เกิดข้อผิดพลาดในการลบไฟล์: {e}")


def download_image_from_line(message_id):
    """ดาวน์โหลดรูปภาพจาก LINE / Download image from LINE"""
    if not PIL_AVAILABLE:
        logger.error("❌ PIL not available / ไม่มีโมดูล PIL สำหรับประมวลผลรูปภาพ")
        return None

    try:
        message_content = line_bot_api.get_message_content(message_id)
        image_data = io.BytesIO()
        for chunk in message_content.iter_content():
            image_data.write(chunk)
        image_data.seek(0)
        image = Image.open(image_data)
        logger.info(f"✅ Downloaded image: {image.size}, mode: {image.mode}")
        return image

    except Exception as e:
        logger.error(f"❌ Error downloading image / เกิดข้อผิดพลาดในการดาวน์โหลดรูปภาพ: {e}")
        return None


def predict_skin_cancer(image):
    """ทำนายโรคผิวหนังจากรูปภาพ / Predict skin disease from image"""
    if model is None:
        return None, None, "❌ โมเดลไม่พร้อมใช้งาน\n❌ Model not available."

    try:
        if not isinstance(image, Image.Image):
            return None, None, "⚠️ รูปภาพไม่ถูกต้อง\n⚠️ Invalid image type."

        if image.mode != 'RGB':
            image = image.convert('RGB')

        img_array = np.array(image)
        results = model(img_array, device='cpu', verbose=False, conf=0.3)
        logger.info("✅ Prediction completed / วิเคราะห์สำเร็จ")

        if len(results) == 0 or not hasattr(results[0], 'boxes') or len(results[0].boxes) == 0:
            return None, image, "🔍 ไม่พบรอยโรคผิวหนังในรูปภาพ\n🔍 No skin lesion detected in the image."

        boxes = results[0].boxes
        best_box = max(boxes, key=lambda b: float(b.conf.item()))
        class_id = int(best_box.cls.item())
        confidence = float(best_box.conf.item())

        prediction_result = {
            'class_id': class_id,
            'class_name': SKIN_CANCER_CLASSES_TH.get(class_id, "ไม่ทราบ / Unknown"),
            'confidence': confidence,
            'risk_level': RISK_LEVELS.get(class_id, "ไม่ทราบ / Unknown"),
            'total_detections': len(boxes)
        }

        return prediction_result, image, None

    except Exception as e:
        logger.error(f"❌ Prediction error / เกิดข้อผิดพลาดในการทำนาย: {e}")
        return None, None, f"เกิดข้อผิดพลาดในการวิเคราะห์\nPrediction error: {str(e)}"
# ================================================
# 🗒️ Result Message Creation / สร้างข้อความผลลัพธ์
# ================================================

def create_result_message(prediction_result):
    """สร้างข้อความผลลัพธ์ (สองภาษา ไทย–อังกฤษ, จัดสวยอ่านง่าย)"""
    if prediction_result is None:
        return (
            "❌ ไม่สามารถวิเคราะห์รูปภาพได้\n"
            "❌ Unable to analyze the image."
        )

    message = f"""🏥 ผลการวิเคราะห์ภาพผิวหนัง  
🏥 Skin Analysis Result  

🔍 ผลการตรวจพบ: {prediction_result['class_name']}  
🔍 Detected: {prediction_result['class_name']}  

📊 ความแม่นยำ: {prediction_result['confidence']:.2%}  
📊 Confidence: {prediction_result['confidence']:.2%}  

⚠️ ระดับความเสี่ยง: {prediction_result['risk_level']}  
⚠️ Risk Level: {prediction_result['risk_level']}  

📍 จำนวนจุดที่ตรวจพบ: {prediction_result.get('total_detections', 1)} จุด  
📍 Detected Areas: {prediction_result.get('total_detections', 1)} spots  

⚕️ คำแนะนำ:  
⚕️ Recommendations:"""

    # แนะนำตาม class
    if prediction_result['class_id'] == 0:  # เมลาโนมา (เสี่ยงสูง)
        message += (
            "\n• ควรปรึกษาแพทย์ผิวหนังโดยเร็ว  \n"
            "• Consult a dermatologist as soon as possible."
            "\n• อาจต้องการการตรวจเพิ่มเติม  \n"
            "• Further examination may be required."
        )
    elif prediction_result['class_id'] == 2:  # เซบอร์รีอิก เคราโทซิส
        message += (
            "\n• ควรติดตามอาการอย่างต่อเนื่อง  \n"
            "• Monitor your condition regularly."
            "\n• หากมีการเปลี่ยนแปลง ควรพบแพทย์  \n"
            "• Consult a doctor if any changes occur."
        )
    else:  # เนวัส (เสี่ยงต่ำ)
        message += (
            "\n• ดูแลสุขภาพผิวหนังอย่างสม่ำเสมอ  \n"
            "• Maintain good skin care regularly."
            "\n• หลีกเลี่ยงแสงแดดจัด  \n"
            "• Avoid strong sunlight."
        )

    message += """\n
🎯 กรอบสีในรูปภาพ (Bounding Box Colors):  
🔴 แดง = ความเสี่ยงสูง (เมลาโนมา)  
🔴 Red = High Risk (Melanoma)  

🟠 ส้ม = ความเสี่ยงปานกลาง (เซบอร์รีอิก เคราโทซิส)  
🟠 Orange = Medium Risk (Seborrheic Keratosis)  

🟢 เขียว = ความเสี่ยงต่ำ (เนวัส)  
🟢 Green = Low Risk (Nevus)  

⚠️ หมายเหตุ: ผลนี้เป็นเพียงการประเมินเบื้องต้น ควรปรึกษาแพทย์เพื่อการวินิจฉัยที่แม่นยำ  
⚠️ Note: This is a preliminary result. Please consult a medical professional for accurate diagnosis.
"""

    return message
# ================================================
# 🌐 Flask Routes / เส้นทางหลักของเว็บเซิร์ฟเวอร์
# ================================================

@app.route("/")
def home():
    return f"""
    <h1>LINE Bot Skin Cancer Detection (Bilingual Edition)</h1>
    <p>Status: ✅ Active / พร้อมใช้งาน</p>
    <p>Model Type: {MODEL_TYPE}</p>
    <p>Model Status: {'✅ Loaded' if model else '❌ Not Loaded'}</p>
    <p>BASE_URL: {BASE_URL}</p>
    <hr>
    <h3>System Modules / โมดูลระบบ:</h3>
    <ul>
        <li>NumPy: {'✅' if NUMPY_AVAILABLE else '❌'}</li>
        <li>PyTorch: {'✅' if TORCH_AVAILABLE else '❌'}</li>
        <li>OpenCV: {'✅' if CV2_AVAILABLE else '❌'}</li>
        <li>PIL: {'✅' if PIL_AVAILABLE else '❌'}</li>
        <li>Ultralytics: {'✅' if ULTRALYTICS_AVAILABLE else '❌'}</li>
    </ul>
    """

@app.route("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_type": MODEL_TYPE,
        "base_url": BASE_URL
    }

@app.route("/webhook", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        logger.error("❌ Invalid signature / ลายเซ็นไม่ถูกต้อง")
        abort(400)
    return 'OK'

# ================================================
# 💬 LINE Event Handlers / การจัดการข้อความจากผู้ใช้
# ================================================

@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    """จัดการข้อความข้อความ (Text Message Handler)"""
    text = event.message.text.lower()

    if 'สวัสดี' in text or 'hello' in text:
        reply_text = f"""สวัสดีครับ 👋  
Hello there! 👋  

ผมเป็นบอทช่วยตรวจโรคผิวหนังเบื้องต้น 🤖  
I’m a bot that helps analyze skin conditions.  

📸 วิธีใช้งาน / How to use:  
1️⃣ ส่งรูปภาพผิวหนังที่ต้องการตรวจ  
1️⃣ Send a photo of your skin for analysis.  
2️⃣ รอผลการวิเคราะห์และรูปภาพพร้อมกรอบสี  
2️⃣ Wait for the analysis result with colored bounding boxes.  

🎯 สีของกรอบ / Box Colors:  
🔴 แดง = ความเสี่ยงสูง (เมลาโนมา)  
🔴 Red = High risk (Melanoma)  
🟠 ส้ม = ความเสี่ยงปานกลาง (เซบอร์รีอิก เคราโทซิส)  
🟠 Orange = Medium risk (Seborrheic Keratosis)  
🟢 เขียว = ความเสี่ยงต่ำ (เนวัส)  
🟢 Green = Low risk (Nevus)  

⚠️ หมายเหตุ: ผลนี้เป็นเพียงการประเมินเบื้องต้น  
⚠️ Note: This is a preliminary analysis. Please consult a doctor for an accurate diagnosis."""
    
    elif 'สถานะ' in text or 'status' in text:
        reply_text = f"""✅ สถานะระบบ / System Status  

🤖 โมเดล: {'✅ พร้อมใช้งาน' if model else '❌ ไม่พร้อมใช้งาน'}  
🤖 Model: {'✅ Active' if model else '❌ Inactive'}  

📦 ประเภทโมเดล / Model Type: {MODEL_TYPE}  
🔥 YOLO Version: YOLOv11n  

📚 โมดูลสำคัญ / Key Modules:  
• NumPy: {'✅' if NUMPY_AVAILABLE else '❌'}  
• PyTorch: {'✅' if TORCH_AVAILABLE else '❌'}  
• OpenCV: {'✅' if CV2_AVAILABLE else '❌'}  
• PIL: {'✅' if PIL_AVAILABLE else '❌'}  
• Ultralytics: {'✅' if ULTRALYTICS_AVAILABLE else '❌'}  

🌐 BASE_URL: {BASE_URL}  
📁 Static Dir: {'✅' if os.path.exists('static/images') else '❌'}  
📁 Temp Dir: {'✅' if os.path.exists('temp_images') else '❌'}  

🎯 ระบบพร้อมรับรูปภาพเพื่อวิเคราะห์และแสดงผล  
🎯 The system is ready to analyze and display results."""
    
    else:
        reply_text = """📸 กรุณาส่งรูปภาพผิวหนังที่ต้องการตรวจ  
📸 Please send a photo of the skin area you want to analyze.  

🧠 คำสั่งที่ใช้ได้ / Available commands:  
• "สถานะ" - ตรวจสอบสถานะระบบ  
• "Status" - Check system status  

🤖 ระบบจะส่งผลลัพธ์กลับพร้อมกรอบสีแสดงความเสี่ยง  
🤖 The system will return an image with colored bounding boxes."""

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply_text)
    )


@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    """จัดการรูปภาพ / Handle incoming image"""
    try:
        # แจ้งผู้ใช้ว่ากำลังวิเคราะห์
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="🔍 กำลังวิเคราะห์รูปภาพด้วย YOLOv11n กรุณารอสักครู่...  
🔍 Analyzing the image with YOLOv11n, please wait...")
        )

        # ดาวน์โหลดรูปภาพ
        image = download_image_from_line(event.message.id)
        if image is None:
            line_bot_api.push_message(
                event.source.user_id,
                TextSendMessage(text="❌ ไม่สามารถดาวน์โหลดรูปภาพได้  
❌ Unable to download the image.")
            )
            return

        # วิเคราะห์รูปภาพ
        prediction, img_with_boxes, error = predict_skin_cancer(image)
        if error:
            line_bot_api.push_message(
                event.source.user_id,
                TextSendMessage(text=f"⚠️ {error}")
            )
            return

        # สร้างข้อความผลลัพธ์
        result_message = create_result_message(prediction)

        # บันทึกและส่งรูปภาพผลลัพธ์
        timestamp = int(time.time())
        random_num = random.randint(1000, 9999)
        filename = f"result_{timestamp}_{random_num}.jpg"
        image_urls, file_path = save_image_temporarily(img_with_boxes, filename)

        if image_urls and file_path and os.path.exists(file_path):
            image_url = image_urls[0]
            messages = [
                ImageSendMessage(original_content_url=image_url, preview_image_url=image_url),
                TextSendMessage(text=result_message)
            ]
            line_bot_api.push_message(event.source.user_id, messages)
        else:
            line_bot_api.push_message(
                event.source.user_id,
                TextSendMessage(text=f"{result_message}\n\n⚠️ ไม่สามารถส่งรูปภาพผลลัพธ์ได้  
⚠️ Unable to send result image.")
            )

    except Exception as e:
        logger.error(f"❌ Error in handle_image_message: {e}")
        line_bot_api.push_message(
            event.source.user_id,
            TextSendMessage(text=f"⚠️ เกิดข้อผิดพลาดในการประมวลผล: {str(e)}  
⚠️ Error occurred during processing: {str(e)}")
        )


# ================================================
# 🚀 Run Flask App
# ================================================
if __name__ == "__main__":
    print("🚀 Starting LINE Bot Server with YOLOv11n (Bilingual Mode)...")
    print(f"📡 BASE_URL: {BASE_URL}")
    print(f"🤖 Model Type: {MODEL_TYPE}")
    print(f"🕒 Status: {'✅ Loaded' if model else '❌ Not Loaded'}")
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
