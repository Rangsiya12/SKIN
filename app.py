import os
import io
import torch
import cv2
import numpy as np
from flask import Flask, request, abort, jsonify
from PIL import Image
import logging
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, ImageMessage, TextSendMessage,
    ImageSendMessage, QuickReply, QuickReplyButton, MessageAction
)
import requests
from ultralytics import YOLO

# ตั้งค่า logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ตั้งค่า LINE Bot
LINE_CHANNEL_ACCESS_TOKEN = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.getenv('LINE_CHANNEL_SECRET')

if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_CHANNEL_SECRET:
    logger.error("LINE credentials not found in environment variables")
    raise ValueError("LINE credentials required")

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# โหลด YOLOv5 model
MODEL_PATH = 'models/best.pt'
MODEL_URL = os.getenv('MODEL_URL')

# ตัวแปร global สำหรับโมเดล
model = None

def initialize_model():
    """Initialize or reload the model"""
    global model
    
    # สร้างโฟลเดอร์ models หากยังไม่มี
    os.makedirs('models', exist_ok=True)
    logger.info(f"Models directory created/exists")
    
    # ตรวจสอบและดาวน์โหลดโมเดล
    if not os.path.exists(MODEL_PATH):
        if MODEL_URL:
            try:
                logger.info(f"Downloading model from {MODEL_URL}")
                response = requests.get(MODEL_URL, timeout=300, stream=True)  # เพิ่ม timeout และ stream
                response.raise_for_status()
                
                # ดาวน์โหลดแบบ chunk เพื่อไฟล์ใหญ่
                total_size = 0
                with open(MODEL_PATH, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            total_size += len(chunk)
                
                logger.info(f"Model downloaded successfully, size: {total_size} bytes")
                
            except requests.RequestException as e:
                logger.error(f"Error downloading model: {e}")
                return False
            except Exception as e:
                logger.error(f"Unexpected error during download: {e}")
                return False
        else:
            logger.warning("MODEL_URL not provided and model file doesn't exist")
            # ลองใช้โมเดล default
            try:
                logger.info("Trying to use YOLOv8n default model")
                model = YOLO('yolov8n.pt')  # จะดาวน์โหลดอัตโนมัติ
                logger.info("Default YOLOv8n model loaded successfully")
                return True
            except Exception as e:
                logger.error(f"Error loading default model: {e}")
                return False
    
    # โหลดโมเดล
    try:
        if os.path.exists(MODEL_PATH):
            # ตรวจสอบขนาดไฟล์
            file_size = os.path.getsize(MODEL_PATH)
            logger.info(f"Model file size: {file_size} bytes")
            
            if file_size < 1000:  # ไฟล์เล็กเกินไป อาจเสียหาย
                logger.error("Model file seems corrupted (too small)")
                os.remove(MODEL_PATH)  # ลบไฟล์เสียหาย
                return False
            
            # โหลดโมเดล
            model = YOLO(MODEL_PATH)
            logger.info("Custom model loaded successfully")
            
            # ทดสอบโมเดลด้วยรูปภาพตัวอย่าง
            test_image = np.zeros((640, 640, 3), dtype=np.uint8)
            results = model(test_image)
            logger.info("Model test prediction successful")
            
            return True
        else:
            logger.error("Model file not found after download attempt")
            return False
            
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model = None
        return False

# คลาสโรคผิวหนัง
SKIN_CANCER_CLASSES = {
    0: "เมลาโนมา (Melanoma)",
    1: "เนวัส (Nevus)",
    2: "เซบอร์รีอิก เคราโทซิส (Seborrheic Keratosis)"
}

RISK_LEVELS = {
    0: "ความเสี่ยงสูง - ควรปรึกษาแพทย์",
    1: "ความเสี่ยงต่ำ",
    2: "ความเสี่ยงปานกลาง"
}

def download_image_from_line(message_id):
    """ดาวน์โหลดรูปภาพจาก LINE"""
    try:
        message_content = line_bot_api.get_message_content(message_id)
        image_data = io.BytesIO()
        for chunk in message_content.iter_content():
            image_data.write(chunk)
        image_data.seek(0)
        return Image.open(image_data)
    except Exception as e:
        logger.error(f"Error downloading image: {e}")
        return None

def predict_skin_cancer(image):
    """ทำนายโรคผิวหนังจากรูปภาพ"""
    if model is None:
        return None, "Model not available"
    
    try:
        # แปลง PIL Image เป็น numpy array
        img_array = np.array(image)
        
        # Resize รูปภาพถ้าใหญ่เกินไป
        if img_array.shape[0] > 640 or img_array.shape[1] > 640:
            image = image.resize((640, 640), Image.Resampling.LANCZOS)
            img_array = np.array(image)
        
        # ทำการทำนาย
        results = model(img_array, conf=0.25)  # ลด confidence threshold
        
        # ดึงผลลัพธ์
        if len(results) > 0 and hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
            # หา detection ที่มี confidence สูงสุด
            boxes = results[0].boxes
            best_idx = torch.argmax(boxes.conf)
            best_detection = boxes[best_idx]
            
            class_id = int(best_detection.cls.item())
            confidence = float(best_detection.conf.item())
            
            return {
                'class_id': class_id,
                'class_name': SKIN_CANCER_CLASSES.get(class_id, "Unknown"),
                'confidence': confidence,
                'risk_level': RISK_LEVELS.get(class_id, "ไม่ทราบ")
            }, None
        else:
            # หากไม่มี detection ลองใช้ classification mode
            logger.info("No detection found, trying classification mode")
            return None, "ไม่พบรอยโรคผิวหนังในรูปภาพ กรุณาลองถ่ายรูปใหม่ที่ชัดเจนขึ้น"
            
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return None, f"เกิดข้อผิดพลาดในการวิเคราะห์: {str(e)}"

def create_result_message(prediction_result):
    """สร้างข้อความผลลัพธ์"""
    if prediction_result is None:
        return "ไม่สามารถวิเคราะห์รูปภาพได้"
    
    message = f"""🏥 ผลการวิเคราะห์ภาพผิวหนัง

🔍 ผลการตรวจพบ: {prediction_result['class_name']}
📊 ความแม่นยำ: {prediction_result['confidence']:.2%}
⚠️ ระดับความเสี่ยง: {prediction_result['risk_level']}

⚕️ คำแนะนำ:"""
    
    if prediction_result['class_id'] == 0:  # เมลาโนมา
        message += "\n• ควรปรึกษาแพทย์ผิวหนังโดยเร็ว\n• อาจต้องการการตรวจเพิ่มเติม"
    elif prediction_result['class_id'] == 2:  # เซบอร์รีอิก เคราโทซิส
        message += "\n• ควรติดตามอาการ\n• หากมีการเปลี่ยนแปลง ควรพบแพทย์"
    else:  # เนวัส
        message += "\n• ดูแลสุขภาพผิวหนังอย่างสม่ำเสมอ\n• หลีกเลี่ยงแสงแดดจัด"
    
    message += "\n\n⚠️ หมายเหตุ: ผลนี้เป็นเพียงการประเมินเบื้องต้น ควรปรึกษาแพทย์เพื่อการวินิจฉัยที่แม่นยำ"
    
    return message

@app.route("/webhook", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        logger.error("Invalid signature")
        abort(400)
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        abort(500)

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
3. ได้รับคำแนะนำเบื้องต้น

⚠️ สำคัญ: ผลการตรวจเป็นเพียงข้อมูลเบื้องต้น ควรปรึกษาแพทย์เพื่อการวินิจฉัยที่แม่นยำ"""
        
    elif 'ช่วยเหลือ' in text or 'help' in text.lower():
        reply_text = """🔧 วิธีใช้งานบอท:

📷 ส่งรูปภาพ:
- ถ่ายรูปผิวหนังที่ชัดเจน
- แสงสว่างเพียงพอ
- ไม่มีสิ่งบดบัง

🔍 การวิเคราะห์:
- ระบบจะตรวจหาความผิดปกติ
- แสดงระดับความเสี่ยง
- ให้คำแนะนำเบื้องต้น

❓ คำถามเพิ่มเติม พิมพ์ "ช่วยเหลือ" """
        
    elif 'status' in text or 'สถานะ' in text:
        model_status = "✅ พร้อมใช้งาน" if model is not None else "❌ ไม่พร้อมใช้งาน"
        reply_text = f"""📊 สถานะระบบ:
        
🤖 บอท: ✅ ทำงานปกติ
🧠 โมเดล AI: {model_status}
        
{model_status}"""
        
    else:
        reply_text = """กรุณาส่งรูปภาพผิวหนังที่ต้องการตรวจ 📸

หรือพิมพ์ "ช่วยเหลือ" เพื่อดูวิธีใช้งาน
พิมพ์ "สถานะ" เพื่อดูสถานะระบบ"""
    
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply_text)
    )

@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    """จัดการรูปภาพ"""
    try:
        # ตรวจสอบว่าโมเดลพร้อมใช้งานหรือไม่
        if model is None:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="❌ ระบบยังไม่พร้อมใช้งาน กรุณาลองใหม่อีกครั้งในอีกสักครู่")
            )
            return
        
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
        
        # ทำการทำนาย
        prediction, error = predict_skin_cancer(image)
        
        if error:
            line_bot_api.push_message(
                event.source.user_id,
                TextSendMessage(text=f"เกิดข้อผิดพลาด: {error}")
            )
            return
        
        # สร้างข้อความผลลัพธ์
        result_message = create_result_message(prediction)
        
        # ส่งผลลัพธ์
        line_bot_api.push_message(
            event.source.user_id,
            TextSendMessage(text=result_message)
        )
        
    except Exception as e:
        logger.error(f"Error handling image: {e}")
        line_bot_api.push_message(
            event.source.user_id,
            TextSendMessage(text="เกิดข้อผิดพลาดในการประมวลผล กรุณาลองใหม่อีกครั้ง")
        )

@app.route("/", methods=['GET'])
def health_check():
    """Health check endpoint with detailed info"""
    model_info = {
        "model_file_exists": os.path.exists(MODEL_PATH),
        "model_path": MODEL_PATH,
        "model_url_provided": MODEL_URL is not None,
        "models_directory_exists": os.path.exists('models')
    }
    
    if os.path.exists(MODEL_PATH):
        model_info["model_file_size"] = os.path.getsize(MODEL_PATH)
    
    return jsonify({
        "status": "ok",
        "message": "Skin Cancer Detection LINE Bot is running",
        "model_loaded": model is not None,
        "model_info": model_info,
        "line_credentials": {
            "access_token_provided": LINE_CHANNEL_ACCESS_TOKEN is not None,
            "secret_provided": LINE_CHANNEL_SECRET is not None
        }
    })

@app.route("/reload_model", methods=['POST'])
def reload_model():
    """Reload model endpoint"""
    success = initialize_model()
    return jsonify({
        "status": "success" if success else "failed",
        "model_loaded": model is not None,
        "message": "Model reloaded successfully" if success else "Failed to reload model"
    })

@app.route("/test_model", methods=['GET'])
def test_model():
    """Test model endpoint"""
    if model is None:
        return jsonify({
            "status": "failed",
            "message": "Model not loaded"
        })
    
    try:
        # สร้างรูปภาพทดสอบ
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        results = model(test_image)
        
        return jsonify({
            "status": "success",
            "message": "Model test successful",
            "detections": len(results[0].boxes) if hasattr(results[0], 'boxes') else 0
        })
    except Exception as e:
        return jsonify({
            "status": "failed",
            "message": f"Model test failed: {str(e)}"
        })

if __name__ == "__main__":
    # โหลดโมเดลตอนเริ่มต้น
    logger.info("Starting application...")
    model_loaded = initialize_model()
    
    if model_loaded:
        logger.info("✅ Application started with model loaded")
    else:
        logger.warning("⚠️ Application started without model")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)