import os
import requests
from flask import Flask, request, jsonify, render_template, send_from_directory
from models.caption_model import generate_caption
from models.ocr_model import extract_text_from_image
from models.tts_model import text_to_speech
from PIL import Image
import io
import traceback

# تنظیم مسیر صحیح برای `templates` و `static`
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TEMPLATES_DIR = os.path.abspath(os.path.join(BASE_DIR, '../templates'))
STATIC_DIR = os.path.abspath(os.path.join(BASE_DIR, '../static'))

# اطمینان از ایجاد `static/` و `templates/`
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)

# سرو کردن فایل‌های `static/`
@app.route('/static/<path:filename>')
def serve_static(filename):
    file_path = os.path.join(STATIC_DIR, filename)
    if not os.path.exists(file_path):
        print(f"❌ فایل یافت نشد: {file_path}")
        return jsonify({'error': 'File not found'}), 404
    return send_from_directory(STATIC_DIR, filename)

@app.route('/')
def index():
    print(f"📌 مسیر قالب‌ها: {TEMPLATES_DIR}")
    return render_template('index.html')

@app.route('/caption', methods=['GET'])
def caption_image():
    image_url = request.args.get("image_url")

    if not image_url:
        return jsonify({'error': 'No image URL provided'}), 400

    try:
        print(f"📌 دریافت تصویر از URL: {image_url}")

        # دانلود تصویر از URL
        response = requests.get(image_url)
        image = Image.open(io.BytesIO(response.content)).convert("RGB")

        # ذخیره موقت تصویر برای OCR
        image_path = os.path.join(STATIC_DIR, "temp_image.jpg")
        image.save(image_path)

        # اجرای Image Captioning
        print("📌 در حال اجرای Image Captioning...")
        caption = generate_caption(image)
        print(f"✅ کپشن تولید شده: {caption}")

        # اجرای OCR
        print("📌 در حال اجرای OCR...")
        extracted_text = extract_text_from_image(image_path)
        print(f"✅ متن OCR شده: {extracted_text}")

        # ترکیب متن‌ها برای TTS
        final_text = f"Image captioning content: {caption}. And OCR content: {extracted_text}."
        print(f"📌 متن نهایی برای TTS: {final_text}")

        # مسیر ذخیره فایل صوتی
        audio_path = os.path.join(STATIC_DIR, "output.wav")

        # تبدیل متن به گفتار و ذخیره فایل صوتی
        print("📌 در حال اجرای TTS...")
        generated_audio = text_to_speech(final_text, output_path=audio_path)

        if not generated_audio or not os.path.exists(audio_path):
            print("❌ فایل صوتی ایجاد نشده است!")
            return jsonify({'error': 'Audio file not found'}), 500

        print(f"✅ فایل صوتی ذخیره شده در: {audio_path}")

        return jsonify({
            'caption': caption,
            'ocr_text': extracted_text,
            'audio_url': "/static/output.wav"
        })

    except Exception as e:
        print("❌ خطا در سرور:")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/tts', methods=['GET'])
def text_to_speech_api():
    text = request.args.get("text")

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        print(f"📌 تبدیل متن به گفتار: {text}")

        # مسیر ذخیره فایل صوتی
        audio_path = os.path.join(STATIC_DIR, "tts_output.wav")

        # تبدیل متن به گفتار و ذخیره فایل صوتی
        generated_audio = text_to_speech(text, output_path=audio_path)

        if not generated_audio or not os.path.exists(audio_path):
            print("❌ فایل صوتی ایجاد نشده است!")
            return jsonify({'error': 'Audio file not found'}), 500

        print(f"✅ فایل صوتی ذخیره شده در: {audio_path}")

        return jsonify({'audio_url': "/static/tts_output.wav"})

    except Exception as e:
        print("❌ خطا در سرور:")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
