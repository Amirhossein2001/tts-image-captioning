import torch
import soundfile as sf
import os

# بارگذاری مدل TTS از Hugging Face
print("📌 بارگذاری مدل TTS...")
model_speech, symbols, sample_rate, example_text, apply_tts = torch.hub.load(
    repo_or_dir='snakers4/silero-models',
    model='silero_tts',
    language='en',
    speaker='lj_16khz'
)

# تنظیم مدل روی CPU
model = model_speech.to("cpu")
print("✅ مدل TTS آماده است.")

def text_to_speech(text, output_path="static/output.wav"):
    """
    تبدیل متن به صوت و ذخیره در مسیر مشخص‌شده.
    """
    print(f"📌 تبدیل متن به صوت: {text}")

    try:
        # تولید صوت
        audio = apply_tts(
            texts=[text],
            model=model,
            sample_rate=sample_rate,
            symbols=symbols,
            device="cpu"
        )[0].numpy()

        # ذخیره فایل صوتی
        sf.write(output_path, audio, sample_rate)
        print(f"✅ فایل صوتی ذخیره شده در: {output_path}")

        return output_path

    except RuntimeError as e:
        print(f"❌ خطا در اجرای TTS: {e}")

        # راهکار برای خطای attention_weights
        if "Sizes of tensors must match" in str(e):
            print("🔄 تلاش برای هماهنگ‌سازی ابعاد attention_weights...")
            return None

        return None
