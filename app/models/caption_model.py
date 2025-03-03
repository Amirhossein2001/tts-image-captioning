from transformers import VisionEncoderDecoderModel, ViTImageProcessor, GPT2TokenizerFast
from PIL import Image
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# بارگذاری مدل
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

def generate_caption(image):
    if isinstance(image, Image.Image):
        inputs = processor(images=image, return_tensors="pt").to(device)
        outputs = model.generate(**inputs)
        caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return caption
    else:
        raise ValueError("Invalid image input. Expected a PIL Image.")
