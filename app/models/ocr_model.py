import easyocr

reader = easyocr.Reader(["en"], gpu=False)

def extract_text_from_image(image_path):
    text = reader.readtext(image_path, detail=0)
    return ' '.join(text) if text else "No text detected."
