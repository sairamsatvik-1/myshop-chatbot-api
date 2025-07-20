
pdf_path = r"C:\Users\kanum\AppData\Roaming\Python\Python312\Scripts\machine learning\ragexp.pdf"

from pdf2image import convert_from_path
import pytesseract



# Convert PDF to images
images = convert_from_path(pdf_path)

# Extract text from each image
full_text = ""
for img in images:
    text = pytesseract.image_to_string(img)
    full_text += text + "\n"

# Save to a file (optional)
with open("ocr_output.txt", "w", encoding="utf-8") as f:
    f.write(full_text)

print("OCR extraction complete.")

