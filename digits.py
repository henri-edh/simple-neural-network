import os
from PIL import Image, ImageDraw, ImageFont

def generate_digit_bmps():
    # 1. Image configurations
    width, height = 28, 25
    os.makedirs("digits_bmp", exist_ok=True)
    
    # 2. Try loading a clean system font (falls back to default if unavailable)
    try:
        # Use an available TrueType font file on your system
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()

    # 3. Generate digits 0 through 9
    for digit in range(10):
        # Create a 1-bit monochrome image ('1') with a black background (0)
        img = Image.new('1', (width, height), color=0)
        draw = ImageDraw.Draw(img)
        
        # Calculate text positioning to center it
        text = str(digit)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = (width - text_w) // 2
        y = (height - text_h) // 2 - 2  # Slight vertical offset tuning
        
        # Draw the digit in white (1)
        draw.text((x, y), text, fill=1, font=font)
        
        # Save as standard monochrome BMP
        filename = f"digits_bmp/digit_{digit}.bmp"
        img.save(filename, format="BMP")
        print(f"Saved: {filename}")

if __name__ == "__main__":
    generate_digit_bmps()
