from PIL import Image, ImageDraw, ImageFont
import random
import string

# Modified version using the default font available in this environment
def generate_captcha_image_default_font(length=5, width=58, height=20, font_size=18, use_numbers=True, use_uppercase=True, use_lowercase=True, tasktype = "train"):
    # Initialize the image and drawing object
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Use the default font and size
    font = ImageFont.load_default()
    # font = ImageFont.truetype("font/captcha_easy.ttf", font_size)
    
    # Available character set
    characters = ""
    if use_numbers:
        characters += string.digits
    if use_uppercase:
        characters += string.ascii_uppercase
    if use_lowercase:
        characters += string.ascii_lowercase
    
    if not characters:
        return "Invalid options"
    
    # Randomly generate captcha text
    captcha_text = ''.join(random.choice(characters) for _ in range(length))
    
    # Draw the captcha text
    for i, char in enumerate(captcha_text):
        x = 2 + i * (width - 4) // length  # Adjusted for smaller width
        y = random.randint(0, height - font_size - 2)
        draw.text((x, y), char, fill=random.choice(['black', 'blue', 'red']), font=font)
    
    # Add a small amount of interference lines and points
    for i in range(5):
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        x2 = random.randint(0, width)
        y2 = random.randint(0, height)
        draw.line((x1, y1, x2, y2), fill=random.choice(['black', 'blue', 'red']))
        
    for i in range(10):
        x = random.randint(0, width)
        y = random.randint(0, height)
        draw.point((x, y), fill=random.choice(['black', 'blue', 'red']))
    
    # Save or display the image
    image_path = f"data/20230831_captcha/{tasktype}/{captcha_text}.png"
    image.save(image_path)
    print(f"{captcha_text} has been generated.")

# Usage example
for i in range(10000):
    generate_captcha_image_default_font(length=5, width=58, height=20, font_size=13, tasktype = "train")
print("Training set done.")

for i in range(1000):
    generate_captcha_image_default_font(length=5, width=58, height=20, font_size=13, tasktype = "test")
print("Test set done.")

for i in range(500):
    generate_captcha_image_default_font(length=5, width=58, height=20, font_size=13, tasktype = "val")
print("Validation set done.")
