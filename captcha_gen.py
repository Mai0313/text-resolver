import os
import string
import secrets
import datetime

from PIL import Image, ImageDraw, ImageFont
from pydantic import Field, BaseModel


class CaptchaGenerator(BaseModel):
    length: int = Field(default=5, description="The length of the captcha image", ge=1, le=10)
    width: int = Field(default=58, description="The width of the captcha image", ge=1, le=100)
    height: int = Field(default=20, description="The width of the captcha image", ge=1, le=100)
    font_size: int = Field(default=18, description="The font size", ge=1, le=100)
    use_numbers: bool = Field(default=True, description="Use numbers in the captcha image")
    use_uppercase: bool = Field(
        default=True, description="Use uppercase letters in the captcha image"
    )
    use_lowercase: bool = Field(
        default=True, description="Use lowercase letters in the captcha image"
    )

    def __generate(self, tasktype: str) -> str:
        # Initialize the image and drawing object
        image = Image.new("RGB", (self.width, self.height), "white")
        draw = ImageDraw.Draw(image)

        # Use the default font and size
        font = ImageFont.load_default()
        # font = ImageFont.truetype("font/captcha_easy.ttf", font_size)

        # Available character set
        characters = ""
        if self.use_numbers:
            characters += string.digits
        if self.use_uppercase:
            characters += string.ascii_uppercase
        if self.use_lowercase:
            characters += string.ascii_lowercase

        if not characters:
            return "Invalid options"

        # Randomly generate captcha text using secrets
        captcha_text = "".join(secrets.choice(characters) for _ in range(self.length))

        # Draw the captcha text
        for i, char in enumerate(captcha_text):
            x = 2 + i * (self.width - 4) // self.length  # Adjusted for smaller width
            y = secrets.randbelow(self.height - self.font_size - 2)
            draw.text((x, y), char, fill=secrets.choice(["black", "blue", "red"]), font=font)

        # Add a small amount of interference lines and points
        for _ in range(5):
            x1 = secrets.randbelow(self.width)
            y1 = secrets.randbelow(self.height)
            x2 = secrets.randbelow(self.width)
            y2 = secrets.randbelow(self.height)
            draw.line((x1, y1, x2, y2), fill=secrets.choice(["black", "blue", "red"]))

        for _ in range(10):
            x = secrets.randbelow(self.width)
            y = secrets.randbelow(self.height)
            draw.point((x, y), fill=secrets.choice(["black", "blue", "red"]))

        # Save or display the image
        today = datetime.datetime.now().strftime("%Y%m%d")
        output_folder = f"./data/{today}_captcha/{tasktype}"
        os.makedirs(output_folder, exist_ok=True)

        image_path = f"./{output_folder}/{captcha_text}.png"
        image.save(image_path)
        return image_path

    def generate(self, image_nums: int, tasktype: str) -> list[str]:
        image_paths = []
        for _ in range(image_nums):
            image_path = self.__generate(tasktype=tasktype)
            image_paths.append(image_path)
        return image_paths


if __name__ == "__main__":
    capt = CaptchaGenerator(length=5, width=58, height=20, font_size=13)
    capt.generate(image_nums=1, tasktype="train")
    capt.generate(image_nums=1, tasktype="test")
    capt.generate(image_nums=1, tasktype="val")
