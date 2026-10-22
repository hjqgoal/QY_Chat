from io import BytesIO

from PIL import Image


def image_bytes(format="PNG", size=(16, 16)):
    buffer = BytesIO()
    Image.new("RGB", size, "white").save(buffer, format=format)
    return buffer.getvalue()
