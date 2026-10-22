"""Bounded screenshot storage and decoder validation before model calls."""

from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool


MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_BATCH_BYTES = 60 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
CHUNK_BYTES = 1024 * 1024
EXTENSIONS = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}


def _verify_image(path: Path) -> str:
    try:
        with Image.open(path, formats=list(EXTENSIONS)) as image:
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise HTTPException(413, "截图像素过多，请拆分为多张后上传（每张最多 4000 万像素）。")
            if getattr(image, "is_animated", False):
                raise HTTPException(400, "请上传静态聊天截图，不支持动图。")
            suffix = EXTENSIONS[image.format]
            image.verify()
        with Image.open(path, formats=list(EXTENSIONS)) as image:
            image.load()
        return suffix
    except Image.DecompressionBombError as exc:
        raise HTTPException(413, "截图像素过多，请拆分后上传。") from exc
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise HTTPException(400, "图片无效或已损坏，请上传 PNG、JPEG 或 WebP 静态截图。") from exc


async def save_screenshot(image: UploadFile, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{uuid4().hex}.upload"
    try:
        if image.size is not None and image.size > MAX_IMAGE_BYTES:
            raise HTTPException(413, "每张截图不能超过 20 MB，请压缩或拆分后上传。")
        size = 0
        with target.open("xb") as output:
            while chunk := await image.read(CHUNK_BYTES):
                size += len(chunk)
                if size > MAX_IMAGE_BYTES:
                    raise HTTPException(413, "每张截图不能超过 20 MB，请压缩或拆分后上传。")
                output.write(chunk)
        suffix = await run_in_threadpool(_verify_image, target)
        destination = target.with_suffix(suffix)
        target.rename(destination)
        return destination
    finally:
        target.unlink(missing_ok=True)
        await image.close()
