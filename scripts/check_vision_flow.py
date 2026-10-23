"""Opt-in screenshot-only check using synthetic pixels and a disposable database."""

import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import monotonic
import unicodedata
from unittest.mock import patch


LINES = [
    "周末一起去公园吗", "可以，我周六有空", "那就周六下午吧", "下午两点怎么样",
    "两点可以", "我们在北门见", "北门离地铁远吗", "出站走五分钟就到",
    "我带一点水果", "我带水和纸巾", "要不要带野餐垫", "带一个吧",
    "你那个垫子大吗", "能坐四个人", "应该够用了", "差不多",
    "差不多", "还有谁一起去", "小林也有空", "那我问问小周",
    "他上次说想散步", "我刚发消息了", "如果不来就我们仨", "好，不用等太久",
    "午饭各自吃吗", "对，吃完再出门", "那我一点半出发", "我也差不多这个时间",
    "到了给你发消息", "好，我在北门等", "周六见", "到时见",
]


def synthetic_screenshot(font_path: str, case: str):
    from PIL import Image, ImageDraw, ImageFont

    lines = LINES[:3] if case == "short" else LINES
    reference = [("other" if index % 2 == 0 else "me", text) for index, text in enumerate(lines)]
    if case == "mixed":
        pattern = ["me", "me", "other", "other", "other", "me"]
        reference = [(pattern[index % len(pattern)], text) for index, text in enumerate(lines)]
    image = Image.new("RGB", (724, 60 + len(lines) * 66), "#f4f4f4")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, 22)
    for index, (speaker, text) in enumerate(reference):
        top = 30 + index * 66
        width = int(draw.textlength(text, font=font)) + 28
        left = 60 if speaker == "other" else 664 - width
        draw.rounded_rectangle((left, top, left + width, top + 48), radius=5,
                               fill="#303030" if speaker == "other" else "#35ce73")
        avatar = 14 if speaker == "other" else 680
        if case != "mixed":
            draw.rectangle((avatar, top, avatar + 30, top + 30), fill="#888888")
        elif index % 8 == 0:
            draw.text((325, top - 19), "17:30", font=ImageFont.truetype(font_path, 12), fill="#888888")
        draw.text((left + 14, top + 10), text, font=font,
                  fill="white" if speaker == "other" else "black")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue(), reference


def normalized(text: str) -> str:
    return "".join(char for char in text if unicodedata.category(char)[0] not in {"P", "Z", "C"})


def main() -> int:
    parser = argparse.ArgumentParser(description="只用合成截图检查识别与校对入口；--live 会产生费用。")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--case", choices=["short", "long", "mixed"], default="long")
    parser.add_argument("--force-recovery", action="store_true", help="首轮模拟格式错误，只真实调用恢复模型，验证失败分支。")
    parser.add_argument("--font", default="/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
    args = parser.parse_args()
    if not args.live:
        print("No API calls. Add --live to test synthetic screenshots; --force-recovery injects a first-attempt format error.")
        return 0
    if not Path(args.font).is_file():
        parser.error("请通过 --font 指定中文字体。")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    screenshot, reference = synthetic_screenshot(args.font, args.case)
    with TemporaryDirectory(prefix="qy-vision-check-") as directory:
        os.environ.update(DATABASE_URL=f"sqlite:///{directory}/test.db", UPLOAD_DIR=f"{directory}/uploads")
        from fastapi.testclient import TestClient
        from chat_reply_agent.config import get_settings
        from chat_reply_agent.database import engine
        from chat_reply_agent.main import app
        from chat_reply_agent.services.llm_client import call_vision_model

        if not get_settings().openai_api_key:
            print("No model credentials configured; no live test performed.")
            return 1
        requests = []

        def recognize(*call_args, **kwargs):
            requests.append(kwargs.get("model_name"))
            if args.force_recovery and len(requests) == 1:
                return '{"messages":'
            return call_vision_model(*call_args, **kwargs)

        try:
            with TestClient(app) as client, patch(
                "chat_reply_agent.services.vision_service.call_vision_model", side_effect=recognize,
            ):
                started = monotonic()
                response = client.post("/api/vision/extract-batch",
                    files={"images": ("synthetic.png", screenshot, "image/png")})
                data = response.json()
                messages = data.get("messages", [])
                result = {
                    "synthetic_only": True, "temporary_database": True,
                    "case": args.case, "first_failure_simulated": args.force_recovery,
                    "vision_timeout_seconds": get_settings().vision_timeout_seconds,
                    "http_status": response.status_code, "elapsed_seconds": round(monotonic() - started, 2),
                    "requested_models": requests,
                    "live_model_calls": len(requests) - int(args.force_recovery),
                    "expected_messages": len(reference), "recognized_messages": len(messages),
                    "preview_available": response.status_code == 200 and bool(messages),
                    "text_matches": [normalized(m["text"]) for m in messages] == [normalized(t) for _, t in reference],
                    "speakers_match": [m["speaker"] for m in messages] == [s for s, _ in reference],
                    "speaker_mismatch_indexes": [index for index, (message, (speaker, _))
                        in enumerate(zip(messages, reference)) if message["speaker"] != speaker],
                    "review_required": data.get("review_required"),
                    "uploads_cleaned": not list(Path(directory, "uploads").glob("*")),
                }
                if response.status_code != 200:
                    result["safe_error"] = data.get("detail")
                result["passed"] = all(result[key] for key in
                    ("preview_available", "text_matches", "speakers_match", "uploads_cleaned"))
                print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
                return 0 if result["passed"] else 1
        finally:
            engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
