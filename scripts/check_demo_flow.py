"""Opt-in provider check using synthetic input and a disposable database."""

import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import monotonic, sleep


REFERENCE_MESSAGES = [
    ("other", "今晚一起打游戏吗"),
    ("me", "我已经约了别人，今晚来不了"),
    ("other", "好，那下次吧"),
]


def synthetic_screenshot(font_path):
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (1000, 460), "#eeeeee")
    draw, font = ImageDraw.Draw(image), ImageFont.truetype(font_path, 28)
    for index, (speaker, text) in enumerate(REFERENCE_MESSAGES):
        top = 50 + index * 130
        left, right = (120, 560) if speaker == "other" else (420, 900)
        draw.rectangle((left, top, right, top + 75), fill="white" if speaker == "other" else "#95ec69")
        avatar = 30 if speaker == "other" else 920
        draw.rectangle((avatar, top, avatar + 50, top + 50), fill="#888888")
        draw.text((left + 15, top + 20), text, font=font, fill="black")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def report(result):
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if result.get("passed") else 1


def main():
    parser = argparse.ArgumentParser(description="合成截图与对话联调，会产生费用；仅使用临时数据库。")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--font", default="/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
    parser.add_argument("--confirm-reference", action="store_true", help="用已知合成原文模拟人工校对，继续测试下游；不会将识图失败改为通过。")
    parser.add_argument("--write-fixture", type=Path, help="仅在指定目录生成合成截图，不调用模型。")
    args = parser.parse_args()
    if not args.live and args.write_fixture is None:
        parser.error("需要 --live 明确允许真实模型调用。")
    if not Path(args.font).is_file():
        parser.error("请通过 --font 指定支持中文的字体文件。")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    screenshot = synthetic_screenshot(args.font)
    if args.write_fixture is not None:
        args.write_fixture.mkdir(parents=True, exist_ok=True)
        target = args.write_fixture / "synthetic-chat.png"
        with target.open("xb") as output:
            output.write(screenshot)
        print(f"Synthetic fixture created (no API calls): {target.resolve()}")
        return 0

    with TemporaryDirectory(prefix="qy-live-demo-") as directory:
        # Set isolation before importing the app or its cached settings/engine.
        os.environ.update(
            DATABASE_URL=f"sqlite:///{directory}/demo.db", UPLOAD_DIR=f"{directory}/uploads",
            MEMORY_SIMILARITY_BACKEND="lexical", WEATHER_TOOL_ENABLED="false", QUALITY_REPAIR_ENABLED="false",
        )
        from fastapi.testclient import TestClient
        from chat_reply_agent.database import engine
        from chat_reply_agent.main import app

        result = {"synthetic_only": True, "temporary_database": True, "repair_disabled_for_cost_control": True,
                  "reference_correction_applied": args.confirm_reference, "passed": False}
        try:
            with TestClient(app) as client:
                print("Checking synthetic screenshot...", flush=True)
                started = monotonic()
                response = client.post("/api/vision/extract", files={"image": ("synthetic.png", screenshot, "image/png")})
                result["vision_seconds"] = round(monotonic() - started, 2)
                result["vision_http_status"] = response.status_code
                chat = response.json() if response.status_code == 200 else {"messages": []}
                result["recognized_messages"] = [{"speaker": m["speaker"], "text": m["text"]} for m in chat["messages"]]
                result["vision_speakers_match"] = [m["speaker"] for m in chat["messages"]] == [m[0] for m in REFERENCE_MESSAGES]
                result["vision_text_matches"] = [m["text"] for m in chat["messages"]] == [m[1] for m in REFERENCE_MESSAGES]
                result["vision_passed"] = result["vision_speakers_match"] and result["vision_text_matches"]
                if not result["vision_passed"] and not args.confirm_reference:
                    result["next_step"] = "OCR 未通过；可加 --confirm-reference 单独验收人工校对后的流程。"
                    return report(result)
                if args.confirm_reference:
                    chat = {"messages": [{"speaker": speaker, "side": "right" if speaker == "me" else "left", "text": text}
                                         for speaker, text in REFERENCE_MESSAGES]}

                print("Checking generation and background jobs...", flush=True)
                started = monotonic()
                route = "/api/reply/corrected" if args.confirm_reference else "/api/reply/from-chat"
                response = client.post(route, json={
                    "user_id": "synthetic-demo", "contact_id": "synthetic-friend", "relation_type": "friend",
                    "background_context": "合成测试：朋友邀请打游戏，我已有安排，只想礼貌收尾。",
                    "extracted_chat": chat, "save_to_memory": True,
                })
                result["generation_seconds"] = round(monotonic() - started, 2)
                result["generation_http_status"] = response.status_code
                if response.status_code != 200:
                    return report(result)
                data = response.json()
                result.update(topic=data["extracted_chat"]["topic"], replies=[r["text"] for r in data["replies"]])
                deadline = monotonic() + 160
                while True:
                    quality = client.get(f'/api/quality-jobs/{data["quality_job_id"]}?user_id=synthetic-demo').json()
                    memory = client.get(f'/api/memory-jobs/{data["memory_job_id"]}?user_id=synthetic-demo').json()
                    finished = quality.get("status") not in {"pending", "running"} and memory.get("status") not in {"pending", "running"}
                    if finished or monotonic() >= deadline:
                        break
                    sleep(.5)
                result.update(
                    quality_status=quality.get("status"), memory_status=memory.get("status"),
                    quality_seconds=quality.get("diagnostics", {}).get("total_ms", 0) / 1000,
                )
                memories = client.get('/api/contacts/synthetic-friend/memories?user_id=synthetic-demo')
                result["stored_episodes"] = len(memories.json().get("memories", [])) if memories.status_code == 200 else 0
                result["confirmed_pipeline_passed"] = (
                    len(result["replies"]) == 3 and result["stored_episodes"] == 1
                    and result["quality_status"] == result["memory_status"] == "completed"
                )
                result["passed"] = result["vision_passed"] and result["confirmed_pipeline_passed"]
                return report(result)
        finally:
            engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
