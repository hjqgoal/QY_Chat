import unicodedata
from pathlib import Path

from chat_reply_agent.schemas import ChatMessage, ExtractedChat
from chat_reply_agent.services.llm_client import call_vision_model, parse_json_object


VISION_PROMPT = """
你是一个聊天截图识别助手。请从这张微信聊天截图中提取可读的聊天内容。
要求：
1. 严格按图片从上到下的顺序输出消息。
2. 逐条观察气泡在整张图片中的横向位置，不要根据文字语义猜说话人：
   - bubble_center_x 是气泡中心相对图片宽度的位置，最左为 0，最右为 1。
   - 右侧绿色气泡、头像在右边，side 使用 right。
   - 左侧灰色/深色气泡、头像在左边，side 使用 left。
   - 居中的时间、系统通知不是聊天消息，不要放进 messages。
   - 同一句话重复出现时必须保留为两条，分别报告各自位置，不能合并或照抄上一条位置。
3. 不要编造看不清的内容。
4. side 只能使用 left、right、center、unknown。程序会根据位置计算 speaker，不需要你推断双方身份。
5. time 如果截图里没有明确时间，请返回 null，不要返回 unknown。
6. text_confidence 表示文字是否看清，position_confidence 表示气泡位置是否看清，均使用 0 到 1。
7. 模糊文字宁可降低 text_confidence 并在 warnings 说明，也不要猜测补全。
8. 请只输出 JSON，不要输出 Markdown。

JSON 格式：
{
  "messages": [
    {
      "side": "right",
      "bubble_center_x": 0.78,
      "text": "消息内容",
      "time": null,
      "text_confidence": 0.96,
      "position_confidence": 0.98,
      "warnings": []
    }
  ],
  "topic": "一句话概括当前对话主题",
  "confidence": 0.86,
  "notes": ["看不清或不确定的信息"]
}
"""


def extract_chat_from_image(image_path: str | Path) -> ExtractedChat:
    result = call_vision_model(image_path, VISION_PROMPT)
    if result:
        return _parse_extracted_chat(result)

    filename = Path(image_path).name
    return ExtractedChat(
        messages=[
            ChatMessage(
                speaker="unknown",
                text=(
                    f"已收到图片 {filename}。当前未配置 OPENAI_API_KEY，"
                    "因此暂时无法真实识别截图文字。"
                ),
                side="unknown",
                text_confidence=0.0,
                speaker_confidence=0.0,
                speaker_source="unknown",
                needs_review=True,
                warnings=["未调用视觉模型，无法判断文字和说话人。"],
            )
        ],
        topic="聊天截图待识别",
        last_message_from_other="",
        confidence=0.0,
        notes=["这是本地占位内容；配置 .env 后可启用真实多模态识别。"],
        review_required=True,
        review_message_indexes=[0],
    )


def extract_chat_content(image_path: str | Path) -> str:
    return extract_chat_from_image(image_path).to_text()


def merge_extracted_chats(chats: list[ExtractedChat]) -> ExtractedChat:
    """Merge screenshots in order, removing overlap only at image boundaries."""
    if not chats:
        raise ValueError("至少需要一张截图的识别结果。")

    merged_messages: list[ChatMessage] = []
    removed_overlap = sum(chat.merged_overlap_count for chat in chats)
    topics: list[str] = []
    notes: list[str] = []

    for chat in chats:
        incoming = [message.model_copy(deep=True) for message in chat.messages]
        overlap_size = _find_boundary_overlap(merged_messages, incoming)
        removed_overlap += overlap_size
        merged_messages.extend(incoming[overlap_size:])

        topic = chat.topic.strip()
        if topic and topic not in topics:
            topics.append(topic)
        for note in chat.notes:
            clean_note = note.strip()
            if clean_note and clean_note not in notes:
                notes.append(clean_note)

    source_image_count = sum(chat.source_image_count for chat in chats)
    review_indexes = [
        index for index, message in enumerate(merged_messages) if message.needs_review
    ]
    last_other = next(
        (
            message.text
            for message in reversed(merged_messages)
            if message.speaker == "other"
        ),
        "",
    )
    confidence = _messages_confidence(merged_messages)

    return ExtractedChat(
        messages=merged_messages,
        topic=" / ".join(topics[-3:])[:160],
        last_message_from_other=last_other,
        confidence=confidence,
        notes=list(dict.fromkeys(notes)),
        review_required=bool(review_indexes),
        review_message_indexes=review_indexes,
        source_image_count=source_image_count,
        merged_overlap_count=removed_overlap,
    )


def _find_boundary_overlap(
    existing: list[ChatMessage],
    incoming: list[ChatMessage],
    max_overlap: int = 12,
) -> int:
    maximum = min(len(existing), len(incoming), max_overlap)
    for size in range(maximum, 0, -1):
        left = existing[-size:]
        right = incoming[:size]
        if not all(
            _message_signature(first) == _message_signature(second)
            and bool(_normalized_message_text(first.text))
            for first, second in zip(left, right, strict=True)
        ):
            continue

        if size > 1 or _single_message_overlap_is_safe(left[0], right[0]):
            return size
    return 0


def _message_signature(message: ChatMessage) -> tuple[str, str]:
    if message.side == "right":
        participant = "me"
    elif message.side == "left":
        participant = "other"
    elif message.speaker in {"me", "other"}:
        participant = message.speaker
    else:
        participant = "unknown"
    return participant, _normalized_message_text(message.text)


def _normalized_message_text(text: str) -> str:
    return "".join(
        character
        for character in text.casefold()
        if unicodedata.category(character)[0] not in {"P", "Z", "C"}
    )


def _single_message_overlap_is_safe(
    first: ChatMessage,
    second: ChatMessage,
) -> bool:
    if first.time and second.time and first.time.strip() == second.time.strip():
        return True
    return len(_normalized_message_text(first.text)) >= 8


def _messages_confidence(messages: list[ChatMessage]) -> float:
    if not messages:
        return 0.0
    return round(
        sum(
            message.text_confidence * 0.7 + message.speaker_confidence * 0.3
            for message in messages
        )
        / len(messages),
        3,
    )


def _parse_extracted_chat(raw: str) -> ExtractedChat:
    fallback = {
        "messages": [
            {
                "speaker": "unknown",
                "side": "unknown",
                "text": raw.strip(),
                "time": None,
                "text_confidence": 0.3,
                "position_confidence": 0.0,
                "warnings": ["模型返回格式异常，消息内容需要人工复核。"],
            }
        ],
        "topic": "模型返回了非标准 JSON，已保留原始识别文本。",
        "confidence": 0.3,
        "notes": ["建议检查视觉模型输出格式。"],
    }
    data = parse_json_object(raw, fallback)
    messages = data.get("messages")
    if not isinstance(messages, list):
        messages = fallback["messages"]

    normalized_messages: list[ChatMessage] = []
    parser_notes: list[str] = []
    overall_model_confidence = _normalize_confidence(
        data.get("confidence"), default=0.65
    )

    for item in messages:
        if not isinstance(item, dict):
            continue

        text = str(item.get("text", "")).strip()
        if not text:
            continue

        reported_side = _normalize_side(item.get("side"))
        bubble_center_x = _normalize_position(item.get("bubble_center_x"))
        side, position_warning = _resolve_side(reported_side, bubble_center_x)
        speaker, speaker_source, speaker_warning = _normalize_speaker(
            item.get("speaker"), side, bubble_center_x
        )
        time = item.get("time")
        if str(time).strip().lower() in {"", "none", "null", "unknown", "未知"}:
            time = None

        text_confidence = _normalize_confidence(
            item.get("text_confidence"), default=overall_model_confidence
        )
        position_confidence = _normalize_confidence(
            item.get("position_confidence"),
            default=0.82 if side in {"left", "right"} else 0.25,
        )
        speaker_confidence = (
            position_confidence if speaker_source in {"geometry", "side"} else 0.35
        )
        warnings = _normalize_warnings(item.get("warnings"))
        warnings.extend(
            warning for warning in (position_warning, speaker_warning) if warning
        )
        if text_confidence < 0.72:
            warnings.append("文字识别置信度较低，请核对原图。")
        if speaker_confidence < 0.75 or speaker == "unknown":
            warnings.append("说话人归属不够确定，请核对气泡左右位置。")
        warnings = list(dict.fromkeys(warnings))

        normalized_messages.append(
            ChatMessage(
                speaker=speaker,
                side=side,
                bubble_center_x=bubble_center_x,
                text=text,
                time=time,
                text_confidence=text_confidence,
                speaker_confidence=speaker_confidence,
                speaker_source=speaker_source,
                needs_review=bool(warnings),
                warnings=warnings,
            )
        )

    if not normalized_messages:
        normalized_messages = [
            ChatMessage(
                speaker="unknown",
                side="unknown",
                text="未识别到可读的聊天消息，请上传微信聊天截图。",
                text_confidence=0.0,
                speaker_confidence=0.0,
                speaker_source="unknown",
                needs_review=True,
                warnings=["没有识别到可用消息。"],
            )
        ]

    review_indexes = [
        index for index, message in enumerate(normalized_messages) if message.needs_review
    ]
    last_other = ""
    for message in normalized_messages:
        if message.speaker == "other":
            last_other = message.text

    derived_confidence = _messages_confidence(normalized_messages)
    if review_indexes:
        parser_notes.append(f"有 {len(review_indexes)} 条消息需要人工复核。")

    return ExtractedChat(
        messages=normalized_messages,
        topic=str(data.get("topic", "")).strip(),
        last_message_from_other=last_other,
        confidence=derived_confidence,
        notes=list(
            dict.fromkeys(_normalize_warnings(data.get("notes")) + parser_notes)
        ),
        review_required=bool(review_indexes),
        review_message_indexes=review_indexes,
    )


def _normalize_side(value: object) -> str:
    side = str(value or "unknown").strip().lower()
    if side in {"left", "right", "center", "unknown"}:
        return side
    if side in {"左", "左侧"}:
        return "left"
    if side in {"右", "右侧"}:
        return "right"
    return "unknown"


def _resolve_side(
    reported_side: str,
    bubble_center_x: float | None,
) -> tuple[str, str]:
    if bubble_center_x is None:
        return reported_side, ""

    if bubble_center_x <= 0.45:
        geometry_side = "left"
    elif bubble_center_x >= 0.55:
        geometry_side = "right"
    else:
        geometry_side = "center"

    warning = ""
    if (
        reported_side in {"left", "right"}
        and geometry_side in {"left", "right"}
        and reported_side != geometry_side
    ):
        warning = "模型报告的左右位置与气泡横向坐标冲突，已按坐标修正。"
    return geometry_side, warning


def _normalize_speaker(
    value: object,
    side: str,
    bubble_center_x: float | None,
) -> tuple[str, str, str]:
    speaker = str(value or "unknown").strip().lower()
    mapping = {
        "我": "me",
        "自己": "me",
        "本人": "me",
        "对方": "other",
        "别人": "other",
    }
    speaker = mapping.get(speaker, speaker)
    if speaker not in {"me", "other", "unknown"}:
        speaker = "unknown"

    warning = ""
    if side == "right":
        if speaker == "other":
            warning = "模型给出的说话人与右侧气泡冲突，已按位置归为我。"
        source = "geometry" if bubble_center_x is not None else "side"
        return "me", source, warning
    if side == "left":
        if speaker == "me":
            warning = "模型给出的说话人与左侧气泡冲突，已按位置归为对方。"
        source = "geometry" if bubble_center_x is not None else "side"
        return "other", source, warning
    if speaker in {"me", "other"}:
        return speaker, "model", "缺少可靠气泡位置，暂时保留模型给出的说话人。"
    return "unknown", "unknown", ""


def _normalize_position(value: object) -> float | None:
    try:
        position = float(value)
    except (TypeError, ValueError):
        return None
    if 0.0 <= position <= 1.0:
        return round(position, 3)
    return None


def _normalize_confidence(value: object, default: float) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return default


def _normalize_warnings(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()][:5]
