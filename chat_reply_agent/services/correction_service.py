import json

from sqlalchemy.orm import Session

from chat_reply_agent.models import ChatCorrection
from chat_reply_agent.schemas import ExtractedChat


def save_chat_correction(
    db: Session,
    user_id: str,
    relation_type: str,
    extracted_chat: ExtractedChat,
) -> ChatCorrection:
    correction = ChatCorrection(
        user_id=user_id,
        relation_type=relation_type,
        corrected_chat_json=json.dumps(
            extracted_chat.model_dump(),
            ensure_ascii=False,
        ),
        corrected_chat_text=extracted_chat.to_text(),
    )
    db.add(correction)
    db.commit()
    db.refresh(correction)
    return correction
