import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from fixtures_v021 import GENERATION, MEMORY
from fixtures_images import image_bytes
from fixtures_quality import reviewed_payload
from chat_reply_agent.config import Settings
from chat_reply_agent.database import Base, get_db
from chat_reply_agent.main import app, _build_reply_response
from chat_reply_agent.models import ContactMemory, ConversationMemory, MemoryEpisode, MemoryJob, UserStyleProfile
from chat_reply_agent.schemas import ChatAnalysis, ChatMessage, ChatScenario, ExtractedChat, ReplyControls, ContactMemoryUpdateRequest
from chat_reply_agent.services.contact_memory_service import update_contact_memory_manually
from chat_reply_agent.services.data_management_service import delete_contact_memory_data, reset_style_data
from chat_reply_agent.services.directed_review_service import review_targeted_replies
from chat_reply_agent.services.generation_service import understand_and_generate
from chat_reply_agent.services.llm_client import ModelCallError, call_text_model, configured_models, model_role
from chat_reply_agent.services.memory_queue_service import (
    process_next_memory_job, acquire_worker_lock, start_memory_worker, stop_memory_worker,
)
from chat_reply_agent.services.quality_service import review_and_refine_replies
from chat_reply_agent.services.skill_loader import load_relation_skill


def generate(raw=GENERATION):
    with patch("chat_reply_agent.services.generation_service.call_text_model", return_value=raw):
        return understand_and_generate("对方：我还能小试一下", load_relation_skill("friend"), "", "", ReplyControls())


class GenerationAndReviewTests(TestCase):
    def test_one_strong_call_receives_history_style_background_and_controls(self):
        with patch("chat_reply_agent.services.generation_service.call_text_model", return_value=GENERATION) as model:
            result = understand_and_generate("recent and background", load_relation_skill("lover_pursuing"),
                                            "past fact", "user wording", ReplyControls(intent="decline"))
        model.assert_called_once()
        for value in ("recent and background", "past fact", "user wording", "拒绝"):
            self.assertIn(value, model.call_args.args[1])
        self.assertEqual(len(result.replies), 3)
        self.assertTrue(all(reply.source == "model" for reply in result.replies))

    def test_invalid_generation_fails_instead_of_faking_analysis(self):
        for raw in ("bad", "{}", '{"replies": []}', GENERATION.replace('"daily_chat"', '"made_up"'),
                    GENERATION.replace('"confidence": 0.95', '"confidence": "0.95"')):
            with self.subTest(raw=raw), self.assertRaises(ModelCallError):
                generate(raw)
        self.assertTrue(all(reply.source == "fallback" for reply in generate(None).replies))

    def review(self, generated=None, raw=None, **kwargs):
        generated = generated or generate()
        with patch("chat_reply_agent.services.quality_service.call_text_model", return_value=raw) as model:
            result = review_targeted_replies("对方：我还能小试一下", generated.replies,
                generated.analysis, generated.scenario, {}, "", "", ReplyControls(), **kwargs)
        return result, model

    def test_ordinary_chat_has_no_review_call_and_no_fake_score(self):
        result, model = self.review()
        model.assert_not_called()
        self.assertEqual(result.report.status, "not_reviewed")
        self.assertIsNone(result.report.average_score)
        self.assertFalse(result.report.passed)

    def test_high_risk_and_uncertainty_trigger_one_batch_not_three_calls(self):
        for high_risk in (True, False):
            draft = generate()
            if high_risk:
                draft.analysis.risk_level = "high"
            result, model = self.review(draft, uncertain_speakers=not high_risk)
            model.assert_called_once()
            self.assertEqual(set(result.report.review_targets), {0, 1, 2})
            self.assertEqual(result.replies, draft.replies)
            self.assertIsNone(result.report.average_score)

    def test_only_flagged_candidate_changes_and_other_scores_stay_null(self):
        draft = generate()
        draft.replies[1].text = "我理解你的感受"
        row = reviewed_payload([r.text for r in draft.replies])["items"][1]
        row["reply"]["text"] = "行，试好了叫我"
        result, model = self.review(draft, json.dumps({"items": [row]}))
        model.assert_called_once()
        self.assertEqual(result.replies[0], draft.replies[0])
        self.assertEqual(result.replies[2], draft.replies[2])
        self.assertEqual(result.replies[1].text, row["reply"]["text"])
        self.assertIsNone(result.report.items[1].overall_score)
        self.assertEqual(result.report.items[1].status, "revised_pending")
        self.assertIsNone(result.report.items[0].overall_score)
        self.assertEqual(result.report.status, "partial")
        self.assertIsNone(result.report.average_score)
        row["candidate_index"] = 0
        invalid, _ = self.review(draft, json.dumps({"items": [row]}))
        self.assertEqual(invalid.report.status, "invalid_response")
        self.assertEqual(invalid.replies, draft.replies)

    def test_single_regeneration_review_does_not_touch_other_candidates(self):
        draft = generate()
        draft.replies[0].text = "我理解你的感受"
        result, model = self.review(draft, candidate_indexes={1})
        model.assert_not_called()
        self.assertEqual(result.report.review_targets, {})

    def test_offline_judge_cannot_rewrite_the_answers_it_is_scoring(self):
        draft = generate()
        payload = reviewed_payload([r.text for r in draft.replies])
        payload["items"][0]["reply"]["text"] = "替换文本"
        with patch("chat_reply_agent.services.quality_service.call_text_model", return_value=json.dumps(payload)):
            result = review_and_refine_replies("chat", draft.replies, draft.analysis, draft.scenario, {}, "", "", audit_only=True)
        self.assertEqual(result.replies, draft.replies)
        self.assertIsNone(result.report.average_score)

    def test_role_routing_and_fallback_configuration(self):
        settings = Settings(_env_file=None, openai_model_name="legacy", openai_strong_model_name="strong",
                            openai_review_model_name="review", openai_memory_model_name="cheap")
        client = Mock()
        client.chat.completions.create.return_value.choices = [Mock(message=Mock(content="{}", tool_calls=None))]
        with patch("chat_reply_agent.services.llm_client.get_settings", return_value=settings), patch(
                "chat_reply_agent.services.llm_client._client", return_value=client):
            for role, expected in (("strong", "strong"), ("memory", "cheap"), ("review", "review")):
                with model_role(role):
                    call_text_model("system", "user")
                self.assertEqual(client.chat.completions.create.call_args.kwargs["model"], expected)
            call_text_model("system", "user")
            self.assertEqual(client.chat.completions.create.call_args.kwargs["model"], "strong")
            settings.openai_review_model_name = settings.openai_memory_model_name = ""
            self.assertEqual(configured_models()["review"], "strong")
            self.assertEqual(configured_models()["memory"], "legacy")


class MemoryQueueTests(TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory(prefix="qy-v021-queue-")
        self.engine = create_engine(f"sqlite:///{self.directory.name}/data.db", connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine)
        self.factory = sessionmaker(bind=self.engine)
        self.db = self.factory()
        app.dependency_overrides[get_db] = self.session_override
        self.client = TestClient(app)

    def session_override(self):
        with self.factory() as db:
            yield db

    def tearDown(self):
        app.dependency_overrides.pop(get_db, None)
        self.client.close()
        self.db.close()
        self.engine.dispose()
        self.directory.cleanup()

    def enqueue(self, save=True, text="我看看呀"):
        with patch("chat_reply_agent.services.generation_service.call_text_model", return_value=GENERATION):
            return _build_reply_response(self.db, "owner", "friend", "contact", "小林",
                ExtractedChat(messages=[ChatMessage(speaker="me", side="right", text=text)], topic="周末火锅"),
                save_to_memory=save)

    def run_job(self):
        with patch("chat_reply_agent.services.contact_memory_service.call_text_model", return_value=MEMORY):
            return process_next_memory_job(self.factory)

    def status(self, job_id):
        return self.client.get(f"/api/memory-jobs/{job_id}?user_id=owner")

    def test_enqueued_response_is_fast_path_and_completion_is_atomic_and_idempotent(self):
        with patch("chat_reply_agent.services.contact_memory_service.call_text_model") as memory_model:
            response = self.enqueue()
        memory_model.assert_not_called()
        self.assertFalse(response.memory_persisted)
        self.assertEqual(response.memory_status, "pending")
        self.assertEqual(self.db.query(ConversationMemory).count(), 0)
        self.assertTrue(self.run_job())
        self.assertFalse(self.run_job())
        for model in (ConversationMemory, ContactMemory, MemoryEpisode, UserStyleProfile):
            self.assertEqual(self.db.query(model).count(), 1)
        state = self.status(response.memory_job_id).json()
        self.assertEqual(state["status"], "completed")
        self.assertTrue(state["memory_persisted"])
        self.assertEqual(self.db.get(MemoryJob, response.memory_job_id).payload_json, "{}")
        self.assertNotIn("memory.contact", [stage["stage"] for stage in response.diagnostics["stages"]])
        self.assertIn("memory.contact", [stage["stage"] for stage in state["diagnostics"]["stages"]])

    def test_temporary_mode_creates_no_jobs(self):
        response = self.enqueue(save=False)
        self.assertIsNone(response.memory_job_id)
        self.assertEqual(self.db.query(MemoryJob).count(), 0)

    def test_model_failure_and_invalid_memory_leave_no_partial_rows(self):
        for value in (ModelCallError("private secret"), "{}", '{"summary": "partial"}'):
            response = self.enqueue()
            with patch("chat_reply_agent.services.contact_memory_service.call_text_model",
                       side_effect=value if isinstance(value, Exception) else None,
                       return_value=value if isinstance(value, str) else None):
                process_next_memory_job(self.factory)
            state = self.status(response.memory_job_id).json()
            self.assertEqual(state["status"], "failed")
            self.assertNotIn("secret", json.dumps(state))
        self.assertEqual(self.db.query(ConversationMemory).count(), 0)
        self.assertEqual(self.db.query(ContactMemory).count(), 0)

    def test_failure_after_contact_write_rolls_back_everything(self):
        response = self.enqueue()
        with patch("chat_reply_agent.services.memory_queue_service.learn_user_style", side_effect=RuntimeError("fail")):
            self.run_job()
        for model in (ConversationMemory, ContactMemory, MemoryEpisode, UserStyleProfile):
            self.assertEqual(self.db.query(model).count(), 0)
        self.assertEqual(self.status(response.memory_job_id).json()["status"], "failed")

    def test_delete_while_model_running_does_not_resurrect_contact(self):
        response = self.enqueue()
        def delete_during_model(*args):
            with self.factory() as other:
                deleted = delete_contact_memory_data(other, user_id="owner", contact_id="contact")
                self.assertTrue(deleted.deleted)
            return MEMORY
        with patch("chat_reply_agent.services.contact_memory_service.call_text_model", side_effect=delete_during_model):
            process_next_memory_job(self.factory)
        self.assertEqual(self.status(response.memory_job_id).status_code, 404)
        self.assertEqual(self.db.query(ContactMemory).count(), 0)
        self.assertEqual(self.db.query(ConversationMemory).count(), 0)

    def test_manual_edit_and_style_reset_cancel_stale_jobs(self):
        self.enqueue()
        self.run_job()
        response = self.enqueue()
        def edit_during_model(*args):
            with self.factory() as other:
                update_contact_memory_manually(other, "owner", "contact", ContactMemoryUpdateRequest(summary="manually edited"))
            return MEMORY
        with patch("chat_reply_agent.services.contact_memory_service.call_text_model", side_effect=edit_during_model):
            process_next_memory_job(self.factory)
        self.db.expire_all()
        self.assertEqual(self.db.scalar(select(ContactMemory)).summary, "manually edited")
        self.assertEqual(self.status(response.memory_job_id).json()["status"], "cancelled")
        pending = self.enqueue()
        reset_style_data(self.db, user_id="owner", relation_type="friend")
        self.assertFalse(self.run_job())
        self.assertEqual(self.status(pending.memory_job_id).json()["status"], "cancelled")

    def test_owner_checks_cancel_and_completed_deletion(self):
        response = self.enqueue()
        path = f"/api/memory-jobs/{response.memory_job_id}"
        self.assertEqual(self.client.get(path + "?user_id=someone-else").status_code, 404)
        self.assertEqual(self.client.delete(path + "?user_id=someone-else").status_code, 404)
        self.assertEqual(self.client.delete(path + "?user_id=owner").status_code, 200)
        self.assertFalse(self.run_job())
        completed = self.enqueue()
        self.run_job()
        delete_contact_memory_data(self.db, user_id="owner", contact_id="contact")
        self.assertEqual(self.status(completed.memory_job_id).status_code, 404)

    def test_worker_ownership_is_exclusive_and_released_on_close(self):
        path = str(Path(self.directory.name) / "lock-test.db")
        first = acquire_worker_lock(path)
        self.assertIsNotNone(first)
        try:
            self.assertIsNone(acquire_worker_lock(path))
        finally:
            first.close()
        second = acquire_worker_lock(path)
        self.assertIsNotNone(second)
        second.close()

    def test_restart_recovers_running_job_and_shuts_down_cleanly(self):
        response = self.enqueue()
        job = self.db.get(MemoryJob, response.memory_job_id)
        job.status = "running"
        self.db.commit()
        with patch("chat_reply_agent.services.memory_queue_service.engine", self.engine), patch(
                "chat_reply_agent.services.memory_queue_service.SessionLocal", self.factory), patch(
                "chat_reply_agent.services.memory_queue_service.process_next_memory_job",
                side_effect=lambda: process_next_memory_job(self.factory)), patch(
                "chat_reply_agent.services.contact_memory_service.call_text_model", return_value=MEMORY):
            try:
                start_memory_worker()
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    if self.status(response.memory_job_id).json()["status"] == "completed":
                        break
                    time.sleep(0.01)
                self.assertEqual(self.status(response.memory_job_id).json()["status"], "completed")
            finally:
                stop_memory_worker()
        lock = acquire_worker_lock(self.engine.url.database)
        self.assertIsNotNone(lock)
        lock.close()

    def test_http_generation_returns_pending_and_invalid_output_never_queues(self):
        payload = {"user_id": "owner", "relation_type": "friend", "contact_id": "contact",
                   "background_context": "对方说还能小试一下游戏联机，我还没回复。",
                   "extracted_chat": {"messages": []}, "save_to_memory": True}
        for raw in ("{}", '{"strategy":"unfinished\n</parameter>\n"replies":[]}',
                    GENERATION.replace('"confidence": 0.95', '"confidence": "0.95"')):
            with self.subTest(raw=raw), patch("chat_reply_agent.services.generation_service.call_text_model", return_value=raw):
                invalid = self.client.post("/api/reply/from-chat", json=payload)
            self.assertEqual(invalid.status_code, 502)
            for model in (MemoryJob, ConversationMemory, ContactMemory, MemoryEpisode, UserStyleProfile):
                self.assertEqual(self.db.query(model).count(), 0)
        with patch("chat_reply_agent.services.generation_service.call_text_model", return_value=GENERATION) as strong, patch(
                "chat_reply_agent.services.quality_service.call_text_model") as reviewer:
            response = self.client.post("/api/reply/from-chat", json=payload)
        self.assertEqual(response.status_code, 200)
        strong.assert_called_once()
        reviewer.assert_not_called()
        self.assertEqual(response.json()["memory_status"], "pending")
        self.assertEqual(self.status(response.json()["memory_job_id"]).json()["status"], "pending")

    def test_legacy_upload_route_uses_same_generation_pipeline(self):
        chat = ExtractedChat(messages=[ChatMessage(speaker="other", side="left", text="我还能小试一下")])
        with patch("chat_reply_agent.main.extract_chat_from_image", return_value=chat), patch(
                "chat_reply_agent.services.generation_service.call_text_model", return_value=GENERATION) as model:
            response = self.client.post("/api/reply", data={"user_id": "owner", "relation_type": "friend", "save_to_memory": "false"},
                                        files={"image": ("test.png", image_bytes(), "image/png")})
        self.assertEqual(response.status_code, 200)
        model.assert_called_once()
        self.assertEqual(response.json()["memory_status"], "disabled")
        self.assertEqual(self.db.query(MemoryJob).count(), 0)

    def test_http_does_not_wait_for_review_or_apply_reviewer_edits(self):
        generation = json.loads(GENERATION)
        generation["analysis"]["risk_level"] = "medium"
        texts = [reply["text"] for reply in generation["replies"]]
        review = reviewed_payload(texts)
        review["items"][0]["reply"]["text"] = "行，你先试一下"
        for row in review["items"]:
            row["review"].update({key: 5 for key in ("naturalness", "relevance", "relation_fit", "scenario_fit", "style_fit")})
        with patch("chat_reply_agent.services.generation_service.call_text_model", return_value=json.dumps(generation)), patch(
                "chat_reply_agent.services.quality_service.call_text_model", return_value=json.dumps(review)) as judge:
            response = self.client.post("/api/reply/from-chat", json={
                "user_id": "owner", "relation_type": "friend", "contact_id": "contact",
                "background_context": "对方说还能小试一下联机，我还没回复。",
                "extracted_chat": {"messages": []}, "save_to_memory": False,
            })
        self.assertEqual(response.status_code, 200, response.text)
        judge.assert_not_called()
        quality = response.json()["quality"]
        self.assertIsNone(quality["average_score"])
        self.assertTrue(all(item["overall_score"] is None for item in quality["items"]))
        self.assertEqual(quality["items"][0]["status"], "not_reviewed")
        self.assertIsNone(quality["items"][0]["evaluated_text"])
        self.assertIsNone(quality["items"][1]["evaluated_text"])
        self.assertEqual([r["text"] for r in response.json()["replies"]], texts)
        self.assertIsNone(quality["items"][1]["style_fit"])
        self.assertFalse(quality["passed"])
        self.assertEqual(self.db.query(MemoryJob).count(), 0)

    def test_corrupt_queued_payload_fails_once_instead_of_blocking_queue(self):
        response = self.enqueue()
        self.db.get(MemoryJob, response.memory_job_id).payload_json = "invalid"
        self.db.commit()
        self.assertTrue(self.run_job())
        self.assertFalse(self.run_job())
        self.assertEqual(self.status(response.memory_job_id).json()["status"], "failed")
