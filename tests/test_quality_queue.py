"""Offline tests for bounded, read-only reviews; never call a provider."""

import json
from tempfile import TemporaryDirectory
from threading import Event
from time import monotonic, sleep
from unittest import TestCase
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from chat_reply_agent.database import Base, get_db
from chat_reply_agent.main import app
from chat_reply_agent.schemas import ReplyCandidate, ReplyControls
from chat_reply_agent.services import quality_queue_service as queue
from chat_reply_agent.services.llm_client import ModelCallError, model_request_timeout, _request_timeout
from fixtures_quality import reviewed_payload
from fixtures_v021 import GENERATION


TEXTS = ["行，那我先去了", "好啦，你们玩开心", "嗯，下次再说"]


def review_input():
    return dict(chat_text="对方：算了算了你去玩", replies=[ReplyCandidate(text=t, style="", reason="") for t in TEXTS],
                relation_rules={}, memory_summary="", style_summary="", controls=ReplyControls(),
                style_evidence_available=False)


class QueueHarness:
    def start_queue(self, **kwargs):
        self.manager = queue.QualityJobManager(**kwargs)
        self.release = Event()
        self.started = Event()
        self.manager.start()
        self.addCleanup(self.close_queue)

    def close_queue(self):
        self.release.set()
        with self.manager._lock:
            futures = [job.future for job in self.manager._jobs.values() if job.future]
        for future in futures:
            if not future.cancelled():
                future.result(timeout=3)
        self.manager.stop()

    def blocked_review(self, *args, **kwargs):
        self.started.set()
        if not self.release.wait(3):
            raise AssertionError("Test did not release reviewer")
        return json.dumps(reviewed_payload(TEXTS))

    def terminal(self, job_id):
        end = monotonic() + 3
        while monotonic() < end:
            state = self.manager.get(job_id, "owner")
            if state.status not in queue.ACTIVE:
                return state
            sleep(.005)
        self.fail("Review did not finish")


class QualityQueueTests(QueueHarness, TestCase):
    def setUp(self):
        self.start_queue(workers=1, max_active=2)

    def test_enqueue_is_local_only_and_every_candidate_is_reviewed(self):
        payload = review_input()
        with patch.object(queue, "quality_jobs", self.manager), patch(
                "chat_reply_agent.services.quality_service.call_text_model", side_effect=self.blocked_review) as model:
            job = queue.enqueue_quality(user_id="owner", **payload)
            self.assertEqual(job.status, "pending")
            self.assertTrue(all(item.status == "pending" for item in job.quality.items))
            self.assertTrue(self.started.wait(1))
            # The queued snapshot must not follow edits to the caller's objects.
            payload["replies"][0].text = "用户后来改了"
            self.release.set()
            state = self.terminal(job.job_id)
        self.assertEqual(state.status, "completed")
        self.assertEqual(state.candidate_texts, TEXTS)
        self.assertTrue(all(i.status == "reviewed" and i.assessment_source == "read_only" for i in state.quality.items))
        self.assertTrue(all(i.overall_score is None and i.style_fit is None for i in state.quality.items))
        self.assertIsNone(state.quality.average_score)
        self.assertIsNone(self.manager._jobs[job.job_id].payload)
        self.assertIn("quality.background", [s["stage"] for s in state.diagnostics["stages"]])
        model.assert_called_once()
        self.assertIn("禁止改写", model.call_args.args[1])
        self.assertIn("禁止改写", model.call_args.args[0])
        self.assertNotIn("另行提出", model.call_args.args[0])

    def test_bad_response_or_attempted_rewrite_never_counts_as_completed(self):
        rewrite = reviewed_payload(TEXTS)
        rewrite["items"][0]["reply"]["text"] = "模型擅自改写"
        for raw in ("{}", "not json", json.dumps(rewrite)):
            with self.subTest(raw=raw), patch("chat_reply_agent.services.quality_service.call_text_model", return_value=raw):
                job_id, _ = self.manager.submit("owner", review_input(), 90)
                state = self.terminal(job_id)
            self.assertEqual(state.status, "failed")
            self.assertEqual(state.candidate_texts, TEXTS)
            self.assertIsNone(state.quality)

    def test_negative_judgment_completes_but_is_not_approval(self):
        data = reviewed_payload(TEXTS)
        data["items"][0]["review"]["hard_issues"] = ["新增了未经确认的安排"]
        with patch("chat_reply_agent.services.quality_service.call_text_model", return_value=json.dumps(data)):
            job_id, _ = self.manager.submit("owner", review_input(), 90)
            state = self.terminal(job_id)
        self.assertEqual(state.status, "completed")
        self.assertFalse(state.quality.passed)
        self.assertEqual(state.quality.items[0].verdict, "needs_attention")

    def test_provider_failure_is_private_and_not_a_success(self):
        with patch("chat_reply_agent.services.quality_service.call_text_model", side_effect=ModelCallError("private token")):
            job_id, _ = self.manager.submit("owner", review_input(), 90)
            state = self.terminal(job_id)
        self.assertEqual(state.status, "failed")
        self.assertNotIn("private token", state.model_dump_json())

    def test_cancel_queued_and_running_jobs_rejects_late_results(self):
        with patch("chat_reply_agent.services.quality_service.call_text_model", side_effect=self.blocked_review) as model:
            first, _ = self.manager.submit("owner", review_input(), 90)
            self.assertTrue(self.started.wait(1))
            second, _ = self.manager.submit("owner", {**review_input(), "chat_text": "另一个待评估对话"}, 90)
            self.assertNotEqual(first, second)
            self.assertIsNone(self.manager.get(first, "stranger"))
            self.assertFalse(self.manager.cancel(first, "stranger"))
            self.assertTrue(self.manager.cancel(second, "owner"))
            self.assertTrue(self.manager.cancel(first, "owner"))
            self.release.set()
            self.manager._jobs[first].future.result(timeout=2)
        model.assert_called_once()
        for job_id in (first, second):
            state = self.manager.get(job_id, "owner")
            self.assertEqual(state.status, "cancelled")
            self.assertIsNone(state.quality)
            self.assertEqual(state.candidate_texts, [])

    def test_separate_timeouts_keep_unfinished_work_bounded(self):
        now = [0]
        self.manager._clock = lambda: now[0]
        with patch("chat_reply_agent.services.quality_service.call_text_model", side_effect=self.blocked_review):
            first, _ = self.manager.submit("owner", review_input(), 5)
            self.assertTrue(self.started.wait(1))
            second, _ = self.manager.submit("owner", {**review_input(), "chat_text": "second"}, 5)
            third_payload = {**review_input(), "chat_text": "third"}
            self.assertEqual(self.manager.submit("owner", third_payload, 5), (None, "busy"))
            now[0] = 6
            self.assertEqual(self.manager.get(first, "owner").status, "timed_out")
            self.assertEqual(self.manager.get(first, "owner").diagnostics["error_code"], "stage_timeout")
            self.assertEqual(self.manager.get(second, "owner").status, "pending")
            now[0] = 31
            self.assertEqual(self.manager.get(second, "owner").diagnostics["error_code"], "queue_timeout")
            self.manager._max_active = 1
            self.assertEqual(self.manager.submit("owner", third_payload, 5), (None, "busy"))
            self.release.set()
            self.manager._jobs[first].future.result(timeout=2)
        self.assertEqual(self.manager.get(first, "owner").status, "timed_out")
        self.assertIsNone(self.manager.get(first, "owner").quality)

    def test_retention_and_restart_remove_private_snapshots(self):
        now = [0]
        self.manager._clock = lambda: now[0]
        self.manager._ttl = 10
        self.manager._max_retained = 1
        with patch("chat_reply_agent.services.quality_service.call_text_model", return_value=json.dumps(reviewed_payload(TEXTS))):
            first, _ = self.manager.submit("owner", review_input(), 90)
            self.terminal(first)
            self.manager._jobs[first].future.result(timeout=2)
            second, _ = self.manager.submit("owner", review_input(), 90)
            self.terminal(second)
            self.manager._jobs[second].future.result(timeout=2)
        self.assertIsNone(self.manager.get(first, "owner"))
        now[0] = 11
        self.assertIsNone(self.manager.get(second, "owner"))
        self.manager.stop()
        self.assertEqual(self.manager.submit("owner", review_input(), 90), (None, "unavailable"))
        self.manager.start()
        self.assertIsNone(self.manager.get(second, "owner"))

    def test_fallback_does_not_use_model(self):
        payload = review_input()
        for reply in payload["replies"]:
            reply.source = "fallback"
        with patch.object(queue, "quality_jobs", self.manager), patch(
                "chat_reply_agent.services.quality_service.call_text_model") as model:
            job = queue.enqueue_quality(user_id="owner", **payload)
        self.assertEqual(job.status, "disabled")
        model.assert_not_called()

    def test_unavailable_worker_keeps_local_checks_honest(self):
        self.manager.stop()
        with patch.object(queue, "quality_jobs", self.manager), patch(
                "chat_reply_agent.services.quality_service.call_text_model") as model:
            job = queue.enqueue_quality(user_id="owner", **review_input())
        self.assertEqual(job.status, "unavailable")
        self.assertFalse(job.quality.passed)
        self.assertIsNone(job.quality.average_score)
        self.assertTrue(all(i.status == "not_reviewed" for i in job.quality.items))
        model.assert_not_called()

    def test_timeout_context_is_restored_even_after_errors(self):
        self.assertIsNone(_request_timeout.get())
        with model_request_timeout(15):
            self.assertEqual(_request_timeout.get(), 15)
            with self.assertRaises(ValueError), model_request_timeout(2):
                self.assertEqual(_request_timeout.get(), 2)
                raise ValueError("test")
            self.assertEqual(_request_timeout.get(), 15)
        self.assertIsNone(_request_timeout.get())


class QualityJobApiTests(QueueHarness, TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory(prefix="qy-quality-api-")
        self.addCleanup(self.directory.cleanup)
        self.engine = create_engine(f"sqlite:///{self.directory.name}/test.db", connect_args={"check_same_thread": False})
        self.addCleanup(self.engine.dispose)
        Base.metadata.create_all(self.engine)
        self.factory = sessionmaker(bind=self.engine)
        app.dependency_overrides[get_db] = self.session_override
        self.addCleanup(app.dependency_overrides.pop, get_db, None)
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        self.start_queue(workers=1)
        for target in ("chat_reply_agent.main.quality_jobs", "chat_reply_agent.services.quality_queue_service.quality_jobs"):
            patcher = patch(target, self.manager)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.payload = dict(user_id="owner", relation_type="friend", contact_id="contact", save_to_memory=False,
                            extracted_chat={"messages": []}, background_context="对方说算了算了你去玩，我想简短收尾。")

    def session_override(self):
        with self.factory() as db:
            yield db

    def assert_no_memory(self):
        from chat_reply_agent.models import MemoryJob, ConversationMemory, ContactMemory, MemoryEpisode, UserStyleProfile, ReplyFeedback
        with self.factory() as db:
            for model in (MemoryJob, ConversationMemory, ContactMemory, MemoryEpisode, UserStyleProfile, ReplyFeedback):
                self.assertEqual(db.query(model).count(), 0, model.__name__)

    def test_generation_returns_before_judge_for_low_and_high_risk(self):
        for risk in ("low", "high"):
            self.release.clear()
            self.started.clear()
            draft = json.loads(GENERATION)
            draft["analysis"]["risk_level"] = risk
            for reply, text in zip(draft["replies"], TEXTS):
                reply["text"] = text
            with self.subTest(risk=risk), patch("chat_reply_agent.services.generation_service.call_text_model", return_value=json.dumps(draft)), patch(
                    "chat_reply_agent.services.quality_service.call_text_model", side_effect=self.blocked_review) as model:
                response = self.client.post("/api/reply/from-chat", json=self.payload)
                self.assertEqual(response.status_code, 200, response.text)
                data = response.json()
                self.assertEqual(data["quality_status"], "pending")
                self.assertEqual(data["quality_gate_required"], risk == "high")
                self.assertTrue(self.started.wait(1))
                self.assertEqual(self.manager.get(data["quality_job_id"], "owner").status, "running")
                self.assertEqual([r["text"] for r in data["replies"]], TEXTS)
                self.assertNotIn("quality.background", [s["stage"] for s in data["diagnostics"]["stages"]])
                self.release.set()
                self.assertEqual(self.terminal(data["quality_job_id"]).status, "completed")
                model.assert_called_once()
        self.assert_no_memory()

    def test_manual_review_is_read_only_and_job_is_owner_scoped(self):
        payload = {**self.payload, "replies": [r.model_dump() for r in review_input()["replies"]],
                   "weather_evidence": [{"tool": "get_weather", "result": {"status": "ok", "provider": "weather-evidence"}}]}
        # Even a client asking to save cannot make this endpoint write memory.
        payload["save_to_memory"] = True
        with patch("chat_reply_agent.services.generation_service.call_text_model") as strong, patch(
                "chat_reply_agent.services.quality_service.call_text_model", side_effect=self.blocked_review) as judge:
            response = self.client.post("/api/quality-jobs", json=payload)
            self.assertEqual(response.status_code, 200, response.text)
            job_id = response.json()["job_id"]
            self.assertTrue(self.started.wait(1))
            self.assertEqual(self.client.get(f"/api/quality-jobs/{job_id}?user_id=stranger").status_code, 404)
            self.assertEqual(self.client.delete(f"/api/quality-jobs/{job_id}?user_id=stranger").status_code, 404)
            self.release.set()
            self.assertEqual(self.terminal(job_id).status, "completed")
            state = self.client.get(f"/api/quality-jobs/{job_id}?user_id=owner").json()
            self.assertEqual(state["candidate_texts"], TEXTS)
            self.assertIn("weather-evidence", judge.call_args.args[1])
            strong.assert_not_called()
            self.assertEqual(self.client.delete(f"/api/quality-jobs/{job_id}?user_id=owner").status_code, 200)
            self.assertEqual(self.client.get(f"/api/quality-jobs/{job_id}?user_id=owner").json()["status"], "cancelled")
        self.assert_no_memory()

    def test_manual_review_validates_empty_and_oversized_text(self):
        for texts in (["", "a", "b"], ["x" * 601, "a", "b"], ["a", "b"]):
            response = self.client.post("/api/quality-jobs", json={**self.payload, "replies": [dict(text=t, style="", reason="") for t in texts]})
            self.assertEqual(response.status_code, 422)
        self.assert_no_memory()

    def test_single_regeneration_queues_entire_group_without_inline_edit(self):
        draft = json.loads(GENERATION)
        replies = [r.model_dump() for r in review_input()["replies"]]
        replies[1]["text"] = "待替换的旧回复"
        payload = {**self.payload, "analysis": draft["analysis"], "scenario": draft["scenario"],
                   "existing_replies": replies, "candidate_index": 1,
                   "weather_evidence": [{"tool": "get_weather", "result": {"status": "ok", "provider": "earlier-weather"}}]}
        with patch("chat_reply_agent.services.reply_service.call_text_model", return_value=json.dumps({"reply": {
                "text": TEXTS[1], "style": "自然", "reason": "简短收尾"}})), patch(
                "chat_reply_agent.services.quality_service.call_text_model", side_effect=self.blocked_review) as judge:
            response = self.client.post("/api/reply/regenerate", json=payload)
            self.assertEqual(response.status_code, 200, response.text)
            data = response.json()
            self.assertEqual(data["quality_status"], "pending")
            self.assertEqual(data["reply"]["text"], TEXTS[1])
            self.assertEqual(len(data["quality_report"]["items"]), 3)
            self.assertTrue(self.started.wait(1))
            self.release.set()
            state = self.terminal(data["quality_job_id"])
            self.assertEqual(state.candidate_texts, TEXTS)
            self.assertEqual(state.status, "completed")
            self.assertIn("earlier-weather", judge.call_args.args[1])
        self.assert_no_memory()
