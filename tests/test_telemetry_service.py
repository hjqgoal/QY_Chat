import asyncio
import json
from unittest import TestCase
from unittest.mock import patch

from chat_reply_agent.services.telemetry_service import (
    current_trace, log_trace, timed_call, trace_scope,
)


class TelemetryServiceTests(TestCase):
    def test_unconfigured_model_is_marked_skipped(self):
        with trace_scope() as trace:
            timed_call("model.text", lambda: None)
        self.assertEqual(trace.stages[0]["status"], "skipped")

    def test_success_failure_and_context_cleanup(self):
        with trace_scope() as trace:
            self.assertEqual(timed_call("test.ok", lambda: 3), 3)
            with self.assertRaises(ValueError):
                timed_call("test.fail", lambda: int("bad"))
            snapshot = trace.snapshot()
        self.assertIsNone(current_trace.get())
        self.assertEqual([item["status"] for item in snapshot["stages"]], ["ok", "error"])
        self.assertTrue(all(item["duration_ms"] >= 0 for item in snapshot["stages"]))

    def test_logs_do_not_include_arguments_or_exception_messages(self):
        with trace_scope() as trace:
            timed_call("test", lambda secret: secret, "private chat and credentials")
            with patch("chat_reply_agent.services.telemetry_service.logger.info") as output:
                log_trace(trace, 200)
        payload = json.loads(output.call_args.args[1])
        self.assertNotIn("private", str(payload))
        self.assertEqual(payload["http_status"], 200)

    def test_concurrent_requests_and_worker_threads_are_isolated(self):
        async def run_one(name):
            with trace_scope() as trace:
                await asyncio.to_thread(timed_call, name, lambda: None)
                await asyncio.sleep(0)
                return trace.snapshot()
        async def run_all():
            return await asyncio.gather(run_one("first"), run_one("second"))
        first, second = asyncio.run(run_all())
        self.assertNotEqual(first["request_id"], second["request_id"])
        self.assertEqual([item["stage"] for item in first["stages"]], ["first"])
        self.assertEqual([item["stage"] for item in second["stages"]], ["second"])
