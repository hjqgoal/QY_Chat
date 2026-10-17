from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from chat_reply_agent.main import app
from chat_reply_agent.schemas import ChatMessage, ExtractedChat
from chat_reply_agent.services.llm_client import ModelCallError


class ApiSmokeTests(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_evaluation_page_is_available(self) -> None:
        response = self.client.get("/evals")
        self.assertEqual(response.status_code, 200)
        self.assertIn("回复质量评测", response.text)

    def test_evaluation_case_list_is_available(self) -> None:
        response = self.client.get("/api/evals/cases")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertGreaterEqual(len(payload["cases"]), 10)

    def test_v020_regression_comparison_uses_real_context_but_no_model(self) -> None:
        with patch("chat_reply_agent.services.llm_client.call_text_model") as model:
            response = self.client.get("/api/evals/regressions")
        model.assert_not_called()
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["source"]["kind"], "user_provided_screenshot")
        self.assertFalse(payload["model_called"])
        self.assertTrue(payload["passed"])
        self.assertEqual(len(payload["results"]), 3)
        self.assertTrue(all(item["legacy_replaced"] for item in payload["results"]))
        self.assertEqual([item["speaker"] for item in payload["chat"]["messages"][1:3]], ["me", "other"])
        cases = self.client.get("/api/evals/cases?dataset=v020").json()["cases"]
        self.assertEqual(cases[0]["id"], "real_friend_game_crossplay")
        self.assertEqual(self.client.get("/api/evals/cases?dataset=unknown").status_code, 422)

    def test_evaluation_run_requires_case_id(self) -> None:
        response = self.client.post("/api/evals/run", json={"case_ids": []})
        self.assertEqual(response.status_code, 422)

    def test_memory_management_page_is_available(self) -> None:
        response = self.client.get("/memory")
        self.assertEqual(response.status_code, 200)
        self.assertIn("记忆与风格", response.text)
        self.assertIn("自动学习到的本人语气", response.text)

    def test_home_supports_ordered_multi_screenshot_uploads(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn('name="images"', response.text)
        self.assertIn("multiple", response.text)
        self.assertIn("/api/vision/extract-batch", response.text)
        self.assertIn('data-upload-action="up"', response.text)
        self.assertIn("真人语气依据", response.text)
        self.assertIn('name="background_context"', response.text)
        self.assertIn("可单独使用", response.text)
        self.assertIn("根据背景生成回复", response.text)

    def test_style_api_exposes_automatic_profile_evidence(self) -> None:
        response = self.client.get(
            "/api/style/examples",
            params={
                "user_id": "v018-empty-profile",
                "relation_type": "friend",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("automatic_summary", payload)
        self.assertIn("automatic_examples", payload)

    def test_status_reports_v022(self) -> None:
        response = self.client.get("/api/status")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["version"], "0.22.0")
        self.assertEqual(set(response.json()["models"]), {"strong", "vision", "review", "memory"})
        self.assertIn("X-Request-ID", response.headers)
        self.assertIn("Server-Timing", response.headers)

    def test_blank_manual_style_example_is_rejected(self) -> None:
        response = self.client.post(
            "/api/style/examples",
            json={
                "user_id": "default",
                "relation_type": "friend",
                "text": "   ",
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_v019_routes_are_registered(self) -> None:
        paths = self.client.get("/openapi.json").json()["paths"]
        self.assertIn("/api/vision/extract", paths)
        self.assertIn("/api/vision/extract-batch", paths)
        self.assertIn("/api/reply/from-chat", paths)
        self.assertIn("/api/reply/regenerate", paths)
        self.assertIn("/api/contacts/{contact_id}/memories", paths)
        self.assertIn("/api/contacts/{contact_id}/memories/{memory_id}", paths)
        self.assertIn("/api/style/examples/{example_id}", paths)
        self.assertIn("/api/style/data", paths)

    def test_background_only_can_generate_without_screenshot(self) -> None:
        targets = (
            "chat_reply_agent.services.generation_service.call_text_model",
            "chat_reply_agent.services.analysis_service.call_text_model",
            "chat_reply_agent.services.scenario_service.call_text_model",
            "chat_reply_agent.services.reply_service.call_text_model",
            "chat_reply_agent.services.quality_service.call_text_model",
        )
        with ExitStack() as stack:
            for target in targets:
                stack.enter_context(patch(target, return_value=None))
            response = self.client.post(
                "/api/reply/from-chat",
                json={
                    "relation_type": "friend",
                    "user_id": "background-only-api-user",
                    "contact_id": "xiaolin",
                    "background_context": "我们昨天因为见面时间吵了几句。",
                    "extracted_chat": {
                        "messages": [],
                        "source_image_count": 0,
                    },
                    "save_to_memory": False,
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["background_context"], "我们昨天因为见面时间吵了几句。")
        self.assertEqual(payload["diagnostics"]["request_id"], response.headers["X-Request-ID"])
        stages = [item["stage"] for item in payload["diagnostics"]["stages"]]
        for stage in ("context.load", "memory.retrieve", "reply.understand_generate", "quality.targeted"):
            self.assertIn(stage, stages)
        self.assertNotIn("memory.contact", stages)
        self.assertIsNone(payload["quality"]["average_score"])
        self.assertFalse(payload["quality"]["passed"])
        self.assertEqual(payload["screenshot_text"], "")
        self.assertEqual(payload["extracted_chat"]["source_image_count"], 0)
        self.assertEqual(payload["style_evidence"]["current_user_message_count"], 0)

    def test_reply_rejects_empty_background_and_empty_chat(self) -> None:
        response = self.client.post(
            "/api/reply/from-chat",
            json={
                "relation_type": "friend",
                "background_context": "   ",
                "extracted_chat": {
                    "messages": [],
                    "source_image_count": 0,
                },
                "save_to_memory": False,
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("上传聊天截图或填写", response.json()["detail"])
        self.assertIn("X-Request-ID", response.headers)

    def test_reply_rejects_background_over_2000_characters(self) -> None:
        response = self.client.post(
            "/api/reply/from-chat",
            json={
                "relation_type": "friend",
                "background_context": "背" * 2001,
                "extracted_chat": {
                    "messages": [],
                    "source_image_count": 0,
                },
                "save_to_memory": False,
            },
        )

        self.assertEqual(response.status_code, 422)

    @patch("chat_reply_agent.main.extract_chat_from_image")
    def test_vision_preview_returns_extraction_without_generation(self, mock_extract) -> None:
        mock_extract.return_value = ExtractedChat(
            messages=[ChatMessage(speaker="other", side="left", text="在吗")],
            last_message_from_other="在吗",
            confidence=0.95,
        )

        response = self.client.post(
            "/api/vision/extract",
            files={"image": ("chat.png", b"fake-image", "image/png")},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["last_message_from_other"], "在吗")
        mock_extract.assert_called_once()

    def test_vision_preview_removes_temporary_upload_on_model_error(self) -> None:
        with TemporaryDirectory() as directory:
            image_path = Path(directory) / "temporary-chat.png"
            image_path.write_bytes(b"temporary-image")
            with (
                patch(
                    "chat_reply_agent.main._save_upload",
                    new_callable=AsyncMock,
                    return_value=image_path,
                ),
                patch(
                    "chat_reply_agent.main.extract_chat_from_image",
                    side_effect=RuntimeError("vision failed"),
                ),
            ):
                with self.assertRaises(RuntimeError):
                    self.client.post(
                        "/api/vision/extract",
                        files={"image": ("chat.png", b"fake-image", "image/png")},
                    )

        self.assertFalse(image_path.exists())

    @patch("chat_reply_agent.main.extract_chat_from_image")
    def test_batch_vision_merges_screenshots_in_upload_order(self, mock_extract) -> None:
        mock_extract.side_effect = [
            ExtractedChat(
                messages=[
                    ChatMessage(speaker="me", side="right", text="第一条"),
                    ChatMessage(speaker="other", side="left", text="重叠一"),
                    ChatMessage(speaker="me", side="right", text="重叠二"),
                ]
            ),
            ExtractedChat(
                messages=[
                    ChatMessage(speaker="other", side="left", text="重叠一"),
                    ChatMessage(speaker="me", side="right", text="重叠二"),
                    ChatMessage(speaker="other", side="left", text="最后一条"),
                ]
            ),
        ]

        response = self.client.post(
            "/api/vision/extract-batch",
            files=[
                ("images", ("01.png", b"first-image", "image/png")),
                ("images", ("02.png", b"second-image", "image/png")),
            ],
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(
            [message["text"] for message in payload["messages"]],
            ["第一条", "重叠一", "重叠二", "最后一条"],
        )
        self.assertEqual(payload["source_image_count"], 2)
        self.assertEqual(payload["merged_overlap_count"], 2)
        self.assertEqual(mock_extract.call_count, 2)

    @patch("chat_reply_agent.main.extract_chat_from_image")
    def test_batch_vision_rejects_more_than_six_images(self, mock_extract) -> None:
        response = self.client.post(
            "/api/vision/extract-batch",
            files=[
                ("images", (f"{index}.png", b"image", "image/png"))
                for index in range(7)
            ],
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("最多上传 6 张", response.json()["detail"])
        mock_extract.assert_not_called()

    @patch("chat_reply_agent.main.extract_chat_from_image")
    def test_batch_vision_reports_which_image_failed(self, mock_extract) -> None:
        mock_extract.side_effect = [
            ExtractedChat(messages=[ChatMessage(text="第一张", speaker="me")]),
            ModelCallError("模型服务暂时不可用。"),
        ]

        response = self.client.post(
            "/api/vision/extract-batch",
            files=[
                ("images", ("01.png", b"first", "image/png")),
                ("images", ("02.png", b"second", "image/png")),
            ],
        )

        self.assertEqual(response.status_code, 502)
        self.assertIn("第 2 张截图识别失败", response.json()["detail"])

    def test_batch_vision_removes_all_uploads_when_later_image_fails(self) -> None:
        with TemporaryDirectory() as directory:
            first_path = Path(directory) / "first.png"
            second_path = Path(directory) / "second.png"
            first_path.write_bytes(b"first")
            second_path.write_bytes(b"second")
            with (
                patch(
                    "chat_reply_agent.main._save_upload",
                    new_callable=AsyncMock,
                    side_effect=[first_path, second_path],
                ),
                patch(
                    "chat_reply_agent.main.extract_chat_from_image",
                    side_effect=[
                        ExtractedChat(
                            messages=[ChatMessage(text="第一张", speaker="me")]
                        ),
                        RuntimeError("vision failed"),
                    ],
                ),
            ):
                with self.assertRaises(RuntimeError):
                    self.client.post(
                        "/api/vision/extract-batch",
                        files=[
                            ("images", ("01.png", b"first", "image/png")),
                            ("images", ("02.png", b"second", "image/png")),
                        ],
                    )

            self.assertFalse(first_path.exists())
            self.assertFalse(second_path.exists())
