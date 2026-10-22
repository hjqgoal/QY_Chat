import asyncio
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient

from fixtures_images import image_bytes
from chat_reply_agent.config import Settings
from chat_reply_agent.main import app
from chat_reply_agent.schemas import ExtractedChat
from chat_reply_agent.services.embedding_service import embedding_runtime_status
from chat_reply_agent.services.skill_loader import load_relation_skill
from chat_reply_agent.services.upload_service import save_screenshot
from chat_reply_agent.services.vision_service import _parse_extracted_chat


class ScreenshotValidationTests(TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory(prefix="qy-upload-check-")
        self.path = Path(self.directory.name)
        self.client = TestClient(app)
        self.override = patch("chat_reply_agent.main.settings.upload_dir", str(self.path))
        self.override.start()

    def tearDown(self):
        self.override.stop()
        self.client.close()
        self.directory.cleanup()

    def test_bad_or_empty_file_never_calls_model_on_any_upload_route(self):
        for route, field in (("/api/vision/extract", "image"), ("/api/vision/extract-batch", "images"), ("/api/reply", "image")):
            for content in (b"", b"not an image", image_bytes()[:40]):
                with self.subTest(route=route, size=len(content)), patch("chat_reply_agent.main.extract_chat_from_image") as model:
                    response = self.client.post(route, data={"relation_type": "friend"},
                                                files={field: ("fake.png", content, "image/png")})
                self.assertEqual(response.status_code, 400, response.text)
                model.assert_not_called()
                self.assertEqual(list(self.path.iterdir()), [])

    def test_malformed_vision_output_stops_before_generation_on_all_routes(self):
        def extract(path):
            return _parse_extracted_chat('```json\n{"messages": [\n```')
        for route, field in (("/api/vision/extract", "image"), ("/api/vision/extract-batch", "images"), ("/api/reply", "image")):
            with self.subTest(route=route), patch("chat_reply_agent.main.extract_chat_from_image", side_effect=extract), patch("chat_reply_agent.main._build_reply_response") as generation:
                response = self.client.post(route, data={"relation_type": "friend"}, files={field: ("test.png", image_bytes(), "image/png")})
            self.assertEqual(response.status_code, 502, response.text)
            self.assertNotIn('```json', response.text)
            generation.assert_not_called()
            self.assertEqual(list(self.path.iterdir()), [])

    def test_supported_formats_use_decoded_type_not_untrusted_filename(self):
        for format, suffix in (("PNG", ".png"), ("JPEG", ".jpg"), ("WEBP", ".webp")):
            def extract(path):
                self.assertEqual(path.suffix, suffix)
                self.assertTrue(path.exists())
                return ExtractedChat()
            with self.subTest(format=format), patch("chat_reply_agent.main.extract_chat_from_image", side_effect=extract):
                response = self.client.post("/api/vision/extract", files={"image": ("../../wrong.jpg", image_bytes(format), "image/png")})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(list(self.path.iterdir()), [])

    def test_size_and_pixel_limits_return_413_and_clean_up(self):
        for constant, limit in (("MAX_IMAGE_BYTES", 10), ("MAX_IMAGE_PIXELS", 10)):
            with patch(f"chat_reply_agent.services.upload_service.{constant}", limit), patch("chat_reply_agent.main.extract_chat_from_image") as model:
                response = self.client.post("/api/vision/extract", files={"image": ("large.png", image_bytes(), "image/png")})
            self.assertEqual(response.status_code, 413)
            model.assert_not_called()
            self.assertEqual(list(self.path.iterdir()), [])

    def test_stream_without_declared_size_is_bounded_and_closed(self):
        upload = UploadFile(file=BytesIO(image_bytes()), filename="test.png")
        with patch("chat_reply_agent.services.upload_service.MAX_IMAGE_BYTES", 10):
            with self.assertRaises(HTTPException) as error:
                asyncio.run(save_screenshot(upload, self.path))
        self.assertEqual(error.exception.status_code, 413)
        self.assertTrue(upload.file.closed)
        self.assertEqual(list(self.path.iterdir()), [])

    def test_later_invalid_file_is_detected_before_first_model_call(self):
        with patch("chat_reply_agent.main.extract_chat_from_image") as model:
            response = self.client.post("/api/vision/extract-batch", files=[
                ("images", ("first.png", image_bytes(), "image/png")),
                ("images", ("second.png", b"broken", "image/png")),
            ])
        self.assertEqual(response.status_code, 400)
        model.assert_not_called()
        self.assertEqual(list(self.path.iterdir()), [])

    def test_batch_limit_is_checked_before_model_calls(self):
        with patch("chat_reply_agent.main.MAX_BATCH_BYTES", len(image_bytes()) + 1), patch("chat_reply_agent.main.extract_chat_from_image") as model:
            response = self.client.post("/api/vision/extract-batch", files=[
                ("images", ("first.png", image_bytes(), "image/png")),
                ("images", ("second.png", image_bytes(), "image/png")),
            ])
        self.assertEqual(response.status_code, 413)
        model.assert_not_called()
        self.assertEqual(list(self.path.iterdir()), [])


class RuntimeTransparencyTests(TestCase):
    def test_relation_skill_body_is_actually_loaded(self):
        for relation in ("friend", "colleague", "family", "lover_committed", "lover_pursuing"):
            rules = load_relation_skill(relation)
            self.assertIn("instructions", rules)
            self.assertIn("##", rules["instructions"])
            self.assertNotIn("description:", rules["instructions"])
        self.assertNotIn("别当真结束", str(load_relation_skill("friend")))

    def test_embedding_readiness_does_not_overpromise_model_availability(self):
        settings = Settings(_env_file=None, memory_similarity_backend="semantic", memory_embedding_provider="local")
        with patch("chat_reply_agent.services.embedding_service.find_spec", return_value=None):
            self.assertEqual(embedding_runtime_status(settings)["readiness"], "dependency_missing")
        with patch("chat_reply_agent.services.embedding_service.find_spec", return_value=object()):
            self.assertEqual(embedding_runtime_status(settings)["readiness"], "not_verified")
        settings.memory_embedding_provider = "dashscope"
        self.assertEqual(embedding_runtime_status(settings)["readiness"], "configuration_missing")
        settings.memory_embedding_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        settings.memory_embedding_api_key = "test-only-key"
        self.assertEqual(embedding_runtime_status(settings)["readiness"], "not_verified")
        settings.memory_similarity_backend = "lexical"
        self.assertEqual(embedding_runtime_status(settings)["readiness"], "lexical")
