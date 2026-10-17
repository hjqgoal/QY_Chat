from pathlib import Path
import os
from unittest import TestCase
from unittest.mock import patch

from pydantic import ValidationError

from chat_reply_agent.config import Settings


class ConfigurationSafetyTests(TestCase):
    def test_embedding_provider_timeout_and_secret_serialization(self) -> None:
        for config in ({"memory_embedding_provider": "unknown"},
                       {"memory_embedding_timeout_seconds": 0},
                       {"memory_embedding_timeout_seconds": 121}):
            with self.subTest(config=config), self.assertRaises(ValidationError):
                Settings(_env_file=None, **config)
        settings = Settings(_env_file=None, memory_embedding_api_key="test-only-secret")
        self.assertNotIn("test-only-secret", repr(settings))
        self.assertNotIn("memory_embedding_api_key", settings.model_dump())

    def test_half_life_configuration_is_validated(self) -> None:
        for value in (0, -10, float("inf"), float("nan")):
            for field in ("memory_decay_half_life_days", "memory_decay_half_life_by_scenario"):
                setting = {"invitation": value} if field.endswith("scenario") else value
                with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                    Settings(_env_file=None, **{field: setting})

    def test_scenario_overrides_can_be_loaded_from_environment(self) -> None:
        with patch.dict(os.environ, {
            "MEMORY_DECAY_HALF_LIFE_DAYS": "60",
            "MEMORY_DECAY_HALF_LIFE_BY_SCENARIO": '{"invitation": 7, "daily_chat": 30}',
        }):
            settings = Settings(_env_file=None)
        self.assertEqual(settings.memory_decay_half_life_days, 60)
        self.assertEqual(settings.memory_decay_half_life_by_scenario, {
            "invitation": 7, "daily_chat": 30,
        })

    def test_example_environment_file_contains_no_api_key(self) -> None:
        example_path = Path(__file__).resolve().parents[1] / ".env.example"
        values = {
            key.strip(): value.strip()
            for line in example_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#") and "=" in line
            for key, value in [line.split("=", 1)]
        }

        self.assertIn("OPENAI_API_KEY", values)
        self.assertEqual(values["OPENAI_API_KEY"], "")
        self.assertEqual(values["MEMORY_EMBEDDING_API_KEY"], "")
