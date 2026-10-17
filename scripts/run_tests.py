"""Run the offline suite without touching personal data or calling paid models."""

import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    sys.dont_write_bytecode = True
    with TemporaryDirectory(prefix="qy-offline-tests-") as directory:
        os.environ.update(
            DATABASE_URL=f"sqlite:///{directory}/test.db",
            UPLOAD_DIR=f"{directory}/uploads",
            OPENAI_API_KEY="",
            MEMORY_EMBEDDING_API_KEY="",
            MEMORY_EMBEDDING_PROVIDER="local",
            MEMORY_SIMILARITY_BACKEND="lexical",
        )
        Path(os.environ["UPLOAD_DIR"]).mkdir()
        from chat_reply_agent.database import engine, init_db

        try:
            init_db()
            suite = unittest.defaultTestLoader.discover(str(root / "tests"))
            result = unittest.TextTestRunner(verbosity=1).run(suite)
            return 0 if result.wasSuccessful() else 1
        finally:
            engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
