from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(Path(__file__).resolve().parents[1] / ".env")


@dataclass(frozen=True)
class Settings:
    base_dir: Path = Path(__file__).resolve().parents[1]
    cors_origins: tuple[str, ...] = tuple(
        value.strip()
        for value in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
        if value.strip()
    )
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "500"))

    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data"

    @property
    def media_dir(self) -> Path:
        return self.base_dir / "media"

    @property
    def database_path(self) -> Path:
        return self.data_dir / "app.db"


settings = Settings()
