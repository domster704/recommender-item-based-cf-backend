from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent


class JWTSettings(BaseSettings):
    secret: SecretStr = Field(..., min_length=32)
    ttl_minutes: int = 14 * 24 * 60
    algorithm: str = "HS256"


class TelegramSettings(BaseSettings):
    token: SecretStr = Field(...)


class SecuritySettings(BaseSettings):
    apikeys: list[str] = Field(...)


class DBSettings(BaseSettings):
    dsn: Path = Field(default=ROOT_DIR / "src" / "infrastructure" / "db" / "db.sqlite")

    @property
    def data_source_name(self) -> str:
        return f"sqlite+aiosqlite:///{self.dsn.as_posix()}"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_prefix="",
        extra="forbid",
        env_nested_delimiter="_",
    )

    debug: bool = False
    bot: TelegramSettings = Field(default_factory=TelegramSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    jwt: JWTSettings = Field(default_factory=JWTSettings)
    db: DBSettings = Field(default_factory=DBSettings)


settings = Settings()
