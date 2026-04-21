from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class AppSettings(BaseSettings):
    index_name: str = Field(default='docs')
    batch_size: int = Field(default=512)
    vector_size: int = Field(default=512)
    es_host: str = Field(default='localhost')
    elastic_user: str = Field(default='elastic')
    elastic_password: str = Field(default='')

    gigachat_api_url: str = "https://gigachat.devices.sberbank.ru/api/v1/embeddings"
    gigachat_model: str = "EmbeddingsGigaR"
    gigachat_auth_token: str | None = None
    gigachat_scope: str | None = None
    gigachat_oauth_url: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"   # игнорируем лишние переменные окружения
    )


settings = AppSettings()