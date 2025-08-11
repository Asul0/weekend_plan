# Файл: src/config.py
import os
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class Settings:
    # GigaChat API
    GIGACHAT_CREDENTIALS: str = os.getenv("GIGACHAT_CREDENTIALS", "")
    GIGACHAT_SCOPE: str = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")

    # GIS (2GIS) API
    GIS_API_KEY: str = os.getenv("GIS_API_KEY", "")

    # Afisha Proxy
    AFISHA_PROXY_BASE_URL: str = os.getenv(
        "AFISHA_PROXY_BASE_URL", "http://localhost:8000"
    )

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    def __init__(self):
        # Логируем статус ключа GigaChat при создании объекта настроек
        if self.GIGACHAT_CREDENTIALS:
            logger.info("Ключ GIGACHAT_CREDENTIALS успешно загружен из .env.")
        else:
            logger.error(
                "КРИТИЧЕСКАЯ ОШИБКА: GIGACHAT_CREDENTIALS не найден в .env файле или переменных окружения."
            )

        if not self.GIS_API_KEY:
            logger.warning("Ключ GIS_API_KEY не найден в .env.")


settings = Settings()
