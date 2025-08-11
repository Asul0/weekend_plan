# Файл: src/gigachat_client.py
from langchain_gigachat import GigaChat
from langchain_core.tools import BaseTool
from typing import List, Optional
import logging

from src.config import settings

logger = logging.getLogger(__name__)


def get_gigachat_client(tools: Optional[List[BaseTool]] = None) -> GigaChat:
    """
    Создает и возвращает НОВЫЙ экземпляр клиента GigaChat при каждом вызове.
    Это гарантирует изоляцию между тестами и предотвращает ошибки с "закрытым циклом событий".
    """
    if not settings.GIGACHAT_CREDENTIALS:
        raise ValueError(
            "GIGACHAT_CREDENTIALS не установлены. Проверьте ваш .env файл."
        )

    client_params = {
        "credentials": settings.GIGACHAT_CREDENTIALS,
        "scope": settings.GIGACHAT_SCOPE,
        "verify_ssl_certs": False,
        "model": "GigaChat-Max",
        "timeout": 120,
        "profanity_check": False,
        "temperature": 0.1,
        "max_tokens": 650,
    }

    logger.debug("Созданыие нового экземпляра GigaChat клиента.")

    # Создаем новый клиент
    new_client = GigaChat(**client_params)

    # Если переданы инструменты, привязываем их
    if tools:
        try:
            return new_client.bind_tools(tools)
        except Exception as e:
            logger.error(f"Не удалось привязать инструменты к GigaChat: {e}")
            return new_client

    # Возвращаем клиент без инструментов
    return new_client
