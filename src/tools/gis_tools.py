# Файл: src/tools/gis_tools.py (ФИНАЛЬНАЯ ИСПРАВЛЕННАЯ ВЕРСИЯ)
import logging
from typing import List
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from src.services.gis_service import get_geocoding_details

# ИСПРАВЛЕНИЕ: Импортируем правильные классы из нашего единого источника правды
from src.schemas.data_schemas import ParkInfo, FoodPlaceInfo
from src.services.gis_service import search_parks, search_food_places

logger = logging.getLogger(__name__)

# --- Pydantic-модели для аргументов инструментов ---


class GeocodeAddressArgs(BaseModel):
    """Аргументы для инструмента геокодинга адреса."""

    address: str = Field(description="Полный адрес для поиска, включая улицу и дом.")
    city: str = Field(
        description="Город, в котором находится адрес, для уточнения поиска."
    )


class ParkSearchArgs(BaseModel):
    """Аргументы для инструмента поиска парков."""

    query: str = Field(description="Поисковый запрос, например, 'парк' или 'сквер'.")
    city: str = Field(description="Город для поиска.")


class FoodSearchArgs(BaseModel):
    """Аргументы для инструмента поиска заведений питания."""

    query: str = Field(
        description="Поисковый запрос, например, 'ресторан', 'кафе' или 'поужинать'."
    )
    city: str = Field(description="Город для поиска.")


# --- Инструмент для поиска парков ---


@tool("park_search_tool", args_schema=ParkSearchArgs)
async def park_search_tool(query: str, city: str) -> List[dict]:
    """
    Инструмент для поиска парков и мест для прогулок.
    Возвращает список словарей с информацией о найденных парках.
    """
    logger.info(f"TOOL: park_search_tool. Запрос: query='{query}', city='{city}'")
    try:
        # ИСПРАВЛЕНИЕ: Указываем правильный тип
        parks: List[ParkInfo] = await search_parks(
            original_query=query, city=city, limit=10
        )
        return [park.model_dump(exclude_none=True) for park in parks]
    except Exception as e:
        logger.error(f"Ошибка в park_search_tool: {e}", exc_info=True)
        return []


@tool("food_place_search_tool", args_schema=FoodSearchArgs)
async def food_place_search_tool(query: str, city: str) -> List[dict]:
    """
    Инструмент для поиска ресторанов, кафе и других мест, где можно поесть.
    Возвращает список словарей с информацией о найденных заведениях.
    """
    logger.info(f"TOOL: food_place_search_tool. Запрос: query='{query}', city='{city}'")
    try:
        # Вызываем соответствующую функцию из сервисного слоя
        food_places: List[FoodPlaceInfo] = await search_food_places(
            original_query=query, city=city, limit=10
        )
        # Возвращаем результат в виде списка словарей для совместимости с LangChain
        return [place.model_dump(exclude_none=True) for place in food_places]
    except Exception as e:
        logger.error(f"Ошибка в food_place_search_tool: {e}", exc_info=True)
        return []
