import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import asyncio

from src.agent_core.state import AgentState
from src.schemas.data_schemas import (
    UserIntent,
    RouteSegment,
    Event,
    FoodPlaceInfo,
    Plan,
)
from src.agent_core.nodes import presenter_node


class TestPlanPresentation:
    """Тест для проверки логики отображения плана"""

    @pytest.fixture
    def mock_external_apis(self):
        """Мокаем внешние API с реалистичными данными"""
        with patch(
            "src.services.afisha_service.fetch_cities"
        ) as mock_fetch_cities, patch(
            "src.services.afisha_service.search_sessions"
        ) as mock_search_sessions, patch(
            "src.services.gis_service.get_coords_from_address"
        ) as mock_geocode, patch(
            "src.services.gis_service.search_food_places"
        ) as mock_search_food, patch(
            "src.services.gis_service.get_route"
        ) as mock_get_route, patch(
            "src.gigachat_client.get_gigachat_client"
        ) as mock_gigachat:

            # Мокаем Афишу
            mock_fetch_cities.return_value = [{"id": 2010, "name": "Воронеж"}]
            mock_search_sessions.return_value = [
                {
                    "id": "movie_1",
                    "name": "Доктор Динозавров",
                    "venue": {"name": "Левый берег", "address": "Ленинский просп., 1д"},
                    "datetime": "2025-09-15T10:00:00",
                    "price": "220 ₽",
                    "duration": 94,
                }
            ]

            # Мокаем 2GIS
            mock_geocode.return_value = (39.199775, 51.660548)
            mock_search_food.return_value = [
                {
                    "id": "rest_1",
                    "name": "ЕваЕла, рестобистро",
                    "address": "улица Комиссаржевской, 7",
                    "schedule": "Пн–Вс 11:11–24:00",
                    "average_check": "1200 ₽",
                    "coordinates": [39.200000, 51.660000],
                }
            ]
            mock_get_route.return_value = {
                "duration": 300,  # 5 минут
                "distance": 2500,  # 2.5 км
                "segments": [{"duration": 300, "distance": 2500}],
            }

            # Мокаем GigaChat
            mock_gigachat.return_value.chat_completion.return_value = {
                "choices": [{"message": {"content": "MOVIE"}}]
            }

            yield mock_fetch_cities, mock_search_sessions, mock_geocode, mock_search_food, mock_get_route, mock_gigachat

    @pytest.fixture
    def test_state(self):
        """Создаем тестовое состояние агента"""
        state = AgentState(
            chat_id=12345,
            user_message="Найди кино и кафе на завтра в Воронеже",
            chat_history=[],
            current_intent=UserIntent.PLAN_REQUEST,
            extracted_criteria={
                "city": "Воронеж",
                "dates_description": "завтра",
                "ordered_activities": [
                    {"activity_type": "MOVIE", "query_details": "кино"},
                    {"activity_type": "RESTAURANT", "query_details": "кафе"},
                ],
                "person_count": 1,
            },
            search_results={
                "MOVIE": [
                    {
                        "id": "movie_1",
                        "name": "Доктор Динозавров",
                        "venue": {
                            "name": "Левый берег",
                            "address": "Ленинский просп., 1д",
                        },
                        "datetime": "2025-09-15T10:00:00",
                        "price": "220 ₽",
                        "duration": 94,
                        "coordinates": [39.199775, 51.660548],
                    }
                ],
                "RESTAURANT": [
                    {
                        "id": "rest_1",
                        "name": "ЕваЕла, рестобистро",
                        "address": "улица Комиссаржевской, 7",
                        "schedule": "Пн–Вс 11:11–24:00",
                        "average_check": "1200 ₽",
                        "coordinates": [39.200000, 51.660000],
                    }
                ],
            },
            current_plan=Plan(
                items=[
                    {
                        "activity": {
                            "activity_type": "MOVIE",
                            "name": "Доктор Динозавров",
                            "venue": "Левый берег",
                            "address": "Ленинский просп., 1д",
                            "start_time": datetime(2025, 9, 15, 10, 0),
                            "duration": 94,
                            "price": "220 ₽",
                            "coordinates": [39.199775, 51.660548],
                        }
                    },
                    {
                        "activity": {
                            "activity_type": "RESTAURANT",
                            "name": "ЕваЕла, рестобистро",
                            "address": "улица Комиссаржевской, 7",
                            "schedule": "Пн–Вс 11:11–24:00",
                            "average_check": "1200 ₽",
                            "coordinates": [39.200000, 51.660000],
                        }
                    },
                ],
                total_travel_seconds=0,
                warnings=[],
            ),
            is_awaiting_start_address=True,
        )
        return state

    @pytest.mark.asyncio
    async def test_current_plan_presentation(self, test_state, mock_external_apis):
        """Тестируем текущее поведение - должно показывать 'Переход ко второму пункту'"""

        # Запускаем presenter_node
        result_state = await presenter_node(test_state)

        # Проверяем что план отображается
        assert "current_plan_display" in result_state
        plan_display = result_state["current_plan_display"]

        # Проверяем что показывается "Переход ко второму пункту" (текущее поведение)
        assert "Переход ко второму пункту" in plan_display
        assert (
            "Переезд" not in plan_display
        )  # Не должно быть реального времени переезда

        # Проверяем что запрашивается стартовый адрес
        assert "Откуда вы планируете начать" in plan_display

    @pytest.mark.asyncio
    async def test_plan_with_route_calculation(self, test_state, mock_external_apis):
        """Тестируем план с рассчитанными маршрутами между точками"""

        # Добавляем маршрут между точками плана
        test_state["current_plan"].items[0]["travel_info_to_next"] = {
            "duration_seconds": 300,  # 5 минут
            "distance_meters": 2500,  # 2.5 км
        }

        # Запускаем presenter_node
        result_state = await presenter_node(test_state)

        # Проверяем что план отображается
        assert "current_plan_display" in result_state
        plan_display = result_state["current_plan_display"]

        # Проверяем что показывается реальное время переезда (желаемое поведение)
        assert "Переезд" in plan_display
        assert "5 мин" in plan_display
        assert "Переход ко второму пункту" not in plan_display

    @pytest.mark.asyncio
    async def test_final_plan_with_start_address(self, test_state, mock_external_apis):
        """Тестируем финальный план с маршрутом от стартового адреса"""

        # Добавляем стартовый адрес и маршрут от него
        test_state["start_address"] = "Пограничная 2"
        test_state["current_plan"].items[0]["travel_info_to_here"] = {
            "duration_seconds": 720,  # 12 минут
            "distance_meters": 7400,  # 7.4 км
        }

        # Добавляем маршрут между точками
        test_state["current_plan"].items[0]["travel_info_to_next"] = {
            "duration_seconds": 300,  # 5 минут
            "distance_meters": 2500,  # 2.5 км
        }

        # Запускаем presenter_node
        result_state = await presenter_node(test_state)

        # Проверяем что план отображается
        assert "current_plan_display" in result_state
        plan_display = result_state["current_plan_display"]

        # Проверяем что показывается маршрут от стартового адреса
        assert "от дома" in plan_display
        assert "12 мин" in plan_display
        assert "7.4 км" in plan_display

        # Проверяем что показывается маршрут между точками
        assert "Переезд" in plan_display
        assert "5 мин" in plan_display

        # Проверяем общее время в пути
        assert "Общее время в пути" in plan_display
        assert "17 мин" in plan_display  # 12 + 5 = 17


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
