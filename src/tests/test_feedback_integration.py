# tests/test_feedback_integration.py

import unittest
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any

# Настраиваем логирование, чтобы видеть ВСЕ сообщения от агента в консоли
logging.basicConfig(level=logging.INFO)

from src.agent_core.state import AgentState
from src.agent_core.graph import agent_app
from src.schemas.data_schemas import (
    ExtractedInitialInfo,
    OrderedActivityItem,
    Plan,
    Event,
    FoodPlaceInfo,
    ParkInfo,
)

# --- ВАЖНО: Это наш "первоначальный" мок-ответ от узла поиска ---
# Мы будем использовать его для создания первого плана
MOCK_INITIAL_CANDIDATES = {
    "2025-08-01": {
        "MOVIE": [
            Event(
                session_id=1,
                name="Дюна: Часть третья",
                user_event_type_key="MOVIE",
                place_name="Кинотеатр 'Космос'",
                start_time_naive_event_tz=datetime(2025, 8, 1, 19, 30),
                start_time_iso="",
                min_price=1000,
            ).model_dump()
        ],
        "RESTAURANT": [
            FoodPlaceInfo(
                id_gis="gis1",
                name="Дорогая столовая",
                avg_bill_str="2000р",
                min_price=2000,
            ).model_dump()
        ],
        "PARK": [ParkInfo(id_gis="gis2", name="Центральный парк").model_dump()],
    }
}


class TestFeedbackIntegration(unittest.TestCase):
    """
    ФИНАЛЬНАЯ ВЕРСИЯ E2E Тестов.
    Проверяет полный цикл: Создание плана -> Фидбек -> Перестроение плана.
    """

    async def _run_graph_and_get_final_state(self, inputs: dict) -> AgentState:
        """Хелпер для запуска графа и корректного извлечения финального состояния."""
        final_state = None
        async for chunk in agent_app.astream(inputs, {"recursion_limit": 25}):
            # Сохраняем последнее валидное состояние
            last_node_key = next(iter(chunk))
            final_state = chunk[last_node_key]
        return final_state

    async def test_e2e_delete_scenario(self):
        """E2E тест: Проверяет всю цепочку от запроса "убери парк" до получения перестроенного плана без парка."""
        print("\n--- Running E2E Scenario for DELETE command ---")

        # --- ШАГ 1: Создаем первоначальный план ---
        print("--- E2E DELETE: Step 1: Building initial plan ---")
        initial_inputs = {
            "user_message": "Найди кино, ресторан и парк в Тестовом Городе на 1 августа 2025",
            # Имитируем, что предыдущие узлы уже отработали
            "search_criteria": ExtractedInitialInfo(
                city="Тестовый Город",
                dates_description="1 августа 2025",
                ordered_activities=[
                    OrderedActivityItem(activity_type="MOVIE", query_details="кино"),
                    OrderedActivityItem(
                        activity_type="RESTAURANT", query_details="ресторан"
                    ),
                    OrderedActivityItem(activity_type="PARK", query_details="парк"),
                ],
            ),
            "cached_candidates": MOCK_INITIAL_CANDIDATES,
            "parsed_dates_iso": [datetime(2025, 8, 1).isoformat()],
        }

        state_after_build = await self._run_graph_and_get_final_state(initial_inputs)
        self.assertIsNotNone(
            state_after_build.get("current_plan"),
            "Первоначальный план должен быть построен",
        )
        self.assertEqual(
            len(state_after_build["current_plan"].items),
            3,
            "В первоначальном плане должно быть 3 элемента",
        )

        # --- ШАГ 2: Даем фидбек и перестраиваем план ---
        print("--- E2E DELETE: Step 2: Applying feedback ---")
        feedback_query = "Отлично, только давай без парка"

        # ВАЖНО: Мы используем состояние ПОСЛЕ первого прогона как вход для второго
        feedback_inputs = {
            **state_after_build,
            "user_message": feedback_query,
            "command_queue": [],
        }

        final_state = await self._run_graph_and_get_final_state(feedback_inputs)

        # 3. Assert
        self.assertIsNotNone(
            final_state.get("current_plan"), "Финальный план не должен быть пустым"
        )
        final_plan = final_state["current_plan"]

        self.assertEqual(
            len(final_plan.items), 2, "В финальном плане должно быть 2 элемента"
        )

        item_names_in_plan = {item.get("name") for item in final_plan.items}
        self.assertNotIn(
            "Центральный парк",
            item_names_in_plan,
            "Парк должен быть удален из финального плана",
        )

        print("--- E2E DELETE command Test Passed ---")

    async def test_e2e_min_price_scenario(self):
        """E2E тест: Проверяет всю цепочку от запроса "самый дешевый" до получения перестроенного плана."""
        print("\n--- Running E2E Scenario for MIN operator ---")

        # 1. Arrange
        # Добавляем "Супер Дешевый Фильм" в кэш
        extended_candidates = MOCK_INITIAL_CANDIDATES.copy()
        extended_candidates["2025-08-01"]["MOVIE"].append(
            Event(
                session_id=10,
                name="Супер Дешевый Фильм",
                user_event_type_key="MOVIE",
                place_name="Кино 2",
                start_time_naive_event_tz=datetime(2025, 8, 1, 20, 0),
                start_time_iso="",
                min_price=100,
            ).model_dump()
        )

        # --- ШАГ 1: Создаем первоначальный план (с дорогим фильмом) ---
        print("--- E2E MIN: Step 1: Building initial plan ---")
        initial_inputs = {
            "user_message": "Найди кино, ресторан и парк",
            "search_criteria": ExtractedInitialInfo(
                city="Тестовый Город",
                ordered_activities=[
                    OrderedActivityItem(activity_type="MOVIE", query_details="фильм"),
                    OrderedActivityItem(
                        activity_type="RESTAURANT", query_details="ресторан"
                    ),
                    OrderedActivityItem(activity_type="PARK", query_details="парк"),
                ],
            ),
            "cached_candidates": MOCK_INITIAL_CANDIDATES,  # Сначала даем только дорогие варианты
            "parsed_dates_iso": [datetime(2025, 8, 1).isoformat()],
        }
        state_after_build = await self._run_graph_and_get_final_state(initial_inputs)
        self.assertEqual(
            state_after_build["current_plan"].items[0]["name"], "Дюна: Часть третья"
        )

        # --- ШАГ 2: Даем фидбек и перестраиваем план ---
        print("--- E2E MIN: Step 2: Applying feedback ---")
        feedback_query = "Это слишком дорого, найди мне самый дешевый фильм"

        # ВАЖНО: Теперь мы подкладываем полный кэш с дешевым фильмом
        state_after_build["cached_candidates"] = extended_candidates
        feedback_inputs = {
            **state_after_build,
            "user_message": feedback_query,
            "command_queue": [],
        }

        final_state = await self._run_graph_and_get_final_state(feedback_inputs)

        # 3. Assert
        self.assertIsNotNone(
            final_state.get("current_plan"), "Финальный план не должен быть пустым"
        )
        final_plan = final_state["current_plan"]
        movie_in_plan = next(
            (
                item
                for item in final_plan.items
                if item.get("user_event_type_key") == "MOVIE"
            ),
            None,
        )

        self.assertIsNotNone(movie_in_plan, "В финальном плане должен быть фильм")
        self.assertEqual(
            movie_in_plan["name"],
            "Супер Дешевый Фильм",
            "В плане должен быть самый дешевый фильм",
        )

        print("--- E2E MIN operator Test Passed ---")


if __name__ == "__main__":
    # Запускаем асинхронные тесты
    asyncio.run(unittest.main())
