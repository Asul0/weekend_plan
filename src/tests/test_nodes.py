# tests/test_nodes.py

import unittest
import asyncio
from datetime import datetime
from unittest.mock import patch, AsyncMock
from src.agent_core.state import AgentState
from src.schemas.data_schemas import (
    ExtractedInitialInfo,
    OrderedActivityItem,
    ChangeRequest,
    Plan,
    Event,
    FoodPlaceInfo,
    ParkInfo,
)
from src.agent_core.nodes import delete_activity_node, add_activity_node


class TestNodeFunctions(unittest.TestCase):
    """
    Набор модульных тестов для отдельных узлов графа.
    Эти тесты НЕ вызывают LLM и работают с подготовленными данными.
    """

    async def test_unit_add_activity_node(self):
        """
        Модульный тест для узла add_activity_node.
        Проверяет, что узел ищет кандидатов и корректно обновляет state.
        """
        print("\n--- Running Unit Test for add_activity_node ---")

        # 1. Arrange
        # Настраиваем, что наш "поддельный" инструмент вернет при вызове
        mock_candidates = [
            FoodPlaceInfo(id_gis="gis_new", name="Новое Кафе").model_dump()
        ]

        # Создаем начальное состояние
        initial_state = self._create_mock_state_with_full_plan()
        initial_state["search_criteria"] = ExtractedInitialInfo(
            city="Тестовый Город",
            ordered_activities=[
                OrderedActivityItem(activity_type="MOVIE", query_details="фильм"),
            ],
        )
        initial_state["parsed_dates_iso"] = [datetime(2025, 8, 1).isoformat()]
        initial_state["cached_candidates"] = {
            "2025-08-01": {"MOVIE": [initial_state["current_plan"].items[0]]}
        }

        # Создаем команду "add"
        add_command = ChangeRequest(
            command="add",
            new_activity=OrderedActivityItem(
                activity_type="RESTAURANT", query_details="кафе"
            ),
        )
        initial_state["last_structured_command"] = add_command

        # 2. Act
        # Используем patch как контекстный менеджер, что более надежно
        with patch(
            "src.agent_core.nodes.food_place_search_tool.ainvoke",
            new_callable=AsyncMock,
        ) as mock_search_tool:
            mock_search_tool.return_value = mock_candidates

            # Вызываем узел напрямую, так как он асинхронный
            final_state = await add_activity_node(initial_state)

        # 3. Assert
        # Проверяем, что инструмент поиска был вызван с правильными параметрами
        mock_search_tool.assert_called_once_with(
            {"query": "кафе", "city": "Тестовый Город"}
        )

        # Проверяем, что кэш был обновлен
        final_cache = final_state["cached_candidates"]["2025-08-01"]
        self.assertIn("RESTAURANT", final_cache)
        self.assertEqual(len(final_cache["RESTAURANT"]), 1)
        self.assertEqual(final_cache["RESTAURANT"][0]["name"], "Новое Кафе")

        # Проверяем, что список дел был обновлен
        final_activities = final_state["search_criteria"].ordered_activities
        self.assertEqual(len(final_activities), 2)
        self.assertEqual(final_activities[-1].activity_type, "RESTAURANT")

        # Проверяем, что план сброшен
        self.assertIsNone(final_state["current_plan"])

        print("--- add_activity_node Unit Test Passed ---")

    def _create_mock_state_with_full_plan(self) -> AgentState:
        """Вспомогательная функция для создания комплексного состояния."""
        # Эта функция может быть вынесена в общий test_utils, но пока оставим здесь
        start_time_movie = datetime(2025, 8, 1, 19, 30, 0)
        mock_movie = Event(
            session_id=1,
            name="Фильм",
            user_event_type_key="MOVIE",
            place_name="Кино",
            start_time_naive_event_tz=start_time_movie,
            start_time_iso="",
        ).model_dump()
        mock_food = FoodPlaceInfo(
            id_gis="gis1", name="Кафе", avg_bill_str=""
        ).model_dump()
        mock_park = ParkInfo(id_gis="gis2", name="Парк").model_dump()
        mock_plan = Plan(
            items=[mock_movie, mock_food, mock_park], total_travel_seconds=0
        )
        state: AgentState = {"current_plan": mock_plan}
        return state

    def test_unit_delete_activity_node(self):
        """
        Модульный тест для узла delete_activity_node.
        Проверяет, что узел корректно удаляет активность из state.
        """

        async def run():
            print("\n--- Running Unit Test for delete_activity_node ---")

            # 1. Arrange
            initial_state = self._create_mock_state_with_full_plan()
            initial_state["search_criteria"] = ExtractedInitialInfo(
                ordered_activities=[
                    OrderedActivityItem(activity_type="MOVIE", query_details="фильм"),
                    OrderedActivityItem(
                        activity_type="RESTAURANT", query_details="ресторан"
                    ),
                    OrderedActivityItem(activity_type="PARK", query_details="парк"),
                ]
            )
            delete_command = ChangeRequest(command="delete", target="PARK")
            initial_state["last_structured_command"] = delete_command

            # 2. Act
            final_state = await delete_activity_node(initial_state)

            # 3. Assert
            self.assertIsNone(final_state["current_plan"], "План должен быть сброшен")
            final_activities = final_state["search_criteria"].ordered_activities
            self.assertEqual(len(final_activities), 2, "Должно остаться 2 активности")
            activity_types = {act.activity_type for act in final_activities}
            self.assertNotIn("PARK", activity_types, "Парк должен быть удален")
            self.assertIn("MOVIE", activity_types, "Фильм должен был остаться")

            print("--- delete_activity_node Unit Test Passed ---")

        asyncio.run(run())

    
if __name__ == "__main__":
    unittest.main()
