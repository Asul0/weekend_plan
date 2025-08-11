# tests/test_planner.py

import unittest
import asyncio
from datetime import datetime

from src.agent_core.state import AgentState
from src.schemas.data_schemas import Event
from src.agent_core.planner import PlanBuilder


class TestPlanBuilder(unittest.TestCase):
    """
    Набор модульных тестов для PlanBuilder.
    Проверяет внутреннюю логику планировщика, особенно сортировку.
    """

    def _create_base_mock_state(self) -> AgentState:
        """Создает базовое состояние с датами для инициализации PlanBuilder."""
        state: AgentState = {
            "parsed_dates_iso": [datetime(2025, 8, 1, 9, 0, 0).isoformat()],
            "cached_candidates": {},
            "sorting_preference": None,
            # Добавляем остальные поля, чтобы избежать ошибок типизации
            "user_message": "",
            "chat_history": [],
            "classified_intent": None,
            "next_action": None,
            "error": None,
            "is_awaiting_clarification": None,
            "plan_presented": False,
            "plan_warnings": [],
            "search_criteria": None,
            "current_plan": None,
            "plan_builder_result": None,
            "analyzed_feedback": None,
            "pinned_items": {},
            "command_queue": [],
            "city_id_afisha": None,
            "parsed_end_dates_iso": None,
            "user_start_coordinates": None,
            "is_awaiting_address": False,
            "status_message_id": None,
        }
        return state

    def test_get_candidates_with_min_price_sorting(self):
        """
        Тест-кейс: Проверяет, что PlanBuilder правильно сортирует кандидатов
        по цене (MIN) согласно инструкции в state.
        """
        print("\n--- Running Unit Test for PlanBuilder Sorting (MIN) ---")

        # 1. Arrange
        state = self._create_base_mock_state()

        # Создаем кандидатов в неправильном порядке
        candidates = [
            Event(
                name="Дорогой фильм",
                min_price=3000,
                session_id=1,
                user_event_type_key="MOVIE",
                place_name="A",
                start_time_naive_event_tz=datetime.now(),
                start_time_iso="",
            ),
            Event(
                name="Дешевый фильм",
                min_price=500,
                session_id=2,
                user_event_type_key="MOVIE",
                place_name="B",
                start_time_naive_event_tz=datetime.now(),
                start_time_iso="",
            ),
            Event(
                name="Средний фильм",
                min_price=1500,
                session_id=3,
                user_event_type_key="MOVIE",
                place_name="C",
                start_time_naive_event_tz=datetime.now(),
                start_time_iso="",
            ),
            Event(
                name="Фильм без цены",
                min_price=None,
                session_id=4,
                user_event_type_key="MOVIE",
                place_name="D",
                start_time_naive_event_tz=datetime.now(),
                start_time_iso="",
            ),
        ]
        state["cached_candidates"] = {"2025-08-01": {"MOVIE": candidates}}

        # Создаем инструкцию по сортировке
        state["sorting_preference"] = {
            "target": "MOVIE",
            "attribute": "price",
            "order": "MIN",
        }

        # 2. Act
        # Инициализируем PlanBuilder и вызываем тестируемый метод
        builder = PlanBuilder(state)
        sorted_candidates = builder._get_candidates_for_slot("MOVIE")

        # 3. Assert
        self.assertEqual(len(sorted_candidates), 4, "Должны вернуться все кандидаты")

        # Проверяем правильность порядка
        self.assertEqual(
            sorted_candidates[0].name,
            "Дешевый фильм",
            "Первым должен быть самый дешевый",
        )
        self.assertEqual(sorted_candidates[1].name, "Средний фильм")
        self.assertEqual(sorted_candidates[2].name, "Дорогой фильм")
        self.assertEqual(
            sorted_candidates[3].name,
            "Фильм без цены",
            "Фильм без цены должен быть последним",
        )

        # Проверяем, что инструкция была очищена
        self.assertIsNone(
            state["sorting_preference"],
            "Инструкция по сортировке должна быть очищена после использования",
        )

        print("--- PlanBuilder Sorting Test Passed ---")


if __name__ == "__main__":
    unittest.main()
