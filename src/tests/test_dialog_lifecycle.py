# tests/test_dialog_lifecycle.py

import unittest
import asyncio
from datetime import datetime
from unittest.mock import patch, AsyncMock

from src.agent_core.state import AgentState
from src.schemas.data_schemas import (
    Plan,
    Event,
    FoodPlaceInfo,
    ParkInfo,
    SemanticConstraint,
    ExtractedInitialInfo,
    OrderedActivityItem,
)
from src.agent_core.command_processor import CommandProcessor
from src.agent_core.nodes import (
    delete_activity_node,
    add_activity_node,
    refine_plan_node,
)


class TestDialogLifecycle(unittest.TestCase):
    """
    Сценарный тест, имитирующий полный диалог с пользователем
    для проверки всей цепочки обработки команд БЕЗ вызова LLM.
    """

    def _create_initial_state(self) -> AgentState:
        """Создает начальное состояние с планом из 3-х элементов."""
        start_time = datetime(2025, 8, 1, 19, 30, 0)
        movie = Event(
            session_id=1,
            name="Фильм",
            user_event_type_key="MOVIE",
            place_name="Кино",
            start_time_naive_event_tz=start_time,
            start_time_iso="",
            min_price=1000,
        ).model_dump()
        food = FoodPlaceInfo(
            id_gis="gis1", name="Ресторан", avg_bill_str=""
        ).model_dump()
        park = ParkInfo(id_gis="gis2", name="Парк").model_dump()

        state: AgentState = {
            "current_plan": Plan(items=[movie, food, park], total_travel_seconds=0),
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
            "cached_candidates": {
                "2025-08-01": {"MOVIE": [movie], "RESTAURANT": [food], "PARK": [park]}
            },
            "parsed_dates_iso": [start_time.isoformat()],
            "pinned_items": {
                "MOVIE": Event.model_validate(movie),
                "RESTAURANT": FoodPlaceInfo.model_validate(food),
                "PARK": ParkInfo.model_validate(park),
            },
            "command_queue": [],
        }
        return state

    def test_full_dialog_scenario(self):
        """Тестирует последовательность команд: delete -> add -> modify."""

        async def run():
            print("\n--- Running Full Dialog Lifecycle Test ---")

            # === ШАГ 1: "удали ресторан" ===
            print("--- Step 1: User says 'удали ресторан' ---")
            state_step1 = self._create_initial_state()

            intents_step1 = [
                SemanticConstraint(command_type="delete", target="RESTAURANT")
            ]

            processor1 = CommandProcessor(state_step1, intents_step1)
            state_step1["command_queue"] = processor1.process()
            state_after_delete = await delete_activity_node(state_step1)

            activities_s1 = state_after_delete["search_criteria"].ordered_activities
            self.assertEqual(len(activities_s1), 2)
            self.assertNotIn("RESTAURANT", {act.activity_type for act in activities_s1})
            self.assertNotIn("RESTAURANT", state_after_delete["pinned_items"])
            print("--- Step 1 PASSED ---")

            # === ШАГ 2: "добавь ресторан" ===
            print("--- Step 2: User says 'добавь ресторан' ---")
            state_step2 = state_after_delete

            intents_step2 = [
                SemanticConstraint(command_type="add", target="RESTAURANT")
            ]

            # --- КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: Используем 'with patch' правильно ---
            # Мы патчим food_place_search_tool в модуле nodes, где он и вызывается
            with patch(
                "src.agent_core.nodes.food_place_search_tool", new_callable=AsyncMock
            ) as mock_tool:
                # Настраиваем, что ВЫЗОВ этого мок-объекта вернет нужное значение
                mock_tool.return_value = [
                    FoodPlaceInfo(id_gis="gis_new", name="Новый Ресторан").model_dump()
                ]

                processor2 = CommandProcessor(state_step2, intents_step2)
                state_step2["command_queue"] = processor2.process()
                state_after_add = await add_activity_node(state_step2)

                # Проверяем, что мок-объект был вызван с правильным словарем
                mock_tool.assert_called_once_with(
                    {"query": "restaurant", "city": "Тестовый Город"}
                )
            # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

            activities_s2 = state_after_add["search_criteria"].ordered_activities
            self.assertEqual(len(activities_s2), 3)
            self.assertEqual(activities_s2[-1].activity_type, "RESTAURANT")
            self.assertIn(
                "RESTAURANT", state_after_add["cached_candidates"]["2025-08-01"]
            )
            print("--- Step 2 PASSED ---")

            # === ШАГ 3: "поменяй фильм на 4 часа позже" ===
            print("--- Step 3: User says 'поменяй фильм на 4 часа позже' ---")
            state_step3 = state_after_add

            intents_step3 = [
                SemanticConstraint(
                    command_type="modify",
                    target="MOVIE",
                    attribute="start_time",
                    operator="GREATER_THAN",
                    value_num=4,
                    value_unit="часа",
                )
            ]

            processor3 = CommandProcessor(state_step3, intents_step3)
            state_step3["command_queue"] = processor3.process()
            state_after_modify = await refine_plan_node(state_step3)

            self.assertNotIn(
                "MOVIE",
                state_after_modify["pinned_items"],
                "Фильм должен быть откреплен",
            )
            self.assertIn("PARK", state_after_modify["pinned_items"])
            print("--- Step 3 PASSED ---")

        asyncio.run(run())


if __name__ == "__main__":
    asyncio.run(unittest.main())
