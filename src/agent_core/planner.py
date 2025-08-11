# --- НАЧАЛО БЛОКА ДЛЯ ЗАМЕНЫ: planner.py (версия с "Pinned Items") ---

import logging
import itertools
from datetime import datetime, timedelta, time
from typing import List, Union, Optional, Dict, Any, Tuple
from collections import Counter

from src.agent_core.schedule_parser import parse_schedule_and_check_open
from src.agent_core.state import AgentState, PlanItem
from src.schemas.data_schemas import (
    Event,
    ParkInfo,
    FoodPlaceInfo,
    Plan,
    PlanBuilderResult,
    ChangeRequest,
    OrderedActivityItem,
)
from src.services.gis_service import get_route

logger = logging.getLogger(__name__)

DEFAULT_POI_DURATION_MINUTES = 30
MAX_ROUTES_TO_CALCULATE = 15
DEFAULT_START_HOUR = 9


class PlanBuilder:
    """
    Умный построитель планов v3.0.
    Работает с концепцией "закрепленных" элементов (pinned_items) для гибкой
    модификации существующих планов.
    """

    def __init__(self, state: AgentState):
        self.state = state
        self.criteria = state.get("search_criteria")
        self.command: Optional[ChangeRequest] = state.get("last_structured_command")
        self.plan_warnings = state.get("plan_warnings", [])
        self.pinned_items: Dict[str, PlanItem] = state.get("pinned_items", {})

        self.is_flexible_start_time = (
            self.criteria.raw_time_description is None if self.criteria else False
        )
        logger.info(
            f"PlanBuilder initialized. Flexible start time: {self.is_flexible_start_time}. Pinned items: {list(self.pinned_items.keys())}"
        )

        # Получаем кэш кандидатов для текущей даты
        current_date_key = datetime.fromisoformat(
            state["parsed_dates_iso"][0]
        ).strftime("%Y-%m-%d")
        all_cached_candidates = state.get("cached_candidates", {})
        self.daily_cache: Dict[str, List[PlanItem]] = all_cached_candidates.get(
            current_date_key, {}
        )

        # Настройка времени начала и конца
        self.user_start_time = datetime.fromisoformat(state["parsed_dates_iso"][0])
        if (
            self.is_flexible_start_time and not self.pinned_items
        ):  # Гибкое время только для первого плана
            self.user_start_time = self.user_start_time.replace(
                hour=DEFAULT_START_HOUR, minute=0, second=0, microsecond=0
            )
            logger.info(
                f"Start time is flexible. Setting default start to {self.user_start_time.strftime('%H:%M')}"
            )

        self.user_end_time = (
            datetime.fromisoformat(state["parsed_end_dates_iso"][0])
            if state.get("parsed_end_dates_iso")
            else self.user_start_time.replace(hour=23, minute=59)
        )
        self.user_start_coords = state.get("user_start_coordinates")
        self.ordered_activities = (
            self.criteria.ordered_activities if self.criteria else []
        )
        self.route_calculations_count = 0

    def _get_item_coords(self, item: PlanItem) -> Optional[Dict[str, float]]:
        # (Без изменений)
        if isinstance(item, Event) and item.place_coords_lon and item.place_coords_lat:
            return {"lon": item.place_coords_lon, "lat": item.place_coords_lat}
        if isinstance(item, (ParkInfo, FoodPlaceInfo)) and item.coords:
            return {"lon": item.coords[0], "lat": item.coords[1]}
        return None

    async def _check_compatibility(
        self, last_item_state: Dict[str, Any], item_to_add: PlanItem
    ) -> Dict[str, Any]:
        # (Без изменений)
        if self.route_calculations_count >= MAX_ROUTES_TO_CALCULATE:
            return {"compatible": False, "reason": "Превышен лимит расчетов маршрутов."}

        last_item_end_time = last_item_state["end_time"]
        last_item_coords = last_item_state["coords"]
        item_to_add_coords = self._get_item_coords(item_to_add)
        item_name = getattr(item_to_add, "name", "N/A")
        item_type = item_to_add.__class__.__name__

        travel_seconds = 0
        if last_item_coords and item_to_add_coords:
            self.route_calculations_count += 1
            route_info = await get_route(points=[last_item_coords, item_to_add_coords])
            if route_info.get("status") != "success":
                return {
                    "compatible": False,
                    "reason": f"Не удалось построить маршрут до '{item_name}'.",
                    "type": item_type,
                }
            travel_seconds = route_info.get("duration_seconds", 0)

        arrival_time = last_item_end_time + timedelta(seconds=travel_seconds)
        travel_minutes = round(travel_seconds / 60)

        if isinstance(item_to_add, Event):
            if arrival_time > item_to_add.start_time_naive_event_tz:
                reason = (
                    f"Вы не успеваете на '{item_name}' (начало в {item_to_add.start_time_naive_event_tz.strftime('%H:%M')}). "
                    f"Дорога занимает около {travel_minutes} мин, прибытие в {arrival_time.strftime('%H:%M')}."
                )
                return {"compatible": False, "reason": reason, "type": item_type}
            start_time = item_to_add.start_time_naive_event_tz
            duration = item_to_add.duration_minutes or 120
            end_time = start_time + timedelta(minutes=duration)
        else:
            start_time = arrival_time
            is_open, possible_end_time, reason_closed = parse_schedule_and_check_open(
                schedule_str=getattr(item_to_add, "schedule_str", None),
                visit_start_dt=start_time,
                desired_duration_minutes=DEFAULT_POI_DURATION_MINUTES,
                item_type_for_schedule=(
                    "park" if isinstance(item_to_add, ParkInfo) else "food"
                ),
                poi_name_for_log=item_name,
            )
            if not is_open:
                reason = f"'{item_name}' будет закрыт к моменту вашего прибытия ({arrival_time.strftime('%H:%M')}). Причина: {reason_closed}."
                return {"compatible": False, "reason": reason, "type": item_type}
            end_time = possible_end_time

        if end_time > self.user_end_time:
            return {
                "compatible": False,
                "reason": f"Посещение '{item_name}' не укладывается в ваше время (до {self.user_end_time.strftime('%H:%M')}).",
                "type": item_type,
            }

        return {
            "compatible": True,
            "item": item_to_add,
            "start_time": start_time,
            "end_time": end_time,
            "coords": item_to_add_coords,
            "name": item_name,
            "travel_to_seconds": travel_seconds,
        }

    def _get_candidates_for_slot(self, activity_type: str) -> List[PlanItem]:
        """
        Извлекает кандидатов для слота, применяя фильтрацию по времени
        и новую логику сортировки из state.
        """
        candidates = self.daily_cache.get(activity_type, [])
        if not candidates:
            return []

        # --- НОВАЯ ЛОГИКА: Применение инструкции по сортировке ---
        sorting_pref = self.state.get("sorting_preference")
        if sorting_pref and sorting_pref["target"] == activity_type:
            logger.info(f"Применяю сортировку для '{activity_type}': {sorting_pref}")

            # Сопоставляем семантический атрибут с реальным полем модели
            attr_map = {
                "price": "min_price",
                "rating": "rating",
                "start_time": "start_time_naive_event_tz",
            }
            sort_attribute = attr_map.get(sorting_pref["attribute"])

            if sort_attribute:
                # Определяем направление сортировки
                reverse_order = True if sorting_pref["order"] == "MAX" else False

                # Сортируем кандидатов, обрабатывая None значения
                candidates = sorted(
                    candidates,
                    key=lambda x: getattr(
                        x, sort_attribute, 0 if reverse_order else float("inf")
                    )
                    or (0 if reverse_order else float("inf")),
                    reverse=reverse_order,
                )
                logger.debug(
                    f"Топ-3 кандидата после сортировки: {[getattr(c, 'name', 'N/A') for c in candidates[:3]]}"
                )

            # Очищаем инструкцию после использования
            self.state["sorting_preference"] = None

        # Старая логика гибкого времени (осталась без изменений)
        if (
            self.is_flexible_start_time
            and activity_type == "MOVIE"
            and not self.pinned_items
        ):
            return [
                c
                for c in candidates
                if isinstance(c, Event)
                and c.start_time_naive_event_tz.time() >= time(hour=DEFAULT_START_HOUR)
            ]

        return candidates

    async def _find_best_plan_for_activities(
        self, activities_to_plan: List[OrderedActivityItem]
    ) -> Tuple[Optional[Plan], List[Dict]]:
        """
        Ключевой метод, переписанный для работы с "закрепленными" элементами.
        """
        self.route_calculations_count = 0
        activity_slots = []

        # --- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Формирование слотов с учетом pinned_items ---
        for act in activities_to_plan:
            pinned_item = self.pinned_items.get(act.activity_type)
            if pinned_item:
                logger.info(
                    f"Использую закрепленный элемент для '{act.activity_type}': {getattr(pinned_item, 'name', 'N/A')}"
                )
                activity_slots.append([pinned_item])
            else:
                logger.info(
                    f"Ищу кандидатов для свободного слота: '{act.activity_type}'"
                )
                candidates = self._get_candidates_for_slot(act.activity_type)
                if not candidates:
                    return None, [
                        {
                            "reason": f"Не найдено подходящих вариантов для '{act.query_details}'.",
                            "type": act.activity_type,
                        }
                    ]

                # Сортируем кандидатов (особенно важно для незакрепленных слотов)
                candidates.sort(
                    key=lambda x: getattr(
                        x, "start_time_naive_event_tz", self.user_start_time
                    )
                )
                activity_slots.append(candidates[:5])  # Берем топ-5 для перебора

        if not activity_slots:
            return None, [
                {"reason": "Нет активностей для планирования.", "type": "General"}
            ]

        possible_plans: List[Plan] = []
        rejection_reasons = []

        # --- Основной цикл перебора (без изменений в логике, но работает с новыми слотами) ---
        for combination in itertools.product(*activity_slots):
            if self.route_calculations_count >= MAX_ROUTES_TO_CALCULATE:
                break

            current_plan_items_objects = []
            total_travel = 0
            first_item = combination[0]

            start_node_time = self.user_start_time
            if (
                self.is_flexible_start_time
                and isinstance(first_item, Event)
                and not self.pinned_items
            ):
                start_node_time = first_item.start_time_naive_event_tz

            last_item_state = {
                "end_time": start_node_time,
                "coords": self.user_start_coords,
            }

            is_possible = True
            for item_to_add in combination:
                check_result = await self._check_compatibility(
                    last_item_state, item_to_add
                )
                if check_result.get("compatible"):
                    current_plan_items_objects.append(check_result["item"])
                    total_travel += check_result["travel_to_seconds"]
                    last_item_state = check_result
                else:
                    is_possible = False
                    rejection_reasons.append(check_result)
                    break

            if is_possible:
                plan_items_as_dicts = [
                    item.model_dump() for item in current_plan_items_objects
                ]
                possible_plans.append(
                    Plan(items=plan_items_as_dicts, total_travel_seconds=total_travel)
                )

        if not possible_plans:
            return None, rejection_reasons

        best_plan = sorted(possible_plans, key=lambda p: p.total_travel_seconds)[0]
        return best_plan, rejection_reasons

    async def build(self) -> PlanBuilderResult:
        """
        Основной метод v3.1. Больше не обрабатывает команды, а напрямую
        использует актуальный список активностей из state['search_criteria'].
        """
        # --- ШАГ 1: ПОЛУЧЕНИЕ АКТУАЛЬНОГО СПИСКА ДЕЛ ---
        # Это ключевое изменение. Мы больше не смотрим на self.command.
        # Мы доверяем `search_criteria` в state, который уже был обновлен
        # узлами `delete_activity_node` или `add_activity_node`.

        if not self.criteria or not self.criteria.ordered_activities:
            logger.warning(
                "PlanBuilder: Нет активностей для планирования в search_criteria."
            )
            return PlanBuilderResult(failure_reason="Нет запланированных дел.")

        activities_to_plan = self.criteria.ordered_activities

        logger.info(
            f"PlanBuilder: Пытаюсь построить план из {len(activities_to_plan)} активностей. Закреплено: {list(self.pinned_items.keys())}"
        )

        # --- ШАГ 2: ПОИСК ПЛАНА (без изменений) ---
        full_plan, rejection_reasons = await self._find_best_plan_for_activities(
            activities_to_plan
        )

        if full_plan:
            logger.info("Полный план успешно построен.")
            full_plan.warnings.extend(self.plan_warnings)
            self._update_pinned_items(full_plan)
            return PlanBuilderResult(best_plan=full_plan)

        # --- ШАГ 3: ФОРМИРОВАНИЕ ОТВЕТА ОБ ОШИБКЕ (без изменений) ---
        logger.warning("Не удалось построить полный план.")
        if rejection_reasons:
            most_common_reason = Counter(
                r["reason"] for r in rejection_reasons if r.get("reason")
            ).most_common(1)
            if most_common_reason:
                summary = f"Не удалось составить план. Основная проблема: {most_common_reason[0][0]}"
            else:
                summary = "Не удалось составить маршрут из найденных вариантов, они несовместимы по времени или расстоянию."
        else:
            summary = "Не удалось составить маршрут. Возможно, для одного из ваших запросов не нашлось подходящих вариантов."

        return PlanBuilderResult(failure_reason=summary)

    def _update_pinned_items(self, plan: Plan):
        """
        Сохраняет элементы успешного плана в state для будущих модификаций.
        """
        new_pinned_items = {}
        # Используем Pydantic для безопасного парсинга словарей из плана
        all_item_types = (Event, ParkInfo, FoodPlaceInfo)

        for item_dict in plan.items:
            parsed_item = None
            for item_class in all_item_types:
                try:
                    parsed_item = item_class.model_validate(item_dict)
                    break
                except Exception:
                    continue

            if parsed_item:
                activity_type_key = ""
                if isinstance(parsed_item, Event):
                    activity_type_key = parsed_item.user_event_type_key
                elif isinstance(parsed_item, ParkInfo):
                    activity_type_key = "PARK"
                elif isinstance(parsed_item, FoodPlaceInfo):
                    activity_type_key = "RESTAURANT"

                if activity_type_key:
                    new_pinned_items[activity_type_key] = parsed_item

        self.state["pinned_items"] = new_pinned_items
        logger.info(
            f"Элементы плана закреплены (pinned): {list(new_pinned_items.keys())}"
        )


# --- КОНЕЦ БЛОКА ДЛЯ ЗАМЕНЫ ---
