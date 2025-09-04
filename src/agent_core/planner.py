# --- НАЧАЛО БЛОКА ДЛЯ ЗАМЕНЫ: planner.py ---

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
    RouteSegment,  # <-- ВАЖНО: Убедитесь, что этот импорт будет работать
)
from src.services.gis_service import get_route

logger = logging.getLogger(__name__)

DEFAULT_POI_DURATION_MINUTES = 30
MAX_ROUTES_TO_CALCULATE = 25
DEFAULT_START_HOUR = 9


class PlanBuilder:
    """
    Умный построитель планов v3.2.
    Сохраняет информацию о маршрутах между мероприятиями.
    """

    def __init__(self, state: AgentState):
        self.state = state
        self.criteria = state.get("search_criteria")
        self.plan_warnings = state.get("plan_warnings", [])
        self.pinned_items: Dict[str, PlanItem] = state.get("pinned_items", {})
        self.is_flexible_start_time = (
            self.criteria.raw_time_description is None if self.criteria else False
        )
        logger.info(
            f"PlanBuilder initialized. Flexible start time: {self.is_flexible_start_time}. Pinned items: {list(self.pinned_items.keys())}"
        )
        # Убедимся, что даты есть в состоянии
        if not state.get("parsed_dates_iso"):
            # Этого не должно происходить, если граф работает правильно, но для защиты...
            logger.error("Критическая ошибка: PlanBuilder вызван без parsed_dates_iso.")
            # Можно либо выбросить исключение, либо установить дату по умолчанию
            # Для примера установим текущую дату
            state["parsed_dates_iso"] = [datetime.now().isoformat()]

        current_date_key = datetime.fromisoformat(
            state["parsed_dates_iso"][0]
        ).strftime("%Y-%m-%d")
        all_cached_candidates = state.get("cached_candidates", {})
        self.daily_cache: Dict[str, List[PlanItem]] = all_cached_candidates.get(
            current_date_key, {}
        )
        self.user_start_time = datetime.fromisoformat(state["parsed_dates_iso"][0])
        if self.is_flexible_start_time and not self.pinned_items:
            self.user_start_time = self.user_start_time.replace(
                hour=DEFAULT_START_HOUR, minute=0, second=0, microsecond=0
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
        if isinstance(item, Event) and item.place_coords_lon and item.place_coords_lat:
            return {"lon": item.place_coords_lon, "lat": item.place_coords_lat}
        if isinstance(item, (ParkInfo, FoodPlaceInfo)) and item.coords:
            return {"lon": item.coords[0], "lat": item.coords[1]}
        return None

    async def _check_compatibility(
        self, last_item_state: Dict[str, Any], item_to_add: PlanItem
    ) -> Dict[str, Any]:
        # (Этот метод остается без изменений, он уже корректен)
        if self.route_calculations_count >= MAX_ROUTES_TO_CALCULATE:
            return {"compatible": False, "reason": "Превышен лимит расчетов маршрутов."}
        last_item_end_time = last_item_state["end_time"]
        last_item_coords = last_item_state["coords"]
        item_to_add_coords = self._get_item_coords(item_to_add)
        item_name = getattr(item_to_add, "name", "N/A")
        travel_seconds, travel_info = 0, None
        if last_item_coords and item_to_add_coords:
            self.route_calculations_count += 1
            route_info = await get_route(points=[last_item_coords, item_to_add_coords])
            if route_info.get("status") != "success":
                return {
                    "compatible": False,
                    "reason": f"Не удалось построить маршрут до '{item_name}'.",
                }
            travel_seconds = route_info.get("duration_seconds", 0)
            travel_info = RouteSegment(
                from_name=last_item_state.get("name", "Точка старта"),
                to_name=item_name,
                duration_seconds=travel_seconds,
                distance_meters=route_info.get("distance_meters", 0),
                from_coords=last_item_coords,
                to_coords=item_to_add_coords,
            )
        arrival_time = last_item_end_time + timedelta(seconds=travel_seconds)
        if isinstance(item_to_add, Event):
            if arrival_time > item_to_add.start_time_naive_event_tz:
                return {
                    "compatible": False,
                    "reason": f"Вы не успеваете на '{item_name}'.",
                }
            start_time = item_to_add.start_time_naive_event_tz
            end_time = start_time + timedelta(
                minutes=item_to_add.duration_minutes or 120
            )
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
                return {
                    "compatible": False,
                    "reason": f"'{item_name}' будет закрыт. Причина: {reason_closed}.",
                }
            end_time = possible_end_time
        if end_time > self.user_end_time:
            return {
                "compatible": False,
                "reason": f"Посещение '{item_name}' не укладывается в ваше время.",
            }
        return {
            "compatible": True,
            "item": item_to_add,
            "start_time": start_time,
            "end_time": end_time,
            "coords": item_to_add_coords,
            "name": item_name,
            "travel_info": travel_info,
        }

    def _get_candidates_for_slot(self, activity_type: str) -> List[PlanItem]:
        # (Этот метод остается без изменений, он уже корректен)
        candidates = self.daily_cache.get(activity_type, [])
        if not candidates:
            return []
        sorting_pref = self.state.get("sorting_preference")
        if sorting_pref and sorting_pref["target"] == activity_type:
            attr_map = {
                "price": "min_price",
                "rating": "rating",
                "start_time": "start_time_naive_event_tz",
            }
            sort_attribute = attr_map.get(sorting_pref["attribute"])
            if sort_attribute:
                reverse_order = True if sorting_pref["order"] == "MAX" else False
                candidates = sorted(
                    candidates,
                    key=lambda x: getattr(
                        x, sort_attribute, 0 if reverse_order else float("inf")
                    )
                    or (0 if reverse_order else float("inf")),
                    reverse=reverse_order,
                )
            self.state["sorting_preference"] = None
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
        Полностью переписанная функция для надежного построения плана.
        Гарантирует корректную передачу состояния (время, координаты) между шагами.
        """
        self.route_calculations_count = 0
        activity_slots = []
        for act in activities_to_plan:
            pinned_item = self.pinned_items.get(act.activity_type)
            if pinned_item:
                activity_slots.append([pinned_item])
            else:
                candidates = self._get_candidates_for_slot(act.activity_type)
                if not candidates:
                    return None, [
                        {
                            "reason": f"Не найдено вариантов для '{act.query_details}'.",
                            "type": act.activity_type,
                        }
                    ]
                # Берем срез кандидатов для ускорения перебора
                activity_slots.append(candidates[:10])

        if not activity_slots:
            return None, [
                {"reason": "Нет активностей для планирования.", "type": "General"}
            ]

        possible_plans: List[Plan] = []
        rejection_reasons = []

        # 1. Перебираем все комбинации кандидатов
        for combination in itertools.product(*activity_slots):
            if self.route_calculations_count >= MAX_ROUTES_TO_CALCULATE:
                break

            current_plan_items = []
            is_combination_possible = True

            # 2. Устанавливаем НАЧАЛЬНОЕ состояние для КАЖДОЙ новой комбинации
            # Это состояние "до" первого мероприятия.
            # Для предварительного плана user_start_coords будет None.
            last_item_state = {
                "end_time": self.user_start_time,
                "coords": self.user_start_coords,
                "name": "Точка старта",
            }

            # 3. Последовательно обрабатываем каждый элемент в комбинации
            for item_to_add in combination:

                # 4. Проверяем совместимость, передавая состояние от ПРЕДЫДУЩЕГО шага
                check_result = await self._check_compatibility(
                    last_item_state, item_to_add
                )

                if check_result["compatible"]:
                    # Если элемент подходит, создаем для него словарь
                    activity_dict = check_result["item"].model_dump()

                    # Явно добавляем информацию о маршруте, если она была рассчитана
                    if travel_info := check_result.get("travel_info"):
                        activity_dict["travel_info_to_here"] = travel_info.model_dump()

                    current_plan_items.append(activity_dict)

                    # 5. ОБНОВЛЯЕМ состояние для СЛЕДУЮЩЕГО шага
                    last_item_state = {
                        "end_time": check_result["end_time"],
                        "coords": check_result["coords"],
                        "name": check_result["name"],
                    }
                else:
                    # Если хоть один элемент не подошел, вся комбинация невозможна
                    is_combination_possible = False
                    rejection_reasons.append(check_result)
                    break  # Прерываем цикл по элементам, переходим к след. комбинации

            # 6. Если вся комбинация успешно обработана, создаем и сохраняем план
            if is_combination_possible:
                total_travel = sum(
                    item.get("travel_info_to_here", {}).get("duration_seconds", 0)
                    for item in current_plan_items
                )
                possible_plans.append(
                    Plan(items=current_plan_items, total_travel_seconds=total_travel)
                )

        if not possible_plans:
            logger.warning("Не найдено ни одной совместимой комбинации.")
            return None, rejection_reasons

        # 7. КРИТИЧЕСКИЙ ШАГ: Выбор лучшего плана
        # Сначала отделяем планы, где есть реальный маршрут
        plans_with_travel = [p for p in possible_plans if p.total_travel_seconds > 0]

        best_plan = None
        if plans_with_travel:
            # Если есть планы с маршрутом, выбираем из них лучший (с минимальным временем в пути)
            best_plan = min(plans_with_travel, key=lambda p: p.total_travel_seconds)
            logger.info(
                f"Найдено {len(plans_with_travel)} планов с маршрутом. Выбран лучший."
            )
        elif possible_plans:
            # Если планов с маршрутом нет, но есть какие-то другие (например, для одного события),
            # берем лучший из них. Это fallback-сценарий.
            best_plan = min(possible_plans, key=lambda p: p.total_travel_seconds)
            logger.warning(
                "Не найдено планов с маршрутом. Выбран план с нулевым временем в пути."
            )

        if not best_plan:
            return None, rejection_reasons

        # Отладочный лог для 3 лучших планов (из всех возможных)
        logger.info(
            f"--- Отладка PlanBuilder: Найдено {len(possible_plans)} возможных планов. ---"
        )
        sorted_for_debug = sorted(possible_plans, key=lambda p: p.total_travel_seconds)
        for idx, p in enumerate(sorted_for_debug[:3]):
            plan_names = [item.get("name", "N/A") for item in p.items]
            # Используем -1 как маркер отсутствия маршрута
            travel_times = [
                item.get("travel_info_to_here", {}).get("duration_seconds", -1)
                for item in p.items
            ]
            logger.info(
                f"  Топ #{idx+1}: { ' -> '.join(plan_names) }, Общее время в пути: {p.total_travel_seconds} сек., Сегменты: {travel_times}"
            )
        logger.info("---------------------------------------------------------")

        logger.info(
            f"Выбран лучший план с временем в пути {best_plan.total_travel_seconds} сек."
        )
        return best_plan, rejection_reasons

    async def build(self) -> PlanBuilderResult:
        if not self.criteria or not self.criteria.ordered_activities:
            return PlanBuilderResult(failure_reason="Нет запланированных дел.")
        activities_to_plan = self.criteria.ordered_activities
        logger.info(
            f"PlanBuilder: Пытаюсь построить план из {len(activities_to_plan)} активностей. Закреплено: {list(self.pinned_items.keys())}"
        )

        full_plan, rejection_reasons = await self._find_best_plan_for_activities(
            activities_to_plan
        )

        if full_plan:
            logger.info("Полный план успешно построен.")
            full_plan.warnings.extend(self.plan_warnings)
            self._update_pinned_items(full_plan)
            return PlanBuilderResult(best_plan=full_plan)

        logger.warning("Не удалось построить полный план.")
        if rejection_reasons:
            # Улучшенное логирование ошибок
            reason_counts = Counter(
                r.get("reason", "Неизвестная ошибка") for r in rejection_reasons
            )
            logger.info(f"Причины отказа комбинаций: {reason_counts.most_common(3)}")
            most_common_reason = reason_counts.most_common(1)[0][0]
            summary = (
                f"Не удалось составить план. Основная проблема: {most_common_reason}"
            )
        else:
            summary = "Не удалось составить маршрут. Возможно, для одного из ваших запросов не нашлось подходящих вариантов."
        return PlanBuilderResult(failure_reason=summary)

    def _update_pinned_items(self, plan: Plan):
        """
        Сохраняет элементы успешного плана в state для будущих модификаций.
        Версия 2.1 с корректным определением типа RESTAURANT.
        """
        new_pinned_items = {}
        # Используем Pydantic для безопасного парсинга словарей из плана
        all_item_types = (Event, ParkInfo, FoodPlaceInfo)

        for item_dict in plan.items:
            # Сначала убираем наше служебное поле, чтобы не мешать валидации
            item_data_for_validation = item_dict.copy()
            if "travel_info_to_here" in item_data_for_validation:
                del item_data_for_validation["travel_info_to_here"]

            parsed_item = None
            for item_class in all_item_types:
                try:
                    # Валидируем очищенные данные
                    parsed_item = item_class.model_validate(item_data_for_validation)
                    break
                except Exception:
                    continue

            if parsed_item:
                activity_type_key = ""
                # --- КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ ЗДЕСЬ ---
                # Определяем тип активности для ключа словаря pinned_items
                if isinstance(parsed_item, Event):
                    # Для событий из Афиши ключ уже задан при поиске
                    activity_type_key = parsed_item.user_event_type_key
                elif isinstance(parsed_item, ParkInfo):
                    # Для парков ключ всегда "PARK"
                    activity_type_key = "PARK"
                elif isinstance(parsed_item, FoodPlaceInfo):
                    # Для еды ключ всегда "RESTAURANT"
                    activity_type_key = "RESTAURANT"
                # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

                if activity_type_key:
                    new_pinned_items[activity_type_key] = parsed_item

        self.state["pinned_items"] = new_pinned_items
        logger.info(
            f"Элементы плана закреплены (pinned): {list(new_pinned_items.keys())}"
        )


# --- КОНЕЦ БЛОКА ДЛЯ ЗАМЕНЫ ---
