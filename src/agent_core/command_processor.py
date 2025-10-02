import logging
import re
from typing import List, Optional, Union, Dict, Any
from datetime import datetime, timedelta

from src.agent_core.state import AgentState, PlanItem
from src.schemas.data_schemas import (
    SemanticConstraint,
    ChangeRequest,
    Constraint,
    NewSearchCriteria,
    OrderedActivityItem,
    Position,
    Event,
    ParkInfo,
    FoodPlaceInfo,
)
import pymorphy3

logger = logging.getLogger(__name__)
morph = pymorphy3.MorphAnalyzer()


class CommandProcessor:
    """
    Преобразует семантическое ядро, извлеченное LLM, в список конкретных,
    валидных и исполняемых команд ChangeRequest для графа состояний.
    Это детерминированный, тестируемый и основной логический узел агента.
    """

    def __init__(self, state: AgentState, semantic_intents: List[SemanticConstraint]):
        self.state = state
        self.semantic_intents = semantic_intents
        self.executable_commands: List[ChangeRequest] = []
        self._structural_change_detected = False

    def process(self) -> List[ChangeRequest]:
        """
        Главный метод-оркестратор. Выполняет обработку в три этапа для
        обеспечения корректного порядка операций.
        """
        # 1. Глобальные изменения (дата, город), которые делают весь кэш невалидным.
        self._process_global_updates()
        if self._structural_change_detected:
            # Если изменилась дата/город, нет смысла обрабатывать другие команды,
            # так как весь контекст (кэш, план) стал невалидным.
            return self.executable_commands

        # 2. Структурные изменения плана (добавление/удаление).
        self._process_structural_changes()

        # 3. Модификации атрибутов существующих элементов.
        self._process_modifications()

        return self.executable_commands

    def _process_global_updates(self):
        """Ищет и обрабатывает команды, инвалидирующие весь кэш."""
        updates = NewSearchCriteria()
        changed_fields = []

        for intent in self.semantic_intents:
            if (
                intent.command_type == "update_criteria"
                and intent.attribute
                and intent.value_str
            ):
                if intent.attribute == "date":
                    updates.dates_description = intent.value_str
                    changed_fields.append("date")
                elif intent.attribute == "city":
                    parsed_city = morph.parse(intent.value_str)[0]
                    updates.city = parsed_city.normal_form.title()
                    changed_fields.append("city")

        if changed_fields:
            self._structural_change_detected = True
            logger.info(
                f"Обнаружено глобальное изменение: {updates.model_dump(exclude_none=True)}. Инвалидирую кэш и план."
            )
            command = ChangeRequest(command="update_criteria", updates=updates)
            self.executable_commands.append(command)

    def _process_structural_changes(self):
        """Обрабатывает добавление и удаление элементов из плана. Улучшенная версия."""
        for intent in self.semantic_intents:
            if intent.command_type == "delete" and intent.target:
                command = ChangeRequest(command="delete", target=intent.target)
                self.executable_commands.append(command)
                logger.debug(
                    f"Сформирована команда на удаление: {command.model_dump_json()}"
                )

            elif intent.command_type == "add" and intent.target:
                # --- УЛУЧШЕННАЯ ЛОГИКА 'add' ---
                # 1. Определяем, что искать. Если есть value_str, и это не просто "мусор", используем его.
                #    Иначе используем target как поисковый запрос.
                query_details = intent.value_str
                # Простой фильтр "мусорных" фраз от LLM
                if query_details in ["после фильма", "до парка", "в конце"]:
                    query_details = None

                if not query_details:
                    # Если пользователь сказал "добавь ресторан", target будет 'RESTAURANT'.
                    # Преобразуем это в поисковый запрос "ресторан".
                    query_details = intent.target.lower()

                # 2. Создаем new_activity
                new_activity = OrderedActivityItem(
                    activity_type=intent.target.upper(), query_details=query_details
                )

                # 3. TODO: Реализовать логику определения позиции (до/после)
                position = Position(after="ALL")

                command = ChangeRequest(
                    command="add", new_activity=new_activity, position=position
                )
                self.executable_commands.append(command)
                logger.debug(
                    f"Сформирована команда на добавление: {command.model_dump_json()}"
                )

    def _process_modifications(self):
        """Обрабатывает команды на изменение атрибутов существующих элементов."""
        for intent in self.semantic_intents:
            if intent.command_type == "modify":
                command = self._create_modify_command(intent)
                if command:
                    self.executable_commands.append(command)

    def _create_modify_command(
        self, intent: SemanticConstraint
    ) -> Optional[ChangeRequest]:
        """Фабрика для создания команды 'modify' на основе семантического интента."""
        if not intent.attribute or not intent.target or not intent.operator:
            logger.warning(
                f"Пропуск неполной команды 'modify': {intent.model_dump_json()}"
            )
            return None

        try:
            final_value = self._calculate_final_value(intent)
            # Для операторов MIN/MAX None - это нормальное значение
            if final_value is None and intent.operator not in ["MIN", "MAX"]:
                raise ValueError("Не удалось вычислить финальное значение.")
        except Exception as e:
            logger.error(
                f"Ошибка при вычислении значения для modify: {e} для интента {intent.model_dump_json()}",
                exc_info=True,
            )
            return None

        # Преобразуем значение в строку для Pydantic модели, но делаем это умно
        if final_value is None:
            value_for_model = None
        elif isinstance(final_value, datetime):
            value_for_model = final_value.isoformat()
        else:
            value_for_model = str(final_value)

        constraint = Constraint(
            attribute=intent.attribute,
            operator=intent.operator,
            value=value_for_model,
        )
        command = ChangeRequest(
            command="modify", target=intent.target, constraints=[constraint]
        )
        logger.debug(
            f"Сформирована команда на модификацию: {command.model_dump_json()}"
        )
        return command

    def _calculate_final_value(
        self, intent: SemanticConstraint
    ) -> Union[str, int, float, datetime, None]:
        """Вычисляет конкретное значение для ограничения, обрабатывая относительные величины."""
        if intent.attribute == "start_time":
            current_item = self._get_item_from_plan(intent.target)
            if not current_item or not hasattr(
                current_item, "start_time_naive_event_tz"
            ):
                raise ValueError(
                    f"Не найден элемент '{intent.target}' в плане для относительного изменения времени."
                )

            current_time = current_item.start_time_naive_event_tz
            delta = timedelta(hours=1)  # Дефолтный шаг для "попозже/пораньше"
            if intent.value_num:
                if "час" in (intent.value_unit or ""):
                    delta = timedelta(hours=intent.value_num)
                elif "мин" in (intent.value_unit or ""):
                    delta = timedelta(minutes=intent.value_num)

            return (
                current_time + delta
                if intent.operator == "GREATER_THAN"
                else current_time - delta
            )

        elif intent.attribute == "price":
            if intent.operator in ["MIN", "MAX"]:
                return 0

            if intent.value_num is not None:
                return intent.value_num

            current_item = self._get_item_from_plan(intent.target)

            # --- НАЧАЛО ИСПРАВЛЕНИЯ ---
            if isinstance(current_item, FoodPlaceInfo):
                if current_item.avg_bill_str:
                    # Извлекаем первое число из строки типа "1000–1500 ₽"
                    match = re.search(
                        r"\d+", current_item.avg_bill_str.replace(" ", "")
                    )
                    if match:
                        return float(match.group(0))

            if isinstance(current_item, Event) and current_item.min_price is not None:
                return current_item.min_price
            # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

            raise ValueError(
                f"Не найден элемент '{intent.target}' или его цена для относительного изменения."
            )

        # --- НАЧАЛО ВТОРОГО ИСПРАВЛЕНИЯ (для смены фильма) ---
        elif intent.attribute == "name":
            current_item = self._get_item_from_plan(intent.target)
            if not current_item or not hasattr(current_item, "name"):
                raise ValueError(
                    f"Не найден элемент '{intent.target}' или его имя в плане."
                )
            return current_item.name

        # Для операторов MIN/MAX не нужно конкретное значение
        if intent.operator in ["MIN", "MAX"]:
            return None

        # Для других атрибутов (имя, рейтинг) просто возвращаем извлеченное значение
        return intent.value_str or intent.value_num

    def _map_attribute_to_model(self, semantic_attribute: str) -> str:
        """Сопоставляет семантический атрибут с реальным именем поля в Pydantic моделях."""
        mapping = {
            "start_time": "start_time_naive_event_tz",
            "price": "min_price",
            "rating": "rating",
            "name": "name",
        }
        return mapping.get(semantic_attribute, semantic_attribute)

    def _get_item_from_plan(self, target_type: str) -> Optional[PlanItem]:
        """
        Надежно находит и парсит элемент в текущем плане по его типу.
        Версия 2.0: Устойчива к неполным данным в плане.
        """
        if not self.state.get("current_plan"):
            return None

        for item_dict in self.state["current_plan"].items:
            # --- НАЧАЛО ИСПРАВЛЕНИЯ ---
            # Определяем тип элемента по наличию уникальных ключей,
            # это быстрее и надежнее, чем обработка исключений Pydantic.

            item_type = None
            if "session_id" in item_dict and "user_event_type_key" in item_dict:
                item_type = item_dict["user_event_type_key"]
            elif "avg_bill_str" in item_dict:
                item_type = "RESTAURANT"
            elif "id_gis" in item_dict:  # Парк - самый общий случай с id_gis
                item_type = "PARK"

            # Если определенный тип совпадает с тем, что мы ищем...
            if item_type == target_type:
                try:
                    # ...только тогда мы пытаемся его распарсить в полную модель.
                    if item_type in [
                        "MOVIE",
                        "CONCERT",
                        "STAND_UP",
                        "PERFORMANCE",
                        "MUSEUM_EXHIBITION",
                    ]:
                        return Event.model_validate(item_dict)
                    elif item_type == "RESTAURANT":
                        return FoodPlaceInfo.model_validate(item_dict)
                    elif item_type == "PARK":
                        return ParkInfo.model_validate(item_dict)
                except Exception as e:
                    # Если даже тут произошла ошибка, логируем ее и продолжаем поиск.
                    logger.warning(
                        f"Не удалось распарсить элемент плана '{item_dict.get('name')}' при поиске: {e}"
                    )
                    continue
            # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

        return None  # Если ничего не найдено в цикле

    @staticmethod
    def _get_activity_type_from_item(item: PlanItem) -> str:
        """Возвращает системный тип (MOVIE, PARK) из объекта PlanItem."""
        if isinstance(item, Event):
            return item.user_event_type_key
        if isinstance(item, ParkInfo):
            return "PARK"
        if isinstance(item, FoodPlaceInfo):
            return "RESTAURANT"
        return "UNKNOWN"
