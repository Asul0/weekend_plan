# Файл: src/schemas/data_schemas.py (ФИНАЛЬНАЯ ПОЛНАЯ ВЕРСИЯ)
from typing import Optional, List, Dict, Any, Union, Literal
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from enum import Enum


class PossibleActions(str, Enum):
    """Перечисление всех возможных действий, которые может принять роутер."""

    EXTRACT_CRITERIA = "EXTRACT_CRITERIA"
    SEARCH_EVENTS = "SEARCH_EVENTS"
    BUILD_PLAN = "BUILD_PLAN"
    ANALYZE_FEEDBACK = "ANALYZE_FEEDBACK"
    PRESENT_RESULTS = "PRESENT_RESULTS"
    CLARIFY_OR_CHITCHAT = "CLARIFY_OR_CHITCHAT"
    REFINE_PLAN = "REFINE_PLAN"
    DELETE_ACTIVITY = "DELETE_ACTIVITY"
    ADD_ACTIVITY = "ADD_ACTIVITY"
    PROCESS_START_ADDRESS = "PROCESS_START_ADDRESS"


class RouteSegment(BaseModel):
    """Представляет один отрезок маршрута от одной точки до другой."""
    model_config = ConfigDict(extra="ignore")

    from_name: str = Field(description="Название начальной точки.")
    to_name: str = Field(description="Название конечной точки.")
    duration_seconds: int = Field(description="Длительность поездки в секундах.")
    distance_meters: float = Field(description="Расстояние в метрах.")
    from_coords: Optional[Dict[str, float]] = Field(None, description="Координаты начальной точки.")
    to_coords: Optional[Dict[str, float]] = Field(None, description="Координаты конечной точки.")


# Файл: src/schemas/data_schemas.py
# ДОБАВЬТЕ ЭТИ ДВА КЛАССА В КОНЕЦ ФАЙЛА


class SimplifiedExtractedInfo(BaseModel):
    """
    Упрощенная схема для LLM. Извлекает активности как простой список строк,
    чтобы повысить надежность распознавания.
    """

    city: Optional[str] = Field(None, description="Название города.")
    dates_description: Optional[str] = Field(
        None, description="Описание дат или периода, как его дал пользователь."
    )
    # КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Просим извлечь просто список строк
    activities_list: Optional[List[str]] = Field(
        None,
        description="Список всех упомянутых активностей, например: ['фильм', 'покушать', 'погулять в парке'].",
    )
    budget: Optional[int] = Field(None, description="Общий бюджет пользователя.")
    person_count: Optional[int] = Field(1, description="Количество человек.")
    raw_time_description: Optional[str] = Field(
        None, description="Необработанное описание времени (e.g., 'после обеда')."
    )


class ActivityClassifier(BaseModel):
    """Классифицирует одну активность в системный тип."""

    activity_type: Literal[
        "MOVIE",
        "PARK",
        "RESTAURANT",
        "CONCERT",
        "STAND_UP",
        "PERFORMANCE",
        "MUSEUM_EXHIBITION",
        "UNKNOWN",
    ] = Field(description="Системный тип активности.")
    reasoning: str = Field(description="Краткое обоснование выбора типа.")


class NewSearchCriteria(BaseModel):
    """Конкретная структура для запроса на 'structural' изменение."""

    city: Optional[str] = None
    dates_description: Optional[str] = None
    raw_time_description: Optional[str] = None
    budget: Optional[int] = None


class OrderedActivityItem(BaseModel):
    """Представление одной активности в запрошенной пользователем последовательности."""

    activity_type: str = Field(description="Тип активности (MOVIE, PARK, etc.).")
    query_details: Optional[str] = Field(
        None, description="Оригинальная формулировка пользователя для этой активности."
    )
    activity_budget: Optional[int] = Field(
        None, description="Бюджет, специфичный для данной активности."
    )


class ExtractedInitialInfo(BaseModel):
    """Извлечённая из пользовательского запроса информация для подбора мероприятий."""

    city: Optional[str] = Field(None, description="Название города.")
    dates_description: Optional[str] = Field(
        None, description="Описание дат или периода, как его дал пользователь."
    )
    ordered_activities: Optional[List[OrderedActivityItem]] = Field(
        None, description="Упорядоченный список запрошенных активностей."
    )
    budget: Optional[int] = Field(None, description="Общий бюджет пользователя.")
    person_count: Optional[int] = Field(1, description="Количество человек.")
    raw_time_description: Optional[str] = Field(
        None, description="Необработанное описание времени (e.g., 'после обеда')."
    )


class Event(BaseModel):
    """Детальная информация о конкретном мероприятии (из API Афиши)."""

    session_id: int = Field(description="Уникальный ID сеанса мероприятия.")
    name: str = Field(description="Полное название мероприятия.")
    user_event_type_key: str = Field(
        description="Ключ типа события, использовавшийся для поиска."
    )
    place_name: str = Field(description="Название места проведения.")
    start_time_iso: str = Field(
        description="Время начала в формате ISO с часовым поясом."
    )
    start_time_naive_event_tz: datetime = Field(
        description="Наивное время начала в локальном часовом поясе события."
    )
    place_address: Optional[str] = Field(
        None, description="Полный адрес места проведения."
    )
    place_coords_lon: Optional[float] = Field(
        None, description="Долгота места проведения."
    )
    place_coords_lat: Optional[float] = Field(
        None, description="Широта места проведения."
    )
    duration_minutes: Optional[int] = Field(
        None, description="Продолжительность в минутах."
    )
    min_price: Optional[int] = Field(
        None, description="Минимальная цена билета в рублях."
    )
    max_price: Optional[int] = Field(
        None, description="Максимальная цена билета в рублях."
    )
    price_text: Optional[str] = Field(
        None, description="Текстовое представление диапазона цен."
    )
    rating: Optional[float] = Field(None, description="Рейтинг события (число).")
    age_restriction: Optional[str] = Field(
        None, description="Возрастное ограничение (e.g., '18+')."
    )

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)


class ParkInfo(BaseModel):
    """Детальная информация о парке или месте для прогулок (из GIS)."""

    id_gis: str = Field(description="Уникальный ID парка в GIS-сервисе.")
    name: str = Field(description="Название парка.")
    address: Optional[str] = Field(None, description="Адрес парка.")
    coords: Optional[List[float]] = Field(
        None, description="Координаты парка [долгота, широта]."
    )
    schedule_str: str = Field(
        "Время работы не указано",
        description="Строковое представление расписания работы.",
    )
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)


class FoodPlaceInfo(BaseModel):
    """Детальная информация о заведении питания (из GIS)."""

    id_gis: str = Field(description="Уникальный ID заведения в GIS-сервисе.")
    name: str = Field(description="Название заведения.")
    address: Optional[str] = Field(None, description="Адрес заведения.")
    coords: Optional[List[float]] = Field(
        None, description="Координаты заведения [долгота, широта]."
    )
    schedule_str: str = Field(
        "Время работы не указано",
        description="Строковое представление расписания работы.",
    )
    rating_str: str = Field(
        "Рейтинг не указан", description="Строковое представление рейтинга."
    )
    avg_bill_str: str = Field(
        "Средний чек не указан", description="Строковое представление среднего чека."
    )
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)


class TravelInfo(BaseModel):
    """Информация о перемещении между точками."""

    duration_seconds: int = Field(description="Время в пути в секундах.")


class PlanItem(BaseModel):
    """Один шаг в плане, включающий само мероприятие и дорогу до него."""

    activity: Union[Event, ParkInfo, FoodPlaceInfo] = Field(
        description="Сам объект мероприятия."
    )
    travel_from_previous: Optional[TravelInfo] = Field(
        None, description="Информация о дороге до этого пункта от предыдущего."
    )
    calculated_start_time: datetime = Field(
        description="Расчетное время начала активности."
    )
    calculated_end_time: datetime = Field(
        description="Расчетное время окончания активности."
    )


class Plan(BaseModel):
    """Представляет собой готовый, упорядоченный и проверенный план мероприятий."""

    items: List[Dict[str, Any]] = Field(
        description="Упорядоченный список шагов в плане. Каждый шаг - это словарь с 'activity' и опциональным 'travel_info'."
    )
    total_travel_seconds: int = Field(
        description="Общее время в пути между точками в секундах."
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Предупреждения, возникшие при построении плана.",
    )


class PlanBuilderResult(BaseModel):
    """Результат работы PlanBuilder. Содержит либо успешный план, либо причину неудачи."""

    best_plan: Optional[Plan] = Field(
        None, description="Самый оптимальный построенный план."
    )
    failure_reason: Optional[str] = Field(
        None, description="Причина, по которой не удалось построить план."
    )


class Constraint(BaseModel):
    """
    Атомарное ограничение для поиска или модификации.
    Поле value сделано строкой для совместимости с GigaChat.
    """

    attribute: Literal["price", "start_time", "rating", "name"] = Field(
        description="Атрибут для фильтрации/сортировки (e.g., 'price', 'start_time', 'rating', 'name')."
    )
    operator: Literal[
        "MIN",
        "MAX",
        "GREATER_THAN",
        "LESS_THAN",
        "NOT_EQUALS",
    ] = Field(description="Оператор сравнения или сортировки.")

    # КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ:
    # Делаем value строкой, чтобы избежать ошибки "Union type not supported".
    # Наша логика на стороне Python будет сама преобразовывать типы.
    value: Optional[str] = Field(
        None,
        description="Значение для сравнения в виде СТРОКИ (например, '500', '2025-07-31T10:15:00' или 'Название фильма').",
    )


class SimplifiedPlanItemForPrompt(BaseModel):
    """Ультра-компактное представление элемента плана для передачи в LLM."""

    type: str = Field(description="Тип активности (MOVIE, PARK, RESTAURANT)")
    name: str = Field(description="Название")
    # Используем Union для времени и цены, т.к. они могут отсутствовать
    start_time: Optional[Union[datetime, str]] = Field(
        None, description="Время начала (ISO формат, если есть)"
    )
    price: Optional[int] = Field(None, description="Минимальная цена, если есть")
    rating: Optional[float] = Field(None, description="Рейтинг, если есть")


class FlatCommand(BaseModel):
    """
    Максимально плоская и простая схема ОДНОЙ команды для LLM.
    Это "таблица", которую легко заполнить.
    """

    command: Literal[
        "modify", "add", "delete", "reorder", "update_criteria", "chitchat"
    ]
    target_activity: Optional[str] = Field(
        None, description="Тип цели (e.g., 'MOVIE', 'PARK')"
    )
    constraint_attribute: Optional[str] = Field(
        None,
        description="Атрибут для modify (e.g., 'start_time_naive_event_tz', 'min_price')",
    )
    constraint_operator: Optional[str] = Field(
        None, description="Оператор для modify (e.g., 'GREATER_THAN', 'NOT_EQUALS')"
    )
    #
    # --- ВОТ ЭТА СТРОКА ---
    constraint_value: Optional[str] = Field(
        None, description="Значение для сравнения в modify (ВСЕГДА КАК СТРОКА)."
    )
    # --- КОНЕЦ ИЗМЕНЕНИЯ. БЫЛО Optional[Any], СТАЛО Optional[str] ---
    #
    new_activity_details: Optional[str] = Field(
        None, description="Детали для команды 'add' (e.g., 'недорогой боулинг')"
    )


# --- ДОБАВЛЕНИЕ В КОНЕЦ ФАЙЛА: Схемы для гибридного анализа фидбека ---


class SemanticConstraint(BaseModel):
    """
    Семантическое ядро, извлеченное LLM из фразы пользователя.
    Это НЕ готовая команда, а сырье для дальнейшей обработки в Command Processor.
    """

    command_type: Literal[
        "modify",
        "delete",
        "reorder",
        "add",
        "update_criteria",
        "chitchat",
        "query_info",  # Важное добавление для вопросов типа "какой рейтинг?"
    ] = Field(description="Общий тип намерения, извлеченный из запроса.")

    target: Optional[str] = Field(
        None,
        description="Целевая активность (e.g., 'MOVIE', 'PARK') или атрибут ('budget').",
    )
    # Поля ниже заполняются преимущественно для команды 'modify'
    attribute: Optional[
        Literal["price", "start_time", "rating", "name", "date", "city", "interests"]
    ] = Field(
        None, description="Конкретный атрибут, который пользователь хочет изменить."
    )

    operator: Optional[
        Literal["GREATER_THAN", "LESS_THAN", "NOT_EQUALS", "MIN", "MAX", "EQUALS"]
    ] = Field(None, description="Направление изменения или сравнения.")

    # Разделение значения на компоненты - ключевое улучшение
    value_str: Optional[str] = Field(
        None,
        description="Извлеченное текстовое значение (e.g., 'завтра', 'Казань', 'комедия').",
    )
    value_num: Optional[float] = Field(
        None, description="Извлеченное числовое значение (e.g., 2, 2000)."
    )
    value_unit: Optional[str] = Field(
        None,
        description="Единица измерения для числового значения (e.g., 'часа', 'рублей').",
    )


class LlmExtractionResult(BaseModel):
    """
    Корневая модель для вывода LLM. Содержит список семантических намерений,
    извлеченных из ОДНОГО сообщения пользователя.
    """

    semantic_intents: List[SemanticConstraint]


class FlatFeedback(BaseModel):
    """
    Корневая модель для вывода LLM, содержит список плоских команд.
    """

    commands: List[FlatCommand]


class Position(BaseModel):
    """Описывает, куда вставить новую активность."""

    after: Optional[str] = Field(
        None,
        description="Тип активности, ПОСЛЕ которой нужно вставить новую (e.g., 'MOVIE').",
    )
    before: Optional[str] = Field(
        None, description="Тип активности, ПЕРЕД которой нужно вставить новую."
    )


class SimplifiedExtractedInfo(BaseModel):
    """
    Упрощенная схема для LLM. Извлекает активности как ОДНУ СТРОКУ,
    чтобы повысить надежность распознавания.
    """

    city: Optional[str] = Field(None, description="Название города.")
    dates_description: Optional[str] = Field(
        None, description="Описание дат или периода, как его дал пользователь."
    )
    # КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Просим извлечь активности как одну строку через запятую.
    activities_str: Optional[str] = Field(
        None,
        description="Перечисление ВСЕХ упомянутых активностей через запятую, например: 'кино, ресторан, погулять в парке'.",
    )
    budget: Optional[int] = Field(None, description="Общий бюджет пользователя.")
    person_count: Optional[int] = Field(1, description="Количество человек.")
    raw_time_description: Optional[str] = Field(
        None, description="Необработанное описание времени (e.g., 'вечером')."
    )


class ChangeRequest(BaseModel):
    """Структурированное описание ОДНОЙ атомарной команды от пользователя."""

    command: Literal[
        "modify",
        "add",
        "delete",
        "reorder",
        "update_criteria",
        "confirmation",
        "chitchat",
    ] = Field(description="Тип команды.")

    target: Optional[str] = Field(
        None,
        description="Тип активности, к которому применяется команда (e.g., 'MOVIE', 'PARK').",
    )
    # ИЗМЕНЕНИЕ: Теперь `constraints` - это список наших новых, мощных ограничений
    constraints: Optional[List[Constraint]] = Field(
        None, description="Список ограничений для команды 'modify'."
    )
    new_activity: Optional[OrderedActivityItem] = Field(
        None, description="Новая активность для команды 'add'."
    )
    position: Optional[Position] = Field(
        None, description="Позиция для добавления новой активности."
    )
    new_order: Optional[List[str]] = Field(
        None, description="Новый порядок для команды 'reorder'."
    )
    updates: Optional[NewSearchCriteria] = Field(
        None, description="Изменения для команды 'update_criteria'."
    )


class AnalyzedFeedback(BaseModel):
    """Результат работы узла analyze_feedback_node, содержащий список команд."""

    user_intent_list: List[ChangeRequest] = Field(
        description="Упорядоченный список проанализированных намерений пользователя."
    )
    response_to_user: Optional[str] = Field(
        None,
        description="Промежуточный ответ пользователю, если требуется (обычно для chitchat).",
    )


# --- ВОССТАНОВЛЕННЫЕ СХЕМЫ ДЛЯ ИНСТРУМЕНТОВ ---


class DateTimeParserToolArgs(BaseModel):
    """Аргументы для инструмента парсинга описания даты и времени."""

    natural_language_date: str = Field(
        description="Описание даты на естественном языке."
    )
    natural_language_time_qualifier: Optional[str] = Field(
        None, description="Дополнительное описание времени (e.g., 'вечером')."
    )
    base_date_iso: Optional[str] = Field(
        None, description="ISO строка базовой даты для относительных вычислений."
    )


class EventSearchToolArgs(BaseModel):
    """Аргументы для инструмента поиска мероприятий (Афиша)."""

    city_id: int = Field(description="Числовой ID города для поиска.")
    city_name: str = Field(description="Название города для дополнительной фильтрации.")
    date_from: datetime = Field(description="Дата и время начала периода поиска.")
    date_to: datetime = Field(description="Дата и время окончания периода поиска.")
    user_creation_type_key: str = Field(
        description="Строковый ключ типа события (MOVIE, CONCERT, etc.)."
    )
    max_budget_per_person: Optional[int] = Field(
        None, description="Максимальный бюджет на одного человека."
    )


class UserIntent(str, Enum):
    """Категории намерений пользователя."""

    PLAN_REQUEST = "PLAN_REQUEST"  # Запрос на создание или изменение плана
    FEEDBACK_ON_PLAN = "FEEDBACK_ON_PLAN"  # Обратная связь по предложенному плану
    CHITCHAT = "CHITCHAT"  # Общий разговор, вопрос о возможностях и т.д.


class ClassifiedIntent(BaseModel):
    """Результат работы узла классификации намерения."""

    intent: UserIntent = Field(description="Классифицированное намерение пользователя.")
    reasoning: str = Field(
        description="Краткое обоснование, почему выбрана эта категория."
    )
