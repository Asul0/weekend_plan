# Файл: src/agent_core/state.py (ПОЛНАЯ ИСПРАВЛЕННАЯ ВЕРСИЯ)
from typing import TypedDict, Optional, List, Annotated, Union, Dict, Any
from operator import add
from langchain_core.messages import BaseMessage
from src.schemas.data_schemas import (
    ExtractedInitialInfo,
    Event,
    ParkInfo,
    FoodPlaceInfo,
    PlanBuilderResult,
    PossibleActions,
    ChangeRequest,
    Plan,
    AnalyzedFeedback,
    ClassifiedIntent,
)
from typing import TypedDict, Optional, Literal

PlanItem = Union[Event, ParkInfo, FoodPlaceInfo]
CachedCandidates = Dict[str, List[PlanItem]]  # Ключ - тип активности (MOVIE, PARK...)
DailyCache = Dict[str, CachedCandidates]


class SortingPreference(TypedDict):
    """Инструкция для PlanBuilder по сортировке кандидатов."""

    target_type: str
    attribute: str
    order: Literal["ascending", "descending"]


class AgentState(TypedDict):
    """Определяет "память" или состояние нашего агента."""

    user_message: str
    chat_history: Annotated[List[BaseMessage], add]
    classified_intent: Optional[ClassifiedIntent]
    next_action: Optional[PossibleActions]
    error: Optional[str]
    is_awaiting_clarification: Optional[str]
    plan_presented: bool
    plan_warnings: List[str]
    search_criteria: Optional[ExtractedInitialInfo]
    cached_candidates: DailyCache
    current_plan: Optional[Plan]
    plan_builder_result: Optional[PlanBuilderResult]
    analyzed_feedback: Optional[AnalyzedFeedback]
    pinned_items: Dict[str, PlanItem]
    command_queue: List[ChangeRequest]
    sorting_preference: Optional[SortingPreference]
    city_id_afisha: Optional[int]
    parsed_dates_iso: Optional[List[str]]
    parsed_end_dates_iso: Optional[List[str]]
    user_start_coordinates: Optional[dict]
    is_awaiting_address: bool
    status_message_id: Optional[Any]
    user_start_address: Optional[str]        
    is_awaiting_start_address: bool
    is_awaiting_criteria_clarification: bool # Флаг, что мы ждем уточнения по обязательным критериям
    missing_criteria_fields: List[str] # Список полей, которые мы ожидаем получить (например, ['city', 'dates_description'])
    last_clarification_question: Optional[str] # Последний вопрос, заданный пользователю для уточнения