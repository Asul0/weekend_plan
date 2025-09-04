# --- НАЧАЛО БЛОКА ДЛЯ ЗАМЕНЫ: graph.py ---
import logging
from langgraph.graph import StateGraph, END
from src.agent_core.state import AgentState
from src.agent_core.nodes import (
    router_node,
    extract_initial_criteria_node,
    prepare_and_search_events_node,
    build_initial_plan_node,
    analyze_feedback_node,
    presenter_node,
    classify_intent_node,
    chitchat_node,
    handle_clarification_node,
    refine_plan_node,
    delete_activity_node,
    add_activity_node,
    process_start_address_node,
)

# --- ИСПРАВЛЕНИЕ ИМПОРТОВ ---
# Мы импортируем оба Enum из одного места для консистентности
from src.schemas.data_schemas import PossibleActions, UserIntent

logger = logging.getLogger(__name__)


# --- НОВАЯ ФУНКЦИЯ-РАЗВИЛКА ---
def should_classify_or_process_address(state: AgentState) -> str:
    """
    Первая и главная развилка графа.
    Проверяет, ждет ли система ввода адреса. Если да, то пропускает
    классификацию и сразу отправляет на обработку адреса.
    """
    if state.get("is_awaiting_start_address"):
        logger.info(
            "Граф: Обнаружен флаг is_awaiting_start_address. Пропускаю классификацию, перехожу к обработке адреса."
        )
        return "PROCESS_START_ADDRESS"
    else:
        logger.info(
            "Граф: Флаг ожидания адреса не установлен. Перехожу к стандартной классификации намерения."
        )
        return "classify_intent"


def decide_after_classification(state: AgentState) -> str:
    """Ключевая развилка ПОСЛЕ определения намерения."""
    # Эта функция остается без изменений
    logger.info(
        f"Граф: Принятие решения после классификации. Намерение: {state.get('classified_intent').intent}"
    )
    if state.get("classified_intent").intent == UserIntent.CHITCHAT:
        return "CLARIFY_OR_CHITCHAT"
    return "router"


def build_agent_graph():
    """Собирает граф агента. v7.2 с корректной обработкой ввода адреса."""
    workflow = StateGraph(AgentState)

    # 1. Добавляем узлы (без изменений)
    nodes = {
        "classify_intent": classify_intent_node,
        "router": router_node,
        "EXTRACT_CRITERIA": extract_initial_criteria_node,
        "SEARCH_EVENTS": prepare_and_search_events_node,
        "BUILD_PLAN": build_initial_plan_node,
        "ANALYZE_FEEDBACK": analyze_feedback_node,
        "PRESENT_RESULTS": presenter_node,
        "CLARIFY_OR_CHITCHAT": chitchat_node,
        "HANDLE_CLARIFICATION": handle_clarification_node,
        "REFINE_PLAN": refine_plan_node,
        "DELETE_ACTIVITY": delete_activity_node,
        "ADD_ACTIVITY": add_activity_node,
        "PROCESS_START_ADDRESS": process_start_address_node,
    }
    for name, node in nodes.items():
        workflow.add_node(name, node)

    # --- ИЗМЕНЕНИЕ: Устанавливаем новую условную точку входа ---
    # workflow.set_entry_point("classify_intent") <-- СТАРАЯ ВЕРСИЯ
    workflow.set_conditional_entry_point(
        should_classify_or_process_address,
        {
            "PROCESS_START_ADDRESS": "PROCESS_START_ADDRESS",
            "classify_intent": "classify_intent",
        },
    )
    # --- КОНЕЦ ИЗМЕНЕНИЯ ---

    # 3. Определяем остальные ребра
    workflow.add_conditional_edges(
        "classify_intent",
        decide_after_classification,
        {"CLARIFY_OR_CHITCHAT": "CLARIFY_OR_CHITCHAT", "router": "router"},
    )
    workflow.add_edge("ANALYZE_FEEDBACK", "router")
    workflow.add_edge("REFINE_PLAN", "BUILD_PLAN")
    workflow.add_edge("DELETE_ACTIVITY", "BUILD_PLAN")
    workflow.add_edge("ADD_ACTIVITY", "BUILD_PLAN")
    workflow.add_edge("BUILD_PLAN", "router")
    workflow.add_edge("EXTRACT_CRITERIA", "router")
    workflow.add_edge("SEARCH_EVENTS", "router")
    workflow.add_edge(
        "PROCESS_START_ADDRESS", "PRESENT_RESULTS"
    )  # Связь от обработки адреса к показу результата

    # Главная развилка - роутер
    # Используем более надежный способ создания словаря для conditional_edges
    action_map = {action.value: action.value for action in PossibleActions}
    workflow.add_conditional_edges(
        "router", lambda state: state["next_action"].value, action_map
    )

    # Конечные узлы графа
    workflow.add_edge("PRESENT_RESULTS", END)
    workflow.add_edge("CLARIFY_OR_CHITCHAT", END)

    app = workflow.compile()
    logger.info("Агентский граф (v7.2, с условной точкой входа) скомпилирован.")
    return app


agent_app = build_agent_graph()
# --- КОНЕЦ БЛОКА ДЛЯ ЗАМЕНЫ ---
