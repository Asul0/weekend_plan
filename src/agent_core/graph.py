import logging
from langgraph.graph import StateGraph, END
from src.agent_core.state import AgentState
from src.agent_core.nodes import (
    router_node,
    extract_initial_criteria_node,
    check_and_ask_for_missing_criteria_node,
    process_clarification_node,
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

from src.schemas.data_schemas import PossibleActions, UserIntent

logger = logging.getLogger(__name__)


# --- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: ОБНОВЛЕННАЯ ФУНКЦИЯ-РАЗВИЛКА ---
def route_initial_message(state: AgentState) -> str:
    """
    Главная входная развилка графа. Проверяет все состояния ожидания.
    """
    if state.get("is_awaiting_start_address"):
        logger.info(
            "Граф: Обнаружен флаг is_awaiting_start_address. Пропускаю классификацию, перехожу к обработке адреса."
        )
        return "PROCESS_START_ADDRESS"
    
    if state.get("is_awaiting_criteria_clarification"):
        logger.info(
            "Граф: Обнаружен флаг is_awaiting_criteria_clarification. Пропускаю классификацию, перехожу к роутеру."
        )
        return "router"
    
    logger.info(
        "Граф: Флаги ожидания не установлены. Перехожу к стандартной классификации намерения."
    )
    return "classify_intent"


def decide_after_classification(state: AgentState) -> str:
    """Ключевая развилка ПОСЛЕ определения намерения."""
    logger.info(
        f"Граф: Принятие решения после классификации. Намерение: {state.get('classified_intent').intent}"
    )
    if state.get("classified_intent").intent == UserIntent.CHITCHAT:
        return "CLARIFY_OR_CHITCHAT"
    return "router"


def decide_after_criteria_check(state: AgentState) -> str:
    """
    Решает, куда идти после проверки критериев:
    - Если нужно задать вопрос, идет на ASK_FOR_CRITERIA_CLARIFICATION.
    - Если все в порядке, возвращается в роутер.
    """
    if state.get("next_action") == PossibleActions.ASK_FOR_CRITERIA_CLARIFICATION:
        logger.info("Граф: После проверки критериев нужно задать вопрос пользователю.")
        return "ASK_FOR_CRITERIA_CLARIFICATION"
    else:
        logger.info("Граф: После проверки критериев все данные полны, возвращаемся в роутер.")
        return "router"


def build_agent_graph():
    """
    Собирает граф агента. v8.6 с финальной корректной маршрутизацией.
    """
    workflow = StateGraph(AgentState)

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
        "CHECK_CRITERIA": check_and_ask_for_missing_criteria_node,
        "PROCESS_CRITERIA_CLARIFICATION": process_clarification_node,
        "ASK_FOR_CRITERIA_CLARIFICATION": presenter_node,
    }
    for name, node in nodes.items():
        workflow.add_node(name, node)

    # --- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: ОБНОВЛЕННАЯ ТОЧКА ВХОДА ---
    workflow.set_conditional_entry_point(
        route_initial_message,
        {
            "PROCESS_START_ADDRESS": "PROCESS_START_ADDRESS",
            "router": "router",
            "classify_intent": "classify_intent",
        },
    )

    workflow.add_conditional_edges(
        "classify_intent",
        decide_after_classification,
        {"CLARIFY_OR_CHITCHAT": "CLARIFY_OR_CHITCHAT", "router": "router"},
    )

    workflow.add_conditional_edges(
        "CHECK_CRITERIA",
        decide_after_criteria_check,
        {
            "ASK_FOR_CRITERIA_CLARIFICATION": "ASK_FOR_CRITERIA_CLARIFICATION",
            "router": "router",
        },
    )

    workflow.add_edge("PROCESS_CRITERIA_CLARIFICATION", "router")
    workflow.add_edge("ANALYZE_FEEDBACK", "router")
    workflow.add_edge("REFINE_PLAN", "BUILD_PLAN")
    workflow.add_edge("DELETE_ACTIVITY", "BUILD_PLAN")
    workflow.add_edge("ADD_ACTIVITY", "BUILD_PLAN")
    workflow.add_edge("BUILD_PLAN", "router")
    workflow.add_edge("EXTRACT_CRITERIA", "router")
    workflow.add_edge("SEARCH_EVENTS", "router")
    workflow.add_edge("PROCESS_START_ADDRESS", "PRESENT_RESULTS")

    action_map = {action.value: action.value for action in PossibleActions}
    workflow.add_conditional_edges(
        "router", lambda state: state["next_action"].value, action_map
    )

    workflow.add_edge("PRESENT_RESULTS", END)
    workflow.add_edge("CLARIFY_OR_CHITCHAT", END)
    workflow.add_edge("ASK_FOR_CRITERIA_CLARIFICATION", END)

    app = workflow.compile()
    logger.info("Агентский граф (v8.6, с финальной маршрутизацией) скомпилирован.")
    return app


agent_app = build_agent_graph()