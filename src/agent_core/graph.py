# --- ЗАМЕНА ВСЕГО ФАЙЛА graph.py ---
import logging
from langgraph.graph import StateGraph, END
from src.agent_core.state import AgentState, UserIntent
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
)
from src.schemas.data_schemas import PossibleActions

logger = logging.getLogger(__name__)


def decide_after_classification(state: AgentState) -> str:
    """Ключевая развилка после определения намерения."""
    logger.info(
        f"Граф: Принятие решения после классификации. Намерение: {state.get('classified_intent').intent}"
    )
    if state.get("classified_intent").intent == UserIntent.CHITCHAT:
        return "CLARIFY_OR_CHITCHAT"
    return "router"


def build_agent_graph():
    """Собирает граф агента. v7.0 с финальной архитектурой потока управления."""
    workflow = StateGraph(AgentState)

    # 1. Добавляем узлы
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
    }
    for name, node in nodes.items():
        workflow.add_node(name, node)

    # 2. Определяем точку входа
    workflow.set_entry_point("classify_intent")

    # 3. Определяем ребра
    workflow.add_conditional_edges(
        "classify_intent",
        decide_after_classification,
        {"CLARIFY_OR_CHITCHAT": "CLARIFY_OR_CHITCHAT", "router": "router"},
    )

    # После анализа фидбека всегда идем в роутер, который начнет исполнять очередь
    workflow.add_edge("ANALYZE_FEEDBACK", "router")

    # --- ЯВНЫЙ ПОТОК ПЕРЕПЛАНИРОВАНИЯ ---
    # После любого изменения состояния мы ОБЯЗАТЕЛЬНО идем строить новый план.
    workflow.add_edge("REFINE_PLAN", "BUILD_PLAN")
    workflow.add_edge("DELETE_ACTIVITY", "BUILD_PLAN")
    workflow.add_edge("ADD_ACTIVITY", "BUILD_PLAN")

    # После построения плана всегда идем в роутер. Он проверит, пуста ли
    # очередь команд, и либо исполнит следующую, либо пойдет на PRESENT_RESULTS.
    workflow.add_edge("BUILD_PLAN", "router")

    # Узлы подготовки данных возвращаются в роутер
    workflow.add_edge("EXTRACT_CRITERIA", "router")
    workflow.add_edge("SEARCH_EVENTS", "router")

    # Главная развилка - роутер
    workflow.add_conditional_edges(
        "router",
        lambda state: state["next_action"].value,
        {
            "HANDLE_CLARIFICATION": "HANDLE_CLARIFICATION",
            "EXTRACT_CRITERIA": "EXTRACT_CRITERIA",
            "SEARCH_EVENTS": "SEARCH_EVENTS",
            "ANALYZE_FEEDBACK": "ANALYZE_FEEDBACK",
            "BUILD_PLAN": "BUILD_PLAN",
            "REFINE_PLAN": "REFINE_PLAN",
            "DELETE_ACTIVITY": "DELETE_ACTIVITY",
            "ADD_ACTIVITY": "ADD_ACTIVITY",
            "PRESENT_RESULTS": "PRESENT_RESULTS",
        },
    )

    # Конечные узлы графа
    workflow.add_edge("PRESENT_RESULTS", END)
    workflow.add_edge("CLARIFY_OR_CHITCHAT", END)

    app = workflow.compile()
    logger.info("Агентский граф (v7.0, Финальная версия) скомпилирован.")
    return app


agent_app = build_agent_graph()
# --- КОНЕЦ ЗАМЕНЫ ---
