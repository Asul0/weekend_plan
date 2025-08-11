# Файл: src/agent_core/nodes.py (ФИНАЛЬНАЯ ЭТАЛОННАЯ ВЕРСИЯ)
import logging
import asyncio
import json
import re
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from typing import List, Optional, Union
from src.agent_core.command_processor import CommandProcessor
from src.schemas.data_schemas import (  # <-- ИЗМЕНИТЬ ЭТОТ БЛОК
    ExtractedInitialInfo,
    Event,
    ParkInfo,
    PlanItem,
    FlatFeedback,
    Constraint,
    FoodPlaceInfo,
    AnalyzedFeedback,
    PlanBuilderResult,
    PossibleActions,
    SimplifiedExtractedInfo,
    ActivityClassifier,
    OrderedActivityItem,
    UserIntent,
    ChangeRequest,
    SemanticConstraint,
    ClassifiedIntent,
    LlmExtractionResult,
)
from src.tools.datetime_parser_tool import datetime_parser_tool
from src.tools.gis_tools import park_search_tool, food_place_search_tool
from src.services.gis_service import get_geocoding_details
from src.agent_core.state import AgentState
from src.gigachat_client import get_gigachat_client
from src.services.afisha_service import fetch_cities
from src.tools.datetime_parser_tool import datetime_parser_tool
from src.tools.event_search_tool import event_search_tool
from langchain_core.messages import HumanMessage, AIMessage
from src.utils.callbacks import TokenUsageCallbackHandler

logger = logging.getLogger(__name__)


class DecomposedIntent(BaseModel):
    """Промежуточная структура. Одно сырое, неразобранное намерение."""

    target_str: str = Field(
        description="К чему относится это намерение (e.g., 'фильм', 'ресторан', 'дата', 'город')."
    )
    raw_query: str = Field(
        description="Точная фраза пользователя, описывающая это намерение (e.g., 'на часик пораньше', 'в Казани', 'убрать')."
    )


class DecomposedResult(BaseModel):
    """Результат работы первого LLM-шага (декомпозиции)."""

    decomposed_intents: List[DecomposedIntent]


async def classify_intent_node(state: AgentState) -> AgentState:
    """
    Узел-Диспетчер. Версия 2.1. Промпт откалиброван для большей точности.
    """
    user_query = state.get("user_message")
    history = state.get("chat_history", [])
    logger.info("--- УЗЕЛ: classify_intent_node ---")
    logger.info(f"Определяю намерение для запроса: '{user_query}'")

    llm = get_gigachat_client()
    structured_llm = llm.with_structured_output(ClassifiedIntent)
    token_callback = TokenUsageCallbackHandler(node_name="classify_intent")
    prompt = f"""
Ты — высокоточный системный диспетчер. Твоя задача — проанализировать запрос пользователя и четко классифицировать его намерение по ОДНОЙ из трех категорий.

### Категории и Ключевые Признаки:
1.  **PLAN_REQUEST**:
    *   **Признаки:** В запросе есть упоминания **АКТИВНОСТЕЙ** (кино, парк, погулять, поесть, концерт, стендап), **МЕСТА** (город, адрес) или **ВРЕМЕНИ** (завтра, на выходных, в 15:00).
    *   **ПРАВИЛО:** Если есть **хотя бы один** из этих признаков — это почти всегда `PLAN_REQUEST`.
    *   **Примеры:**
        - "Хочу на фильм и покушать послезавтра в Воронеже" -> PLAN_REQUEST
        - "Найди стендап в Москве" -> PLAN_REQUEST
        - "Чем заняться на выходных?" -> PLAN_REQUEST
        - "Поесть в центре" -> PLAN_REQUEST

2.  **FEEDBACK_ON_PLAN**:
    *   **Признаки:** Пользователь отвечает на **только что предложенный план**. Ответ содержит прямые указания на изменение плана.
    *   **Примеры:** "Поменяй ресторан на более дешевый", "А есть сеанс попозже?", "Убери парк", "Да, этот план подходит".

3.  **CHITCHAT**:
    *   **Признаки:** Запрос **НЕ содержит** признаков `PLAN_REQUEST` или `FEEDBACK_ON_PLAN`. Это общие фразы.
    *   **Примеры:** "Привет", "Спасибо", "Что ты умеешь?", "Как дела?", "Ты бот?".

### ЗАДАЧА:
Проанализируй следующий запрос пользователя и верни JSON с категорией намерения.
**Запрос:** "{user_query}"
"""
    # --- КОНЕЦ БЛОКА ДЛЯ ЗАМЕНЫ ---

    try:
        # Проверяем, есть ли в state уже предложенный план. Это важный контекст.
        if state.get("current_plan"):
            # Если план есть, высока вероятность, что это фидбек. Уточняем промпт.
            prompt += "\n**Дополнительный контекст:** Ассистент только что предложил пользователю план. Проанализируй, является ли запрос фидбеком на этот план или сменой темы."

        result = await structured_llm.ainvoke(
            prompt, config={"callbacks": [token_callback]}
        )
        state["classified_intent"] = result
        logger.info(
            f"Намерение определено как: {result.intent}. Причина: {result.reasoning}"
        )
    except Exception as e:
        logger.error(f"Ошибка при классификации намерения: {e}", exc_info=True)
        state["classified_intent"] = ClassifiedIntent(
            intent=UserIntent.PLAN_REQUEST,
            reasoning="Произошла ошибка при классификации, выбран сценарий по умолчанию.",
        )
    return state


async def chitchat_node(state: AgentState) -> AgentState:
    """
    Узел для обработки общих вопросов и поддержания диалога.
    """
    user_query = state.get("user_message")
    logger.info("--- УЗЕЛ: chitchat_node ---")
    logger.info(f"Генерирую ответ на общий вопрос: '{user_query}'")

    llm = get_gigachat_client()
    token_callback = TokenUsageCallbackHandler(node_name="chitchat_node")
    prompt = f"""
Ты — дружелюбный и услужливый ассистент по планированию досуга.
Пользователь написал тебе сообщение, которое не является запросом на составление плана.
Твоя задача — вежливо и по делу ответить ему.

Ключевые моменты, которые нужно упомянуть в ответе:
- Ты можешь помочь найти мероприятия (кино, концерты, стендапы), места для прогулок (парки) и отдыха.
- Ты умеешь составлять из них последовательный план с учетом времени и маршрутов.
- Предложи пользователю сформулировать свой запрос, например: "Найди мне стендап в Воронеже на этих выходных".

Запрос пользователя: "{user_query}"

Сформируй краткий, дружелюбный и полезный ответ.
"""
    try:
        response = await llm.ainvoke(prompt, config={"callbacks": [token_callback]})
        ai_response = response.content
    except Exception as e:
        logger.error(f"Ошибка при генерации chitchat-ответа: {e}")
        ai_response = "Извините, у меня возникли трудности с ответом. Попробуйте, пожалуйста, сформулировать запрос на планирование."

    # Добавляем ответ в историю и завершаем работу
    state["chat_history"].append(AIMessage(content=ai_response))
    logger.info(f"Сформирован chitchat-ответ: '{ai_response[:100]}...'")
    return state


class FlatExtraction(BaseModel):
    """Схема с плоской структурой для максимальной надежности извлечения."""

    location: Optional[str] = Field(None, description="Город.")
    date: Optional[str] = Field(None, description="Описание даты.")
    time: Optional[str] = Field(None, description="Описание времени.")
    person_count_str: Optional[str] = Field(
        None, description="Фрагмент о кол-ве людей."
    )
    budget_str: Optional[str] = Field(None, description="Фрагмент о бюджете.")
    # Мы просим извлечь активности как ОДНУ строку, которую потом разделим кодом
    activities_str: Optional[str] = Field(
        None,
        description="Перечисление всех активностей через запятую, например: 'кино, парк'.",
    )


async def extract_initial_criteria_node(state: AgentState) -> AgentState:
    """
    Извлекает и классифицирует критерии из запроса пользователя в 2 этапа.
    Версия 2.2 с супер-надежным извлечением активностей в виде строки.
    """
    user_query = state.get("user_message")
    logger.info("--- УЗЕЛ: extract_initial_criteria_node (v2.2, надежный) ---")
    logger.info(f"Обработка запроса: '{user_query}'")

    llm = get_gigachat_client()

    # --- ЭТАП 1: Упрощенное извлечение с activities_str для надежности ---
    try:
        simple_extractor = llm.with_structured_output(SimplifiedExtractedInfo)
        token_callback_extract = TokenUsageCallbackHandler(
            node_name="extract_criteria_step1"
        )

        # ОБНОВЛЕННЫЙ ПРОМПТ: Просим строку, а не список
        prompt_extract = f"""
Ты — системный анализатор. Твоя задача — извлечь из запроса пользователя ключевую информацию.
Извлеки ВСЕ упоминания активностей (дел, занятий) как ОДНУ СТРОКУ, разделенную запятыми.

### Пример 1:
- Запрос: "Хочу на фильм и покушать еще в парке погулять послезавтра в воронеже"
- Результат (JSON):
  {{
    "city": "Воронеж",
    "dates_description": "послезавтра",
    "activities_str": "фильм, покушать, в парке погулять"
  }}

### Пример 2:
- Запрос: "Найди стендап в Москве на выходных, а потом сходим в бар"
- Результат (JSON):
  {{
    "city": "Москва",
    "dates_description": "на выходных",
    "activities_str": "стендап, сходим в бар"
  }}

### ЗАДАЧА:
Проанализируй следующий запрос пользователя и верни ТОЛЬКО JSON-объект.
**Запрос:** "{user_query}"
"""
        simplified_data = await simple_extractor.ainvoke(
            prompt_extract, config={"callbacks": [token_callback_extract]}
        )
        logger.info(
            f"Этап 1: Успешно извлечены упрощенные данные: {simplified_data.model_dump_json(indent=2)}"
        )

        # КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Проверяем новую строку, а не список
        if not simplified_data.activities_str:
            raise ValueError("Не удалось извлечь ни одной активности из запроса.")

        # Превращаем строку в список прямо здесь
        activities_list = [
            act.strip()
            for act in simplified_data.activities_str.split(",")
            if act.strip()
        ]
        if not activities_list:
            raise ValueError(
                "Извлеченная строка активностей оказалась пустой после обработки."
            )

    except Exception as e:
        error_message = f"Ошибка на этапе 1 (извлечение): {e}"
        logger.error(error_message, exc_info=True)
        state["error"] = (
            "Не смог понять, чем вы хотите заняться. Попробуйте переформулировать."
        )
        return state

    # --- ЭТАП 2: Классификация каждой активности (без изменений) ---
    try:
        activity_classifier = llm.with_structured_output(ActivityClassifier)
        ordered_activities = []

        # КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Итерируемся по списку, созданному из строки
        for activity_str in activities_list:
            token_callback_classify = TokenUsageCallbackHandler(
                node_name=f"extract_criteria_step2_{activity_str[:10]}"
            )
            prompt_classify = f"""
Твоя задача — классифицировать активность пользователя, выбрав ОДИН наиболее подходящий системный тип.
### Системные типы и их ключевые слова:
- **MOVIE**: Кино, фильм, сеанс, кинотеатр.
- **PARK**: Парк, сквер, погулять на природе, сад, набережная, аллея.
- **RESTAURANT**: Поесть, покушать, ресторан, кафе, бар, ужин, обед, перекусить, выпить кофе.
- **CONCERT**: Концерт, музыкальное выступление, опен-эйр.
- **STAND_UP**: Стендап, комедийное шоу, открытый микрофон.
- **PERFORMANCE**: Спектакль, театр, балет, опера.
- **MUSEUM_EXHIBITION**: Музей, выставка, галерея, экспозиция.
- **UNKNOWN**: Если не подходит ни один из вышеперечисленных.
### ЗАДАЧА: Классифицируй следующую активность: "{activity_str}"
"""
            classified_activity = await activity_classifier.ainvoke(
                prompt_classify, config={"callbacks": [token_callback_classify]}
            )
            logger.info(
                f"Активность '{activity_str}' классифицирована как {classified_activity.activity_type}"
            )

            if classified_activity.activity_type != "UNKNOWN":
                ordered_activities.append(
                    OrderedActivityItem(
                        activity_type=classified_activity.activity_type,
                        query_details=activity_str,
                    )
                )

        if not ordered_activities:
            raise ValueError("Ни одна из извлеченных активностей не была распознана.")

        final_criteria = ExtractedInitialInfo(
            city=simplified_data.city,
            dates_description=simplified_data.dates_description,
            ordered_activities=ordered_activities,
            budget=simplified_data.budget,
            person_count=simplified_data.person_count,
            raw_time_description=simplified_data.raw_time_description,
        )

        logger.info(
            f"Этап 2: Успешно собраны финальные критерии: {final_criteria.model_dump_json(indent=2)}"
        )
        state["search_criteria"] = final_criteria
        state["error"] = None

    except Exception as e:
        error_message = f"Ошибка на этапе 2 (классификация): {e}"
        logger.error(error_message, exc_info=True)
        state["error"] = (
            "Возникла проблема при анализе ваших пожеланий. Пожалуйста, попробуйте еще раз."
        )
        return state

    return state


async def build_initial_plan_node(state: AgentState) -> AgentState:
    """
    Узел-Строитель. Вызывает PlanBuilder для построения оптимального
    первоначального плана из собранных кандидатов.
    """
    logger.info("NODE: build_initial_plan_node. Начало построения оптимального плана.")

    # --- ИСПРАВЛЕНА ЛОГИКА ЧТЕНИЯ КАНДИДАТОВ ---
    # Проверяем, существует ли вообще кэш
    if not state.get("cached_candidates"):
        state["plan_builder_result"] = PlanBuilderResult(
            failure_reason="Не найдено ни одного подходящего варианта (кэш пуст)."
        )
        return state
    # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

    # Импортируем PlanBuilder здесь, чтобы избежать циклических импортов
    from src.agent_core.planner import PlanBuilder

    # PlanBuilder сам разберется с состоянием, передаем его целиком
    builder = PlanBuilder(state)
    result = await builder.build()
    state["plan_builder_result"] = result

    if result.best_plan:
        state["current_plan"] = result.best_plan
        logger.info(
            f"PlanBuilder успешно построил план из {len(result.best_plan.items)} пунктов."
        )
    else:
        state["current_plan"] = None
        logger.warning(
            f"PlanBuilder не смог построить план. Причина: {result.failure_reason}"
        )

    # Очищаем фидбек после использования
    if "analyzed_feedback" in state:
        state["analyzed_feedback"] = None

    return state


async def prepare_and_search_events_node(state: AgentState) -> AgentState:
    """
    Узел-Сборщик. Версия 2.0.
    Вызывает все необходимые инструменты поиска (Афиша, 2ГИС для парков и еды)
    и собирает ВСЕХ кандидатов в единый кэш.
    """
    logger.info("--- УЗЕЛ: prepare_and_search_events_node ---")
    logger.info("Начало сбора всех кандидатов (включая заведения питания).")

    criteria = state.get("search_criteria")

    if not criteria or not criteria.city or not criteria.dates_description:
        error_msg = "Отсутствуют необходимые критерии (город или дата) для поиска."
        logger.error(error_msg)
        state["error"] = error_msg
        return state

    if not criteria.ordered_activities:
        error_msg = "Не указаны интересы для поиска."
        logger.error(error_msg)
        state["error"] = error_msg
        return state

    try:
        # --- Шаг 1: Определение города и времени (без изменений) ---
        cities = await fetch_cities()
        city_info = next(
            (c for c in cities if c["name"].lower() == criteria.city.lower()), None
        )

        if not city_info:
            state["error"] = f"Не удалось найти город '{criteria.city}' в базе."
            return state
        city_id, city_name = city_info["id"], city_info["name"]

        parsed_time = await datetime_parser_tool.ainvoke(
            {
                "natural_language_date": criteria.dates_description,
                "natural_language_time_qualifier": criteria.raw_time_description,
            }
        )
        if not parsed_time or not parsed_time.get("datetime_iso"):
            state["error"] = (
                f"Не удалось распознать дату: '{criteria.dates_description}'"
            )
            return state

        state["parsed_dates_iso"] = [parsed_time["datetime_iso"]]
        state["parsed_end_dates_iso"] = (
            [parsed_time["end_datetime_iso"]]
            if parsed_time.get("end_datetime_iso")
            else []
        )

        start_date = datetime.fromisoformat(parsed_time["datetime_iso"])
        end_date = (
            datetime.fromisoformat(parsed_time["end_datetime_iso"])
            if parsed_time.get("end_datetime_iso")
            else start_date.replace(hour=23, minute=59, second=59)
        )

        # --- Шаг 2: Формирование очереди задач для всех активностей ---
        search_tasks = []
        for activity in criteria.ordered_activities:
            activity_type = activity.activity_type.upper()
            query = activity.query_details or ""

            if activity_type in [
                "MOVIE",
                "CONCERT",
                "PERFORMANCE",
                "STAND_UP",
                "MUSEUM_EXHIBITION",
            ]:
                logger.info(
                    f"Добавляю в очередь задачу на поиск события: type='{activity_type}'"
                )
                budget_to_use = activity.activity_budget or criteria.budget
                budget_per_person = (
                    budget_to_use / criteria.person_count
                    if budget_to_use and criteria.person_count
                    else None
                )
                task = event_search_tool.ainvoke(
                    {
                        "city_id": city_id,
                        "city_name": city_name,
                        "date_from": start_date,
                        "date_to": end_date,
                        "user_creation_type_key": activity_type,
                        "max_budget_per_person": budget_per_person,
                    }
                )
                search_tasks.append((activity_type, task))

            elif activity_type == "PARK":
                logger.info(
                    f"Добавляю в очередь задачу на поиск парка: query='{query}'"
                )
                task = park_search_tool.ainvoke(
                    {"query": query or "парк", "city": city_name}
                )
                search_tasks.append((activity_type, task))

            # --- НОВЫЙ БЛОК: Логика для поиска еды ---
            elif activity_type == "RESTAURANT":
                logger.info(f"Добавляю в очередь задачу на поиск еды: query='{query}'")
                task = food_place_search_tool.ainvoke(
                    {"query": query or "ресторан", "city": city_name}
                )
                search_tasks.append((activity_type, task))
            # --- КОНЕЦ НОВОГО БЛОКА ---

        # --- Шаг 3: Асинхронное выполнение всех задач ---
        logger.info(f"Запускаю {len(search_tasks)} задач на поиск...")
        results = await asyncio.gather(*[t for _, t in search_tasks])
        logger.info("Все задачи на поиск завершены.")

        # --- Шаг 4: Сбор и валидация результатов в кэш ---
        date_key = start_date.strftime("%Y-%m-%d")
        daily_cache = {}

        for (activity_type, _), result_list in zip(search_tasks, results):
            if result_list and isinstance(result_list, list):
                if activity_type not in daily_cache:
                    daily_cache[activity_type] = []

                logger.info(
                    f"Обрабатываю {len(result_list)} кандидатов для типа '{activity_type}'"
                )
                for item_dict in result_list:
                    try:
                        if activity_type == "PARK":
                            daily_cache[activity_type].append(ParkInfo(**item_dict))

                        # --- НОВЫЙ БЛОК: Логика для валидации еды ---
                        elif activity_type == "RESTAURANT":
                            daily_cache[activity_type].append(
                                FoodPlaceInfo(**item_dict)
                            )
                        # --- КОНЕЦ НОВОГО БЛОКА ---

                        else:  # Для всех типов событий из Афиши
                            daily_cache[activity_type].append(Event(**item_dict))
                    except Exception as e:
                        logger.warning(
                            f"Ошибка валидации кандидата типа '{activity_type}': {e} для данных {str(item_dict)[:200]}..."
                        )

        # Записываем собранных кандидатов в состояние
        state["cached_candidates"] = {date_key: daily_cache}
        logger.info(
            f"Сбор кандидатов завершен. Найдено для даты {date_key}: { {k: len(v) for k, v in daily_cache.items()} }"
        )
        state["error"] = None

    except Exception as e:
        logger.error(f"Критическая ошибка в узле сбора кандидатов: {e}", exc_info=True)
        state["error"] = "Произошла ошибка при поиске мероприятий."

    return state


async def handle_clarification_node(state: AgentState) -> AgentState:
    """
    Обрабатывает ответ пользователя на уточняющий вопрос. (ИСПРАВЛЕННАЯ ВЕРСИЯ)
    """
    logger.info("--- УЗЕЛ: handle_clarification_node ---")
    user_response = state.get("user_message")
    field_to_clarify = state.get("is_awaiting_clarification")
    criteria = state.get("search_criteria")

    if field_to_clarify == "city":
        logger.info(
            f"Получен ответ для уточнения города: '{user_response}'. Обновляю критерии."
        )

        # --- КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ ---
        # Обращаемся к 'search_criteria' как к объекту, а не словарю.
        if criteria:
            criteria.city = user_response.strip()
        else:
            # На случай, если критерии не были созданы, создаем их.
            state["search_criteria"] = ExtractedInitialInfo(city=user_response.strip())
        # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

        # Сбрасываем флаг ожидания и ошибку, чтобы граф мог двигаться дальше
        state["is_awaiting_clarification"] = None
        state["error"] = None

    return state


async def router_node(state: AgentState) -> AgentState:
    """
    Главный узел-оркестратор v8.0. ФИНАЛЬНАЯ ВЕРСИЯ.
    Реализована защита от цикла и корректная обработка новых PLAN_REQUEST.
    """
    logger.info(
        f"--- УЗЕЛ: router_node ---. Вход: plan={'Да' if state.get('current_plan') else 'Нет'}, queue={len(state.get('command_queue', []))}, intent={state.get('classified_intent').intent if state.get('classified_intent') else 'N/A'}"
    )

    # Шаг 0: Очистка "одноразовых" полей
    state["last_structured_command"] = None
    state["error"] = None

    # --- ИЕРАРХИЯ ПРИНЯТИЯ РЕШЕНИЙ ---

    # Приоритет 1: Новый запрос на ПОЛНОЕ перепланирование
    classified_intent = state.get("classified_intent")
    if classified_intent and classified_intent.intent == UserIntent.PLAN_REQUEST:
        logger.info(
            "Приоритет 1: Получен новый PLAN_REQUEST. Полный сброс и переход к извлечению критериев."
        )
        # Полная очистка старого контекста
        state["search_criteria"] = None
        state["cached_candidates"] = {}
        state["current_plan"] = None
        state["pinned_items"] = {}
        state["plan_builder_result"] = None
        state["classified_intent"] = None  # Сбрасываем интент
        state["next_action"] = PossibleActions.EXTRACT_CRITERIA
        return state

    # Приоритет 2: Новый ФИДБЕК на существующий план
    if classified_intent and classified_intent.intent == UserIntent.FEEDBACK_ON_PLAN:
        logger.info("Приоритет 2: Обнаружен FEEDBACK_ON_PLAN. -> ANALYZE_FEEDBACK")
        state["classified_intent"] = None  # Сбрасываем интент
        state["next_action"] = PossibleActions.ANALYZE_FEEDBACK
        return state

    # Приоритет 3: Обработка ОЧЕРЕДИ команд
    command_queue = state.get("command_queue", [])
    if command_queue:
        command_type = command_queue[0].command
        logger.info(f"Приоритет 3: Обработка команды '{command_type}' из очереди.")
        action_map = {
            "modify": PossibleActions.REFINE_PLAN,
            "delete": PossibleActions.DELETE_ACTIVITY,
            "add": PossibleActions.ADD_ACTIVITY,
            "update_criteria": PossibleActions.SEARCH_EVENTS,
        }
        if command_type in action_map:
            if command_type == "update_criteria":
                state["cached_candidates"] = {}
                state["pinned_items"] = {}
                state["current_plan"] = None
            state["next_action"] = action_map[command_type]
        else:
            state["next_action"] = PossibleActions.PRESENT_RESULTS
        return state

    # Приоритет 4: Проверка на ФАТАЛЬНУЮ ошибку PlanBuilder
    builder_result = state.get("plan_builder_result")
    if builder_result and builder_result.failure_reason:
        logger.error(
            f"Приоритет 4: PlanBuilder не смог построить план. Причина: {builder_result.failure_reason}. -> PRESENT_RESULTS"
        )
        state["error"] = builder_result.failure_reason  # Передаем ошибку для показа
        state["next_action"] = PossibleActions.PRESENT_RESULTS
        return state

    # Приоритет 5: Стандартный путь построения/показа плана
    if not state.get("search_criteria"):
        state["next_action"] = PossibleActions.EXTRACT_CRITERIA
    elif not state.get("cached_candidates"):
        state["next_action"] = PossibleActions.SEARCH_EVENTS
    elif not state.get("current_plan"):
        state["next_action"] = PossibleActions.BUILD_PLAN
    else:
        state["next_action"] = PossibleActions.PRESENT_RESULTS

    logger.info(f"Роутер решил: следующее действие -> {state['next_action'].value}")
    return state























async def presenter_node(state: AgentState) -> AgentState:
    """
    Финальный узел. Формирует ответ для пользователя с учетом контекста и предупреждений.
    """
    logger.info("--- УЗЕЛ: presenter_node (v2.1) ---")

    state["plan_presented"] = False

    error = state.get("error")
    criteria = state.get("search_criteria")
    builder_result = state.get("plan_builder_result")
    current_plan = state.get("current_plan")
    user_message = state.get("user_message")
    response_text = ""

    plan_to_show = None
    if builder_result and builder_result.best_plan:
        plan_to_show = builder_result.best_plan
        logger.debug("Presenter: Использую 'best_plan' из 'plan_builder_result'.")
    elif current_plan:
        plan_to_show = current_plan
        logger.debug("Presenter: Использую сохраненный 'current_plan'.")

    # --- НОВАЯ ЛОГИКА: Сбор всех предупреждений ---
    all_warnings = state.get("plan_warnings", [])
    if plan_to_show and plan_to_show.warnings:
        all_warnings.extend(plan_to_show.warnings)
    # --- КОНЕЦ НОВОЙ ЛОГИКИ ---

    if criteria and not criteria.city:
        response_text = (
            "Отличный план! Осталось только уточнить, в каком городе будем искать?"
        )
        state["is_awaiting_clarification"] = "city"
    elif error:
        # Теперь сюда попадают только настоящие ошибки
        response_text = f"К сожалению, произошла ошибка: {error}"
    elif builder_result and builder_result.failure_reason:
        response_text = f"К сожалению, не удалось составить план. Причина: {builder_result.failure_reason}"
    elif not plan_to_show:
        response_text = "Что-то пошло не так, и я не смог составить для вас план. Давайте попробуем еще раз с другими параметрами."
    else:
        # --- ШАГ 2: СОЗДАНИЕ УМНОГО ПРОМПТА С КОНТЕКСТОМ И ПРЕДУПРЕЖДЕНИЯМИ ---
        llm = get_gigachat_client()
        token_callback = TokenUsageCallbackHandler(node_name="presenter_node")
        plan_json = plan_to_show.model_dump_json(indent=2)

        warnings_text = ""
        if all_warnings:
            # Убираем дубликаты, если они есть
            unique_warnings = list(dict.fromkeys(all_warnings))
            warnings_text = "\n\n### Важные уточнения:\n" + "\n".join(
                f"- {w}" for w in unique_warnings
            )

        prompt = f"""
Ты — "Голос" умного ассистента-планировщика. Твоя задача — сформировать финальный, дружелюбный и полезный ответ.

### Контекст:
- **Последний запрос пользователя:** "{user_message}"
- **Системные данные (ГОТОВЫЙ ПЛАН):**
{plan_json}
{warnings_text}

### Твоя задача:
1.  **Проанализируй запрос пользователя и выбери подходящую вводную фразу:**
    - Если запрос первичный ("составь план"), начни с "Отлично, вот что я смог для вас подобрать:".
    - Если просят напомнить ("напомни план"), начни с "Конечно, вот ваш план:".
    - Если это подтверждение ("да, подходит"), начни с "Рад, что вам понравилось! Вот финальный план:" и в конце добавь "Приятного отдыха!".

2.  **Красиво и структурированно выведи каждый пункт плана** из `items`, используя Markdown и эмодзи (📍, 🕒, 💰).

3.  **Если в системных данных есть "Важные уточнения", обязательно добавь их в свой ответ** после списка мероприятий. Начни этот блок с эмодзи ⚠️ и фразы "Обратите внимание:".

4.  **В конце, если это не финальное подтверждение, задай уточняющий вопрос**, например: "Как вам такой план? Устроит или что-то поменяем?"

Твой ответ должен быть ТОЛЬКО текстом для пользователя.
"""
        try:
            llm_response = await llm.ainvoke(
                prompt, config={"callbacks": [token_callback]}
            )
            response_text = llm_response.content
            state["plan_presented"] = True
            logger.info("Флаг 'plan_presented' установлен в True.")
        except Exception as e:
            logger.error(
                f"Ошибка при генерации ответа в presenter_node: {e}", exc_info=True
            )
            response_text = "Я составил для вас план, но у меня возникли трудности с его красивым описанием."

    # --- ШАГ 3: ОБНОВЛЕНИЕ ИСТОРИИ И ОЧИСТКА СОСТОЯНИЯ ---
    if response_text:
        ai_message = AIMessage(content=response_text)
        history = state.get("chat_history", [])
        if not history or not (
            isinstance(history[-1], AIMessage) and history[-1].content == response_text
        ):
            state["chat_history"].append(ai_message)
        logger.info(f"Сформирован финальный ответ: '{response_text[:150]}...'")
    else:
        logger.error(
            "Критическая ошибка в presenter_node: не удалось сформировать текст ответа."
        )
        state["chat_history"].append(
            AIMessage(content="Извините, произошла внутренняя ошибка.")
        )

    # Очищаем "одноразовые" и временные поля
    state["plan_builder_result"] = None
    state["last_structured_command"] = None
    state["plan_warnings"] = []  # Очищаем предупреждения после их показа

    return state










































# в файле nodes.py
async def analyze_feedback_node(state: AgentState) -> AgentState:
    """
    Узел-Анализатор v7.0 (Chain of Thought + Text Marking).
    Это наиболее надежная архитектура.
    ШАГ 1: LLM генерирует рассуждения и простую текстовую разметку команд.
    ШАГ 2: Python-парсер преобразует разметку в семантическое ядро.
    ШАГ 3: CommandProcessor превращает ядро в исполняемые команды.
    """
    logger.info("--- УЗЕЛ: analyze_feedback_node (v7.0, CoT + Marking) ---")
    user_query = state.get("user_message")
    current_plan = state.get("current_plan")

    # Подготовка контекста (этот блок вы заполняете своим кодом)
    simplified_plan_json = "План еще не составлен."
    if current_plan and current_plan.items:
        # Ваша логика формирования simplified_plan_json остается здесь
        pass

    llm = get_gigachat_client()
    # Нам больше не нужен structured_llm, так как мы работаем с сырым текстом

    # === ШАГ 1: РАССУЖДЕНИЕ И РАЗМЕТКА (Chain of Thought) ===
    logger.info("Шаг 1: Запрос на рассуждение и текстовую разметку.")

    # Новый, самый надежный промпт
    prompt = f"""
Ты — высокоточный системный аналитик. Твоя задача - проанализировать запрос пользователя на изменение плана, сначала пошагово рассуждая, а затем выдав четкие команды в специальном формате.

### Контекст:
- **Предложенный план:** {simplified_plan_json}
- **Запрос пользователя:** "{user_query}"

### ЗАДАЧА:
Выполни два действия в указанном порядке:
1.  **Внутри тега `<reasoning>`:** Напиши свои пошаговые рассуждения. Определи, сколько намерений у пользователя и что он хочет сделать с каждым элементом.
2.  **Внутри тега `<commands>`:** Для каждого намерения напиши ОДНУ команду на новой строке в строго заданном формате с 6 полями, разделенными точкой с запятой. Используй 'None' для пустых полей.

### Формат команды:
`command_type;target;attribute;operator;value_str;value_num_unit`
(value_num_unit - это число и единица измерения через пробел, например '2 часа' или '1500 рублей')

### СЛОВАРЬ ТЕРМИНОВ:
- **command_type**: `modify`, `delete`, `add`, `update_criteria`, `chitchat`.
- **target**: `MOVIE`, `RESTAURANT`, `PARK`, `date`, `city`.
- **attribute**: `start_time`, `price`, `rating`, `date`, `city`, `name`.
- **operator**: `GREATER_THAN`, `LESS_THAN`, `NOT_EQUALS`, `MIN`, `MAX`.

### ПРИМЕР:
- **Запрос:** "Фильм слишком поздний, давай на часик пораньше. И ресторан найди подешевле, до 1500 рублей, а парк убери. И давай все в Казани."
- **Твой ответ:**
<reasoning>
Пользователь выразил четыре намерения.
1. Фильм: "на часик пораньше" - это изменение времени в меньшую сторону (modify, LESS_THAN).
2. Ресторан: "подешевле, до 1500" - это изменение цены в меньшую сторону (modify, LESS_THAN).
3. Парк: "убери" - это удаление (delete).
4. Глобальное изменение: "в Казани" - это смена города (update_criteria), что является приоритетной командой.
</reasoning>
<commands>
modify;MOVIE;start_time;LESS_THAN;None;1 час
modify;RESTAURANT;price;LESS_THAN;None;1500 рублей
delete;PARK;None;None;None;None
update_criteria;city;city;None;Казань;None
</commands>

ЗАДАЧА: Проанализируй запрос и сгенерируй ответ в указанном формате.
"""

    try:
        response_text = (await llm.ainvoke(prompt)).content
        logger.info(f"Получен ответ от LLM для разметки:\n---\n{response_text}\n---")
        logger.debug(f"Получен сырой ответ от LLM:\n{response_text}")
    except Exception as e:
        logger.error(f"Критическая ошибка при вызове LLM: {e}", exc_info=True)
        state["error"] = "Не удалось получить ответ от языковой модели."
        return state

    # === ШАГ 2: PYTHON-ПАРСЕР РАЗМЕТКИ ===
    logger.info("Шаг 2: Парсинг текстовой разметки.")
    all_semantic_intents = []
    try:
        # Используем re.DOTALL, чтобы . соответствовал и символу новой строки
        command_text_match = re.search(
            r"<commands>(.*?)</commands>", response_text, re.DOTALL
        )
        if not command_text_match:
            logger.warning("Тег <commands> не найден в ответе LLM.")
        else:
            command_text = command_text_match.group(1).strip()
            lines = [line.strip() for line in command_text.split("\n") if line.strip()]

            for line in lines:
                parts = [
                    p.strip() if p.strip().lower() != "none" else None
                    for p in line.split(";")
                ]
                if len(parts) != 6:
                    logger.warning(f"Пропуск некорректной строки команды: '{line}'")
                    continue

                command_type, target, attribute, operator, value_str, value_num_unit = (
                    parts
                )

                value_num, value_unit = None, None
                if value_num_unit:
                    num_match = re.search(r"[\d.]+", value_num_unit)
                    unit_match = re.search(r"[а-яА-Яa-zA-Z]+", value_num_unit)
                    if num_match:
                        value_num = float(num_match.group(0))
                    if unit_match:
                        value_unit = unit_match.group(0)

                all_semantic_intents.append(
                    SemanticConstraint(
                        command_type=command_type,
                        target=target,
                        attribute=attribute,
                        operator=operator,
                        value_str=value_str,
                        value_num=value_num,
                        value_unit=value_unit,
                    )
                )
    except Exception as e:
        logger.error(f"Ошибка на Шаге 2 (Парсинг): {e}", exc_info=True)
        state["error"] = "Не удалось разобрать внутреннюю структуру ответа."
        return state

    logger.debug(
        f"Итоговое семантическое ядро: {[i.model_dump() for i in all_semantic_intents]}"
    )

    # === ШАГ 3: ПЕРЕДАЧА В COMMAND PROCESSOR (без изменений) ===
    logger.info("Шаг 3: Передача семантического ядра в CommandProcessor.")
    processor = CommandProcessor(state, all_semantic_intents)
    state["command_queue"] = processor.process()
    state["error"] = None
    logger.info(
        f"CommandProcessor сформировал {len(state['command_queue'])} исполняемых команд."
    )

    return state


async def delete_activity_node(state: AgentState) -> AgentState:
    """
    Узел-Исполнитель для команды 'delete' v2.0.
    Удаляет активность из search_criteria и pinned_items.
    """
    command: Optional[ChangeRequest] = (
        state["command_queue"].pop(0) if state.get("command_queue") else None
    )

    logger.info(
        f"--- УЗЕЛ: delete_activity_node ---. Команда: {command.model_dump_json(indent=2)}"
    )

    if not command or command.command != "delete" or not command.target:
        logger.warning("Некорректная или отсутствующая команда 'delete'.")
        return state

    target_type = command.target
    criteria = state.get("search_criteria")

    if criteria and criteria.ordered_activities:
        initial_count = len(criteria.ordered_activities)
        criteria.ordered_activities = [
            act
            for act in criteria.ordered_activities
            if act.activity_type != target_type
        ]
        if len(criteria.ordered_activities) < initial_count:
            logger.info(f"Активность '{target_type}' удалена из search_criteria.")
        else:
            logger.warning(f"Активность '{target_type}' не найдена в search_criteria.")

    if target_type in state.get("pinned_items", {}):
        del state["pinned_items"][target_type]
        logger.info(f"Элемент '{target_type}' откреплен (unpinned).")

    # План больше не сбрасывается здесь. BUILD_PLAN будет вызван следующим.
    logger.info("Узел delete_activity_node завершил работу. Переход к BUILD_PLAN.")
    return state


async def refine_plan_node(state: AgentState) -> AgentState:
    """
    Узел-Исполнитель v4.2. ФИНАЛЬНАЯ ВЕРСИЯ.
    Исправлена логика получения команды и добавлена проверка на пустой результат.
    """
    # --- КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: Берем команду из очереди, а не из last_structured_command ---
    command: Optional[ChangeRequest] = (
        state["command_queue"].pop(0) if state.get("command_queue") else None
    )

    logger.info(
        f"--- УЗЕЛ: refine_plan_node ---. Команда: {command.model_dump_json(indent=2) if command else 'None'}"
    )

    if not command or command.command != "modify":
        logger.warning("Некорректная или отсутствующая команда 'modify'.")
        return state
    # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

    target_type = command.target
    constraints = command.constraints

    if not target_type or not constraints:
        logger.warning("Пропуск неполной команды 'modify'.")
        return state

    date_key = next(iter(state.get("cached_candidates", {})), None)
    if not date_key:
        state["error"] = "Кэш кандидатов пуст, не могу выполнить изменение."
        logger.error(state["error"])
        return state

    for constr in constraints:
        if constr.operator in ["MIN", "MAX"]:
            logger.info(
                f"Создание инструкции по сортировке: {constr.operator} для '{target_type}' по '{constr.attribute}'"
            )
            state["sorting_preference"] = {
                "target": target_type,
                "attribute": constr.attribute,
                "order": constr.operator,
            }
            continue

        all_candidates = state["cached_candidates"][date_key].get(target_type, [])
        logger.info(
            f"Фильтрация {len(all_candidates)} кандидатов для '{target_type}' по '{constr.attribute} {constr.operator} {constr.value}'"
        )

        try:
            precise_filtered = _apply_constraint(all_candidates, constr)
            filtered_candidates = precise_filtered

            if not precise_filtered and constr.attribute == "start_time":
                logger.info(
                    "Точных совпадений по времени нет, расширяю диапазон поиска на ±15 минут."
                )
                filtered_candidates = _apply_constraint(
                    all_candidates, constr, expansion_minutes=15
                )
                if filtered_candidates:
                    state["plan_warnings"] = (state.get("plan_warnings") or []) + [
                        "Не нашлось вариантов на точное время, но я подобрал ближайший."
                    ]

            logger.info(
                f"После фильтрации осталось {len(filtered_candidates)} кандидатов."
            )

            if not filtered_candidates:
                state["plan_warnings"] = (state.get("plan_warnings") or []) + [
                    f"Не удалось найти вариантов для '{target_type}' по вашим новым критериям. Эта активность удалена из плана."
                ]
                criteria = state.get("search_criteria")
                if criteria and criteria.ordered_activities:
                    criteria.ordered_activities = [
                        act
                        for act in criteria.ordered_activities
                        if act.activity_type != target_type
                    ]
                    logger.info(
                        f"Активность '{target_type}' удалена из списка дел из-за отсутствия кандидатов."
                    )

            state["cached_candidates"][date_key][target_type] = filtered_candidates
        except Exception as e:
            state["error"] = "Ошибка при поиске по новым критериям."
            return state

    if target_type in state.get("pinned_items", {}):
        del state["pinned_items"][target_type]
        logger.info(f"Элемент '{target_type}' откреплен.")

    logger.info("Узел refine_plan_node завершил работу. Переход к BUILD_PLAN.")
    return state


async def add_activity_node(state: AgentState) -> AgentState:
    """
    Узел-Исполнитель для команды 'add'.
    Ищет кандидатов для новой активности и добавляет их в кэш и критерии.
    """
    logger.info("--- УЗЕЛ: add_activity_node ---")
    command: Optional[ChangeRequest] = (
        state["command_queue"].pop(0) if state.get("command_queue") else None
    )

    if not command or command.command != "add" or not command.new_activity:
        logger.warning("add_activity_node: нет команды 'add' для исполнения.")
        return state

    new_activity = command.new_activity
    activity_type = new_activity.activity_type
    query = new_activity.query_details
    criteria = state.get("search_criteria")

    if not criteria or not criteria.city or not state.get("parsed_dates_iso"):
        state["error"] = (
            "Недостаточно контекста (город/дата) для добавления активности."
        )
        return state

    logger.info(
        f"Ищу кандидатов для новой активности '{activity_type}' с запросом '{query}'"
    )

    # --- Логика поиска (упрощенная версия prepare_and_search_events_node) ---
    try:
        # TODO: Добавить поддержку всех типов активностей, включая Афишу
        # TODO: Вынести логику поиска в отдельный инструмент/сервис
        if activity_type in ["PARK", "RESTAURANT"]:
            tool = (
                park_search_tool if activity_type == "PARK" else food_place_search_tool
            )
            candidates = await tool.ainvoke({"query": query, "city": criteria.city})

            if candidates:
                date_key = datetime.fromisoformat(
                    state["parsed_dates_iso"][0]
                ).strftime("%Y-%m-%d")
                if date_key not in state["cached_candidates"]:
                    state["cached_candidates"][date_key] = {}
                if activity_type not in state["cached_candidates"][date_key]:
                    state["cached_candidates"][date_key][activity_type] = []

                state["cached_candidates"][date_key][activity_type].extend(candidates)
                logger.info(
                    f"Добавлено {len(candidates)} кандидатов для '{activity_type}' в кэш."
                )
        else:
            logger.warning(
                f"Поиск для типа '{activity_type}' пока не реализован в add_activity_node."
            )

        # Добавляем новую активность в список дел
        if criteria.ordered_activities:
            # TODO: Реализовать вставку в корректную позицию (до/после)
            criteria.ordered_activities.append(new_activity)
            logger.info(f"Активность '{activity_type}' добавлена в search_criteria.")

    except Exception as e:
        logger.error(
            f"Ошибка при поиске кандидатов для новой активности: {e}", exc_info=True
        )
        state["error"] = f"Не удалось найти варианты для '{query}'."

    state["current_plan"] = None
    state["plan_builder_result"] = None
    state["last_structured_command"] = None

    return state


def _apply_constraint(
    candidates: List[PlanItem], constr: Constraint, expansion_minutes: int = 0
) -> List[PlanItem]:
    """
    Применяет одно семантическое ограничение к списку кандидатов.
    ФИНАЛЬНАЯ ВЕРСИЯ: Работает с семантическими именами атрибутов.
    """
    new_filtered_list = []

    # --- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Сопоставляем семантику с моделью ЗДЕСЬ ---
    attr_map = {
        "start_time": "start_time_naive_event_tz",
        "price": "min_price",
        "rating": "rating",
        "name": "name",
    }
    model_attribute = attr_map.get(constr.attribute)
    if not model_attribute:
        logger.warning(f"Неизвестный атрибут для фильтрации: {constr.attribute}")
        return candidates  # Возвращаем без изменений, если атрибут не знаком
    # --- КОНЕЦ ИЗМЕНЕНИЯ ---

    # Вспомогательная функция для безопасного приведения типов
    def get_typed_value(value_str: str, target_type: str):
        if target_type == "datetime":
            return datetime.fromisoformat(value_str)
        if target_type == "float":
            return float(value_str)
        return value_str

    is_numeric = constr.attribute in ["price", "rating"]
    is_datetime = constr.attribute == "start_time"

    try:
        if is_datetime:
            target_value = get_typed_value(constr.value, "datetime")
        elif is_numeric:
            target_value = get_typed_value(constr.value, "float")
        else:
            target_value = constr.value
    except (ValueError, TypeError):
        logger.error(
            f"Не удалось распарсить значение '{constr.value}' для атрибута '{constr.attribute}'"
        )
        return []

    for candidate in candidates:
        cand_value = getattr(
            candidate, model_attribute, None
        )  # <-- Используем model_attribute
        if cand_value is None:
            continue

        if constr.operator in ["MIN", "MAX"]:
            new_filtered_list.append(candidate)
            continue

        is_match = False
        try:
            if is_datetime:
                lower_bound = target_value - timedelta(minutes=expansion_minutes)
                upper_bound = target_value + timedelta(minutes=expansion_minutes)
                if constr.operator == "GREATER_THAN":
                    is_match = cand_value > lower_bound
                if constr.operator == "LESS_THAN":
                    is_match = cand_value < upper_bound
            elif is_numeric:
                if constr.operator == "GREATER_THAN":
                    is_match = cand_value > target_value
                if constr.operator == "LESS_THAN":
                    is_match = cand_value < target_value
            else:
                if constr.operator == "NOT_EQUALS":
                    is_match = cand_value != target_value
                if constr.operator == "EQUALS":
                    is_match = cand_value == target_value
        except TypeError:
            continue

        if is_match:
            new_filtered_list.append(candidate)

    return new_filtered_list
