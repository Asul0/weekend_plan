# Анализ кодовой базы AI-приложения: Weekend Plan Assistant

## 📁 Структура проекта

```
weekend_plan/
├── main.py                    # Точка входа Telegram-бота
├── server.py                  # FastAPI прокси-сервер для API Афиши
├── src/
│   ├── config.py             # Конфигурация и настройки
│   ├── gigachat_client.py    # Клиент для работы с GigaChat API
│   ├── agent_core/           # Ядро AI-агента
│   │   ├── graph.py          # Определение графа состояний LangGraph
│   │   ├── nodes.py          # Узлы графа (обработчики)
│   │   ├── state.py          # Схема состояния агента
│   │   ├── planner.py        # Построитель планов мероприятий
│   │   ├── command_processor.py # Обработчик команд пользователя
│   │   └── schedule_parser.py # Парсер расписаний
│   ├── schemas/              # Pydantic схемы данных
│   │   └── data_schemas.py   # Все модели данных
│   ├── services/             # Внешние сервисы
│   │   ├── afisha_service.py # Интеграция с API Афиши
│   │   └── gis_service.py    # Интеграция с 2ГИС API
│   ├── tools/                # LangChain инструменты
│   │   ├── datetime_parser_tool.py
│   │   ├── event_search_tool.py
│   │   ├── gis_tools.py
│   │   └── route_builder_tool.py
│   └── utils/
│       └── callbacks.py      # Обработчики коллбэков
└── test/                     # Тесты (15 файлов)
    ├── test_full_scenario.py # Интеграционные тесты
    └── test_*.py             # Unit-тесты для каждого узла
```

**Принципы организации кода:**

- **Layer-based архитектура** с четким разделением на слои (агент, сервисы, схемы)
- **Domain-driven подход** с выделением доменов (планирование, поиск, маршруты)
- **Модульная структура** с изолированными компонентами

## 🛠 Технологический стек

| Технология              | Версия/Назначение | Описание                                               |
| ----------------------- | ----------------- | ------------------------------------------------------ |
| **Python**              | 3.13+             | Основной язык программирования                         |
| **LangGraph**           | Latest            | Фреймворк для построения AI-агентов с графом состояний |
| **LangChain**           | Latest            | Интеграция с LLM и инструментами                       |
| **GigaChat**            | API               | Российская языковая модель для обработки запросов      |
| **Pydantic**            | v2                | Валидация и сериализация данных                        |
| **FastAPI**             | Latest            | Прокси-сервер для API Афиши                            |
| **aiohttp**             | Latest            | Асинхронные HTTP-запросы                               |
| **python-telegram-bot** | Latest            | Telegram Bot API                                       |
| **2ГИС API**            | v3.0              | Поиск мест и построение маршрутов                      |
| **Афиша.ру API**        | v3                | Поиск мероприятий                                      |

**Управление зависимостями:** Отсутствует файл requirements.txt или pyproject.toml, что является недостатком проекта.

## 🏗 Архитектура

### Основные архитектурные паттерны

**1. Граф состояний (StateGraph)**

```python
# src/agent_core/graph.py
workflow = StateGraph(AgentState)
workflow.set_conditional_entry_point(
    should_classify_or_process_address,
    {
        "PROCESS_START_ADDRESS": "PROCESS_START_ADDRESS",
        "classify_intent": "classify_intent",
    },
)
```

**2. Узлы-обработчики (Nodes)**

```python
# src/agent_core/nodes.py
async def classify_intent_node(state: AgentState) -> AgentState:
    """Классификация намерения пользователя"""
    llm = get_gigachat_client()
    structured_llm = llm.with_structured_output(ClassifiedIntent)
    # ... обработка
```

**3. Управление состоянием**

```python
# src/agent_core/state.py
class AgentState(TypedDict):
    user_message: str
    chat_history: Annotated[List[BaseMessage], add]
    classified_intent: Optional[ClassifiedIntent]
    search_criteria: Optional[ExtractedInitialInfo]
    current_plan: Optional[Plan]
    # ... другие поля
```

**4. Маршрутизация и условные переходы**

```python
def decide_after_classification(state: AgentState) -> str:
    if state.get("classified_intent").intent == UserIntent.CHITCHAT:
        return "CLARIFY_OR_CHITCHAT"
    return "router"
```

**5. Обработка ошибок и retry**

```python
# src/services/afisha_service.py
try:
    async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=60)) as response:
        # ... обработка ответа
except aiohttp.ClientConnectorError as e:
    logger.error(f"Afisha Proxy ClientConnectorError: {e}")
```

## 🔌 Интеграции и работа с данными

### Взаимодействие с LLM (GigaChat)

**Формирование промптов:**

```python
# src/agent_core/nodes.py
prompt = f"""
Ты — высокоточный системный диспетчер. Твоя задача — проанализировать запрос пользователя
и четко классифицировать его намерение по ОДНОЙ из трех категорий.

### Категории и Ключевые Признаки:
1. **PLAN_REQUEST**: В запросе есть упоминания АКТИВНОСТЕЙ, МЕСТА или ВРЕМЕНИ
2. **FEEDBACK_ON_PLAN**: Пользователь отвечает на предложенный план
3. **CHITCHAT**: Общий разговор, вопрос о возможностях

**Запрос:** "{user_query}"
"""
```

**Обработка ответов:**

```python
structured_llm = llm.with_structured_output(ClassifiedIntent)
result = await structured_llm.ainvoke(prompt, config={"callbacks": [token_callback]})
```

### Интеграция с внешними сервисами

**2ГИС API:**

```python
# src/services/gis_service.py
async def search_parks(original_query: str, city: Optional[str] = None) -> List[ParkInfo]:
    params = {
        "q": api_q,
        "type": relevant_types_str,
        "fields": "items.id,items.name,items.full_name,items.address_name,items.geometry.centroid",
        "key": GIS_API_KEY,
    }
```

**Афиша.ру API (через прокси):**

```python
# src/services/afisha_service.py
async def _make_afisha_request(session: aiohttp.ClientSession, endpoint: str) -> Optional[Dict]:
    full_url = f"{settings.AFISHA_PROXY_BASE_URL.rstrip('/')}/{endpoint.lstrip('/')}"
    headers_to_afisha["X-ApiAuth-PartnerKey"] = PROXY_INTERNAL_PARTNER_KEY
```

### Валидация данных

**Pydantic схемы:**

```python
# src/schemas/data_schemas.py
class Event(BaseModel):
    session_id: int = Field(description="Уникальный ID сеанса мероприятия")
    name: str = Field(description="Полное название мероприятия")
    start_time_iso: str = Field(description="Время начала в формате ISO")
    # ... другие поля

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)
```

## ✅ Качество кода

### Сильные стороны

1. **Типизация:** Активное использование type hints и Pydantic для валидации
2. **Асинхронность:** Полностью асинхронная архитектура с aiohttp
3. **Логирование:** Детальное логирование на всех уровнях
4. **Тестирование:** Покрытие unit и integration тестами
5. **Модульность:** Четкое разделение ответственности между модулями

### Области для улучшения

1. **Отсутствие файла зависимостей** (requirements.txt/pyproject.toml)
2. **Дублирование кода** в некоторых узлах
3. **Отсутствие документации** в формате docstring
4. **Хардкод значений** в некоторых местах

### Покрытие тестами

**Структура тестов:**

- 15 тестовых файлов
- Unit-тесты для каждого узла графа
- Integration тесты для полных сценариев
- Мокирование внешних API

```python
# test/test_full_scenario.py
class TestFullScenario(unittest.IsolatedAsyncioTestCase):
    @patch("src.agent_core.nodes.fetch_cities", new_callable=AsyncMock)
    @patch("src.agent_core.nodes.datetime_parser_tool", new_callable=AsyncMock)
    async def test_scenario_plan_refine_address(self, mock_fetch_cities, mock_datetime_parser):
        # ... тест полного сценария
```

## ⚙️ Ключевые узлы и графы

### 1. Классификатор намерений (classify_intent_node)

**Назначение:** Определяет тип запроса пользователя (план, фидбек, общение)

```python
async def classify_intent_node(state: AgentState) -> AgentState:
    llm = get_gigachat_client()
    structured_llm = llm.with_structured_output(ClassifiedIntent)

    prompt = f"""
    Ты — высокоточный системный диспетчер.
    Проанализируй запрос: "{user_query}"
    """

    result = await structured_llm.ainvoke(prompt)
    state["classified_intent"] = result
    return state
```

**Входные данные:** `user_message`, `chat_history`
**Выходные данные:** `ClassifiedIntent` с типом намерения

### 2. Извлечение критериев (extract_initial_criteria_node)

**Назначение:** Извлекает и классифицирует критерии поиска из запроса

```python
async def extract_initial_criteria_node(state: AgentState) -> AgentState:
    # Этап 1: Упрощенное извлечение
    simple_extractor = llm.with_structured_output(SimplifiedExtractedInfo)
    simplified_data = await simple_extractor.ainvoke(prompt_extract)

    # Этап 2: Классификация активностей
    for activity_str in activities_list:
        classified_activity = await activity_classifier.ainvoke(prompt_classify)
        ordered_activities.append(OrderedActivityItem(
            activity_type=classified_activity.activity_type,
            query_details=activity_str,
        ))
```

**Входные данные:** `user_message`
**Выходные данные:** `ExtractedInitialInfo` с критериями поиска

### 3. Построитель планов (PlanBuilder)

**Назначение:** Создает оптимальный план мероприятий с учетом времени и маршрутов

```python
class PlanBuilder:
    async def _find_best_plan_for_activities(self, activities_to_plan: List[OrderedActivityItem]):
        # Перебираем все комбинации кандидатов
        for combination in itertools.product(*activity_slots):
            # Проверяем совместимость каждого элемента
            for item_to_add in combination:
                check_result = await self._check_compatibility(last_item_state, item_to_add)
                if check_result["compatible"]:
                    # Добавляем элемент в план
                    current_plan_items.append(activity_dict)
```

**Входные данные:** `cached_candidates`, `search_criteria`
**Выходные данные:** `Plan` с оптимальным маршрутом

### 4. Анализатор фидбека (analyze_feedback_node)

**Назначение:** Анализирует отзывы пользователя и генерирует команды для изменения плана

```python
async def analyze_feedback_node(state: AgentState) -> AgentState:
    # Chain of Thought подход
    prompt = f"""
    <reasoning>
    Пользователь выразил четыре намерения...
    </reasoning>
    <commands>
    modify;MOVIE;start_time;LESS_THAN;None;1 час
    delete;PARK;None;None;None;None
    </commands>
    """

    # Парсинг команд
    command_text_match = re.search(r"<commands>(.*?)</commands>", response_text, re.DOTALL)
    # ... обработка команд
```

**Входные данные:** `user_message`, `current_plan`
**Выходные данные:** `command_queue` с командами для выполнения

### 5. Презентер результатов (presenter_node)

**Назначение:** Форматирует и представляет план пользователю

```python
async def presenter_node(state: AgentState) -> AgentState:
    if not user_start_address:
        # Запрос адреса для расчета маршрута
        state["is_awaiting_start_address"] = True
        response_text = await llm.ainvoke(prompt)
    else:
        # Показ итогового плана с маршрутом
        response_text = await llm.ainvoke(prompt_with_route)

    state["chat_history"].append(AIMessage(content=response_text))
    return state
```

## 📋 Выводы и рекомендации

### Уровень сложности: **Middle-Senior**

Проект демонстрирует высокий уровень архитектурной зрелости с использованием современных паттернов AI-разработки.

### Сильные стороны

1. **Современная архитектура:** LangGraph + LangChain для построения AI-агентов
2. **Качественная типизация:** Pydantic v2 с полной валидацией данных
3. **Асинхронность:** Эффективная обработка множественных API-запросов
4. **Тестируемость:** Хорошее покрытие тестами с мокированием
5. **Модульность:** Четкое разделение ответственности между компонентами

### Рекомендации по улучшению

1. **Добавить файл зависимостей** (requirements.txt или pyproject.toml)
2. **Улучшить документацию** - добавить docstrings в формате Google/NumPy
3. **Рефакторинг дублирования** - вынести общую логику в утилиты
4. **Добавить конфигурацию** - вынести хардкод в конфигурационные файлы
5. **Улучшить обработку ошибок** - добавить retry механизмы для внешних API
6. **Добавить мониторинг** - метрики производительности и использования

### Потенциал развития

Проект имеет отличную основу для масштабирования:

- Добавление новых типов активностей
- Интеграция с дополнительными API
- Улучшение алгоритмов оптимизации маршрутов
- Добавление машинного обучения для персонализации

**Общая оценка:** 8.5/10 - Высококачественный проект с современной архитектурой и хорошими практиками разработки.
