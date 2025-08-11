import logging
import os
import sys
from typing import Dict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from src.agent_core.graph import agent_app
from src.agent_core.state import AgentState


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)


logging.getLogger("src").setLevel(logging.INFO)


logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram.ext").setLevel(logging.WARNING)
logging.getLogger("apscheduler").setLevel(logging.WARNING)
logging.getLogger("dateparser").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


user_states: Dict[int, AgentState] = {}


def get_initial_state() -> AgentState:
    """Создает начальное состояние для нового пользователя."""
    return AgentState(
        chat_history=[],
        user_message="",
        classified_intent=None,
        next_action=None,
        error=None,
        is_awaiting_clarification=None,
        search_criteria=None,
        cached_candidates={},
        current_plan=None,
        plan_builder_result=None,
        analyzed_feedback=None,
        pinned_items={},
        command_queue=[],
        city_id_afisha=None,
        parsed_dates_iso=None,
        parsed_end_dates_iso=None,
        user_start_coordinates=None,
        is_awaiting_address=False,
        status_message_id=None,
        plan_presented=False,
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /start. Сбрасывает состояние пользователя."""
    chat_id = update.effective_chat.id
    if chat_id in user_states:
        del user_states[chat_id]
        logger.info(f"Состояние для чата {chat_id} сброшено по команде /start.")

    await update.message.reply_text(
        "Привет! Я помогу тебе спланировать досуг. 😊\n\n"
        "Напиши, чем бы ты хотел заняться, например: \n"
        "'Хочу сходить в кино и погулять в парке в Москве завтра вечером'."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Основной обработчик всех текстовых сообщений."""
    if not update.message or not update.message.text:
        return

    chat_id = update.effective_chat.id
    user_message_text = update.message.text
    logger.info(f"Получено сообщение от chat_id={chat_id}: '{user_message_text}'")

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    current_state = user_states.get(chat_id)
    if not current_state:
        logger.info(f"Новая сессия для chat_id={chat_id}. Создаю начальное состояние.")
        current_state = get_initial_state()

    current_state["user_message"] = user_message_text

    final_state = None
    try:
        final_state = await agent_app.ainvoke(current_state, {"recursion_limit": 25})
    except Exception as e:
        logger.error(
            f"Критическая ошибка при выполнении графа для chat_id={chat_id}: {e}",
            exc_info=True,
        )
        await update.message.reply_text(
            "Произошла непредвиденная ошибка. Попробуйте начать заново с команды /start."
        )
        if chat_id in user_states:
            del user_states[chat_id]
        return

    response_message = "Извините, я не смог обработать ваш запрос."
    if final_state and final_state.get("chat_history"):
        last_message = final_state["chat_history"][-1]
        if isinstance(last_message, AIMessage):
            response_message = last_message.content
        else:
            logger.warning(
                f"Последнее сообщение в истории было не от AI: {last_message}"
            )
    else:
        logger.error(
            f"Финальное состояние не содержит ответа для пользователя: {final_state}"
        )

    await update.message.reply_text(response_message)

    if final_state:
        user_states[chat_id] = final_state
        logger.info(f"Состояние для чата {chat_id} обновлено.")
    else:
        if chat_id in user_states:
            del user_states[chat_id]


def main() -> None:
    """Запускает бота."""
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not telegram_token:
        logger.critical("Токен TELEGRAM_BOT_TOKEN не найден в .env файле!")
        sys.exit(1)

    application = Application.builder().token(telegram_token).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )

    logger.info("Бот запускается...")
    application.run_polling()
    logger.info("Бот остановлен.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Бот остановлен с критической ошибкой: {e}", exc_info=True)
