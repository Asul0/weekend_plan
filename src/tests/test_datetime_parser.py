# tests/test_datetime_parser.py

import unittest
import asyncio
from datetime import datetime, time
from unittest.mock import patch, MagicMock

# Импортируем тестируемый инструмент и его внутренние компоненты
from src.tools.datetime_parser_tool import datetime_parser_tool, TimeRangeResult


class TestDateTimeParserTool(unittest.TestCase):
    """
    Набор модульных тестов для datetime_parser_tool.
    Тестирует как Python-логику, так и взаимодействие с LLM (через моки).
    """

    def test_simple_relative_date(self):
        """Проверяет простые относительные даты без времени."""

        async def run():
            # Запускаем без time_qualifier
            result = await datetime_parser_tool.ainvoke(
                {"natural_language_date": "завтра"}
            )

            self.assertFalse(result["is_ambiguous"])
            self.assertIsNotNone(result["datetime_iso"])

            # Проверяем, что время установлено на начало дня
            parsed_dt = datetime.fromisoformat(result["datetime_iso"])
            self.assertEqual(parsed_dt.time(), time(0, 0))

        asyncio.run(run())

    def test_time_only_qualifier(self):
        """Проверяет простое указание времени (без даты), мокируя ответ LLM."""

        async def run():
            # 1. Arrange: Готовим мок-ответ от LLM
            mock_llm_response = TimeRangeResult(start_hour=18, start_minute=0)

            # 2. Act: Патчим LLM-функцию и вызываем инструмент
            with patch(
                "src.tools.datetime_parser_tool._parse_time_with_llm_flexible",
                new_callable=MagicMock,
            ) as mock_parse_time:
                # Настраиваем мок, чтобы он возвращал наш подготовленный ответ
                mock_parse_time.return_value = asyncio.Future()
                mock_parse_time.return_value.set_result(mock_llm_response)

                result = await datetime_parser_tool.ainvoke(
                    {
                        "natural_language_date": "сегодня",
                        "natural_language_time_qualifier": "вечером",
                    }
                )

            # 3. Assert
            self.assertFalse(result["is_ambiguous"])
            self.assertIsNotNone(result["datetime_iso"])
            parsed_dt = datetime.fromisoformat(result["datetime_iso"])
            self.assertEqual(parsed_dt.time(), time(18, 0))
            # Конец дня не должен быть установлен
            self.assertIsNone(result["end_datetime_iso"])

        asyncio.run(run())

    def test_date_with_after_time_qualifier(self):
        """
        ГЛАВНЫЙ ТЕСТ: Проверяет наш проблемный случай "завтра после 15:00".
        """

        async def run():
            # 1. Arrange: Мокируем LLM, чтобы она вернула 'is_after_indicator'=True
            mock_llm_response = TimeRangeResult(
                start_hour=15, start_minute=0, is_after_indicator=True
            )

            # 2. Act
            with patch(
                "src.tools.datetime_parser_tool._parse_time_with_llm_flexible",
                new_callable=MagicMock,
            ) as mock_parse_time:
                mock_parse_time.return_value = asyncio.Future()
                mock_parse_time.return_value.set_result(mock_llm_response)

                result = await datetime_parser_tool.ainvoke(
                    {
                        "natural_language_date": "завтра",
                        "natural_language_time_qualifier": "после 15:00",
                    }
                )

            # 3. Assert
            self.assertFalse(result["is_ambiguous"])
            self.assertIsNotNone(result["datetime_iso"])
            self.assertIsNotNone(
                result["end_datetime_iso"],
                "End datetime должен быть установлен для 'после'",
            )

            start_dt = datetime.fromisoformat(result["datetime_iso"])
            end_dt = datetime.fromisoformat(result["end_datetime_iso"])

            # Проверяем, что время начала и конца правильные
            self.assertEqual(start_dt.time(), time(15, 0))
            self.assertEqual(end_dt.time(), time(23, 59, 59))

        asyncio.run(run())

    def test_full_time_range(self):
        """Проверяет полный диапазон времени, например, 'с 10 до 18'."""

        async def run():
            # 1. Arrange
            mock_llm_response = TimeRangeResult(
                start_hour=10,
                start_minute=0,
                end_hour=18,
                end_minute=0,
                is_range_indicator=True,
            )

            # 2. Act
            with patch(
                "src.tools.datetime_parser_tool._parse_time_with_llm_flexible",
                new_callable=MagicMock,
            ) as mock_parse_time:
                mock_parse_time.return_value = asyncio.Future()
                mock_parse_time.return_value.set_result(mock_llm_response)

                result = await datetime_parser_tool.ainvoke(
                    {
                        "natural_language_date": "сегодня",
                        "natural_language_time_qualifier": "с 10 до 18",
                    }
                )

            # 3. Assert
            self.assertFalse(result["is_ambiguous"])
            self.assertIsNotNone(result["datetime_iso"])
            self.assertIsNotNone(result["end_datetime_iso"])

            start_dt = datetime.fromisoformat(result["datetime_iso"])
            end_dt = datetime.fromisoformat(result["end_datetime_iso"])

            self.assertEqual(start_dt.time(), time(10, 0))
            self.assertEqual(end_dt.time(), time(18, 0))

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
