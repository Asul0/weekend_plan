# Файл: src/utils/callbacks.py

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)


class TokenUsageCallbackHandler(BaseCallbackHandler):
    """Простой Callback Handler для логирования использования токенов."""

    def __init__(self, node_name: str = "UnknownNode"):
        super().__init__()
        self.node_name = node_name
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Вызывается после каждого завершения работы LLM."""
        token_usage = response.llm_output.get("token_usage", {})
        if token_usage:
            prompt_tokens = token_usage.get("prompt_tokens", 0)
            completion_tokens = token_usage.get("completion_tokens", 0)
            total_tokens = token_usage.get("total_tokens", 0)

            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_tokens += total_tokens

            logger.info(
                f"TOKEN_USAGE ({self.node_name}): "
                f"Prompt Tokens: {prompt_tokens}, "
                f"Completion Tokens: {completion_tokens}, "
                f"Total Tokens: {total_tokens}"
            )
