from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

from .gpt4v_model import GPT4VModel

load_dotenv()
logger = logging.getLogger(__name__)


class Flash2V(GPT4VModel):
    def _create_client(self):
        # Using same key for Flash2 and Flash2V for simplicity, adjust if needed
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning(
                "GEMINI_API_KEY environment variable not set. Flash2V may not work.",
            )
        return OpenAI(api_key=api_key, base_url=self.api_base)
