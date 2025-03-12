from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

from .gpt4o_model import GPT4oModel

load_dotenv()
logger = logging.getLogger(__name__)


class Flash2(GPT4oModel):
    def _create_client(self):
        # Using same key for Flash2 and Flash2V for simplicity, adjust if needed
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning(
                "GEMINI_API_KEY environment variable not set. Flash2 may not work.",
            )
        return OpenAI(api_key=api_key, base_url=self.api_base)
