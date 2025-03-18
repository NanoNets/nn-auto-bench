from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

from .gpt4o_model import GPT4oModel

load_dotenv()
logger = logging.getLogger(__name__)


class Gemma3_27B(GPT4oModel):
    def _create_client(self):
        api_key = os.getenv("GEMMA3_27B_API_KEY")
        if not api_key:
            logger.warning(
                "GEMMA3_27B_API_KEY environment variable not set. Gemma 3 may not work.",
            )
        return OpenAI(api_key=api_key, base_url=self.api_base)
