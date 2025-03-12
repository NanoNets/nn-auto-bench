from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

from .gpt4o_model import GPT4oModel

logger = logging.getLogger(__name__)


class MistralLarge(GPT4oModel):
    def _create_client(self):
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            logger.warning(
                "MISTRAL_API_KEY environment variable not set. Mistral Large may not work.",
            )
        return OpenAI(api_key=api_key, base_url=self.api_base)
