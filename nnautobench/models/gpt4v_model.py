from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

from nnautobench.models.qwen2_model import Qwen2Model

load_dotenv()
logger = logging.getLogger(__name__)


class GPT4VModel(Qwen2Model):
    def _create_client(self):
        # Using generic OpenAI key as GPT4V is also OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning(
                "OPENAI_API_KEY environment variable not set. GPT4V may not work.",
            )
        return OpenAI(api_key=api_key, base_url=self.api_base)
