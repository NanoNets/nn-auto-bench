from __future__ import annotations

import json
import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

from .base_model import BaseModel
from .qwen2_model import Qwen2Model
from nnautobench.utils.prompt_utils import create_field_extraction_prompt_ocr
from nnautobench.utils.prompt_utils import get_sample_output

load_dotenv()
logger = logging.getLogger(__name__)


class GPT4oModel(Qwen2Model):
    def __init__(self, model_name, api_base):
        self.model_name = model_name
        self.api_base = api_base
        self.client = self._create_client()

    def _create_client(self):
        # Using generic OpenAI key as GPT4o is also OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning(
                "OPENAI_API_KEY environment variable not set. GPT4o may not work.",
            )
        return OpenAI(api_key=api_key, base_url=self.api_base)

    def create_prompt(
        self,
        fields,
        descriptions=None,
        image_paths=None,
        ctx=[],
        input_text="",
        layout="vision_default",
    ):
        actual_few_shot = len(ctx)
        if actual_few_shot > 0 and not ctx[0]["accepted"]:
            actual_few_shot = 0
        question = create_field_extraction_prompt_ocr(
            fields,
            descriptions,
            disable_output_format=False,
            ocr_text=input_text,
        )  # change text_textract_table
        if actual_few_shot == 0:
            # zeroshot
            messages = [
                {
                    "role": "user",
                    "content": question,
                },
            ]
        else:
            messages = []
            for i in range(actual_few_shot):

                sample_output = get_sample_output(
                    fields,
                    ctx[i]["accepted"],
                )
                answer = f"{json.dumps(sample_output, ensure_ascii=False)}"
                sample_prompt = create_field_extraction_prompt_ocr(
                    fields,
                    descriptions,
                    disable_output_format=False,
                    ocr_text=ctx[i]["text"],
                )  # change textract_table
                messages.append({"role": "user", "content": sample_prompt})
                messages.append({"role": "assistant", "content": answer})
            messages.append({"role": "user", "content": question})
        return messages
