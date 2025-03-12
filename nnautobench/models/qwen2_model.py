from __future__ import annotations

import json
import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

from nnautobench.models.base_model import BaseModel
from nnautobench.utils.image_utils import encode_image_base64
from nnautobench.utils.prompt_utils import create_field_extraction_prompt
from nnautobench.utils.prompt_utils import get_sample_output

load_dotenv()
logger = logging.getLogger(__name__)


class Qwen2Model(BaseModel):
    def __init__(self, model_name, api_base):
        self.model_name = model_name
        self.api_base = api_base
        self.client = self._create_client()

    def _create_client(self):
        api_key = os.getenv("QWEN2_API_KEY")  # Specific API key for Qwen2
        if not api_key:
            logger.warning(
                "QWEN2_API_KEY environment variable not set. Qwen2 may not work.",
            )
        return OpenAI(api_key=api_key, base_url=self.api_base)

    def create_prompt(
        self,
        fields,
        descriptions=None,
        image_paths=None,
        ctx=[],
        input_text=None,
        layout="vision_default",
    ):
        actual_few_shot = len(ctx)
        question = create_field_extraction_prompt(fields, descriptions)
        base64_images = [encode_image_base64(img) for img in image_paths]
        data_urls = [
            f"data:image/jpeg;base64,{base64_image}" for base64_image in base64_images
        ]
        if actual_few_shot == 0:
            # zeroshot
            content = []
            for image_url in data_urls:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                        },
                    },
                )  # vllm
            content.append({"type": "text", "text": question})

            messages = [
                {
                    "role": "user",
                    "content": content,
                },
            ]
        else:
            messages = []
            for i in range(actual_few_shot):
                # for i in range(actual_few_shot - 1, -1, -1):

                sample_output = get_sample_output(
                    fields,
                    ctx[i]["accepted"],
                )
                answer = f"{json.dumps(sample_output, ensure_ascii=False)}"
                sample_base64_image = encode_image_base64(ctx[i]["image_path"])
                sample_image_url = f"data:image/jpeg;base64,{sample_base64_image}"
                content = []
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": sample_image_url,
                        },
                    },
                )
                content.append({"type": "text", "text": question})
                messages.append({"role": "user", "content": content})
                messages.append({"role": "assistant", "content": answer})
            image_url = data_urls[0]
            content2 = []
            content2.append(
                {"type": "image_url", "image_url": {"url": image_url}},
            )
            content2.append({"type": "text", "text": question})
            messages.append({"role": "user", "content": content2})

        return messages
