from __future__ import annotations

import json
import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

from .base_model import BaseModel
from .qwen2_model import Qwen2Model
from nnautobench.utils.conf_score_prompts import get_conf_score_prob_prompt
from nnautobench.utils.conf_score_prompts import get_conf_score_yes_no_prompt
from nnautobench.utils.prompt_utils import create_field_extraction_prompt_ocr
from nnautobench.utils.prompt_utils import get_sample_output

load_dotenv()
logger = logging.getLogger(__name__)


class GPTo3MiniModel(Qwen2Model):
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
        )
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
                )
                messages.append({"role": "user", "content": sample_prompt})
                messages.append({"role": "assistant", "content": answer})
            messages.append({"role": "user", "content": question})
        return messages

    def score_conf_prob_score(self, messages, answer, **kwargs):
        return self._score_conf_score(messages, answer, get_conf_score_prob_prompt())

    def score_conf_yes_no_score(self, messages, answer, **kwargs):
        return self._score_conf_score(messages, answer, get_conf_score_yes_no_prompt())

    def _score_conf_score(self, messages, answer, prompt):
        messages.append({"role": "assistant", "content": answer})
        messages.append({"role": "user", "content": prompt})
        conf_score_response = self.completions_with_backoff(
            model=self.model_name,
            messages=messages,
            reasoning_effort="low",
            response_format={"type": "json_object"},
        )
        return json.loads(conf_score_response.choices[0].message.content)

    def get_consistency_conf_score(
        self,
        messages,
        answer,
        choices,
        parsed_answer,
        **kwargs,
    ):
        consistency_dict = {}
        for choice in choices:
            answer, is_parsable = self.post_process(choice.message.content)
            if not is_parsable:
                logger.error(
                    "Error in parsing choice.message.content in get_consistency_conf_score",
                )
                continue
            for key, field_ans in answer.items():
                consistency_dict.setdefault(key, []).append(
                    str(field_ans.get("value", "")),
                )
        return {
            key: 1 if len(set(answers)) == 1 else 0
            for key, answers in consistency_dict.items()
        }

    def get_conf_score(
        self,
        conf_score_method,
        messages,
        choices,
        parsed_answer,
        **kwargs,
    ):
        answer = choices[0].message.content
        try:
            if conf_score_method == "prob":
                return self.score_conf_prob_score(messages, answer, **kwargs)
            elif conf_score_method == "yes_no":
                return self.score_conf_yes_no_score(messages, answer, **kwargs)
            elif conf_score_method == "nanonets":
                print(
                    "This is a proprietary algorithm, share the model with nanonets if you want to evaluate with this method",
                )
                return RuntimeError("Not implemented")
            elif conf_score_method == "consistency":
                return self.get_consistency_conf_score(
                    messages,
                    answer,
                    choices,
                    parsed_answer,
                    **kwargs,
                )
        except Exception as e:
            logger.error(e)
            return {}

    def predict(self, messages, conf_score_method):
        conf_score = {}
        try:
            response = self.completions_with_backoff(
                model=self.model_name,
                messages=messages,
                max_completion_tokens=80000,
                logprobs=False,
                reasoning_effort="medium",
                n=5 if conf_score_method == "consistency" else 1,
            )
            if conf_score_method == "consistency":
                print(f"len(response.choices): {len(response.choices)}")
            parsed_answer, is_parsable = self.post_process(
                response.choices[0].message.content,
            )
            conf_score = self.get_conf_score(
                conf_score_method=conf_score_method,
                messages=messages,
                choices=response.choices,
                parsed_answer=parsed_answer,
            )
        except Exception as e:
            print(e)
            print(messages)
            raise e

        logger.debug(f"conf_score: {conf_score}")
        logger.debug(
            f"response.choices[0].message.content: {response.choices[0].message.content}",
        )
        logger.debug(
            f"Completion Token Details: {response.usage.completion_tokens_details}",
        )

        return response.choices[0].message.content, response.usage.dict(), conf_score
