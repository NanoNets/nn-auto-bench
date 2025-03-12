from __future__ import annotations

import json
import logging
import os
from abc import ABC
from abc import abstractmethod

import backoff
import dotenv
import openai
from openai import OpenAI

from nnautobench.utils.common_utils import clean_gpt_response
from nnautobench.utils.conf_score_prompts import get_conf_score_prob_prompt
from nnautobench.utils.conf_score_prompts import get_conf_score_yes_no_prompt
from nnautobench.utils.prompt_utils import create_field_extraction_prompt

dotenv.load_dotenv()

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    def __init__(self, model_name, api_base):
        self.model_name = model_name
        self.api_base = api_base
        self.client = self._create_client()

    @backoff.on_exception(backoff.expo, openai.RateLimitError)
    def completions_with_backoff(self, **kwargs):
        # logger.info({key:val for key, val in kwargs.items() if key != 'messages'})

        if (
            self.model_name == "gemini-2.0-flash"
            or self.model_name == "mistral-large-latest"
        ):
            if "top_logprobs" in kwargs:
                kwargs.pop("top_logprobs")
            if "logprobs" in kwargs:
                kwargs.pop("logprobs")
        else:
            kwargs["seed"] = 42
        return self.client.chat.completions.create(**kwargs)

    def _create_client(self):
        api_key = os.getenv("BASE_API_KEY")  # Generic API Key
        if not api_key:
            logger.warning(
                "OPENAI_API_KEY environment variable not set. API calls may fail.",
            )
            api_key = "EMPTY"  # Fallback to "EMPTY" if no key is set

        return OpenAI(api_key=api_key, base_url=self.api_base)

    def create_prompt(self, fields, descriptions=None):
        return create_field_extraction_prompt(fields, descriptions)

    def score_conf_prob_score(self, messages, answer, **kwargs):
        # returns confidence score for each field
        prompt = get_conf_score_prob_prompt()
        messages.append({"role": "assistant", "content": answer})
        messages.append({"role": "user", "content": prompt})
        conf_score_response = self.completions_with_backoff(
            model=self.model_name,
            messages=messages,
            temperature=0,
            max_tokens=3000,
            response_format={"type": "json_object"},
        )
        conf_score_response = json.loads(
            conf_score_response.choices[0].message.content,
        )
        return conf_score_response

    def score_conf_yes_no_score(self, messages, answer, **kwargs):
        # returns 1 if the field is correct and 0 if it is incorrect
        prompt = get_conf_score_yes_no_prompt()
        messages.append({"role": "assistant", "content": answer})
        messages.append({"role": "user", "content": prompt})
        conf_score_response = self.completions_with_backoff(
            model=self.model_name,
            messages=messages,
            temperature=0,
            max_tokens=3000,
            response_format={"type": "json_object"},
        )
        conf_score_response = self.completions_with_backoff(
            model=self.model_name,
            messages=messages,
            temperature=0,
            max_tokens=3000,
            response_format={"type": "json_object"},
        )
        conf_score_response = json.loads(
            conf_score_response.choices[0].message.content,
        )
        return conf_score_response

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
                print(
                    f"error in parsing choice.message.content in  get_consistency_conf_score",
                )
                continue
            for key, field_ans in answer.items():
                try:
                    if key not in consistency_dict:
                        consistency_dict[key] = []
                    consistency_dict[key].append(str(field_ans["value"]))
                except:
                    consistency_dict[key].append("")  # unparsable format
        conf_score = {
            key: (
                1
                if len(
                    set(answers),
                )
                == 1
                else 0
            )
            for key, answers in consistency_dict.items()
        }
        return conf_score

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
                    f"This is a proprietary algorithm, share the model with nanonets if you want to evaluate with this method",
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
            print(e)
            return {}

    def predict(self, messages, conf_score_method):
        conf_score = {}
        try:
            response = self.completions_with_backoff(
                model=self.model_name,
                messages=messages,
                temperature=1.25 if conf_score_method == "consistency" else 0,
                max_tokens=3000,
                logprobs=True,
                top_logprobs=2,
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

        return response.choices[0].message.content, response.usage.dict(), conf_score

    def post_process(self, content):
        return clean_gpt_response(content)
