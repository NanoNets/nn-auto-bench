from __future__ import annotations

import ast
import json
import logging
from time import time

from nnautobench.utils.conf_score_prompts import (
    compute_conf_score_approval_and_precision,
)
from nnautobench.utils.metrics import calculate_metrics
from nnautobench.utils.prompt_utils import get_prompt_string

logger = logging.getLogger(__name__)


class Predictor:
    def __init__(self, model, conf_score_method="prob"):
        self.model = model
        self.conf_score_method = conf_score_method

    def process_single_image(
        self,
        image_path,
        annotation,
        few_shot=0,
        ctx=[],
        input_text=None,
        keys=None,
        layout="vision_default",
    ):

        if isinstance(image_path, str):
            image_paths = [image_path]
        else:
            image_paths = image_path

        actual_few_shot = 0
        if few_shot > 0 and ctx:  # remove redundant checking
            if input_text is not None:
                col = "text"
            else:
                col = "image_path"
            if ctx[0][col] is None:
                actual_few_shot = 0
            elif ctx[few_shot - 1][col]:  # image_path
                actual_few_shot = few_shot
            elif ctx[few_shot - 2][col] and few_shot - 2 >= 0:
                actual_few_shot = few_shot - 1
            elif ctx[few_shot - 3][col] and few_shot - 3 >= 0:
                actual_few_shot = few_shot - 2
            else:
                actual_few_shot = 0

        if actual_few_shot == 0:
            messages = self.model.create_prompt(
                keys,
                descriptions={},
                image_paths=image_paths,
                ctx=[],
                input_text=input_text,
                layout=layout,
            )
        else:
            messages = self.model.create_prompt(
                keys,
                descriptions={},
                image_paths=image_paths,
                ctx=ctx[:actual_few_shot],
                input_text=input_text,
                layout=layout,
            )

        start_time = time()
        content, usage, conf_score = self.model.predict(
            messages,
            self.conf_score_method,
        )
        end_time = time()
        try:
            preds, is_parsable = self.model.post_process(content)

        except json.JSONDecodeError:
            preds = {}
            is_parsable = False
        except Exception as e:
            print(e)
            print("image path", image_path)
            raise
        metrics = calculate_metrics(annotation, preds, keys)
        metrics["pred"] = preds

        if conf_score is not None:
            (
                correct_approved,
                incorrect_approved,
            ) = compute_conf_score_approval_and_precision(
                conf_score,
                annotation,
                preds,
            )

        return {
            "time_taken": end_time - start_time,
            "usage": usage,
            "path": image_path,
            "layout": layout,
            "annotation": annotation,
            "queried_labels": keys,
            "prompt": get_prompt_string(messages),
            "ctx": ctx[0] if actual_few_shot > 0 else {},
            # remove
            "ctx_1_accepted": ctx[0]["accepted"] if actual_few_shot > 0 else {},
            "content": content,
            "actual_few_shot": actual_few_shot,
            "parsing_accuracy": int(is_parsable),
            "predicted_field_conf_scores": conf_score,
            "total_fields": len(ast.literal_eval(keys)),
            "queried_labels": ast.literal_eval(keys),
            "correct_approved": correct_approved,
            "incorrect_approved": incorrect_approved,
            **metrics,
        }
