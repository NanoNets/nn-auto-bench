from __future__ import annotations

import logging
import re
from typing import Union

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    try:
        return (
            re.sub(r"\s", " ", re.sub(r"\s{2,}", " ", text)).strip().replace(" / ", " ")
        )
    except:
        logger.debug(text)
        return text


def clean_string(value):
    # if its a currency value remove the common prefix
    value = re.sub(r"[$€£¥₹₩₽₨₦₫₴]", "", value)
    value = (
        value.replace(",", "")
        .replace(
            "USD",
            "",
        )
        .replace("EUR ", "")
        .replace("RM", "")
    )
    return value.strip()


def is_zero_tax(tax_value: str | float):
    if isinstance(tax_value, float):
        return tax_value == 0.0
    if isinstance(tax_value, str):
        try:
            float_value = float(tax_value)
            return float_value == 0.0
        except ValueError:
            return False
    return False


def compute_conf_score_approval_and_precision(
    predicted_field_conf_scores: dict,
    gt_ans: dict,
    pred_ans: dict,
    threshold: float = 0.99,
    print_incorrect: bool = False,
):
    for key, value in predicted_field_conf_scores.items():
        if isinstance(value, float):
            predicted_field_conf_scores[key] = value
        elif isinstance(value, dict):
            predicted_field_conf_scores[key] = value.get("confidence", 0.0)
        else:
            try:
                predicted_field_conf_scores[key] = float(value)
            except:
                predicted_field_conf_scores[key] = 0.0

    gt_ans = gt_ans["fields"]
    if isinstance(gt_ans, list):
        gt_ans = gt_ans[0]
    gt_ans = {key: str(ans["value"]) for key, ans in gt_ans.items()}
    pred_ans = {key: str(ans["value"]) for key, ans in pred_ans.items()}
    total_correct_approved = 0
    total_incorrect_approved = 0
    all_keys = set(list(gt_ans.keys()) + list(pred_ans.keys()))
    for field_name in all_keys:
        if field_name not in gt_ans:
            gt_ans[field_name] = ""
        if field_name not in pred_ans:
            pred_ans[field_name] = ""
        if "amount" in field_name.lower() or "total_tax" in field_name.lower():
            # remove any currency symbols
            gt_ans[field_name] = clean_string(gt_ans[field_name])
            pred_ans[field_name] = clean_string(pred_ans[field_name])
        if "date" in field_name.lower():
            pred_ans[field_name] = (
                pred_ans[field_name]
                .replace(
                    " ",
                    "",
                )
                .replace("/", "-")
                .replace(".", "-")
                .rstrip("-")
            )
            gt_ans[field_name] = (
                gt_ans[field_name]
                .replace(
                    " ",
                    "",
                )
                .replace("/", "-")
                .replace(".", "-")
                .rstrip("-")
            )
        if "currency" in field_name.lower():
            pred_ans[field_name] = (
                pred_ans[field_name]
                .replace(
                    "DEM",
                    "DM",
                )
                .replace("U. S. DOLLARS", "$")
                .replace("US Dollars", "$")
            )
            gt_ans[field_name] = (
                gt_ans[field_name]
                .replace("DEM", "DM")
                .replace(
                    "U. S. DOLLARS",
                    "$",
                )
                .replace("US Dollars", "$")
            )
        if "total_tax" in field_name.lower():
            # if the tax is 0, gt annotations does not have this sometimes
            gt_ans[field_name] = (
                ""
                if is_zero_tax(
                    gt_ans[field_name],
                )
                else gt_ans[field_name]
            )
            pred_ans[field_name] = (
                ""
                if is_zero_tax(
                    pred_ans[field_name],
                )
                else pred_ans[field_name]
            )

        predicted_conf_score = predicted_field_conf_scores.get(field_name, 0.0)
        # print(normalize_text(gt_ans[field_name]).lower(), normalize_text(pred_ans[field_name]).lower(), predicted_conf_score)
        # correct and approved
        if (
            normalize_text(gt_ans[field_name]).lower()
            == normalize_text(pred_ans[field_name]).lower()
            and predicted_conf_score >= threshold
        ):
            total_correct_approved += 1
        # incorrect and approved
        elif (
            normalize_text(gt_ans[field_name]).lower()
            != normalize_text(pred_ans[field_name]).lower()
            and predicted_conf_score >= threshold
        ):
            if print_incorrect:
                print(field_name, predicted_conf_score)
            total_incorrect_approved += 1
    return total_correct_approved, total_incorrect_approved


def get_conf_score_prob_prompt():
    return "Please provide the confidence score for each of the fields from the above answer in the following JSON format: {'field_name': 'confidence_score'}. The confidence score should be a number between 0 and 1."


def get_conf_score_yes_no_prompt():
    return "For each field mentioned in the above answer, return 1 if the field is correct and 0 if it is incorrect. Return the result in the following JSON format: {'field_name': '1/0'}"
