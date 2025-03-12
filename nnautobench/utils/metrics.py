from __future__ import annotations

import logging

from Levenshtein import distance as edit_distance

logger = logging.getLogger(__name__)


def calculate_field_metrics(annotation, preds, keys=None):
    gt = annotation
    metrics_dict = {}
    tp = 0.0
    fp = 0.0
    fn = 0.0
    total_gt_fields = len(gt)
    total_pred_fields = len(preds) if isinstance(preds, dict) else 0

    tp_strict = 0.0
    fp_strict = 0.0
    fn_strict = 0.0

    if isinstance(preds, dict):
        for key in keys:
            gt_val = gt.get(key, {"value": ""}).get("value", "")
            field_name = key
            gt_val = "" if gt_val is None else gt_val
            gt_val = str(gt_val)
            if field_name in preds or gt_val == "":
                pred_item = preds.get(field_name, {"value": ""})
                if isinstance(pred_item, dict) and "value" in pred_item:
                    pred_val = str(pred_item["value"])
                else:
                    pred_val = str(pred_item)
                if pred_val == gt_val:
                    score = 1.0
                    tp_strict += 1.0
                else:
                    edit_dist = edit_distance(pred_val, gt_val)
                    max_len = max(len(pred_val), len(gt_val))
                    score = 1 - (edit_dist / max_len if max_len > 0 else 1)
                    fp_strict += 1.0
                    fn_strict += 1.0
                tp += score
                fn += 1 - score
                fp += 1 - score
            else:
                fn += 1.0
                fn_strict += 1.0

    # metrics change
    if total_gt_fields > 0:
        field_accuracy = tp / len(keys)
    else:
        field_accuracy = 1.0 if total_pred_fields == 0 else 0.0

    metrics_dict["tp"] = tp
    metrics_dict["fp"] = fp
    metrics_dict["fn"] = fn
    metrics_dict["total_gt_fields"] = total_gt_fields
    metrics_dict["total_pred_fields"] = total_pred_fields
    metrics_dict["file_accuracy"] = field_accuracy

    metrics_dict["tp_strict"] = tp_strict
    metrics_dict["fp_strict"] = fp_strict
    metrics_dict["fn_strict"] = fn_strict

    logger.debug(
        f" tp: {tp}, fp: {fp}, fn: {fn}, tp_strict: {tp_strict}, fp_strict: {fp_strict}, fn_strict: {fn_strict}",
    )
    return metrics_dict


def calculate_metrics(annotation, preds, keys=None):
    gt = annotation.get("fields", {})
    logger.debug(keys)
    return calculate_field_metrics(gt, preds, keys)
