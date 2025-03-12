from __future__ import annotations

import ast
import json
from typing import Any


def create_field_extraction_prompt(fields, descriptions=None, multiple_images=False):
    if isinstance(fields, str):
        fields = ast.literal_eval(fields)
    if descriptions is None:
        field_instructions = "\n".join(
            f"{i+1}. {field}" for i, field in enumerate(fields)
        )
    else:
        field_instructions = ""
        for i, field in enumerate(fields):
            label_string = field.replace("\n", " ")
            field_instructions += f"\n{i+1}. {label_string}"
            if field in descriptions:
                field_instructions += f": {descriptions[field]}"
        # field_instructions = "\n".join(f"{i+1}. {field}" for i, field in enumerate(fields))
    json_dict = {field: {"value": ".."} for field in fields}

    prompt = f""""Follow the below instructions and extract field(s) from the document image{"" if not multiple_images else "s"} and if value is not present for a field then \"\" should be provided. If there are more than 1 value for a field, give all the values as an array.
Extract the following fields from this document

{field_instructions}

The output should be formatted as a flattened JSON output. Do not give any additional explanation.{"If there are multiple images, DO NOT give a list of dictionary for each image, consider all images and extract one single json dictionary." if multiple_images else ""}
OUTPUT JSON FORMAT
{json.dumps(json_dict)}
"""

    return prompt


def create_field_extraction_prompt_ocr(
    fields,
    descriptions=None,
    disable_output_format=False,
    ocr_text="",
    **kwargs,
):
    if isinstance(fields, str):
        fields = ast.literal_eval(fields)
    if descriptions:
        label_instructions = "\n".join(
            f"{i+1}. {col}: {descriptions[col] if col in descriptions else ''}"
            for i, col in enumerate(fields)
        )
    else:
        label_instructions = "\n".join(f"{i+1}. {col}" for i, col in enumerate(fields))
    json_dict = {field: {"value": ".."} for field in fields}
    prompt = f"""Consider the following document text
--------------------------------
{ocr_text}
--------------------------------
From this document, extract the following fields

{label_instructions}

The output should be formatted as the flattened JSON format. If value is not present for a field then \"\" should be provided. If there are more than 1 value for a field, give all the values as an array. Do not give any additional explanation
"""
    if not disable_output_format:
        prompt += f"""OUTPUT JSON FORMAT
{json.dumps(json_dict)}
    """
    return prompt


def create_ocr_prompt():
    prompt = f"""You are an OCR system. Your task is to extract all the ocr text from the document image in reading order. Extract the ocr text from the image provided."""
    return prompt


def create_number_count_prompt():
    prompt = f"""Follow the instructions below and perform number counting
    Ignore the image and Print all integers from 1 to 500. eg: 1,2,3 ... 499,500.
    Do not skip any number.
    Do not give any additional explanation. Print all numbers from 1 till 500.
    """
    return prompt


def messages_to_string(msg):
    role = msg["role"]
    content = msg["content"]

    if isinstance(content, str):
        return f"""role: {role}\ncontent: {content}"""
    content_strings = []
    for item in content:
        # if item['type'] == 'text':
        if "text" in item:
            content_strings.append(item["text"])
        elif item["type"] == "image_url":
            image_tokens_repr = (
                "<|image_pad|>"  # * 1280 #1280 = max num of visual tokens
            )
            content_strings.append(image_tokens_repr)
    content_str = " ".join(content_strings)
    return f"""role: {role}\ncontent: {content_str}"""


def get_prompt_string(messages) -> str:
    string_messages_list = []
    for message in messages:
        string_messages_list.append(messages_to_string(message))
    return "\n\n".join(string_messages_list)


def get_sample_output(fields, annotations) -> dict:
    """
    {
        "invoice_number": "JB120",
        "amount": ["110", "120"],
        "date": "21-10-1876",
        "buyer_address": "Street 1, Random Country"
    }
    """
    json_dict: dict[str, Any] = {}
    try:
        annotations = annotations["fields"]
    except Exception as e:
        print(e)
        print(annotations)
        raise e

    for key, val in annotations.items():
        label = key
        if isinstance(val, list):  # TODO: handle multiple values
            val = val[0]
        value = {"value": val["value"]}
        # if label in fields:# should this condition be removed?
        if label in json_dict:
            if type(json_dict[label]) == list:
                json_dict[label].append(value)
            else:
                json_dict[label] = [json_dict[label], value]
        else:
            json_dict[label] = value
    return json_dict


def create_field_extraction_prompt_text_and_image(
    fields,
    descriptions=None,
    multiple_images=False,
    input_text=None,
):
    if descriptions is None:
        field_instructions = "\n".join(
            f"{i+1}. {field}" for i, field in enumerate(fields)
        )
    else:
        field_instructions = ""
        for i, field in enumerate(fields):
            label_string = field.replace("\n", " ")
            field_instructions += f"\n{i+1}. {label_string}"
            if field in descriptions:
                field_instructions += f": {descriptions[field]}"
        # field_instructions = "\n".join(f"{i+1}. {field}" for i, field in enumerate(fields))
    json_dict = {field: ".." for field in fields}

    prompt = f""""Consider the document image and the following ocr text of the image in reading order.
-------------------
{input_text}
-------------------
Follow the below instructions and extract field(s) from the document image{"" if not multiple_images else "s"} and give priority to image over text for extraction. If value is not present for a field then \"\" should be provided. If there are more than 1 value for a field, give all the values as an array.
Extract the following fields from this document

{field_instructions}

The output should be formatted as a flattened JSON output. Do not give any additional explanation.{"If there are multiple images, DO NOT give a list of dictionary for each image, consider all images and extract one single json dictionary." if multiple_images else ""}
OUTPUT JSON FORMAT
{json.dumps(json_dict)}
"""

    return prompt
