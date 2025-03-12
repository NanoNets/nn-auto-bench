from __future__ import annotations

import ast
import json
import re
from json import JSONDecodeError

import pandas as pd


def fix_malformed_json(json_str):

    for index, i in enumerate(json_str):
        if i == "{" or i == "[":
            index2 = index
            break
    json_str = json_str[index2:]

    # {AAAAA}1111 -> {AAAAA}
    for index, i in enumerate(reversed(json_str)):
        if i == "}" or i == "]":
            index2 = len(json_str) - index
            break
    json_str = json_str[:index2]

    json_str = re.sub(r'\\(?![/bfnrtu"])', "", json_str)
    json_str = re.sub(r"\s+", " ", json_str)
    json_str = re.sub(r",\s*(}|\])", r"\1", json_str)
    json_str = re.sub(r":\s*,", ": null,", json_str)

    stack = []
    result = ""
    quote_counter = 0

    quotes_after_comma = 0
    found_comma = False

    for index, i in enumerate(json_str):
        if i == '"':
            if json_str[index - 1] != "\\":
                quote_counter += 1
                if found_comma:
                    quotes_after_comma += 1
                    if quotes_after_comma == 2:
                        found_comma = False
                        quotes_after_comma = 0

        if i in ["{", "[", '"']:
            if i == '"' and stack and stack[-1] == '"':
                stack.pop()
            else:
                stack.append(i)
            result += i
        elif i in ["}", "]"]:
            if stack and (
                i == "}" and stack[-1] == "{" or i == "]" and stack[-1] == "["
            ):
                stack.pop()
            result += i
        elif i == ",":
            found_comma = True
            result += i
        else:
            result += i

    while quote_counter % 2 != 0 or quote_counter % 4 != 0:

        result += '"'
        quote_counter += 1

    if found_comma:
        for index, i in enumerate(reversed(result)):
            if i == ",":
                index2 = len(result) - index
                result = result[: index2 - 1] + result[index2:]
                break

    for i in reversed(stack):
        if i == "{":
            result += "}"
        elif i == "[":
            result += "]"

    return result


def fix_json(malformed_json):
    try:
        last_pos = malformed_json.rfind("}")
        cropped_malformed_json = malformed_json
        if last_pos != -1:
            # Remove text after last }
            cropped_malformed_json = malformed_json[: last_pos + 1]
        open_count = cropped_malformed_json.count("{")
        closed_count = cropped_malformed_json.count("}")
        if closed_count > open_count:
            new_malformed_json = cropped_malformed_json[:-1]
            valid_json = json.loads(new_malformed_json)
        else:
            valid_json = json.loads(cropped_malformed_json)
        return valid_json
    except Exception as e:
        print(f"Error in fix_json: {e}")
        fixed_json_str = fix_malformed_json(malformed_json)
        data = json.loads(fixed_json_str)
        return data


def remove_extra_chars(gpt_response: str) -> str:

    # Handles the case where internal llm predicting multiple jsons instead of 1 json when sample output format is not provided in the prompt
    pattern = re.compile(r'{"table#\d+')
    matches = pattern.finditer(gpt_response)
    indices = [match.start() for match in matches]
    if len(indices) > 1:
        gpt_response = gpt_response[indices[0] : indices[1]]

    # Remove extra characters at the start and end

    # Find the index of the first occurrence of '{'
    start_index = gpt_response.find("{")

    # Find the index of the last occurrence of '}'
    end_index = gpt_response.rfind("}")

    # Check if both characters are found
    if start_index != -1 and end_index != -1:
        gpt_response = gpt_response[start_index : end_index + 1]

    # regex = r"{.*}"
    # matches = re.findall(regex, gpt_response, re.MULTILINE)
    # if len(matches)==1:
    #     gpt_response=matches[0]

    # Remove extra backslash characters only if they are not new line or Unicode escape sequence (eg: '{\\"delta_z\\": {\\"value\\": \\"192.474\\\\u03bcm\\", \\"line_number\\": [16]}}')
    gpt_response = re.sub(r"\\(?![nu])", "", gpt_response)
    # Removes extra comma before closing braces
    gpt_response = re.sub(r",(\n*?)(\s*?)}", "}", gpt_response)
    # This regex helps find the unescaped inch symbols in the string
    inch_pattern = re.compile(r'(?<=(\d)[\s|\s\s|\s\s\s|])"(?!( )*[}:,])')
    gpt_response = inch_pattern.sub(r'\\"', gpt_response)  # Escape inch symbol

    # # remove ```json pattern
    start_pattern = r"^```json"
    end_pattern = r"```$"
    gpt_response = re.sub(start_pattern, "", gpt_response)
    gpt_response = re.sub(end_pattern, "", gpt_response)

    return gpt_response


def get_json(gpt_response: str) -> dict:
    gpt_response = remove_extra_chars(gpt_response)
    # print(f"after removing extra chars:\n{gpt_response}")
    try:
        json_gpt_response = json.loads(gpt_response, strict=False)
        return json_gpt_response
    except JSONDecodeError as e:
        # llama vision 3.2
        if "Expecting property name enclosed in double quotes" in str(e):
            try:
                parsed_dict = ast.literal_eval(gpt_response)
                print("Successfully parsed after handling the error.")
                return parsed_dict
            except Exception as inner_e:
                print(f"Error while handling: {inner_e}")
                raise
        else:
            raise


def clean_gpt_response(
    gpt_response: str,
    regex=None,
    extraction_type="",
) -> tuple[dict, bool]:
    """
    Clean and parse GPT response into JSON format.

    Args:
        gpt_response (str): Raw GPT response to clean
        regex (Optional): Regular expression pattern for cleaning
        extraction_type (str, optional): Type of extraction being performed

    Returns:
        tuple[dict, bool]: Tuple containing (parsed JSON dict, success flag)
    """
    try:
        json_gpt_response = get_json(gpt_response)
        return json_gpt_response, True
    except Exception as json_e:  # pylint: disable=broad-except
        extra = {"exception": str(json_e), "gpt_response": gpt_response}
        print(
            f"Extraction_Type:{extraction_type} JSON clean step 1, parsing json unsuccessful, trying more clean functions, {extra}",
        )
        try:
            if gpt_response.endswith("{}") and gpt_response != "{}":
                # Internal table llm small can have response like '{"table#1": {"columns": ["col1", "col2"], "data": [{"row_data": ["val1", "val2"], "line_number": ["1"]}]}}{}{}'
                gpt_response = gpt_response.replace("{}", "")
            gpt_response = remove_extra_chars(gpt_response)
            # print(f"after removing extra chars again:\n{gpt_response}")
            json_gpt_response = fix_json(gpt_response)
        except Exception as fix_json_e:  # pylint: disable=broad-except
            extra = {"exception": str(fix_json_e)}
            print(
                f"Extraction_Type:{extraction_type} JSON clean step 2, fix_json unsuccessful, trying more clean functions, {extra}",
            )
            # fields sometimes has outputs like
            # op = {"invoice_amount": {"value": ["34.56"], "line_number": [32]}, "invoice_date": {"value": "07/15/2024"], "line_number": [3]}}
            # This does not work with fix_json, so we need to remove the extra square brackets, for now this only comes for ending square brackets
            # if extraction_type == "Fields":
            try:
                indices = [
                    i
                    for i in range(
                        len(gpt_response),
                    )
                    if gpt_response.startswith("],", i)
                ]
                for index in indices:
                    part_string = gpt_response[: index + 1]
                    # check the last occurance of string value in the part_string
                    last_occ = part_string.rfind('"value":')
                    if last_occ == -1:
                        continue  # if not found, continue
                    value_field = part_string[last_occ + len('"value":') + 1 :]
                    if "[" not in value_field:
                        # remove the extra square bracket
                        gpt_response = gpt_response[:index] + gpt_response[index + 1 :]

                gpt_response = remove_extra_chars(gpt_response)
                # print(f"after removing extra chars again 3:\n{gpt_response}")
                json_gpt_response = fix_json(gpt_response)
                return json_gpt_response, True
            except Exception as fix_sq_br:  # pylint: disable=broad-except
                extra = {"exception": str(fix_sq_br)}
                print(
                    f"Extraction_Type:{extraction_type} JSON clean step 3, fix_json unsuccessful, returning empty dict, {extra}",
                )
            return {}, False
        return json_gpt_response, True


def convert_fields2str(fields):
    if isinstance(fields, str):
        return fields
    try:
        for field, value in fields.items():
            if isinstance(value, dict):
                if "value" in value:
                    fields[field]["value"] = str(fields[field]["value"])
                if "line_number" in value:
                    fields[field]["line_number"] = [
                        str(line) for line in fields[field]["line_number"]
                    ]
            elif isinstance(value, list):
                for item in value:
                    if "value" in item:
                        item["value"] = str(item["value"])
                    if "line_number" in item:
                        item["line_number"] = [
                            str(line) for line in item["line_number"]
                        ]
    except:
        print(f"Error in converting to str {fields}")
    return fields


def load_data(file_path):
    df = pd.read_json(file_path, orient="records", lines=True)
    return df


def filter_data(df, categories):
    df_filtered = df[df.category.isin(categories)].copy()
    df_filtered = df_filtered[df_filtered.s3_path_exists == True]
    return df_filtered


symbols_to_trim = [
    " ,",
    " %",
    " - ",
    " .",
    " : ",
    "( ",
    " )",
    "[ ",
    " ]",
    "{ ",
    " }",
    " ;",
    "€ ",
    "£ ",
    "₹ ",
    "$ ",
    " ?",
    " !",
    " / ",
]


def remove_erroneous_spaces(text, symbols_to_trim=symbols_to_trim):
    for sym in symbols_to_trim:
        text = text.replace(sym, sym.strip())
    return text


def preprocess_cell_text(cell_text):
    if "LO 5" in cell_text:
        cell_text = cell_text.replace("LO 5", "5")
    cell_text = remove_erroneous_spaces(cell_text)
    return cell_text
