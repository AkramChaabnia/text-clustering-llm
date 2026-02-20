"""
given_label_classification.py — Step 2 of the clustering pipeline.

Given the merged label set produced by label_generation.py, this script:
  1. Loads the dataset.
  2. Reads the merged labels from --output_path.
  3. For each text, asks the LLM to pick the best matching label.
  4. Collects results into a dict {label: [text, ...]} and writes it to JSON.

Outputs one file per run to --output_path:
    <data>_<size>_find_labels.json

Original source: ECNU-Text-Computing/Text-Clustering-via-LLM
Modifications:
  - ini_client() now delegates to openrouter_adapter.make_client()
  - Fixed ini_client() call signature (original passed api_key arg incorrectly)
  - response_format is opt-in via LLM_FORCE_JSON_MODE (default: off)
  - _strip_fenced_json() handles models that wrap JSON in markdown fences
  - chat() has a 5-attempt retry with linear backoff on 429 errors
  - LLM_MODEL env var controls the model
"""

import json
import os
import re
import time
import argparse

from dotenv import load_dotenv

load_dotenv()

_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo-0125")
_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))
_FORCE_JSON_MODE = os.getenv("LLM_FORCE_JSON_MODE", "false").lower() == "true"
_REQUEST_DELAY = float(os.getenv("LLM_REQUEST_DELAY", "0"))


def ini_client():
    from openrouter_adapter import make_client
    return make_client()


def _strip_fenced_json(text: str) -> str:
    """Remove markdown code fences if the model wrapped its JSON output in them."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def chat(prompt, client):
    kwargs = dict(
        model=_MODEL,
        temperature=_TEMPERATURE,
        max_tokens=_MAX_TOKENS,
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": prompt},
        ],
    )
    if _FORCE_JSON_MODE:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(5):
        try:
            completion = client.chat.completions.create(**kwargs)
            raw = completion.choices[0].message.content
            if _REQUEST_DELAY > 0:
                time.sleep(_REQUEST_DELAY)
            return _strip_fenced_json(raw)
        except Exception as e:
            if "429" in str(e) and attempt < 4:
                wait = 20 * (attempt + 1)
                print(f"  [rate limit] attempt {attempt + 1}/5, retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  [api error] {e}")
                return None
    return None


def load_dataset(data_path, data, use_large):
    data_file = os.path.join(
        data_path, data, "large.jsonl" if use_large else "small.jsonl"
    )
    print(f"Loading {data_file}")
    with open(data_file, "r") as f:
        data_list = [json.loads(line) for line in f]
    print(f"  {len(data_list)} samples loaded")
    return data_list


def get_label_list(data_list):
    seen = []
    for item in data_list:
        if item["label"] not in seen:
            seen.append(item["label"])
    return seen


def get_predict_labels(output_path, data):
    data_file = os.path.join(output_path, data + "_small_llm_generated_labels_after_merge.json")
    with open(data_file, "r") as f:
        data_list = json.load(f)
    return list(set(data_list))


def prompt_construct(label_list, sentence):
    json_example = {"label_name": "label"}
    prompt = "Given the label list and the sentence, please categorize the sentence into one of the labels.\n"
    prompt += f"Label list: {label_list}.\n"
    prompt += f"Sentence:{sentence}.\n"
    prompt += f"You should only return the label name, please return in json format like: {json_example}"
    return prompt


def answer_process(response, label_list):
    label = "Unsuccessful"
    try:
        parsed = eval(response)  # noqa: S307
    except Exception:
        parsed = str(response)

    if isinstance(parsed, dict):
        for value in parsed.values():
            if value in label_list:
                return value
    else:
        for candidate in label_list:
            if candidate in parsed:
                return candidate

    return label


def known_label_categorize(args, client, data_list, label_list):
    answer = {"Unsuccessful": []}
    for label in label_list:
        answer[label] = []

    length = args.test_num if args.print_details else len(data_list)

    for i in range(length):
        sentence = data_list[i]["input"]
        prompt = prompt_construct(label_list, sentence)
        response = chat(prompt, client)

        if response is None:
            predicted = "Unsuccessful"
        else:
            predicted = answer_process(response, label_list)

        if predicted in label_list:
            answer[predicted].append(sentence)
        else:
            answer["Unsuccessful"].append(sentence)

        if args.print_details:
            print(f"--- Sample {i + 1} ---")
            print(f"Input    : {sentence}")
            print(f"Response : {response}")
            print(f"Predicted: {predicted}")

        if i % 200 == 0:
            print(f"Progress: {i}/{length}")
            write_answer_to_json(args, answer, args.output_path, args.output_file_name)

    return answer


def write_answer_to_json(args, answer, output_path, output_name):
    size = "large" if args.use_large else "small"
    file_name = os.path.join(output_path, "_".join([args.data, size, output_name]))
    with open(file_name, "w") as f:
        json.dump(answer, f, indent=2)
    print(f"Wrote {file_name}")


def load_predict_data(data_path, file_name):
    with open(os.path.join(data_path, file_name), "r") as f:
        return json.load(f)


def describe_final_output(answer):
    for key, values in answer.items():
        print(f"  {key}: {len(values)}")


def main(args):
    print(f"Dataset : {args.data}  |  split: {'large' if args.use_large else 'small'}")
    start = time.time()

    client = ini_client()
    data_list = load_dataset(args.data_path, args.data, args.use_large)
    label_list = get_predict_labels(args.output_path, args.data)
    print(f"Labels to classify into: {len(label_list)}")

    answer = known_label_categorize(args, client, data_list, label_list)
    answer = {k: v for k, v in answer.items() if v}  # drop empty buckets
    write_answer_to_json(args, answer, args.output_path, args.output_file_name)

    print("Classification results:")
    describe_final_output(answer)
    print(f"Done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 2: Classify texts into generated labels.")
    parser.add_argument("--data_path", type=str, default="./dataset/")
    parser.add_argument("--data", type=str, default="arxiv_fine")
    parser.add_argument("--output_path", type=str, default="./generated_labels")
    parser.add_argument("--output_file_name", type=str, default="find_labels.json")
    parser.add_argument("--use_large", action="store_true")
    parser.add_argument("--print_details", type=bool, default=False)
    parser.add_argument("--test_num", type=int, default=5)
    # --api_key kept for backward compatibility but ignored; key comes from .env
    parser.add_argument("--api_key", type=str, default="", help="ignored — use OPENAI_API_KEY in .env")
    args = parser.parse_args()
    main(args)
