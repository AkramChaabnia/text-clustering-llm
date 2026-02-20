"""
label_generation.py — Step 1 of the clustering pipeline.

For each dataset, the script:
  1. Loads the dataset and shuffles it.
  2. Reads the seed labels produced by select_part_labels.py.
  3. Iterates over the texts in chunks of --chunk_size (default 15).
     For each chunk, it asks the LLM to propose new label names for texts
     that don't fit any existing label.
  4. Merges the resulting label set with a second LLM call that deduplicates
     near-synonymous labels.
  5. Writes three JSON files to --output_path:
       <data>_<size>_true_labels.json
       <data>_<size>_llm_generated_labels_before_merge.json
       <data>_<size>_llm_generated_labels_after_merge.json

The LLM is configured entirely through environment variables (see .env.example).
Model selection, temperature, and token limits are read at startup — no code
change is needed to switch models.

Original source: ECNU-Text-Computing/Text-Clustering-via-LLM
Modifications:
  - ini_client() now delegates to openrouter_adapter.make_client()
  - response_format is opt-in via LLM_FORCE_JSON_MODE (default: off)
  - _strip_fenced_json() handles models that wrap JSON in markdown fences
  - chat() has a 5-attempt retry with linear backoff on 429 errors
  - LLM_MODEL env var controls the model (default keeps original for compat)
"""

import json
import os
import random
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


def prompt_construct_generate_label(sentence_list, given_labels):
    json_example = {"labels": ["label name", "label name"]}
    prompt = f"Given the labels, under a text classicifation scenario, can all these text match the label given? If the sentence does not match any of the label, please generate a meaningful new label name.\n \
            Labels: {given_labels}\n \
            Sentences: {sentence_list} \n \
            You should NOT return meaningless label names such as 'new_label_1' or 'unknown_topic_1' and only return the new label names, please return in json format like: {json_example}"
    return prompt


def prompt_construct_merge_label(label_list):
    json_example = {"merged_labels": ["label name", "label name"]}
    prompt = f"Please analyze the provided list of labels to identify entries that are similar or duplicate, considering synonyms, variations in phrasing, and closely related terms that essentially refer to the same concept. Your task is to merge these similar entries into a single representative label for each unique concept identified. The goal is to simplify the list by reducing redundancies without organizing it into subcategories or altering its fundamental structure. \n"
    prompt += f"Here is the list of labels for analysis and simplification::{label_list}.\n"
    prompt += f"Produce the final, simplified list in a flat, JSON-formatted structure without any substructures or hierarchical categorization like: {json_example}"
    return prompt


def get_sentences(sentence_list):
    return [item["input"] for item in sentence_list]


def label_generation(args, client, data_list, chunk_size):
    with open(args.given_label_path, "r") as f:
        given_labels = json.load(f)

    all_labels = list(given_labels.get(args.data, []))
    count = 0

    for i in range(0, len(data_list), chunk_size):
        chunk = data_list[i : i + chunk_size]
        sentences = get_sentences(chunk)
        prompt = prompt_construct_generate_label(sentences, given_labels[args.data])
        raw = chat(prompt, client)
        if raw is None:
            continue
        count += 1
        try:
            response = eval(raw)  # noqa: S307
        except Exception:
            continue

        first_key = list(response.keys())[0]
        if isinstance(response[first_key], list):
            for label in response[first_key]:
                if "unknown_topic" in label or "new_label" in label:
                    continue
                if label not in all_labels:
                    all_labels.append(label)
        else:
            all_labels.append(response[first_key])

        if args.print_details:
            print(f"Prompt:\n{prompt}")
            print(f"Raw response: {raw}")
            print(f"Parsed: {response}")
            print(f"Label count so far: {len(all_labels)}")
            if count >= args.test_num:
                break

    return all_labels


def merge_labels(args, all_labels, client):
    prompt = prompt_construct_merge_label(all_labels)
    response = chat(prompt, client)
    try:
        response = eval(response)  # noqa: S307
        merged = []
        for sub_list in response.values():
            merged.extend(sub_list)
        return merged
    except Exception:
        return all_labels


def write_dict_to_json(args, data, output_path, output_name):
    size = "large" if args.use_large else "small"
    file_name = os.path.join(output_path, "_".join([args.data, size, output_name]) + ".json")
    with open(file_name, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {file_name}")


def main(args):
    print(f"Dataset : {args.data}  |  split: {'large' if args.use_large else 'small'}")
    start = time.time()

    client = ini_client()
    data_list = load_dataset(args.data_path, args.data, args.use_large)
    random.shuffle(data_list)

    true_labels = get_label_list(data_list)
    print(f"True cluster count: {len(true_labels)}")
    write_dict_to_json(args, true_labels, args.output_path, "true_labels")

    all_labels = label_generation(args, client, data_list, args.chunk_size)
    print(f"Labels proposed by LLM (before merge): {len(all_labels)}")
    write_dict_to_json(args, all_labels, args.output_path, "llm_generated_labels_before_merge")

    final_labels = merge_labels(args, all_labels, client)
    write_dict_to_json(args, final_labels, args.output_path, "llm_generated_labels_after_merge")
    print(f"Labels after merge: {len(final_labels)}")
    print(f"Done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 1: LLM label generation and merge.")
    parser.add_argument("--data_path", type=str, default="./dataset/")
    parser.add_argument("--data", type=str, default="arxiv_fine")
    parser.add_argument("--output_path", type=str, default="./generated_labels")
    parser.add_argument("--given_label_path", type=str, default="./generated_labels/chosen_labels.json")
    parser.add_argument("--output_file_name", type=str, default="test.json")
    parser.add_argument("--use_large", action="store_true")
    parser.add_argument("--print_details", type=bool, default=False)
    parser.add_argument("--test_num", type=int, default=5)
    parser.add_argument("--chunk_size", type=int, default=15)
    # --api_key kept for backward compatibility but ignored; key comes from .env
    parser.add_argument("--api_key", type=str, default="", help="ignored — use OPENAI_API_KEY in .env")
    args = parser.parse_args()
    main(args)
