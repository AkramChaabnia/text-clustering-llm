"""
prompts.py — Prompt-construction helpers for the LLM pipeline.

All functions are pure (no I/O, no side-effects) and return a prompt string.

Functions
---------
prompt_construct_generate_label(sentence_list, given_labels)
    Prompt for Step 1 — propose new label names for unchategorised texts.

prompt_construct_merge_label(label_list, target_k=None)
    Prompt for Step 1 — deduplicate/merge near-synonymous label names.

    The base prompt is identical to the original paper (ECNU-Text-Computing).
    target_k is an optional escape hatch: if provided, an extra sentence
    instructs the model to produce approximately that many final labels.
    This should NOT be used with capable models (gemini, GPT-4) — it forces
    the model to fill all k slots with spurious categories instead of
    producing a naturally consolidated set.  It exists only as a fallback
    for weaker models that under-consolidate without guidance.

prompt_construct_classify(label_list, sentence)
    Prompt for Step 2 — classify a single text into one of the known labels.
"""


def prompt_construct_generate_label(sentence_list, given_labels):
    json_example = {"labels": ["label name", "label name"]}
    prompt = f"Given the labels, under a text classicifation scenario, can all these text match the label given? If the sentence does not match any of the label, please generate a meaningful new label name.\n \
            Labels: {given_labels}\n \
            Sentences: {sentence_list} \n \
            You should NOT return meaningless label names such as 'new_label_1' or 'unknown_topic_1' and only return the new label names, please return in json format like: {json_example}"
    return prompt


def prompt_construct_merge_label(label_list, target_k: int | None = None):
    json_example = {"merged_labels": ["label name", "label name"]}
    prompt = (
        "Please analyze the provided list of labels to identify entries that are similar or "
        "duplicate, considering synonyms, variations in phrasing, and closely related terms "
        "that essentially refer to the same concept. Your task is to merge these similar entries "
        "into a single representative label for each unique concept identified. The goal is to "
        "simplify the list by reducing redundancies without organizing it into subcategories or "
        "altering its fundamental structure.\n"
    )
    if target_k is not None:
        prompt += (
            f"The final list should contain approximately {target_k} labels — "
            f"merge aggressively until you reach roughly that number.\n"
        )
    prompt += f"Here is the list of labels for analysis and simplification: {label_list}.\n"
    prompt += f"Produce the final, simplified list in a flat, JSON-formatted structure without any substructures or hierarchical categorization like: {json_example}"
    return prompt


def prompt_construct_classify(label_list, sentence):
    json_example = {"label_name": "label"}
    prompt = "Given the label list and the sentence, please categorize the sentence into one of the labels.\n"
    prompt += f"Label list: {label_list}.\n"
    prompt += f"Sentence:{sentence}.\n"
    prompt += f"You should only return the label name, please return in json format like: {json_example}"
    return prompt
