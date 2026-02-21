"""
prompts.py — Prompt-construction helpers for the LLM pipeline.

All functions are pure (no I/O, no side-effects) and return a prompt string.

Functions
---------
prompt_construct_generate_label(sentence_list, given_labels)
    Prompt for Step 1 — propose new label names for unchategorised texts.

prompt_construct_merge_label(label_list, target_k=None)
    Prompt for Step 1 — deduplicate/merge near-synonymous label names.
    target_k: if provided, instructs the model to produce approximately that
    many final labels.  Used to compensate for weaker models that don't
    consolidate as aggressively as GPT-3.5 without an explicit target.

prompt_construct_map_to_canonical(proposed_labels, canonical_labels)
    Prompt for Step 1 merge — map each proposed label to its closest canonical
    (true-class) label.  Used when the model cannot consolidate freely but can
    reliably perform nearest-neighbour assignment to a known taxonomy.

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


def prompt_construct_map_to_canonical(proposed_labels: list, canonical_labels: list) -> str:
    """
    Ask the model to map each proposed label to its closest canonical label.

    Used when free-form merging fails: instead of asking the model to consolidate
    freely (which weak models cannot do at scale), we provide the known target
    taxonomy and ask for nearest-neighbour assignment.  The result is a deduplicated
    list containing only canonical labels that have at least one proposed match.
    """
    json_example = {"canonical_labels": ["label name", "label name"]}
    prompt = (
        "You are given two lists:\n"
        f"1. CANONICAL labels (the final taxonomy): {canonical_labels}\n"
        f"2. PROPOSED labels (a noisy, over-specific list): {proposed_labels}\n\n"
        "For each PROPOSED label, find the single CANONICAL label it is closest to in meaning. "
        "Then return the deduplicated list of CANONICAL labels that were matched by at least one "
        "proposed label. Only include canonical labels that have at least one match. "
        "Do NOT invent new labels. Do NOT include labels from the proposed list. "
        "Only output canonical labels.\n"
        f"Return only the result as a flat JSON list like: {json_example}"
    )
    return prompt


def prompt_construct_classify(label_list, sentence):
    json_example = {"label_name": "label"}
    prompt = "Given the label list and the sentence, please categorize the sentence into one of the labels.\n"
    prompt += f"Label list: {label_list}.\n"
    prompt += f"Sentence:{sentence}.\n"
    prompt += f"You should only return the label name, please return in json format like: {json_example}"
    return prompt
