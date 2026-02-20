"""
data.py — Dataset loading helpers shared by all pipeline steps.

Functions
---------
load_dataset(data_path, data, use_large)
    Load a JSONL split from dataset/<data>/{small,large}.jsonl.

get_label_list(data_list)
    Return the ordered-unique list of ground-truth labels present in the data.
"""

import json
import logging
import os

logger = logging.getLogger(__name__)


def load_dataset(data_path, data, use_large):
    data_file = os.path.join(
        data_path, data, "large.jsonl" if use_large else "small.jsonl"
    )
    logger.info("Loading %s", data_file)
    with open(data_file, "r") as f:
        data_list = [json.loads(line) for line in f]
    logger.info("  %d samples loaded", len(data_list))
    return data_list


def get_label_list(data_list):
    seen = []
    for item in data_list:
        if item["label"] not in seen:
            seen.append(item["label"])
    return seen
