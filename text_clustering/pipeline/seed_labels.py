"""
seed_labels.py — Step 0 of the clustering pipeline.

Selects 20 % of the ground-truth labels from each dataset and writes the
chosen seed labels to ``runs/chosen_labels.json``.  These seeds
are passed as ``--given_label_path`` to Step 1 (label_generation).

Original source: ECNU-Text-Computing/Text-Clustering-via-LLM
Modifications:
  - moved to text_clustering.pipeline package; logic unchanged
"""

import json
import logging
import os
import random

from text_clustering.logging_config import setup_logging

logger = logging.getLogger(__name__)


def find_sorted_folders(directory):
    folders = []
    for entry in os.scandir(directory):
        if entry.is_dir():
            folders.append(entry.name)
    folders.sort()
    return folders


def load_dataset(data_path, data):
    data_file = os.path.join(data_path, data, "small.jsonl")
    logger.info("Loading %s", data_file)
    with open(data_file, "r") as f:
        data_list = []
        for line in f:
            json_object = json.loads(line)
            data_list.append(json_object)
    logger.info("  %d samples loaded", len(data_list))
    return data_list


def get_label_list(data_list):
    label_list = []
    for data in data_list:
        if data["label"] not in label_list:
            label_list.append(data["label"])
    return label_list


def main():  # select 20% of labels to be given to the LLMs
    data_path = "./dataset/"
    runs_dir = "./runs"
    os.makedirs(runs_dir, exist_ok=True)
    setup_logging(os.path.join(runs_dir, "seed_labels.log"))

    output_path = os.path.join(runs_dir, "chosen_labels.json")
    logger.info("Seed label selection — output: %s", output_path)

    total_chosen_labels = dict()
    for data in find_sorted_folders(data_path):
        data_list = load_dataset(data_path, data)
        data_labels = get_label_list(data_list)
        choose_num = int(0.2 * len(data_labels))
        logger.info("  %s: %d classes → %d seeds", data, len(data_labels), choose_num)
        total_chosen_labels[data] = random.choices(data_labels, k=choose_num)
    with open(output_path, "w") as f:
        json.dump(total_chosen_labels, f, indent=2)
    logger.info("Done — wrote %s", output_path)


if __name__ == "__main__":
    main()
