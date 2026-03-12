"""Quick utility to re-run ONLY the merge step with a target_k."""
import json
import sys

from text_clustering.llm import chat, ini_client
from text_clustering.prompts import prompt_construct_merge_label


def main():
    run_dir = sys.argv[1]
    target_k = int(sys.argv[2])

    client = ini_client()
    with open(f"{run_dir}/labels_proposed.json") as f:
        all_labels = json.load(f)

    print(f"Proposed labels: {len(all_labels)}")
    prompt = prompt_construct_merge_label(all_labels, target_k=target_k)
    response = chat(prompt, client, max_tokens=4096)

    try:
        parsed = eval(response)  # noqa: S307
        if isinstance(parsed, dict):
            merged = []
            for val in parsed.values():
                if isinstance(val, list):
                    merged.extend(val)
            final = merged if merged else all_labels
        elif isinstance(parsed, list):
            final = parsed
        else:
            final = all_labels
    except Exception:
        final = all_labels

    print(f"Merged labels (target_k={target_k}): {len(final)}")
    print(final)
    with open(f"{run_dir}/labels_merged.json", "w") as f:
        json.dump(final, f, indent=2)
    print("Saved labels_merged.json")


if __name__ == "__main__":
    main()
