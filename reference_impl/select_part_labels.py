"""
select_part_labels.py — Step 0 entry-point shim.

All logic lives in text_clustering.pipeline.seed_labels.
This file is kept at the project root for backward compatibility
(``python select_part_labels.py`` continues to work).
"""

from text_clustering.pipeline.seed_labels import main

if __name__ == "__main__":
    main()
