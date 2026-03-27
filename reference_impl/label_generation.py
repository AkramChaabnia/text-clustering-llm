"""
label_generation.py — Step 1 entry-point shim.

All logic lives in text_clustering.pipeline.label_generation.
This file is kept at the project root for backward compatibility
(``python label_generation.py --data foo`` continues to work).
"""

from text_clustering.pipeline.label_generation import build_parser, main

if __name__ == "__main__":
    main(build_parser().parse_args())
