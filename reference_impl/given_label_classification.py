"""
given_label_classification.py — Step 2 entry-point shim.

All logic lives in text_clustering.pipeline.classification.
This file is kept at the project root for backward compatibility
(``python given_label_classification.py --data foo`` continues to work).
"""

from text_clustering.pipeline.classification import build_parser, main

if __name__ == "__main__":
    main(build_parser().parse_args())
