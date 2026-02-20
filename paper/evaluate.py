"""
evaluate.py — Step 3 entry-point shim.

All logic lives in text_clustering.pipeline.evaluation.
This file is kept at the project root for backward compatibility
(``python evaluate.py --data foo --predict_file ...`` continues to work).
"""

from text_clustering.pipeline.evaluation import build_parser, main

if __name__ == "__main__":
    main(build_parser().parse_args())
