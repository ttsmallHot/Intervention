"""Training script specifically for VQA intervention plugins (extracting digits)."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

from src.common.extractors import extract_digit
from src.train.engine import run_main


if __name__ == "__main__":
    run_main(extract_digit)
