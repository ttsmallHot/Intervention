"""Output-parsing helpers shared by train and eval."""

import re


def extract_digit(text: str) -> str:
    nums = re.findall(r"\d+", text)
    return nums[0] if nums else ""


def extract_option(text: str) -> str:
    match = re.search(r"\b([A-D])\b", text.upper())
    return match.group(1) if match else ""
