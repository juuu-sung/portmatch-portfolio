"""Deterministic text views; no external dictionary or generated training data.

Romanization and a deliberately coarse consonant key are separate retrieval
signals, not translations. Original names and numeric tokens remain available.
"""

import re
import unicodedata
from functools import lru_cache

from unidecode import unidecode


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


@lru_cache(maxsize=200_000)
def _normalize(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKC", text).upper() if c.isalnum())


def normalize_name(value: object) -> str:
    return _normalize(_text(value))


@lru_cache(maxsize=200_000)
def _romanize(text: str) -> str:
    return normalize_name(unidecode(unicodedata.normalize("NFKC", text)))


def romanize_name(value: object) -> str:
    return _romanize(_text(value))


@lru_cache(maxsize=200_000)
def _phonetic(text: str) -> str:
    text = romanize_name(text)
    for source, target in (("PH", "F"), ("CK", "K"), ("QU", "K")):
        text = text.replace(source, target)
    text = text.translate(str.maketrans({"F": "P", "V": "B", "Z": "S", "Q": "K", "C": "K", "R": "L"}))
    text = re.sub(r"[AEIOUY]", "", text)
    # Do not collapse repeated digits: vessel numbers are identity evidence.
    return re.sub(r"([A-Z])\1+", r"\1", text)


def phonetic_key(value: object) -> str:
    return _phonetic(_text(value))


def name_numbers(value: object) -> tuple[str, ...]:
    return tuple(str(int(v)) for v in re.findall(r"[0-9]+", normalize_name(value)))


def has_hangul(value: object) -> bool:
    return bool(re.search(r"[가-힣ㄱ-ㅎㅏ-ㅣ]", _text(value)))
