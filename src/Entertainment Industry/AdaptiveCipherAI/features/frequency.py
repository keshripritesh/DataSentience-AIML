"""Frequency analysis utilities."""

from collections import Counter
from typing import Dict, Tuple


def only_letters(text: str) -> str:

	return ''.join(ch.upper() for ch in text if ch.isalpha())


def unigram_frequencies(text: str) -> Dict[str, float]:

	letters = only_letters(text)
	if not letters:
		return {chr(ord('A') + i): 0.0 for i in range(26)}
	c = Counter(letters)
	n = len(letters)
	return {chr(ord('A') + i): c.get(chr(ord('A') + i), 0) / n for i in range(26)}


def bigram_counts(text: str) -> Dict[str, int]:

	letters = only_letters(text)
	c = Counter(a + b for a, b in zip(letters, letters[1:]))
	return dict(c)


def trigram_counts(text: str) -> Dict[str, int]:

	letters = only_letters(text)
	c = Counter(a + b + c for a, b, c in zip(letters, letters[1:], letters[2:]))
	return dict(c)


