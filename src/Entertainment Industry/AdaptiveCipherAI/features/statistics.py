"""Statistical measures useful for cryptanalysis."""

import math
from typing import Dict

from .frequency import only_letters, unigram_frequencies


ENGLISH_FREQ = {
	'A': 0.08167, 'B': 0.01492, 'C': 0.02782, 'D': 0.04253, 'E': 0.12702,
	'F': 0.02228, 'G': 0.02015, 'H': 0.06094, 'I': 0.06966, 'J': 0.00153,
	'K': 0.00772, 'L': 0.04025, 'M': 0.02406, 'N': 0.06749, 'O': 0.07507,
	'P': 0.01929, 'Q': 0.00095, 'R': 0.05987, 'S': 0.06327, 'T': 0.09056,
	'U': 0.02758, 'V': 0.00978, 'W': 0.02360, 'X': 0.00150, 'Y': 0.01974, 'Z': 0.00074,
}


def index_of_coincidence(text: str) -> float:

	letters = only_letters(text)
	n = len(letters)
	if n < 2:
		return 0.0
	counts: Dict[str, int] = {}
	for ch in letters:
		counts[ch] = counts.get(ch, 0) + 1
	num = sum(c * (c - 1) for c in counts.values())
	den = n * (n - 1)
	return num / den


def entropy(text: str) -> float:

	freqs = unigram_frequencies(text)
	return -sum(p * math.log2(p) for p in freqs.values() if p > 0)


def chi_squared_english_score(text: str) -> float:

	letters = only_letters(text)
	n = len(letters)
	if n == 0:
		return float('inf')
	counts = {chr(ord('A') + i): 0 for i in range(26)}
	for ch in letters:
		counts[ch] += 1
	score = 0.0
	for ch, expected_freq in ENGLISH_FREQ.items():
		expected = expected_freq * n
		observed = counts[ch]
		if expected > 0:
			score += (observed - expected) ** 2 / expected
	return score


