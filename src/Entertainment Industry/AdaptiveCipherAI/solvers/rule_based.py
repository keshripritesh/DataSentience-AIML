"""Rule-based solvers for classical ciphers.

Includes:
- Caesar shift estimation via chi-squared over English frequencies.
- Simple substitution hill-climbing using n-gram scoring (unigram + bigram freq proxy).
"""

from __future__ import annotations

import math
import random
from typing import Dict, Tuple

from features.statistics import chi_squared_english_score
from features.frequency import unigram_frequencies, only_letters
from ciphers import caesar, substitution


def solve_caesar(ciphertext: str) -> Tuple[int, str]:

	return caesar.crack(ciphertext)


# Basic English scoring using unigrams; lightweight but effective for short texts
def english_score(text: str) -> float:

	# Lower score is better (reuse chi-squared)
	return chi_squared_english_score(text)


def _swap(key: str, a: int, b: int) -> str:

	if a == b:
		return key
	letters = list(key)
	letters[a], letters[b] = letters[b], letters[a]
	return ''.join(letters)


def solve_substitution(ciphertext: str, max_iter: int = 5000, seed: int | None = None) -> Tuple[str, str, float]:

	"""Attempt to solve monoalphabetic substitution using hill-climbing.

	Returns (key, plaintext, score). Lower score is better.
	"""
	rng = random.Random(seed)
	key = substitution.random_key(rng)
	plaintxt = substitution.decrypt(ciphertext, key)
	best_score = english_score(plaintxt)

	no_improve = 0
	for i in range(max_iter):
		# Propose swapping two positions in key
		a, b = rng.randrange(26), rng.randrange(26)
		cand_key = _swap(key, a, b)
		cand_plain = substitution.decrypt(ciphertext, cand_key)
		s = english_score(cand_plain)
		if s < best_score:
			key, plaintxt, best_score = cand_key, cand_plain, s
			no_improve = 0
		else:
			no_improve += 1
		# Occasional random restart
		if no_improve > 500 and i < max_iter - 500:
			key = substitution.random_key(rng)
			plaintxt = substitution.decrypt(ciphertext, key)
			best_score = english_score(plaintxt)
			no_improve = 0

	return key, plaintxt, best_score


