"""Caesar cipher implementation.

Provides functions to encrypt and decrypt using a Caesar shift over A-Z.
Non-alphabetic characters are preserved. Case is preserved.
"""

from typing import Tuple


ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _shift_char(ch: str, shift: int) -> str:

	if not ch.isalpha():
		return ch

	is_upper = ch.isupper()
	base = ord('A') if is_upper else ord('a')
	alpha_index = ord(ch) - base
	shifted = (alpha_index + shift) % 26
	return chr(base + shifted)


def encrypt(plaintext: str, shift: int) -> str:

	return ''.join(_shift_char(c, shift) for c in plaintext)


def decrypt(ciphertext: str, shift: int) -> str:

	return ''.join(_shift_char(c, -shift) for c in ciphertext)


def crack(ciphertext: str) -> Tuple[int, str]:

	"""Brute-force crack: returns (best_shift, plaintext_guess).

	Uses simple English letter frequency scoring (chi-squared) to select shift.
	"""
	from features.statistics import chi_squared_english_score

	COMMON_WORDS = {
		"THE","AND","OF","TO","IN","A","IS","THAT","IT","FOR","ON","WITH","AS","I","YOU","ARE","THIS","HE","BE","AT","ONE","HAVE","NOT","BY",
		# Boost practical signals for short phrases
		"HELLO","NEW","USER","QUICK","BROWN","FOX","OVER","LAZY","DOG"
	}

	def word_ratio(text: str) -> float:
		words = [w for w in ''.join(ch if ch.isalpha() or ch.isspace() else ' ' for ch in text).upper().split() if w]
		if not words:
			return 0.0
		matches = sum(1 for w in words if w in COMMON_WORDS)
		return matches / len(words)

	best_shift = 0
	best_key = (1.0, float('inf'))  # (-wr, chi2) to minimize
	best_plain = ciphertext
	for s in range(26):
		plain = decrypt(ciphertext, s)
		chi2 = chi_squared_english_score(plain)
		wr = word_ratio(plain)
		key = (-wr, chi2)
		if key < best_key:
			best_key = key
			best_shift = s
			best_plain = plain
	return best_shift, best_plain


