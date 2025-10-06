"""Monoalphabetic substitution cipher.

Encrypts/decrypts via a key that is a permutation of A-Z.
Non-letters are preserved; case is preserved.
"""

from typing import Dict
import random
import string


ALPHABET = string.ascii_uppercase


def normalize_key(key: str) -> str:

	key = ''.join(ch for ch in key.upper() if ch.isalpha())
	if len(key) == 26 and set(key) == set(ALPHABET):
		return key
	raised = ValueError("Key must be a 26-letter permutation of A-Z")
	raised.args = (*raised.args, {"key": key})
	raise raised


def random_key(rng: random.Random | None = None) -> str:

	r = rng or random
	letters = list(ALPHABET)
	r.shuffle(letters)
	return ''.join(letters)


def _build_maps(key: str) -> tuple[Dict[str, str], Dict[str, str]]:

	key = normalize_key(key)
	enc_map = {p: k for p, k in zip(ALPHABET, key)}
	dec_map = {v: k for k, v in enc_map.items()}
	return enc_map, dec_map


def encrypt(plaintext: str, key: str) -> str:

	enc_map, _ = _build_maps(key)
	result_chars: list[str] = []
	for ch in plaintext:
		if ch.isalpha():
			is_upper = ch.isupper()
			mapped = enc_map[ch.upper()]
			result_chars.append(mapped if is_upper else mapped.lower())
		else:
			result_chars.append(ch)
	return ''.join(result_chars)


def decrypt(ciphertext: str, key: str) -> str:

	_, dec_map = _build_maps(key)
	result_chars: list[str] = []
	for ch in ciphertext:
		if ch.isalpha():
			is_upper = ch.isupper()
			mapped = dec_map[ch.upper()]
			result_chars.append(mapped if is_upper else mapped.lower())
		else:
			result_chars.append(ch)
	return ''.join(result_chars)


