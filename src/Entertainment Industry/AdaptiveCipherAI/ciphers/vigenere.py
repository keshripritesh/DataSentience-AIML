"""Vigenère cipher implementation.

Supports A-Z keys; non-letters preserved; case preserved.
"""

from typing import Tuple


def _shift_char(ch: str, shift: int) -> str:

	if not ch.isalpha():
		return ch
	is_upper = ch.isupper()
	base = ord('A') if is_upper else ord('a')
	alpha_index = ord(ch) - base
	shifted = (alpha_index + shift) % 26
	return chr(base + shifted)


def _key_stream(text: str, key: str):

	filtered_key = [k.upper() for k in key if k.isalpha()]
	if not filtered_key:
		raise ValueError("Key must contain at least one alphabetic character")
	ki = 0
	for ch in text:
		if ch.isalpha():
			k = filtered_key[ki % len(filtered_key)]
			yield ord(k) - ord('A')
			ki += 1
		else:
			yield 0


def encrypt(plaintext: str, key: str) -> str:

	return ''.join(_shift_char(c, s) for c, s in zip(plaintext, _key_stream(plaintext, key)))


def decrypt(ciphertext: str, key: str) -> str:

	return ''.join(_shift_char(c, -s) for c, s in zip(ciphertext, _key_stream(ciphertext, key)))


