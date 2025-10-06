import pytest

from ciphers import caesar, substitution, vigenere


def test_caesar_roundtrip():

	plain = "Hello, World!"
	for shift in [0, 1, 13, 25]:
		ct = caesar.encrypt(plain, shift)
		pt = caesar.decrypt(ct, shift)
		assert pt == plain


def test_substitution_roundtrip():

	key = "QWERTYUIOPASDFGHJKLZXCVBNM"
	plain = "The Quick Brown Fox Jumps Over The Lazy Dog."
	ct = substitution.encrypt(plain, key)
	pt = substitution.decrypt(ct, key)
	assert pt == plain


def test_vigenere_roundtrip():

	key = "LEMON"
	plain = "ATTACK AT DAWN"
	ct = vigenere.encrypt(plain, key)
	pt = vigenere.decrypt(ct, key)
	assert pt == plain


