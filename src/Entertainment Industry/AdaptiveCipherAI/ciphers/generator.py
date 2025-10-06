"""Synthetic dataset generator for classical ciphers.

Writes a JSON list with entries: {type, plaintext, key, ciphertext}.
"""

import json
import random
import argparse
from pathlib import Path

from ciphers import caesar, substitution, vigenere


DEFAULT_TEXTS = [
	"lorem ipsum dolor sit amet consectetur adipiscing elit",
	"cryptography and machine learning meet in fascinating ways",
	"adaptive systems can learn patterns from ciphertext alone",
	"the quick brown fox jumps over the lazy dog",
	"reinforcement learning can guide search over key spaces",
]


def _rand_plain(rng: random.Random) -> str:

	text = rng.choice(DEFAULT_TEXTS)
	# Randomize casing and add punctuation occasionally
	chars = []
	for ch in text:
		if ch.isalpha() and rng.random() < 0.2:
			chars.append(ch.upper())
		else:
			chars.append(ch)
	if rng.random() < 0.5:
		chars.append('.')
	return ''.join(chars)


def generate_sample(cipher_type: str, rng: random.Random) -> dict:

	plain = _rand_plain(rng)
	if cipher_type == 'caesar':
		shift = rng.randrange(1, 26)
		cipher = caesar.encrypt(plain, shift)
		key = str(shift)
		return {"type": cipher_type, "plaintext": plain, "key": key, "ciphertext": cipher}
	elif cipher_type == 'substitution':
		key = substitution.random_key(rng)
		cipher = substitution.encrypt(plain, key)
		return {"type": cipher_type, "plaintext": plain, "key": key, "ciphertext": cipher}
	elif cipher_type == 'vigenere':
		length = rng.randint(3, 8)
		key = ''.join(chr(ord('A') + rng.randrange(26)) for _ in range(length))
		cipher = vigenere.encrypt(plain, key)
		return {"type": cipher_type, "plaintext": plain, "key": key, "ciphertext": cipher}
	else:
		raise ValueError(f"Unsupported cipher type: {cipher_type}")


def main():

	parser = argparse.ArgumentParser(description="Generate synthetic cipher dataset")
	parser.add_argument('--num_samples', type=int, default=1000)
	parser.add_argument('--types', nargs='+', default=['caesar', 'substitution', 'vigenere'])
	parser.add_argument('--out', type=str, default='examples/sample_ciphers.json')
	parser.add_argument('--seed', type=int, default=1337)
	args = parser.parse_args()

	rng = random.Random(args.seed)
	data = []
	for _ in range(args.num_samples):
		ctype = rng.choice(args.types)
		data.append(generate_sample(ctype, rng))

	out_path = Path(args.out)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	with out_path.open('w', encoding='utf-8') as f:
		json.dump(data, f, ensure_ascii=False, indent=2)
	print(f"Wrote {len(data)} samples to {out_path}")


if __name__ == '__main__':

	main()


