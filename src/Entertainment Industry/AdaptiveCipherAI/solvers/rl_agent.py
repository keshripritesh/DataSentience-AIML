"""Reinforcement learning agent stub for decryption.

For now, this provides a CLI that evaluates random substitution keys
using an Englishness score, reporting the best found. This is a
lightweight stand-in for a full RL loop.
"""

import argparse
import random

from features.statistics import chi_squared_english_score
from ciphers import caesar, substitution


def reward(text: str) -> float:

	# Higher reward is better: invert chi-squared
	s = chi_squared_english_score(text)
	return -s


def crack_caesar(ciphertext: str) -> str:

	best_shift, plain = caesar.crack(ciphertext)
	return plain


def random_search_substitution(ciphertext: str, steps: int = 2000, seed: int | None = None) -> tuple[str, str, float]:

	rng = random.Random(seed)
	best_key = substitution.random_key(rng)
	best_plain = substitution.decrypt(ciphertext, best_key)
	best_reward = reward(best_plain)
	for _ in range(steps):
		key = substitution.random_key(rng)
		plain = substitution.decrypt(ciphertext, key)
		r = reward(plain)
		if r > best_reward:
			best_key, best_plain, best_reward = key, plain, r
	return best_key, best_plain, best_reward


def main():

	parser = argparse.ArgumentParser(description="RL Agent (stub) for decryption")
	parser.add_argument('--cipher', type=str, required=True)
	parser.add_argument('--type', type=str, choices=['caesar', 'substitution'], required=True)
	parser.add_argument('--steps', type=int, default=2000)
	args = parser.parse_args()

	if args.type == 'caesar':
		plain = crack_caesar(args.cipher)
		print(f"Plain: {plain}")
	else:
		key, plain, r = random_search_substitution(args.cipher, args.steps)
		print(f"Best key: {key}\nPlain: {plain}\nReward: {r:.3f}")


if __name__ == '__main__':

	main()


