"""ML-based decoder stubs.

Provides CLI hooks to train a simple cipher-type classifier
and a placeholder seq2seq model. Implementations are minimal to
avoid heavy dependencies; serves as extension points.
"""

import argparse
import json
from pathlib import Path


def train_classifier(data_path: str, epochs: int = 5):

	# Placeholder: simply compute majority class
	data = json.loads(Path(data_path).read_text(encoding='utf-8'))
	counts = {}
	for row in data:
		counts[row['type']] = counts.get(row['type'], 0) + 1
	major = max(counts, key=counts.get)
	print(f"Trained trivial classifier. Majority class: {major}")
	return {"majority": major}


def decode_seq2seq(ciphertext: str) -> str:

	# Placeholder passthrough
	return ciphertext


def main():

	parser = argparse.ArgumentParser(description="ML decoder CLI (stub)")
	parser.add_argument('--data', type=str, help='Path to dataset json')
	parser.add_argument('--epochs', type=int, default=5)
	args = parser.parse_args()

	if not args.data:
		parser.error('--data is required for training in this stub')
	train_classifier(args.data, args.epochs)


if __name__ == '__main__':

	main()


