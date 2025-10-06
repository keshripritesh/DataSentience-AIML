from ciphers import caesar, substitution
from solvers import rule_based


def test_rule_based_caesar_crack():

	plain = "THE QUICK BROWN FOX"
	ct = caesar.encrypt(plain, 10)
	shift, guess = rule_based.solve_caesar(ct)
	assert guess == plain
	assert shift == 10


def test_rule_based_substitution_improves_score():

	# Construct a simple substitution
	key = "QWERTYUIOPASDFGHJKLZXCVBNM"
	plain = "THIS IS A LONGER ENGLISH SENTENCE USED FOR TESTING"
	ct = substitution.encrypt(plain, key)
	k, guess, score = rule_based.solve_substitution(ct, max_iter=500)
	# We expect at least some improvement over random; ensure chi-squared better than naive.
	from features.statistics import chi_squared_english_score
	assert chi_squared_english_score(guess) <= chi_squared_english_score(ct)


