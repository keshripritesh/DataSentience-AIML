from features.frequency import unigram_frequencies
from features.statistics import index_of_coincidence, entropy, chi_squared_english_score


def test_unigram_frequencies_sum_to_one():

	freqs = unigram_frequencies("ABBCCC")
	assert abs(sum(freqs.values()) - 1.0) < 1e-9


def test_index_of_coincidence_ranges():

	ioc_letters = index_of_coincidence("A" * 10 + "B" * 10)
	ioc_randomish = index_of_coincidence("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
	assert 0.0 <= ioc_letters <= 1.0
	assert 0.0 <= ioc_randomish <= 1.0


def test_entropy_bounds():

	assert entropy("AAAAAA") == 0.0
	assert entropy("AB") > 0.0


def test_chi_squared_lower_for_english_like():

	english_like = "THIS IS A SIMPLE ENGLISH SENTENCE"
	non_english = "ZZZZZ QQQQQ XXXXX"
	assert chi_squared_english_score(english_like) < chi_squared_english_score(non_english)


