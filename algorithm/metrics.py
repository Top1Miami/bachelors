
def compare_results(features_from_semi, features_from_baseline, known_features, good_features):
	semi_found_good = 0
	for feature in features_from_semi:
		if feature in good_features:
			semi_found_good += 1
	accuracy_semi = semi_found_good / len(features_from_semi) if len(features_from_semi) != 0 else 0
	fullness_semi = semi_found_good / len(good_features)
	baseline_found_good = 0
	for feature in features_from_baseline:
		if feature in good_features or feature in known_features:
			baseline_found_good += 1
			# print(feature)
	accuracy_baseline = baseline_found_good / len(features_from_baseline)
	fullness_baseline = baseline_found_good / (len(good_features) + len(known_features))
	return accuracy_semi, fullness_semi, accuracy_baseline, fullness_baseline