import random
from bayes_kalman_trace import RecursiveBayesianClassifier


def test_recursive_bayesian_classification_with_dropouts() -> None:
    clf = RecursiveBayesianClassifier()
    rng = random.Random(7)

    class0_series = [None if rng.random() < 0.1 else rng.gauss(11.2, 3.9) for _ in range(300)]
    class1_series = [None if rng.random() < 0.1 else rng.gauss(22.4, 1.0) for _ in range(300)]

    res0 = clf.filter_series(class0_series)
    res1 = clf.filter_series(class1_series)

    assert res0.predicted_class == 0
    assert res0.final_posterior_class_1 < 0.05
    assert res1.predicted_class == 1
    assert res1.final_posterior_class_1 > 0.95
    assert res1.missing_steps > 0
