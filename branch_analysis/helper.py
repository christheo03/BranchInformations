def branch_wmr(prediction, rate, weight):
    """
    Weighted error of a branch, as a fraction of its benchmark (0 to 1).
    Multiply by 100 to get the benchmark's 0-100 scale.

    prediction: 1 if the branch is predicted taken, 0 if not taken
    rate:       how often the branch really went taken (Taken / Executed)
    weight:     the branch's share of the benchmark's executions

    Same formula as get_binary_error. Works for single numbers and for
    pandas columns.
    """
    return (1 - prediction) * rate * weight + prediction * (1 - rate) * weight


def branch_wmr_if_wrong(actual, rate, weight):
    """
    The most the branch can cost: its wmr if it were predicted the
    opposite of its real direction (actual: 1 taken, 0 not taken).
    """
    return branch_wmr(1 - actual, rate, weight)
