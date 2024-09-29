import numpy as np

def discrete_pmf_sum_pmf(pmf_range, pmf, n: int):
    a = pmf_range[0]
    b = pmf_range[-1]
    final_size = (n * (b - a) + 1)
    du_sum_pmf = np.zeros(final_size)
    du_sum_pmf[0] = 1

    temp_du_sum_pmf = np.zeros(final_size)
    for i in range(1, n+1):
        prior_range = (i-1) * (b - a) + 1
        for dice in range(a, b+1):
            dice_idx = dice - a
            temp_du_sum_pmf[dice_idx: dice_idx + prior_range] += du_sum_pmf[:prior_range] * pmf[dice_idx]

        (du_sum_pmf, temp_du_sum_pmf) = (temp_du_sum_pmf, du_sum_pmf)
        temp_du_sum_pmf[:i * (b - a)+1] = 0

    all_counts = np.arange(a * n, len(du_sum_pmf))
    return (all_counts, du_sum_pmf)