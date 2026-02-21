from typing import Callable, Dict
import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
from statistics import multimode
from scipy.stats import moment
import json
# from matplotlib import pyplot as plt

# @TODO:
# 1) эмпирическая ф-я распределения
# 2) сравнить оценку плотности распределения среднего
#  арифметического ЦПТ и bootstrap

# ============= TASK CONFIG ===============
# Mersenne twister init
GENERATOR = RandomState(MT19937(SeedSequence(2)))


# ======== PROBABILITY FUNCTIONS ==========
def p(
    x: np.ndarray
) -> np.ndarray:
    return 0 if x < 0 else np.exp(-x)


def F(
    x: np.ndarray
) -> np.ndarray:
    return 0 if x < 0 else 1 - np.exp(-x)


def F_reversed(
    P: np.ndarray
) -> np.ndarray:
    if np.any(P == 0):
        return -np.inf * np.ones_like(P)
    return -np.log(1-P)


# ============ AUXILIARY FUNCTIONS ==========
def generate_sample(
    reversed_distribution_function: Callable,
    size: int = 25,
) -> np.ndarray:
    """
    Function model sample with `lenght` values of continuous random variable.
    """
    return reversed_distribution_function(GENERATOR.rand(size))


def calculate_stats(
    sample: np.ndarray
) -> Dict:
    stats = dict()

    stats["mode"] = multimode(sample)  # all elems are mode
    stats["median"] = float(np.median(sample))
    stats["range"] = float(np.max(sample) - np.min(sample))
    stats["skewness"] = moment(sample, 3) / moment(sample, 2) ** 1.5

    return stats


def save_stats(
    sample: np.ndarray
) -> None:
    with open(
        r"first_task\2\sample_stats.json", "w", encoding="utf-8"
    ) as json_file:
        json.dump(calculate_stats(sample), json_file)


def main():
    sample = generate_sample(F_reversed)
    save_stats(sample)


if __name__ == "__main__":
    main()
