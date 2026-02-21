from typing import Callable, Dict
import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
from statistics import multimode
from scipy.stats import moment, bootstrap, Normal
import json
from matplotlib import pyplot as plt


# ============= TASK CONFIG ===============
# Mersenne twister init
GENERATOR = RandomState(MT19937(SeedSequence(2)))


# ======== PROBABILITY FUNCTIONS ==========
def p(
    x: np.ndarray
) -> np.ndarray:
    return np.exp(-x) * (x > np.zeros_like(x))


def F(
    x: np.ndarray
) -> np.ndarray:
    return (1 - np.exp(-x)) * (x > np.zeros_like(x))


def F_reversed(
    P: np.ndarray
) -> np.ndarray:
    # can't get 1, but it doesn't matter
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
    # can't be 1, but it doesn't matter
    return reversed_distribution_function(GENERATOR.rand(size))


def calculate_stats(
    sample: np.ndarray,
    x_range: np.float64
) -> Dict:
    stats = dict()

    stats["mode"] = multimode(sample)  # all elems are mode
    stats["median"] = float(np.median(sample))
    stats["range"] = float(x_range)
    stats["skewness"] = moment(sample, 3) / moment(sample, 2) ** 1.5

    return stats


def save_stats(
    sample: np.ndarray,
    x_range: np.float64
) -> None:
    with open(
        r"first_task\2\sample_stats.json", "w", encoding="utf-8"
    ) as json_file:
        json.dump(calculate_stats(sample, x_range), json_file)


def save_emp_distrib_func(
    sample: np.ndarray,
    x_min: np.float64,
    x_max: np.float64,
    x_range: np.float64
) -> None:
    fig, ax = plt.subplots()

    def F_emp(
        x: np.ndarray
    ) -> np.ndarray:
        n = sample.size
        values = sample.copy().reshape(1, -1)
        return (x[:, np.newaxis] > values).sum(axis=-1) / n

    x = np.linspace(x_min - 0.5 * x_range, x_max + 0.5 * x_range, num=2000)
    y = F(x)
    y_emp = F_emp(x)

    ax.plot(x, y, color='r', label='F(x)', linewidth=1, ls='--')
    ax.plot(x, y_emp, color='b', label='F_emp(x)', linewidth=2, alpha=0.7)
    ax.set_title('Distribution function comparsion')
    ax.grid(True)

    ax.set_ylabel("F(x), F_emp(x)")
    ax.set_xlabel("x")

    ax.set_xlim(x_min - 0.5 * x_range, x_max + 0.5 * x_range)
    ax.set_ylim(-0.05, 1.05)  # just for look

    ax.legend()

    fig.savefig(r"first_task\2\plots\distribution_func_comp.png")


def save_hist(
    sample: np.ndarray,
    x_min: np.float64,
    x_max: np.float64,
    x_range: np.float64
) -> None:

    fig, ax = plt.subplots()

    k = int(1 + np.log2(sample.size))
    # mu_i = m_i / n / delta
    w = np.ones_like(sample) * 1 / (x_range / k) / sample.size
    x = np.linspace(x_min - 0.5 * x_range, x_max + 0.5 * x_range, num=1000)
    y = p(x)

    ax.hist(sample, bins=k, weights=w, label='Histogramm')
    ax.plot(x, y, color='r', label='p(x)', linewidth=1, ls='--')
    ax.set_title('Histogramm')
    ax.grid(True)

    ax.set_ylabel("Histogramm, p(x)")
    ax.set_xlabel("x")

    ax.set_xlim(x_min - 0.5 * x_range, x_max + 0.5 * x_range)
    ax.set_ylim(-0.05, 1.05)

    ax.legend()

    fig.savefig(r"first_task\2\plots\hist.png")


def save_boxplot(
    sample: np.ndarray,
) -> None:

    fig, ax = plt.subplots()

    ax.boxplot(sample, orientation='horizontal')

    fig.savefig(r"first_task\2\plots\boxplot.png")


def save_plots(
    sample: np.ndarray,
    x_min: np.float64,
    x_max: np.float64,
    x_range: np.float64
) -> None:
    save_emp_distrib_func(sample, x_min, x_max, x_range)
    save_hist(sample, x_min, x_max, x_range)
    save_boxplot(sample)


def mean_valuation_compasrion(
    sample: np.ndarray,
) -> None:
    n = 1000
    bootstrap_result = bootstrap(
        (sample,),
        statistic=np.mean,
        n_resamples=n,
        rng=GENERATOR
    )
    CPT_result = Normal(mu=1, sigma=1/5)  # class uses sigma, not sigma^2

    fig, ax = plt.subplots()

    new_sample = bootstrap_result.bootstrap_distribution.reshape(-1)
    x_min, x_max = new_sample.min(), new_sample.max()
    x_range = x_max - x_min

    k = int(1 + np.log2(new_sample.size))
    # mu_i = m_i / n / delta
    w = np.ones_like(new_sample) * 1 / (x_range / k) / new_sample.size
    x = np.linspace(x_min - 0.5 * x_range, x_max + 0.5 * x_range, num=1000)
    y = CPT_result.pdf(x)

    ax.hist(new_sample, bins=k, weights=w, label='Bootstrap')
    ax.plot(x, y, color='r', label='CPT', linewidth=1, ls='--')
    ax.set_title('Bootstrap vs CPT')
    ax.grid(True)

    ax.set_ylabel("Bootstrap, CPT")
    ax.set_xlabel("x")

    ax.set_xlim(x_min - 0.5 * x_range, x_max + 0.5 * x_range)

    ax.legend()

    fig.savefig(r"first_task\2\plots\bootstrap_vs_cpt.png")


def main():
    sample = generate_sample(F_reversed)

    x_min, x_max = np.min(sample), np.max(sample)
    x_range = x_max - x_min

    save_stats(sample, x_range)
    save_plots(sample, x_min, x_max, x_range)
    mean_valuation_compasrion(sample)


if __name__ == "__main__":
    main()
