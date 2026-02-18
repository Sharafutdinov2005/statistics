from typing import Callable, Union
import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
# from matplotlib import pyplot as plt


# ============= TASK CONFIG ===============
# Mersenne twister init
GENERATOR = RandomState(MT19937(SeedSequence(2)))


# ======== PROBABILITY FUNCTIONS ==========
def p(
    x: Union[np.ndarray[np.float64], np.float64]
) -> np.ndarray[np.float64]:
    return 0 if x < 0 else np.exp(-x)


def F(
    x: Union[np.ndarray[np.float64], np.float64]
) -> np.ndarray[np.float64]:
    return 0 if x < 0 else 1 - np.exp(-x)


def F_reversed(
    P: Union[np.ndarray[np.float64], np.float64]
) -> np.ndarray[np.float64]:
    if np.any(P == 0):
        return -np.inf * np.ones_like(P)
    return -np.log(1-P)


# ============ AUXILIARY FUNCTIONS ==========

def generate_sample(
    reversed_distribution_function: Callable[[np.float64], np.float64],
    size: int = 25,
) -> np.ndarray[np.float64]:
    """
    Function model sample with `lenght` values of continuous random variable.
    """
    return reversed_distribution_function(GENERATOR.rand(size))


def main():
    # sample generating
    # sample = generate_sample(F_reversed)


if __name__ == "__main__":
    main()
