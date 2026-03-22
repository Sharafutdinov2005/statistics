# Численное построение доверительного интервала

```python
# non-parametric bootsrap confidence interval
result = bootstrap(
    sample.reshape(1, -1),
    lambda x : 2/3 * np.mean(x),
    n_resamples=1000,
    confidence_level=0.95,
    method='basic',
    rng=GENERATOR
)
```

Согласно документации библиотеки, 'basic' обозначает 'reverse percentile', который строит доверительный интервал как

$$
[
    2\hat\theta - \theta^*_{({1+\beta \over 2} 1000)},
    2\hat\theta - \theta^*_{({1-\beta \over 2} 1000)}
]
$$

С учётом того, что

$$
\Delta_{(i)} = \theta^*_{(i)} - \hat\theta
$$

получаем

$$
[
    \hat\theta - \Delta_{({1+\beta \over 2}1000)}, \hat\theta - \Delta_{({1-\beta \over 2}1000)}
]
$$
\- метод с лекции.

# Вывод

Интервал, полученный точным методом получился лучше, численный на 2м месте, а худший результат показал асимптотический. Параметр $\theta=6$.