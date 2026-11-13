# polymarket-nvidia-ceo-forecast-01 entry — Gecko (w4)

A forecaster for Polymarket's "Next CEO of NVIDIA?" event. `main.py`
(Python 3.13, standard library only) reads one case from the input
directory and writes, for each market and each of the 240 minutes
after the checkpoint, five quantiles (5%, 25%, 50%, 75%, 95%) of
that minute's price to the output directory.

## Model

The long-view variant:

- **Median = persistence.** The last observed price.
- **Long memory:** the last 2160 minutes (36 hours) feed both the
  empirical change table (anchors 1, 5, 10, 30, 45, 90, 180, 240 —
  shifted toward the mid horizons a 4-hour forecast lives at) and
  the sigma estimate.
- **Spread = an even 50/50 blend** of the empirical quantiles
  (interpolated in sqrt(h)) and a Gaussian walk with sigma from
  the mean absolute one-minute change, floored at 0.0015 and
  capped at 0.005.
- **Volatility regime:** the last 12 hours vs the whole history,
  clamped to [0.7, 1.6].
- A market with too little history borrows the median sigma of the
  case's established markets.

Quantiles are clamped to [0, 1], kept non-decreasing and rounded to
6 decimals. Deterministic; well under a second per case.

Backtested on the kit's practice inputs (later cases as ground
truth), this scores a mean pinball loss of about 0.382 against the
house baseline's 0.399 on the same folds.

## License

MIT — see `LICENSE`.
