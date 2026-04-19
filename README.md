# Replication Starter for 2408.01898v2

This workspace contains a minimal Python replication scaffold for:

Choi, Hu, Kwok, "Efficient and accurate simulation of the stochastic-alpha-beta-rho model"

The current scaffold focuses on reproducing the paper's core Monte Carlo machinery:

- Algorithm 1: shifted-lognormal sampling of conditional integrated variance
- Algorithm 2: martingale-preserving CEV approximation for the conditional forward
- Algorithm 3: exact sampling of a CEV transition via a shifted-Poisson mixture gamma law
- Algorithm 4: full SABR terminal simulation over a time grid

## Files

- `sabr_replicate.py`: core implementation
- `run_experiments.py`: CLI entrypoint for tables, figures, and validation

## What is implemented

- Exact volatility stepping
- Conditional moment formulas for the normalized integrated variance
- Shifted-lognormal approximation with fixed shift `lambda = 5/6`
- Exact CEV sampling for `0 < beta < 1`
- Special handling for `beta = 1` and `|rho| = 1`
- European call pricing from Monte Carlo samples
- Table 3 parameter presets as named cases
- Direct experiment entrypoints for Tables 1, 2, 4, 5, 6, 7
- Figure 1 moment-comparison dataset
- Figure 2 / Table 7 convergence dataset
- Figure 3 comparison dataset between the paper scheme and Islah's approximation
- Paper-reference rows for analytic approximations and legacy Monte Carlo baselines
- A 2D SABR PDE / finite-difference benchmark solver in `(F, log sigma)` coordinates
- CLI support for switching benchmark sources with `--benchmark-source paper|fdm|mc|none`

## What is not implemented yet

- Direct reimplementation of competing baselines such as Euler, low-bias, PSE, Hagan, ZC Map, or Hyb ZC Map
- Variance reduction and full performance tuning

## Run

Run with any Python that has `numpy` and `pandas` installed:

```powershell
python .\run_experiments.py --experiment table1 --paper-scale
python .\run_experiments.py --experiment table1 --paper-scale --benchmark-source fdm
python .\run_experiments.py --experiment table4 --paper-scale
python .\run_experiments.py --experiment table7 --n-paths 20000 --repeats 3 --benchmark-source fdm
python .\run_experiments.py --experiment figure3 --n-paths 50000 --repeats 5 --benchmark-source fdm
python .\run_experiments.py --experiment validate --quick --benchmark-source fdm
```

You can also save any tabular output:

```powershell
python .\run_experiments.py --experiment table7 --output-csv .\outputs\table7.csv
```

## Notes

- The paper's formulas were transcribed from the PDF and implemented directly.
- `--benchmark-source paper` uses the tabulated paper benchmarks when they are available.
- `--benchmark-source fdm` recomputes benchmark prices with the built-in PDE/FDM solver.
- `table7` / `figure3` fall back to the internal high-resolution Monte Carlo benchmark unless `--benchmark-source fdm` is requested.
- This is now a stronger reproduction scaffold with direct table/figure entrypoints and a built-in PDE benchmark, but it is still not a finished paper-grade reproduction package.
- The fastest path from here is:
  1. replace paper-reference baseline rows with actual baseline implementations,
  2. tune variance reduction and runtime for paper-scale sweeps,
  3. compare the new PDE benchmarks against an independent reference implementation.
