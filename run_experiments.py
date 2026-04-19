from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sabr_replicate import (
    MonteCarloConfig,
    SABRParams,
    case_table_3,
    european_call_price,
    figure1_moment_comparison,
    figure2_runtime_tradeoff,
    martingale_test,
    run_figure3_experiment,
    run_full_validation,
    run_table1_experiment,
    run_table2_experiment,
    run_table4_experiment,
    run_table5_experiment,
    run_table6_experiment,
    run_table7_experiment,
    simulate_terminal_forward,
)


def _print_frame(df: pd.DataFrame) -> None:
    if df.empty:
        print("(empty DataFrame)")
        return
    print(df.to_string(index=False))


def _maybe_save(df: pd.DataFrame, output_csv: str | None) -> None:
    if output_csv is None:
        return
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\nSaved CSV to {path}")


def _paper_scale_defaults(args: argparse.Namespace) -> None:
    if not args.paper_scale:
        return
    if args.experiment in {"table1", "table2", "table4", "table5", "table6"}:
        args.n_paths = 100_000
        args.repeats = 50
    elif args.experiment in {"table7", "figure2", "figure3"}:
        args.n_paths = 100_000
        args.repeats = 2
    elif args.experiment == "validate":
        args.quick = False


def run_case_i_starter(n_paths: int, seed: int) -> pd.DataFrame:
    case = case_table_3()["Case I"]
    params = SABRParams(
        f0=case["f0"],
        sigma0=case["sigma0"],
        nu=case["nu"],
        rho=case["rho"],
        beta=case["beta"],
    )
    mc = MonteCarloConfig(maturity=case["maturity"], step=1.0, n_paths=n_paths, seed=seed)
    terminal = simulate_terminal_forward(params, mc)
    strikes = [0.2, 0.4, 0.8, 1.0, 1.2, 1.6, 2.0]
    return pd.DataFrame(
        {
            "strike": strikes,
            "call_price": [european_call_price(terminal, strike) for strike in strikes],
            "mean_terminal_forward": float(terminal.mean()),
        }
    )


def run_case_v_martingale(n_paths: int, seed: int) -> pd.DataFrame:
    case = case_table_3()["Case V"]
    params = SABRParams(
        f0=case["f0"],
        sigma0=case["sigma0"],
        nu=case["nu"],
        rho=case["rho"],
        beta=case["beta"],
    )
    return martingale_test(params, maturities=list(range(1, 11)), step=1.0, n_paths=n_paths, seed0=seed)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SABR replication experiments.")
    parser.add_argument(
        "--experiment",
        required=True,
        choices=[
            "starter-case1",
            "martingale-case5",
            "figure1",
            "figure2",
            "figure3",
            "table1",
            "table2",
            "table4",
            "table5",
            "table6",
            "table7",
            "validate",
        ],
    )
    parser.add_argument("--n-paths", type=int, default=20_000)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--paper-scale", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--hat-nu", type=float, default=0.4)
    parser.add_argument("--output-csv", type=str, default=None)
    args = parser.parse_args()

    _paper_scale_defaults(args)

    if args.experiment == "starter-case1":
        df = run_case_i_starter(args.n_paths, args.seed)
        _print_frame(df)
        _maybe_save(df, args.output_csv)
        return 0

    if args.experiment == "martingale-case5":
        df = run_case_v_martingale(args.n_paths, args.seed)
        _print_frame(df)
        _maybe_save(df, args.output_csv)
        return 0

    if args.experiment == "figure1":
        df = figure1_moment_comparison(hat_nu=args.hat_nu)
        _print_frame(df.head(15))
        if len(df) > 15:
            print(f"\n... showing first 15 of {len(df)} rows")
        _maybe_save(df, args.output_csv)
        return 0

    if args.experiment == "figure2":
        df = figure2_runtime_tradeoff(n_paths_base=args.n_paths, n_repeats=args.repeats, seed0=args.seed)
        _print_frame(df)
        _maybe_save(df, args.output_csv)
        return 0

    if args.experiment == "figure3":
        df = run_figure3_experiment(n_paths=args.n_paths, n_repeats=args.repeats, seed0=args.seed)
        _print_frame(df)
        _maybe_save(df, args.output_csv)
        return 0

    if args.experiment == "table1":
        df = run_table1_experiment(n_paths=args.n_paths, n_repeats=args.repeats, seed0=args.seed)
    elif args.experiment == "table2":
        df = run_table2_experiment(n_paths=args.n_paths, n_repeats=args.repeats, seed0=args.seed)
    elif args.experiment == "table4":
        df = run_table4_experiment(n_paths=args.n_paths, n_repeats=args.repeats, seed0=args.seed)
    elif args.experiment == "table5":
        df = run_table5_experiment(n_paths=args.n_paths, n_repeats=args.repeats, seed0=args.seed)
    elif args.experiment == "table6":
        df = run_table6_experiment(n_paths=args.n_paths, n_repeats=args.repeats, seed0=args.seed)
    elif args.experiment == "table7":
        df = run_table7_experiment(n_paths_base=args.n_paths, n_repeats=args.repeats, seed0=args.seed)
    elif args.experiment == "validate":
        out = run_full_validation(quick_mode=args.quick)
        for key in ("table1_df", "table2_df", "martingale_df"):
            print(f"\n[{key}]")
            _print_frame(out[key])
            if args.output_csv is not None:
                base = Path(args.output_csv)
                stem = base.stem if base.suffix else base.name
                parent = base.parent if base.suffix else base
                parent.mkdir(parents=True, exist_ok=True)
                path = parent / f"{stem or 'validation'}_{key}.csv"
                out[key].to_csv(path, index=False)
                print(f"Saved CSV to {path}")
        return 0
    else:
        raise ValueError(f"Unsupported experiment: {args.experiment}")

    _print_frame(df)
    _maybe_save(df, args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
