from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sabr_replicate import (
    FDMConfig,
    MonteCarloConfig,
    SABRParams,
    build_table1_fdm_benchmark,
    build_table2_fdm_benchmark,
    case_table_3,
    european_call_price,
    fdm_benchmark_prices,
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


def _default_fdm_config() -> FDMConfig:
    return FDMConfig()


def _strike_benchmark_for_case(case_name: str, strike_ratios: list[float]) -> dict[float, float]:
    case = case_table_3()[case_name]
    params = SABRParams(
        f0=case["f0"],
        sigma0=case["sigma0"],
        nu=case["nu"],
        rho=case["rho"],
        beta=case["beta"],
    )
    strikes = [params.f0 * x for x in strike_ratios]
    return fdm_benchmark_prices(
        params=params,
        maturity=case["maturity"],
        strikes=strikes,
        config=_default_fdm_config(),
    )


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
    parser.add_argument(
        "--benchmark-source",
        choices=["paper", "fdm", "mc", "none"],
        default="paper",
        help="Benchmark source: paper tables, PDE/FDM solver, internal high-resolution MC, or none.",
    )
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
        if args.benchmark_source == "none":
            raise ValueError("figure2 requires a benchmark source of 'mc' or 'fdm'.")
        benchmark_source = "fdm" if args.benchmark_source == "fdm" else "mc"
        df = figure2_runtime_tradeoff(
            n_paths_base=args.n_paths,
            n_repeats=args.repeats,
            seed0=args.seed,
            benchmark_source=benchmark_source,
            fdm_config=_default_fdm_config() if benchmark_source == "fdm" else None,
        )
        _print_frame(df)
        _maybe_save(df, args.output_csv)
        return 0

    if args.experiment == "figure3":
        if args.benchmark_source == "none":
            raise ValueError("figure3 requires a benchmark source of 'mc' or 'fdm'.")
        benchmark_source = "fdm" if args.benchmark_source == "fdm" else "mc"
        df = run_figure3_experiment(
            n_paths=args.n_paths,
            n_repeats=args.repeats,
            seed0=args.seed,
            benchmark_source=benchmark_source,
            fdm_config=_default_fdm_config() if benchmark_source == "fdm" else None,
        )
        _print_frame(df)
        _maybe_save(df, args.output_csv)
        return 0

    if args.experiment == "table1":
        if args.benchmark_source == "fdm":
            benchmark = build_table1_fdm_benchmark(config=_default_fdm_config())
        elif args.benchmark_source == "none":
            benchmark = None
        elif args.benchmark_source == "paper":
            benchmark = None
        else:
            raise ValueError("table1 benchmark source must be 'paper', 'fdm', or 'none'.")
        kwargs = {"benchmark": benchmark} if args.benchmark_source != "paper" else {}
        df = run_table1_experiment(n_paths=args.n_paths, n_repeats=args.repeats, seed0=args.seed, **kwargs)
    elif args.experiment == "table2":
        if args.benchmark_source == "fdm":
            benchmark = build_table2_fdm_benchmark(config=_default_fdm_config())
        elif args.benchmark_source == "none":
            benchmark = None
        elif args.benchmark_source == "paper":
            benchmark = None
        else:
            raise ValueError("table2 benchmark source must be 'paper', 'fdm', or 'none'.")
        kwargs = {"benchmark": benchmark} if args.benchmark_source != "paper" else {}
        df = run_table2_experiment(n_paths=args.n_paths, n_repeats=args.repeats, seed0=args.seed, **kwargs)
    elif args.experiment == "table4":
        kwargs = {}
        if args.benchmark_source == "fdm":
            kwargs["benchmark_prices"] = _strike_benchmark_for_case("Case I", [0.2, 0.4, 0.8, 1.0, 1.2, 1.6, 2.0])
        elif args.benchmark_source == "none":
            kwargs["benchmark_prices"] = {}
        df = run_table4_experiment(n_paths=args.n_paths, n_repeats=args.repeats, seed0=args.seed, **kwargs)
    elif args.experiment == "table5":
        kwargs = {}
        if args.benchmark_source == "fdm":
            kwargs["benchmark_prices"] = _strike_benchmark_for_case("Case II", [0.2, 0.4, 0.8, 1.0, 1.2, 1.6, 2.0])
        elif args.benchmark_source == "none":
            kwargs["benchmark_prices"] = {}
        df = run_table5_experiment(n_paths=args.n_paths, n_repeats=args.repeats, seed0=args.seed, **kwargs)
    elif args.experiment == "table6":
        kwargs = {}
        if args.benchmark_source == "fdm":
            kwargs["benchmark_prices"] = _strike_benchmark_for_case("Case III", [0.4, 0.8, 1.0, 1.2, 1.6, 2.0])
        elif args.benchmark_source == "none":
            kwargs["benchmark_prices"] = {}
        df = run_table6_experiment(n_paths=args.n_paths, n_repeats=args.repeats, seed0=args.seed, **kwargs)
    elif args.experiment == "table7":
        if args.benchmark_source == "none":
            raise ValueError("table7 requires a benchmark source of 'mc' or 'fdm'.")
        benchmark_source = "fdm" if args.benchmark_source == "fdm" else "mc"
        df = run_table7_experiment(
            n_paths_base=args.n_paths,
            n_repeats=args.repeats,
            seed0=args.seed,
            benchmark_source=benchmark_source,
            fdm_config=_default_fdm_config() if benchmark_source == "fdm" else None,
        )
    elif args.experiment == "validate":
        kwargs = {"quick_mode": args.quick}
        if args.benchmark_source == "fdm":
            kwargs["table1_benchmark"] = build_table1_fdm_benchmark(config=_default_fdm_config())
            kwargs["table2_benchmark"] = build_table2_fdm_benchmark(config=_default_fdm_config())
        elif args.benchmark_source == "none":
            kwargs["table1_benchmark"] = None
            kwargs["table2_benchmark"] = None
        elif args.benchmark_source == "mc":
            raise ValueError("validate benchmark source must be 'paper', 'fdm', or 'none'.")
        out = run_full_validation(**kwargs)
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
