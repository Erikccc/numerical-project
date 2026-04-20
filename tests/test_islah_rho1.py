from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sabr_replicate import MonteCarloConfig, SABRParams, simulate_terminal_forward_islah


def test_islah_rho_one_no_divide_by_zero() -> None:
    """The Islah branch should handle the |rho| = 1 limit without singularities."""
    for beta in (0.4, 0.6, 0.8):
        params = SABRParams(
            f0=1.0,
            sigma0=0.2,
            nu=0.2,
            rho=1.0,
            beta=beta,
        )
        mc = MonteCarloConfig(maturity=1.0, step=1.0, n_paths=2_000, seed=123)

        terminal = simulate_terminal_forward_islah(params, mc)

        assert terminal.shape == (mc.n_paths,)
        assert np.isfinite(terminal).all()
        assert np.all(terminal >= 0.0)
