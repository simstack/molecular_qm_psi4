"""Picklable xTB workers; native libraries and calculators stay in child processes."""

import os


def initialize_xtb_worker(threads):
    """Set native thread limits before importing NumPy, SciPy, or xTB."""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(threads)


def calculate_xtb_molecule(payload):
    import numpy as np
    from xtb.interface import Calculator, Param
    from xtb.libxtb import VERBOSITY_MUTED

    numbers = np.asarray(payload["numbers"], dtype=int)
    positions = np.asarray(payload["positions"], dtype=float)
    method = getattr(Param, payload["method"])
    calc = Calculator(method, numbers, positions, payload["charge"], payload["uhf"])
    calc.set_verbosity(VERBOSITY_MUTED)
    if payload["solvent_model"] != "none":
        calc.set_solvent(payload["solvent"], getattr(Param, payload["solvent_model"].upper()))

    optimization = None
    if payload["optimize"]:
        from scipy.optimize import minimize

        def objective(flat_positions):
            calc.update(flat_positions.reshape(-1, 3))
            result = calc.singlepoint()
            return result.get_energy(), result.get_gradient().flatten()

        initial_energy, _ = objective(positions.flatten())
        optimized = minimize(
            objective, positions.flatten(), jac=True, method="L-BFGS-B",
            options={"maxiter": payload["max_iters"]},
        )
        positions = optimized.x.reshape(-1, 3)
        calc.update(positions)
        optimization = {
            "initial_energy": float(initial_energy),
            "converged": bool(optimized.success),
            "iterations": int(optimized.nit),
            "message": str(optimized.message),
        }
    result = calc.singlepoint()
    return {
        "energy": float(result.get_energy()),
        "positions": positions.tolist(),
        "gradients": result.get_gradient().tolist() if payload["compute_gradients"] else None,
        "optimization": optimization,
    }
