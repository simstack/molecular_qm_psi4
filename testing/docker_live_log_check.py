"""Run inside molecular-qm-psi4:latest to verify live driver/opt logs.

    docker run --rm --entrypoint python ^
      -v "%CD%/molecular_qm_psi4/testing:/work" ^
      molecular-qm-psi4:latest /work/docker_live_log_check.py
"""
from pathlib import Path

import psi4

from molecular_qm_models import BasisSet, Functional, Molecule, QMInput
from molecular_qm_psi4.nodes.psi4_calculator import redirect_psi4_logs
from molecular_qm_psi4.util.psi4_calculator import Psi4Calculator


class CaptureRunner:
    def __init__(self):
        self.msgs = []

    def info(self, msg):
        self.msgs.append(str(msg))
        print(f"LIVE {msg}", flush=True)

    def warning(self, msg):
        print(f"LIVE-WARN {msg}", flush=True)

    def error(self, msg):
        print(f"LIVE-ERR {msg}", flush=True)


def main():
    water = Molecule.from_sites(
        elements=["O", "H", "H"],
        sites=[[0.0, 0.0, 0.117], [0.0, 0.755, -0.471], [0.0, -0.755, -0.471]],
    )
    qm_input = QMInput(
        molecule=water,
        basis_set=BasisSet(basis_set="STO3G"),
        functional=Functional(functional="PBE"),
        optimization=True,
        print_level=1,
        max_optimization_iterations=20,
        max_scf_iterations=50,
    )
    runner = CaptureRunner()
    log_path = Path("psi4.log")
    seen_before_exit = []

    with redirect_psi4_logs(log_path, qm_input.print_level, node_runner=runner):
        calc = Psi4Calculator(qm_input, node_runner=runner)
        calc.set_resources("1 GB", 1)
        calc.set_molecule()
        calc.set_options()
        psi4.optimize(calc.get_method(), return_wfn=True)
        seen_before_exit.extend(runner.msgs)

    if not any("Return gradient()" in msg for msg in seen_before_exit):
        raise SystemExit(
            "FAIL: no live Return gradient() messages during optimize()\n"
            f"captured={seen_before_exit[:20]!r}"
        )
    log_size = log_path.stat().st_size if log_path.exists() else 0
    print(f"PASS live gradient logs={sum('Return gradient()' in m for m in seen_before_exit)} "
          f"psi4.log bytes={log_size}", flush=True)


if __name__ == "__main__":
    main()
