# CHANGELOG

<!-- version list -->

## v0.10.11 (2026-10-08)

### Bug Fixes

- Give large Hessians a fixed memory reserve and record what they need
  ([`f7d0910`](https://github.com/simstack/molecular_qm_psi4/commit/f7d09109b20604b89e13b18cc259b8c452a8f487))

A 15 percent cut of a 125 GB allocation rejected a density-fitted Hessian that fits after a fixed OS
  and Docker reserve. Store atom count, basis, functional, the PySCF budget, and the estimated peak
  for the parent and each child.

Co-authored-by: Cursor <cursoragent@cursor.com>


## v0.10.10 (2026-10-08)

### Bug Fixes

- Bound the Hessian make_h1 aux block so a 32 GB container is not SIGKILLed
  ([`6ad64fc`](https://github.com/simstack/molecular_qm_psi4/commit/6ad64fc4fe11ab131491363eb1c850bd1023e088))

Co-authored-by: Cursor <cursoragent@cursor.com>


## v0.10.9 (2026-10-07)

### Bug Fixes

- Stop test scripts from importing psi4_calculator
  ([`7721534`](https://github.com/simstack/molecular_qm_psi4/commit/77215344bb33c37a3dad10dfae7ba423172aa44a))

create_model_table loads every testing module, and the package no longer exports that name.

Co-authored-by: Cursor <cursoragent@cursor.com>


## v0.10.8 (2026-10-07)

### Bug Fixes

- Size the JK aux block after the Coulomb vectors are resident
  ([`b2654dc`](https://github.com/simstack/molecular_qm_psi4/commit/b2654dc12f5a3bd908a896d2fb7107bae7b72da3))

Re-measuring memory inside the contraction rejected a shell window that was one function over the
  new limit.

Co-authored-by: Cursor <cursoragent@cursor.com>


## v0.10.7 (2026-10-06)

### Bug Fixes

- Pin PySCF BLAS to one thread and OpenMP to the Slurm CPU count
  ([`0da8654`](https://github.com/simstack/molecular_qm_psi4/commit/0da8654fcef7f5124e4c6277af8c077708d754a0))

Importing the package no longer loads Psi4, so the BLAS pool is sized before NumPy and does not nest
  inside the OpenMP team.

Co-authored-by: Cursor <cursoragent@cursor.com>


## v0.10.6 (2026-10-05)

### Bug Fixes

- Clear the range-separated density-fitting cache before CPHF
  ([`e737d33`](https://github.com/simstack/molecular_qm_psi4/commit/e737d3301f3c69a719f9cc0a7aeceea763aaf2eb))

make_h1 leaves a stale omega on the cached auxmol, and the next exchange build asserts. Drop that
  cache after the AO derivatives.

Co-authored-by: Cursor <cursoragent@cursor.com>

- Log Hessian assembly while atom results are loaded
  ([`2d0f676`](https://github.com/simstack/molecular_qm_psi4/commit/2d0f67635f16dd26275335fb271be0121e36866e))

The parent task was silent after the atom batches finished, so a database download looked idle.
  Report each load, the Coulomb contraction, and the final assembly.

Co-authored-by: Cursor <cursoragent@cursor.com>


## v0.10.5 (2026-10-05)

### Bug Fixes

- Drop the range-separated DF cache before Hessian CPHF
  ([`390d968`](https://github.com/simstack/molecular_qm_psi4/commit/390d968b64c93852b97bf1b70017d0384a036bd2))

make_h1 leaves a cached auxiliary molecule at omega 0, so the following get_k asserts. Clear that
  cache before solve_mo1.

Co-authored-by: Cursor <cursoragent@cursor.com>


## v0.10.4 (2026-10-05)

### Bug Fixes

- Stop calling BLAS thread setters through ctypes
  ([`c117f09`](https://github.com/simstack/molecular_qm_psi4/commit/c117f0926765a36669c55859c166b2bef510c7b4))

mkl_set_num_threads and openblas_set_num_threads segfault the Psi4 process (exit 139). Set the
  OpenMP environment and use PySCF lib.num_threads.

Co-authored-by: Cursor <cursoragent@cursor.com>


## v0.10.3 (2026-10-05)

### Bug Fixes

- Stop PySCF set_resources from loading a second BLAS
  ([`d754c50`](https://github.com/simstack/molecular_qm_psi4/commit/d754c50f0c2b3df2428cb2579f182e489368ddff))

Opening libmkl_rt.so or libopenblas.so segfaults the Psi4 process (exit 139). Call thread setters
  only if they are already linked.

Co-authored-by: Cursor <cursoragent@cursor.com>


## v0.10.2 (2026-10-05)

### Bug Fixes

- Use the full Slurm task count for PySCF Hessian threads
  ([`f27cc1e`](https://github.com/simstack/molecular_qm_psi4/commit/f27cc1e54f4a84e20c669d323da4d145835a0fc3))

A default tasks of 1 hid tasks_per_node, so Hessian nodes ran on one thread. Apply that count before
  the molecule is built and to the BLAS pools.

Co-authored-by: Cursor <cursoragent@cursor.com>


## v0.10.1 (2026-10-04)

### Bug Fixes

- Finish the DF Hessian partial without a missing auxmol or rc=137
  ([`6e9f0c7`](https://github.com/simstack/molecular_qm_psi4/commit/6e9f0c72d677dbbfa6434f9c75183d291d475722))

Attach the aux molecule density_fit() leaves unset, copy only the aux window so slices do not retain
  the full tensor, and run ProcessHeartbeat as a context manager.

Co-authored-by: Cursor <cursoragent@cursor.com>


## v0.10.0 (2026-10-03)

### Features

- Introduce `pyscf_hessian_partial` utility for density-fitted RKS partial Hessians
  ([`34c00b1`](https://github.com/simstack/molecular_qm_psi4/commit/34c00b1f002477246c8bd62595f0c587b97e344d))

- Implements functions for aux-shell grouping, memory handling, and Hessian computation. - Supports
  hybrid and range-separated functionals for accurate calculations. - Enhances auxiliary response
  handling and integration consistency checking.

- Split the DF partial Hessian into memory-sized cloud chunks.
  ([`8a7f75f`](https://github.com/simstack/molecular_qm_psi4/commit/8a7f75f98fd8dd45521d7da1f78753976e2b293d))

Aux, XC, and NLC work leaves the parent, which only sums the stored pieces with the CPHF response.

Co-authored-by: Cursor <cursoragent@cursor.com>


## v0.9.1 (2026-10-02)

### Bug Fixes

- Check DF Hessian memory on the Hessian node, not the cloud calculator.
  ([`4baf7f4`](https://github.com/simstack/molecular_qm_psi4/commit/4baf7f4f1681b6b3a4ee8d537ae7f8cef3f76601))

Co-authored-by: Cursor <cursoragent@cursor.com>


## v0.9.0 (2026-10-01)

### Bug Fixes

- Adding missing files for frequency
  ([`9b0b4fa`](https://github.com/simstack/molecular_qm_psi4/commit/9b0b4fa6038db89c808aa60609f564a821fca5ac))

- Drop snapshot.wfn.npy from successful Psi4 and PySCF jobs.
  ([`016d1ff`](https://github.com/simstack/molecular_qm_psi4/commit/016d1ff7bb3f3be61ea96a763cca17e78ce308bc))

Keep result.wfn.npy as the reusable wavefunction and attach the snapshot only when that result file
  was not produced.

Co-authored-by: Cursor <cursoragent@cursor.com>

### Features

- Add `pyscf_hessian_for_atoms_ext` node and improve batch resource assignment
  ([`85d3bef`](https://github.com/simstack/molecular_qm_psi4/commit/85d3bef7831f122198a663124259a0c825453a01))

- Introduce `pyscf_hessian_for_atoms_ext` node for cloud-based Hessian atom batch computation. -
  Modify resource assignment rules to distinguish between local and cloud resources for Hessian
  batches. - Refactor batch handling logic, removing `_batch_parameters` for streamlined workflow. -
  Update export list and docstrings to reflect new node and functionality.

- Add entropy unit conversion, max geometry optimization iterations, and PySCF Hessian support
  ([`8812bf0`](https://github.com/simstack/molecular_qm_psi4/commit/8812bf021dc93f731affdd730607c72aa2f9b2dc))

- Introduce `entropy_to_kcal_per_mol_k` for converting entropy units (Hartree/K or mHartree/K) to
  kcal/(mol·K). - Add `max_iters` field for controlling xTB geometry optimization iterations. -
  Implement `pyscf_hessian` and `pyscf_hessian_for_atoms` nodes for analytical Hessian and frequency
  analysis. - Enhance `compare_conformers` to compute entropy differences in kcal/mol/K. - Extend
  `crest_input` and `__init__.py` to support new features and models. - Update dependencies with
  added support for `molecular-qm-dftb`.

- Cap concurrent Hessian cloud VMs at three, including the master.
  ([`c44191e`](https://github.com/simstack/molecular_qm_psi4/commit/c44191e796dcdb345f7c691b28d04e18a191d62c))

Co-authored-by: Cursor <cursoragent@cursor.com>

- Reuse a finished PySCF optimization and assign Hessian batches to free VMs.
  ([`f9a141f`](https://github.com/simstack/molecular_qm_psi4/commit/f9a141f1249e0925355074459d5aefb0d113df81))

Co-authored-by: Cursor <cursoragent@cursor.com>

- Support configurable PySCF scratch directory and enhance Hessian node documentation
  ([`c6a1a8c`](https://github.com/simstack/molecular_qm_psi4/commit/c6a1a8c1ca6154b73367c7a82684294acde394e2))

- Add `PYSCF_TMPDIR` handling to isolate PySCF scratch files from orchestrator NFS volume. - Extend
  `pyscf_hessian` and `pyscf_hessian_for_atoms` node docstrings with called node references.


## v0.8.1 (2026-09-17)

### Bug Fixes

- Count DF Hessian peak RAM before choosing that path.
  ([`210ab9d`](https://github.com/simstack/molecular_qm_psi4/commit/210ab9dac485185737c98b071433b9e4e5500258))

The rhok0-only estimate reported 1.3 GB for 38-atom TZVP B3LYP, then the DF Hessian SIGKILL -9 on
  int3c IPIP blocks.

Co-authored-by: Cursor <cursoragent@cursor.com>


## v0.8.0 (2026-09-17)

### Features

- Store batch optimization geometries on QMResult.structures.
  ([`e81f095`](https://github.com/simstack/molecular_qm_psi4/commit/e81f095f6e6ed9b19dfdf0b6bc2e1e8875886379))

Co-authored-by: Cursor <cursoragent@cursor.com>


## v0.7.0 (2026-09-06)

### Features

- Add `thermo_benchmark` workflow and support for DFTB integration
  ([`912ed4e`](https://github.com/simstack/molecular_qm_psi4/commit/912ed4e8083445e1cdc69d8476ba5cf86fad62b0))

- Introduce new `thermo_benchmark` node enabling combined DFTB and PBE/def2-SVP optimization
  workflows. - Add support for `molecular_qm_dftb` with hierarchical method step execution. - Extend
  `CompareConformersResult` and outputs to include final molecule structures. - Enhance test
  coverage to validate `thermo_benchmark` functionality and related utilities. - Bump dependencies:
  `molecular-qm-models` to `0.4.0`, `Simstack` to latest `ww/new-features`.


## v0.6.4 (2026-09-06)

### Bug Fixes

- Adjust PySCF memory handling to prevent SIGKILL in memory-constrained environments
  ([`9f70957`](https://github.com/simstack/molecular_qm_psi4/commit/9f70957b5add036168475dd5f34ca2e833807439))

- Apply `_PYSCF_MEMORY_FRACTION_OF_CONTAINER` (0.85) for `max_memory` to reserve space for other
  processes. - Update logs and error messages to reflect adjusted memory budget. - Modify unit tests
  to validate new memory calculation approach.


## v0.6.3 (2026-09-05)

### Bug Fixes

- Loosen engine-specific optimization iteration timeouts.
  ([`7ee5829`](https://github.com/simstack/molecular_qm_psi4/commit/7ee58291c41f11c054c360b52dbee06bd67b3f04))

Psi4 and PySCF used a 1 h per-gradient cap that killed large TZVP jobs.

Co-authored-by: Cursor <cursoragent@cursor.com>


## v0.6.2 (2026-09-04)


## v0.6.1 (2026-09-04)

### Bug Fixes

- Keep PySCF live logs during long Hessian and opt steps
  ([`20ac21a`](https://github.com/simstack/molecular_qm_psi4/commit/20ac21a20e3834fa3ac4e289717605e69dead582))

Spawn a GIL-free heartbeat process and log iteration/Hessian start so Simstack still shows progress
  when C code holds the interpreter.

Co-authored-by: Cursor <cursoragent@cursor.com>


## v0.6.0 (2026-09-04)

### Features

- Persist Cartesian forces from Psi4 gradient single-points
  ([`80318e7`](https://github.com/simstack/molecular_qm_psi4/commit/80318e71ef093423f3458c65f2ef39620ad64a08))

Co-authored-by: Cursor <cursoragent@cursor.com>


## v0.5.1 (2026-09-04)

### Bug Fixes

- Log every PySCF optimization step to the Simstack log
  ([`312c87e`](https://github.com/simstack/molecular_qm_psi4/commit/312c87e9d2c08e206a008c41697f036f190b5c91))

Emit energy, gradient norm, and wall/CPU timings from each scanner evaluation so steps appear in
  node_runner.log like Psi4.

Co-authored-by: Cursor <cursoragent@cursor.com>


## v0.5.0 (2026-09-04)

### Refactoring

- Replace QMThermoResult with thermodynamics SimpleTable
  ([`80c0044`](https://github.com/simstack/molecular_qm_psi4/commit/80c0044635601955b78d48bb9153c71855320a23))

- Streamlined thermochemistry outputs by replacing `QMThermoResult` with `SimpleTable` for
  consistency and future extensibility. - Updated Psi4 and PySCF calculators, result classes, and
  node logic to use the new table format. - Added utility for attaching total thermochemistry values
  (e.g., G_tot, ZPE_tot) to NodeRunner. - Adjusted tests and fallback mechanisms to support the new
  format while maintaining legacy compatibility.


## v0.4.0 (2026-09-03)

### Features

- Enhance Psi4 error handling and debugging with detailed logs
  ([`bd59e68`](https://github.com/simstack/molecular_qm_psi4/commit/bd59e68cd5b773bb526f6f7a93317d556bf3a4ae))

- Added `_attach_psi4_result_files` and `_log_energy_gradient_summary` for improved artifact
  management and runtime summaries. - Introduced `psi4_out_progress` to parse and track Psi4 output
  progress, including convergence and errors. - Enhanced error reporting with
  `_meaningful_psi4_error` for clearer diagnostics during failures. - Updated artifact handling to
  prevent duplicates and ensure all relevant files are attached. - Improved optimization steps
  logging with timestamps and detailed summaries.


## v0.3.2 (2026-09-03)

### Bug Fixes

- Add logging and snapshot tracking for PySCF optimization
  ([`c329a76`](https://github.com/simstack/molecular_qm_psi4/commit/c329a766d2c45284414ac7f4b4ad6a27848b50ac))

- Introduced a `PySCFOptCycleReporter` to log and track optimization cycle details, including
  energy, delta energy, and gradient norm. - Enhanced `OptimizationSnapshotter` with streamlined
  chart recording and stdout redirection using `_TeeStdout`. - Added regex-based parsing for PySCF
  geometry optimization log lines. - Updated tests to validate new logging, charting, and reporting
  functionalities.


## v0.3.1 (2026-09-03)

### Bug Fixes

- Add orbital energy utilities and optimization geometry tracking
  ([`26952e8`](https://github.com/simstack/molecular_qm_psi4/commit/26952e80ca9e847d58ab29d123aed0c2e7eeb035))

- Introduced `orbital_energies.py` for HOMO/LUMO energy extraction and gap calculations. - Added
  `opt_structures.py` to build MoleculeList from optimization snapshots and final geometries. -
  Enhanced QM result handling with orbital energy outputs and molecule tracking.


## v0.3.0 (2026-09-03)


## v0.2.0 (2026-09-03)


## v0.1.0 (2026-09-03)

### Bug Fixes

- Add missing files for vibrational frequency and optimization timing table generation
  ([`d384326`](https://github.com/simstack/molecular_qm_psi4/commit/d3843268afa5992e7fbd8cf8f1a31e636dc4f013))

- Introduced utility functions for vibrational frequency calculations and anomaly detection. -
  Implemented `attach_vibrational_frequencies` to integrate frequency data with QM results and node
  runner. - Added table generation for optimization timing metrics, including iteration and overall
  statistics. - Developed comprehensive unit tests to validate new features.

- Add opt_coordinates option for optimization in Psi4Calculator
  ([`4883981`](https://github.com/simstack/molecular_qm_psi4/commit/4883981244caeeebe5dcb83ecb3f64d3737949ea))

- Introduced the "optking__opt_coordinates" option set to "cartesian" when optimization is enabled,
  ensuring proper handling of redundant internals during optimization. - Updated tests to verify the
  presence of this option when optimization is active and its absence when optimization is disabled.

- Add psi4 import handling and clean up molecule function signature
  ([`28df91a`](https://github.com/simstack/molecular_qm_psi4/commit/28df91ac8710242632dd669cc1a121ab77b5d85c))

- Implemented a try-except block for importing psi4 to handle potential ImportError gracefully. -
  Simplified the function signature of `_molecule_from_psi4_molecule` by removing unnecessary blank
  lines, improving code readability.

- Enhance molecule snapshot handling and optimize Psi4 integration
  ([`52b0cf4`](https://github.com/simstack/molecular_qm_psi4/commit/52b0cf45458924841a223fe0ddc2a383d4ffac2f))

- Added functions to append snapshots to dataset sections while preserving existing rows, improving
  dataset management. - Implemented conformer labeling for molecules based on RMSD thresholds,
  enhancing the identification of molecular conformers. - Refactored database interaction for
  snapshot persistence, ensuring robust handling of existing datasets and improved error management.
  - Updated tests to validate the new snapshot appending functionality and confirm correct behavior
  under various conditions.

- Enhance multistep optimizer with DFTB input persistence and validation
  ([`6508b6d`](https://github.com/simstack/molecular_qm_psi4/commit/6508b6d970d86da58329f12bb34d5458328f76f9))

- Added `compute_gradients` and `optimization` fields to `dftb_input` handling in
  `PreOptimizerInput`. - Implemented `_persist_dftb_input` function to save DFTB settings for nested
  docker reloading. - Updated `_dftb_preopt_input` to utilize `model_copy` for better field
  management. - Ensured that DFTB input is persisted during the multistep optimization process. -
  Refactored related functions for improved clarity and maintainability.

- Enhance QMInput handling and persistence in multistep optimizer
  ([`432554d`](https://github.com/simstack/molecular_qm_psi4/commit/432554d86ad133e5d13255220a25c3370d69d2c3))

- Updated `_qm_input_for_step` to use `model_copy` for better field management and ensure all fields
  are marked as modified. - Introduced `_persist_qm_input` function to save QMInput for nested
  docker reloading, improving data persistence. - Enhanced multistep optimizer to call
  `_persist_qm_input`, ensuring QMInput is saved during the optimization process. - Added tests for
  `_persist_qm_input` to validate the correct saving of iteration limits and field modifications.

- Improve Dockerfile and dependency handling logic
  ([`16aa548`](https://github.com/simstack/molecular_qm_psi4/commit/16aa548e69e2976346d0f6f7e519046b70828853))

- Adjusted Dockerfile to dynamically locate and copy `pyproject.docker` to `pyproject.toml` based on
  file location. - Corrected dependencies in `pyproject.docker` to use hyphenated names
  (`molecular-qm-models` and `molecular-qm-util`) for consistency. - Updated `tool.uv.sources` and
  `override-dependencies` to reflect corrected naming conventions.

- Prefix OptKing step-limit keywords so they reach the optimizer.
  ([`0fe15d0`](https://github.com/simstack/molecular_qm_psi4/commit/0fe15d0731ccf457ef283fb4b50901336857f564))

Unprefixed intrafrag/dynamic_level options were dropped by the Psi4 task planner, so the trust
  radius still grew to 1.0 and back-transformation aborted.

Co-authored-by: Cursor <cursoragent@cursor.com>

- Refactor molecule snapshot inspector and enhance multistep optimizer
  ([`d4fa0e6`](https://github.com/simstack/molecular_qm_psi4/commit/d4fa0e63ce4ea7220b6749aef38c64a5226f208e))

- Introduced `MoleculeSnapshotInspectorInput` and `SnapshotMethod` for improved dataset management.
  - Updated `_snapshot_row` to synchronize smiles and formula between molecule and snapshot. -
  Enhanced `aligned_rmsd` function for better geometry comparison. - Modified `PreOptimizerInput` to
  include full `dftb_input` and updated related validation logic. - Added tests for new
  functionalities in the molecule snapshot inspector and multistep optimizer. - Refactored existing
  tests for clarity and to accommodate new input structures.

- Update PreOptimizerInput schema for DFTB input handling
  ([`28f7335`](https://github.com/simstack/molecular_qm_psi4/commit/28f73351374007457539404e80b97d53a1d7ae32))

- Refactored the `json_schema` method in `PreOptimizerInput` to utilize `DftbInput.json_schema()`
  for improved schema representation. - Removed the previous handling of `dftb_input` and integrated
  a more structured definition with appropriate titles and descriptions. - Updated tests to validate
  the new schema structure and ensure correct properties are present in the DFTB input.

### Documentation

- Update return types in compare_conformers and xtb optimization functions
  ([`616482e`](https://github.com/simstack/molecular_qm_psi4/commit/616482ea747e83c8b04abe1a11f42437b9702de2))

- Enhanced docstrings in `compare_conformers` and `compare_conformers_over_temperature` to specify
  the result types returned from the calculations. - Added detailed return information for the
  `xtb_optimize_molecule_list` and `xtb_molecule_list` functions, clarifying the structure of the
  results and parameters. - Updated the `delta_g_table` function documentation to include the result
  types for better clarity on output expectations.

### Features

- Add `pyproject.toml` for project metadata and configuration
  ([`00ce3d6`](https://github.com/simstack/molecular_qm_psi4/commit/00ce3d62a3e11b2b55162fbdc729e86fb5b8ea03))

- Introduced `pyproject.toml` with project details, dependencies, and entry points for Simstack
  modules. - Defined compatibility with Python 3.12 and added MIT license.

- Add `QMEngineInput` support to `relax_harmonic` function
  ([`b3884c1`](https://github.com/simstack/molecular_qm_psi4/commit/b3884c1f4274a6a554d606dd3594e48dd8dacde1))

- Updated `relax_harmonic_testing.py` to include `QMEngineInput` for consistent quantum mechanics
  engine handling.

- Add accuracy and grid settings support in table generation
  ([`d07f6ce`](https://github.com/simstack/molecular_qm_psi4/commit/d07f6ce3034d399a09d0eb38d0466209e329afb0))

- Introduced new attributes (`scf_accuracy`, `optimization_accuracy`, and `grid_type`) to conformer
  comparison results. - Enhanced table entries to include these additional settings. - Updated
  dependencies in `uv.lock` to align with new table generation features.

- Add deploy workflow and resolve container deps from git
  ([`1223f28`](https://github.com/simstack/molecular_qm_psi4/commit/1223f2881419844eb9f5fcedb8b53216b6d3fde6))

Co-authored-by: Cursor <cursoragent@cursor.com>

- Add fallbacks for missing DFTB dependencies and improve Psi4 logging
  ([`36035a0`](https://github.com/simstack/molecular_qm_psi4/commit/36035a004d94e0332b4dad5c67fa33201b5e4bd1))

- Implemented fallbacks to handle missing DFTB binaries or inputs, enabling graceful degradation of
  features in unsupported environments. - Simplified Psi4 logging by removing `OptKingSummaryFilter`
  and consolidating log handling logic. - Updated Dockerfile with enhanced dependency management and
  streamlined build process. Added conda Open Babel support (`BABEL_DATADIR`) for improved
  compatibility. - Introduced `resolve_uv_git_shas.py` enhancements to normalize and expand cache
  key generation with `HEAD` commits. - Improved Docker documentation and build sequence for both
  local and Simstack contexts. - Refactored `pyproject.docker` to include `molecular_qm_util` and
  `simstack` dependencies for better integration.

- Add PySCF calculator, thermochemistry, and shared QM engine modules
  ([`0056784`](https://github.com/simstack/molecular_qm_psi4/commit/0056784494d3dc50579d5dd465f580581ad75291))

Co-authored-by: Cursor <cursoragent@cursor.com>

- Add SCF/optimization accuracy and grid settings to multistep optimizer
  ([`2716319`](https://github.com/simstack/molecular_qm_psi4/commit/27163190d4111a874bdee6c026907a4301bbb676))

- Introduced `scf_accuracy`, `optimization_accuracy`, and `grid_type` to multistep optimizer and
  corresponding QM inputs. - Enhanced table generation with detailed timing metrics (wall and CPU).
  - Added functions for extracting and attaching timing data from calculations. - Updated Psi4
  optimizations to record timing history and support new settings. - Improved tests to validate the
  integration of accuracy, grid settings, and timing data.

- Extend Psi4 functionality and enhance dependency management
  ([`6e99d49`](https://github.com/simstack/molecular_qm_psi4/commit/6e99d497f9313428cce3fc8abdfa90fd4e77cdca))

- Updated `Dockerfile` to include additional dependencies (`molecular-qm-dftb`, `pubchempy`, etc.)
  and refined build sequence for flexibility. - Introduced `_SNAPSHOT_WFN_NAME` and
  `_cleanup_snapshot_files` for better wavefunction snapshot handling and cleanup. - Added fallback
  attributes (`smiles`, `formula`, `basis_set`, `functional`) in `compare_conformers` for improved
  metadata resilience. - Enhanced `temperature_analysis` to handle broader parent call paths
  (`compare_energy` and `compare_conformers`) and improved child node validation with robust
  logging. - Implemented additional safeguards for molecule and QMInput metadata handling across
  modules.

- Honor DFT grid and abort stalled Psi4 optimizations.
  ([`02b2040`](https://github.com/simstack/molecular_qm_psi4/commit/02b204067ab44cb54ed68933cba7203da8ba48cf))

Map grid_type and scf_accuracy onto Psi4 options, fail hung or oscillating geometry steps, and log
  timestamped energy and gradient each iteration.

Co-authored-by: Cursor <cursoragent@cursor.com>

- Integrate PySCF engine support alongside Psi4
  ([`acb4968`](https://github.com/simstack/molecular_qm_psi4/commit/acb496850f7da82099c8db3143a58ca31027f721))

- Added support for PySCF as an alternative quantum mechanics engine. - Updated
  `multistep_optimizer`, `geometric_neb`, and `relax_harmonic` to respect the selected engine. -
  Introduced `QMEngineInput` for unified engine handling. - Enhanced templates for both Psi4 and
  PySCF in DFT and gradient computations. - Updated tests and schemas to include PySCF-related
  configurations and validation.

- Integrate UV Git SHA resolution into build workflow
  ([`cab9f88`](https://github.com/simstack/molecular_qm_psi4/commit/cab9f88307262807f5565578e87a273bd9058d5d))

- Added `resolve_uv_git_shas.py` step to fetch UV Git SHAs dynamically. - Passed UV Git SHAs as
  build arguments to Docker images for improved version tracking and reproducibility.

- Rename project to `molecular_qm_psi4` and update metadata
  ([`ccdff56`](https://github.com/simstack/molecular_qm_psi4/commit/ccdff5674540be1d0d29d3556d3067e49a9731f0))

- Updated `pyproject.toml` to reflect the project's renaming from `molecular_qm_orca` to
  `molecular_qm_psi4`. - Enhanced dependency and entry-point configurations, introducing
  `molecular_qm_util` and `molecular_qm_dftb`. - Simplified Python compatibility and added Hatch
  build system support. - Defined repository dependencies and improved package structure inclusion
  for wheel builds.

- Update Dockerfile with new dependencies and Simstack workdir improvements
  ([`346721f`](https://github.com/simstack/molecular_qm_psi4/commit/346721f38ff171a514ead2739d4c3e109a962aa0))

- Added `crest` and `xtb-python` dependencies to the Dockerfile. - Created and linked the
  `/tmp/simstack` directory to maintain compatibility with Simstack workdir configurations.

### Refactoring

- Remove DFTB fallbacks and standardize dependencies in multistep optimizer
  ([`b2aa0d5`](https://github.com/simstack/molecular_qm_psi4/commit/b2aa0d59880d36587f64ebffded975ef6ea86d2e))

- Eliminated fallback logic for missing DFTB dependencies, ensuring standardization across
  environments. - Updated `pyproject.docker` to include `molecular_qm_dftb` as a required
  dependency. - Simplified `_dftb_preopt_input` by removing redundant parameter assignments for DFTB
  inputs.

- Replace `compare_conformers_over_temperature` with `temperature_analysis`
  ([`9ae1480`](https://github.com/simstack/molecular_qm_psi4/commit/9ae14808880e3ef4b727147684f414c9ced6875c))

- Removed `compare_conformers_over_temperature` and introduced `temperature_analysis` for
  streamlined thermochemical property analysis over temperature ranges. - Refactored database
  interactions to directly utilize completed `psi4_calculator` results. - Updated tests to validate
  `temperature_analysis` functionality and ensure correct integration.

- Standardize iteration limits, add dispersion and iteration metrics to tables
  ([`9fed963`](https://github.com/simstack/molecular_qm_psi4/commit/9fed963be843df251d0b66eeba48f5679e506079))

- Replaced per-step configurable iteration limits with a consistent global default of 1000 for SCF
  and geometry optimization iterations. - Updated tables to include dispersion correction and
  per-step iteration counts. - Simplified timing-table logic; removed redundant optimization fields
  (`wall_time_s`, `cpu_time_s`) in Psi4 and PySCF integration. - Enhanced timing data handling,
  limiting charts to the last 20 steps for better clarity. - Adjusted tests to align with the new
  behavior and metrics.
