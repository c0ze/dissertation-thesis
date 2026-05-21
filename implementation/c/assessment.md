# implementation/c -- coverage of thesis claims

Updated end of each implementation phase. File:line references point at
the canonical implementation site for each claim.

## §8 (sparse-gate strategy, 2026 thesis)

| Claim | Status | Location |
|---|---|---|
| In-place single-qubit gate, O(2^n) | ✓ | `matrix.c::apply_u_local` |
| Distributed state vector by top-p bits | ✓ | `matrix.c::qreg_create` + `parallel.c::is_local_qubit` |
| Pairwise MPI_Sendrecv for global qubit | ✓ | `parallel.c::exchange_amplitudes` |
| Controlled gate four-case dispatch | ✓ | `matrix.c::apply_cu` |
| Modular_exp via MPI_Alltoallv | ✓ | `parallel.c::redistribute_pairs` + `shor.c::apply_modular_exp` |
| qreg API per §12 | ✓ | `matrix.h` |

## §9 (2004 library API)

| Claim | Status | Notes |
|---|---|---|
| matrix create/init/print | ✓ (functional equiv) | qreg_create / qreg_init_basis / qreg_dump |
| tensor_product, dot_product | ✗ by design | dense operators are not materialised in v1 |
| send_matrix / get_matrix / broadcast_matrix | ✗ by design | replaced by exchange_amplitudes + redistribute_pairs |
| H, CNOT gates | ✓ | apply_h, apply_cnot |
| Deconstructor (~matrix) | ✓ | qreg_destroy |
| QFT promised but unfinished in 2004 | ✓ | `qft.c::apply_qft` |

## §9 (2026 QFT)

| Claim | Status | Location |
|---|---|---|
| QFT forward + inverse | ✓ | `qft.c::apply_qft` / `apply_qft_inverse` |
| Includes bit-reversal swaps | ✓ | `qft.c::apply_qft` final swap loop |
| Period detection on known periodic input | ✓ (tested) | `tests/test_qft.c::test_qft_detects_period` |

## §10 (2026 Grover)

| Claim | Status | Location |
|---|---|---|
| Phase-oracle callback API | ✓ | `grover.h::oracle_fn` |
| H^n -> oracle/diffusion loop | ✓ | `grover.c::apply_grover` |
| Optimum-stop tested | ✓ | `tests/test_grover.c::test_grover_over_iteration_hurts` |
| Multiple marked items | ✓ | `tests/test_grover.c::test_grover_4_marked_in_16` |

## §11 (2026 Shor)

| Claim | Status | Location |
|---|---|---|
| apply_modular_exp with y>=N pass-through | ✓ | `shor.c::apply_modular_exp` |
| Distributed via MPI_Alltoallv | ✓ | via `parallel.c::redistribute_pairs` |
| apply_shor_period (period finding) | ✓ | `shor.c::apply_shor_period` |
| shor_factor (end-to-end) | ✓ | `shor.c::shor_factor` |
| Continued-fraction post-processing | ✓ | `standart.c::continued_fraction` |
| Factor N=15 reliably | ✓ (tested) | `tests/test_shor.c::test_shor_factor_15` |

## §12 (qreg API)

Every entry in spec §6.1 is implemented in matrix.h/c. The disclosed
extensions over thesis §12 (apply_y/s/t/rx/ry/rz, apply_cz, apply_multi_*,
qreg_clone/dump, sample_distribution, shor_factor) are committed back to
the thesis in Task 41.

## Out of scope for v1

* Density matrices / mixed states.
* Noise models.
* Tensor-network / stabilizer-formalism shortcuts.
* GPU offload.
* Python bindings.

## Test matrix

All test binaries pass at NP = 1, 2, 4 via `make test`. `make test-large`
additionally runs at NP = 8 and sets `RUN_SHOR_21=1`, which unlocks the
16-qubit `test_shor_factor_21` end-to-end factoring test
(`test_shor.c::test_shor_factor_21`). On Apple Silicon `shor_factor(21)`
finishes in ~10 ms per call; on the GitHub Actions ubuntu-latest
runners budget another ~50-100 ms. The Shor-21 test is gated behind
the env var so the default `make test` loop stays fast for tight
iteration.
