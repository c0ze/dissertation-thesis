"""Measurement, sampling, clone, and diagnostic dump.

Five public functions, all also exposed as :class:`Qreg` methods:

* :func:`measure_qubit` -- single-qubit projective measurement; the
  register collapses onto the measured branch and renormalises.
* :func:`measure_all` -- sample a full basis index from the
  ``|amp|^2`` distribution and collapse onto it.
* :func:`sample_distribution` -- run ``shots`` independent measurements
  *without* mutating the original register (snapshot + restore).
* :func:`clone` -- independent copy with the same amplitudes and RNG
  state. Mutations on the clone don't reach the original; both
  generators produce the same future sequence until one advances.
* :func:`dump` -- structured diagnostic: list of
  ``(basis_index, amplitude)`` pairs with ``|amp| > threshold``.

Measurement is the documented "device -> host" boundary: we deliberately
``.cpu()`` the probability vector once per call to feed the CPU
:class:`torch.Generator` (the cross-device-portable RNG choice from
Phase 0+1). Gate ops stay on device; only the measurement readout
crosses to host.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from ._assert import raise_value
from ._axis import qubit_axis

if TYPE_CHECKING:
    from .qreg import Qreg


# ---------------------------------------------------------------------------
# measure_qubit: single-qubit projective measurement with collapse.
# ---------------------------------------------------------------------------


def measure_qubit(q: Qreg, target: int) -> int:
    """Perform a projective measurement on the ``target`` qubit.

    Returns ``0`` or ``1``. The register is collapsed onto the chosen
    branch and renormalised so :meth:`Qreg.norm` remains 1.

    Uses the per-Qreg CPU RNG ``q._gen`` for the sample. Two scalar
    syncs to host: the two slice-probability sums (so we can sample on
    CPU). Raises :class:`RuntimeError` with the ``qubit:`` prefix if
    the chosen branch has probability ``<= 0``, which only happens if
    the state has drifted from normalisation.
    """
    raise_value(
        0 <= target < q._n,
        "measure_qubit: target=%d out of [0, %d)", target, q._n,
    )

    n = q._n
    axis = qubit_axis(target, n)

    # n-D view; select gives non-contiguous views of the two halves
    # along the target axis. In-place ops on these slices modify
    # q._amp's underlying storage.
    view = q._amp.view((2,) * n)
    slice_0 = view.select(axis, 0)
    slice_1 = view.select(axis, 1)

    # Compute both probabilities from the actual slice contents (rather
    # than p1 = 1 - p0) so the renormalisation is exact relative to the
    # chosen branch even if the state has drifted slightly.
    p0 = float(
        (slice_0.real * slice_0.real + slice_0.imag * slice_0.imag).sum().item()
    )
    p1 = float(
        (slice_1.real * slice_1.real + slice_1.imag * slice_1.imag).sum().item()
    )

    # Guard zero-norm states (e.g. a fresh Qreg before init_basis) up
    # front, so the sampling step below cannot pick a zero-probability
    # branch and divide by sqrt(0) at the renormalisation step.
    total = p0 + p1
    if total <= 0:
        raise RuntimeError(
            f"qubit: measure_qubit: total probability {total:.6g} <= 0; "
            f"the state has not been initialised or has drifted from "
            f"normalisation (check q.norm())"
        )

    # CPU random draw; q._gen is the per-Qreg CPU generator from Phase 0+1.
    # Sample against the normalised CDF p0/total so an artificially scaled
    # (or slightly drifted) state still produces an unbiased outcome:
    # plain `u < p0` would silently misbehave for any state where
    # p0 + p1 != 1 (the obvious case is a scaled state with p0 = p1 = 2
    # where the un-normalised comparison u < 2 is always True).
    u = float(torch.rand((), generator=q._gen).item())
    outcome = 0 if u < (p0 / total) else 1
    p_outcome = p0 if outcome == 0 else p1

    # Given total > 0 above, the chosen-branch probability is itself > 0
    # because `u < p0/total` cannot be True when p0 == 0 (the CDF cut
    # would be 0 and u is in [0, 1)), and likewise outcome cannot be 1
    # when p1 == 0 (the CDF cut would be 1.0 and u < 1.0 is always True).
    # The check below is defence in depth for floating-point oddities.
    if p_outcome <= 0:
        raise RuntimeError(
            f"qubit: measure_qubit: outcome={outcome} sampled with "
            f"p_outcome={p_outcome:.6g} <= 0; this should not happen "
            f"given total={total:.6g} > 0 (numerical edge case)"
        )

    inv_norm = 1.0 / math.sqrt(p_outcome)

    # Collapse: zero the other half, rescale the chosen half. Both
    # operations are in-place on slices that view q._amp's storage,
    # so q._amp is updated without any reshape/flatten work.
    if outcome == 0:
        slice_1.zero_()
        slice_0.mul_(inv_norm)
    else:
        slice_0.zero_()
        slice_1.mul_(inv_norm)

    return outcome


# ---------------------------------------------------------------------------
# measure_all: sample a full basis index, collapse to it.
# ---------------------------------------------------------------------------


def measure_all(q: Qreg) -> int:
    """Sample a full basis index from the ``|amp|^2`` distribution.

    The register collapses to a pure basis state at the chosen index;
    after this call ``q.norm()`` is 1.0 and exactly one amplitude is
    ``1 + 0j``.

    The probability vector is moved to CPU once so :func:`torch.multinomial`
    can use the per-Qreg CPU generator.
    """
    amp = q._amp
    # Real-valued probability vector on device, then one .cpu() sync
    # so multinomial can pull a sample using q._gen (CPU generator).
    # multinomial accepts non-normalised weights -- it renormalises
    # internally -- so we don't pre-normalise here, but we DO catch
    # the zero-norm case before calling multinomial: otherwise
    # torch.multinomial raises a raw RuntimeError without the
    # "qubit: " prefix, leaking a generic torch error to callers.
    probs = (amp.real * amp.real + amp.imag * amp.imag).detach().cpu()
    total = float(probs.sum().item())
    if total <= 0:
        raise RuntimeError(
            f"qubit: measure_all: total probability {total:.6g} <= 0; "
            f"the state has not been initialised or has drifted from "
            f"normalisation (check q.norm())"
        )

    chosen = int(torch.multinomial(probs, 1, generator=q._gen).item())

    # Collapse on device: zero everything, set the chosen index to 1+0j.
    q._amp.zero_()
    q._amp[chosen] = 1 + 0j
    return chosen


# ---------------------------------------------------------------------------
# sample_distribution: shots independent measurements, original untouched.
# ---------------------------------------------------------------------------


def sample_distribution(q: Qreg, shots: int) -> list[int]:
    """Draw ``shots`` samples from ``|amp|^2`` and return them as a list.

    Does **not** mutate ``q``: the amplitude tensor is read once into a
    probability vector and ``torch.multinomial`` produces all samples
    from it in one call, so the register is never collapsed. ``q._gen``
    is advanced by exactly ``shots`` draws.

    O(2**n + shots) work and O(2**n) extra memory for the probability
    vector. (The previous implementation cloned the full amplitude
    tensor once per shot, which scaled as O(shots * 2**n) instead.)

    ``shots=0`` is valid and returns the empty list. ``shots < 0`` raises.
    Returns a plain ``list[int]`` for ergonomic use with
    :class:`collections.Counter`.
    """
    raise_value(
        shots >= 0,
        "sample_distribution: shots=%d must be >= 0", shots,
    )
    if shots == 0:
        return []

    # Build the probability vector once. q._gen lives on CPU so we move
    # the probs over too -- multinomial requires both on the same device
    # as the generator. For complex64/128 the cast to real-valued probs
    # is straightforward.
    amp = q._amp.detach()
    probs = (amp.real * amp.real + amp.imag * amp.imag).to(
        device="cpu", dtype=torch.float64
    )
    total = float(probs.sum().item())
    if total <= 0.0:
        raise RuntimeError(
            "qubit: sample_distribution: total probability is zero "
            "(forgot to init_basis(), or non-unitary gate sequence?)"
        )

    samples = torch.multinomial(
        probs, num_samples=shots, replacement=True, generator=q._gen
    )
    return [int(x) for x in samples.tolist()]


# ---------------------------------------------------------------------------
# clone: independent Qreg with same amplitudes and RNG state.
# ---------------------------------------------------------------------------


def clone(q: Qreg) -> Qreg:
    """Return an independent copy of ``q``.

    The clone has its own ``_amp`` tensor (a detached copy of ``q._amp``)
    and its own ``_gen`` generator initialised with the same state as
    ``q._gen``. From the moment of cloning, the two registers are
    independent: mutating one's amplitudes doesn't affect the other's,
    and advancing one's RNG (by measurement) doesn't advance the
    other's. They produce the **same** future random sequence until
    one of them draws.

    Bypasses :class:`Qreg`'s ``__init__`` (via ``__new__``) to avoid
    the otherwise-wasted ``torch.zeros(2**n)`` allocation that would
    immediately get overwritten by the amplitude copy. Same trick is
    used by :func:`sample_distribution` indirectly via its snapshot
    pattern.
    """
    from .qreg import Qreg as _Qreg

    new_q = _Qreg.__new__(_Qreg)
    new_q._n = q._n
    new_q._device = q._device
    new_q._dtype = q._dtype
    new_q._amp = q._amp.detach().clone()

    # Match RNG state. Both generators are independent objects from
    # here on; only their initial state is shared.
    new_q._gen = torch.Generator()
    new_q._gen.set_state(q._gen.get_state())
    return new_q


# ---------------------------------------------------------------------------
# dump: structured diagnostic list.
# ---------------------------------------------------------------------------


def dump(q: Qreg, *, threshold: float = 0.0) -> list[tuple[int, complex]]:
    """Return a list of ``(basis_index, amplitude)`` pairs with ``|amp| > threshold``.

    Diagnostic API; does no printing (callers can pretty-print the list
    however they like). Threshold defaults to 0, which returns every
    non-zero amplitude. Pass a positive threshold to filter out small
    amplitudes that you don't care about.

    Pulls the amplitude tensor to CPU once via ``amplitudes_copy``-style
    detach/copy. The boolean mask + ``nonzero`` scan still costs
    ``O(2**n)`` work to find which amplitudes to emit; only the final
    Python ``list`` construction is bounded by the number of selected
    amplitudes. Treat ``dump()`` as a small-state diagnostic helper,
    not a sparse iterator for large registers.
    """
    raise_value(
        threshold >= 0,
        "dump: threshold=%g must be non-negative", threshold,
    )

    cpu_amp = q._amp.detach().cpu()
    # Find indices whose |amp| exceeds the threshold. .abs() on a
    # complex tensor returns magnitudes; the boolean mask is on the
    # real magnitude space.
    mask = cpu_amp.abs() > threshold
    indices = mask.nonzero(as_tuple=True)[0].tolist()
    return [(int(i), complex(cpu_amp[i].item())) for i in indices]
