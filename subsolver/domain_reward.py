"""Grade a student answer with the grader that owns its data_source.

The subsolver reward was written for MATH: normalize the \\boxed{...} payload and
string-compare it to the ground truth. Reasoning360 spans four domains we care
about — math, codegen, logic, stem — and each has its own official grader behind
`verl.utils.reward_score.default_compute_score`, keyed by the `data_source` prefix:

    math__*                  naive_dapo (or prime_math / llm_judge via extra_info)
    codegen__*               coder1 — RUNS the code against tests
    logic__zebra_puzzle*     zebra_puzzle
    logic__ordering_puzzle*  puzzles_dataset
    logic__graph*            graph_dataset
    stem__gpqa*              gpqa / supergpqa
    stem__supergpqa*         supergpqa

Using those instead of our string compare is what makes the pipeline portable: a
zebra-puzzle answer is a grid, a codegen answer is a program, and neither survives
`\\boxed{}` extraction plus normalization.

Two things follow from that, and they are the reason this module exists rather than
a one-line swap:

  WHAT gets graded. Our reward hands the grader `parse_boxed(continuation)`, the
  payload of the last box. Every Reasoning360 grader does its own extraction from
  the RAW completion, so it must receive the whole continuation. `needs_boxed()`
  marks the domains where the boxed payload is still the right input (math keeps
  the old behaviour so existing runs stay reproducible).

  Code EXECUTION. codegen grading runs model-written programs. The coder1 default
  backend is `unsafe_local`, i.e. straight `exec` on this host. Never train with
  that: set CODER1_EXEC=bwrap (available here) or point it at a sandbox server.
  `assert_exec_backend_safe()` fails loudly rather than silently running untrusted
  code, and is called from the launchers.
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# data_source prefixes whose grader wants the \boxed{...} payload, not the raw text.
# Everything else is handed the full continuation.
_BOXED_PREFIXES = ("math", "aime", "lighteval/MATH", "DigitalLearningGmbH/MATH")

# Backends that actually confine the process. "unsafe_local" is `exec` on the host.
_SAFE_EXEC_BACKENDS = ("bwrap", "firejail", "docker", "sandboxfusion", "ces", "kira")


def needs_boxed(data_source: str) -> bool:
    """True iff this domain's grader expects the boxed payload rather than raw text."""
    ds = (data_source or "").strip()
    return ds.startswith(_BOXED_PREFIXES)


def is_code_domain(data_source: str) -> bool:
    ds = (data_source or "").strip()
    return ds.startswith("codegen") or ds.startswith("simulation__code")


_unsafe_warned = False


def assert_exec_backend_safe() -> None:
    """Refuse to grade code with the host-exec backend, unless explicitly opted in.

    coder1 defaults to CODER1_EXEC=unsafe_local, which runs generated programs
    directly on this machine; training generates thousands of them. The guard
    fails closed so nobody inherits that by accident.

    SUBSOLVER_ALLOW_UNSAFE_EXEC=1 overrides it. That opt-in exists because no
    namespace sandbox can run in this container: the host sets
    kernel.apparmor_restrict_unprivileged_userns=1, which transitions any
    unprivileged user namespace to the enforced `unprivileged_userns` AppArmor
    profile and denies the uid-map/mount/loopback operations bwrap (and firejail,
    nsjail) are built on. unsafe_local still applies a 10GB ulimit and a per-run
    timeout, but no filesystem or network isolation. It is an explicit opt-in.
    """
    global _unsafe_warned
    backend = os.environ.get("CODER1_EXEC", "unsafe_local")
    if backend in _SAFE_EXEC_BACKENDS:
        return
    if os.environ.get("SUBSOLVER_ALLOW_UNSAFE_EXEC", "") == "1":
        if not _unsafe_warned:
            logger.warning("CODER1_EXEC=%s with SUBSOLVER_ALLOW_UNSAFE_EXEC=1: "
                           "model-written code runs on the host (ulimit+timeout only)",
                           backend)
            _unsafe_warned = True
        return
    raise RuntimeError(
        f"CODER1_EXEC={backend!r} executes model-written code on the host. "
        f"Set one of {_SAFE_EXEC_BACKENDS}, or acknowledge the risk explicitly "
        "with SUBSOLVER_ALLOW_UNSAFE_EXEC=1."
    )


def _as_float(res: Any) -> float:
    """default_compute_score returns a float, a dict with 'score', or a sequence."""
    if isinstance(res, dict):
        for k in ("score", "acc"):
            if k in res:
                return float(res[k])
        raise ValueError(f"grader dict has no score/acc: {sorted(res)}")
    if isinstance(res, (int, float, bool)):
        return float(res)
    if isinstance(res, (list, tuple)) and res:
        return _as_float(res[0])
    raise ValueError(f"cannot read a score out of {type(res).__name__}")


def score(data_source: str, solution_str: str, ground_truth: Any,
          extra_info: dict[str, Any] | None = None) -> float:
    """Domain grader score in [0, 1]. Returns 0.0 rather than raising.

    A grader that throws — a malformed program, a puzzle parser that trips on an
    empty answer — must not kill the rollout, since the reward is computed inside
    the training loop. The exception is logged once per kind.
    """
    if is_code_domain(data_source):
        assert_exec_backend_safe()
    try:
        from verl.utils.reward_score import default_compute_score
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            "verl.utils.reward_score.default_compute_score is required for "
            "domain-aware grading; this looks like a verl tree without it"
        ) from e
    try:
        return _as_float(default_compute_score(
            data_source, solution_str, ground_truth, extra_info))
    except Exception as e:  # noqa: BLE001
        logger.warning("grader failed for data_source=%s (%s); scoring 0",
                       data_source, type(e).__name__)
        return 0.0


def is_correct(data_source: str, solution_str: str, ground_truth: Any,
               extra_info: dict[str, Any] | None = None,
               threshold: float = 1.0) -> bool:
    """Bool wrapper. Default threshold 1.0 = fully correct.

    coder1 gives partial credit (format_reward + answer_reward), and the logic
    graders can return per-cell fractions, so "correct" has to mean the top of the
    scale rather than any positive score — the is-cut reward multiplies by it.
    """
    return score(data_source, solution_str, ground_truth, extra_info) >= threshold - 1e-9
