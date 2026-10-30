"""Solo pass-rate measurement (avg@n, no tools), pipelined + resumable.

`--system_prompt` picks the system message instead of always replaying the
row's own:

  row      the parquet row's system message ("" if none)   [R360 behaviour]
  code     prompts.STUDENT_SYSTEM_PROMPT_CODE   (fenced-program clause)
  default  prompts.STUDENT_SYSTEM_PROMPT        (\\boxed clause)
  none     no system message at all

The label a gate trains on must be "can the student solve this under the prompt
it will actually run with", so the coding-only pool is measured with `code`.
Grading goes through subsolver.domain_reward (codegen -> coder1, executes the
program; CODER1_EXEC=unsafe_local + SUBSOLVER_ALLOW_UNSAFE_EXEC=1 required here).

Usage:
  SUBSOLVER_STUDENT_ENDPOINT=http://127.0.0.1:8501,... SUBSOLVER_ALLOW_UNSAFE_EXEC=1 \\
  python subsolver/measure_solo_fast.py --parquet data/guru_code_pool/all.parquet \\
      --system_prompt code --n 16 --out logs/guru_code/solo_code_prompt

Modes (--mode; default: pipeline):
  pipeline  generate and grade concurrently -> results.jsonl
  generate  generation only, no grading pool -> <out>/gens.jsonl (index + texts);
            GPUs run without CPU contention from graders
  grade     no servers needed: read <out>/gens.jsonl, grade everything with the
            full process pool -> results.jsonl (same record format as pipeline)
All three resume: rows already in results.jsonl are skipped everywhere; generate
also skips rows already in gens.jsonl. A partial pipelined results.jsonl composes
with generate+grade for the remainder.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import httpx
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from subsolver.domain_reward import score as domain_score  # noqa: E402
from subsolver.prompts import STUDENT_SYSTEM_PROMPT, STUDENT_SYSTEM_PROMPT_CODE  # noqa: E402
# LLM solo-solve prompt (verbatim from the e2e eval harness, eval_code_e2e.py, which is not part of this repo).
ADVISOR_CP_SOLVER_PROMPT = """You are an expert competitive programming solver.

Solve the given programming problem and return a correct submission.

Important rules:

Reason carefully about the algorithm before writing the final code.

Check all important edge cases, including indexing, boundaries, impossible cases, modulo arithmetic, and integer size where relevant.

Follow the required interface exactly:

Preserve the provided class and method signature if one is given.

If the problem requires stdin/stdout, provide a complete standalone program.

Include every required import, including imports needed by type annotations.

Before answering, verify that the final code is syntactically valid Python and can run as written.

Return exactly ONE final Python code block.

Do not output multiple implementations.

Do not repeat the code.

Do not include sample tests or sample execution.

Do not continue explaining or reconsidering the solution after the code block.

Your response should have only:

A brief explanation of the algorithm and its complexity.

# one complete final submission

The Python code block must be the final content of your response."""


def _warm_worker():
    """Pre-import the graders once per spawned worker, off the first task's clock."""
    import subsolver.domain_reward  # noqa: F401
    from verl.utils.reward_score import default_compute_score  # noqa: F401


def _text(prompt, role):
    msgs = prompt.tolist() if hasattr(prompt, "tolist") else prompt
    for m in msgs:
        if m.get("role") == role:
            return m["content"]
    return "" if role == "system" else (msgs[-1]["content"] if msgs else "")


def system_for(row, mode: str) -> str | None:
    if mode == "row":
        return _text(row["prompt"], "system")
    if mode == "code":
        return STUDENT_SYSTEM_PROMPT_CODE
    if mode == "cpsolver":
        return ADVISOR_CP_SOLVER_PROMPT
    if mode == "default":
        return STUDENT_SYSTEM_PROMPT
    return None


class Endpoints:
    def __init__(self, raw: str) -> None:
        self.urls = [u.strip().rstrip("/") for u in raw.split(",") if u.strip()]
        assert self.urls, "SUBSOLVER_STUDENT_ENDPOINT is empty"
        self._i = 0

    def pick(self) -> str:
        u = self.urls[self._i % len(self.urls)]
        self._i += 1
        return u


async def gen_one(sem, client, eps, model, messages, args):
    async with sem:
        payload = {"model": model, "messages": messages, "temperature": args.temperature,
                   "top_p": args.top_p, "max_tokens": args.max_tokens}
        if args.disable_thinking:  # LLM (Qwen3.5) solo solve: no hidden reasoning trace
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        for attempt in range(3):
            try:
                r = await client.post(f"{eps.pick()}/v1/chat/completions", json=payload)
                r.raise_for_status()
                return r.json()["choices"][0]["message"].get("content") or ""
            except Exception:  # noqa: BLE001
                if attempt == 2:
                    return None
                await asyncio.sleep(2 * (attempt + 1))


async def run_problem(row, sem, client, eps, model, pool, loop, args, out_fh, state):
    ei = dict(row["extra_info"])
    ds = ei.get("guru_data_source") or row["data_source"]
    gt = dict(row["reward_model"]).get("ground_truth")
    system = system_for(row, args.system_prompt)
    msgs = ([{"role": "system", "content": system}] if system is not None else []) + \
           [{"role": "user", "content": _text(row["prompt"], "user")}]
    texts = await asyncio.gather(*[
        gen_one(sem, client, eps, model, msgs, args) for _ in range(args.n)])
    errs = sum(1 for t in texts if t is None)
    if args.mode == "generate":
        rec = {"index": ei["index"], "domain": row["ability"], "data_source": ds,
               "guru_pass_rate": ei.get("guru_pass_rate"), "n_err": errs,
               "system_prompt": args.system_prompt,
               "texts": [t for t in texts if t is not None]}
        out_fh.write(json.dumps(rec) + "\n")
        out_fh.flush()
        _tick(state, args)
        return rec
    scores = await asyncio.gather(*[
        loop.run_in_executor(pool, domain_score, ds, t, gt, ei)
        for t in texts if t is not None])
    rate = (sum(1 for s in scores if s >= 1.0 - 1e-9) / len(scores)) if scores else float("nan")
    rec = {"index": ei["index"], "domain": row["ability"], "data_source": ds,
           "solo_pass_rate": rate, "guru_pass_rate": ei.get("guru_pass_rate"),
           "n_ok": len(scores), "n_err": errs,
           "mean_score": (sum(scores) / len(scores)) if scores else float("nan"),
           "system_prompt": args.system_prompt}
    out_fh.write(json.dumps(rec) + "\n")
    out_fh.flush()
    _tick(state, args)
    return rec


def _tick(state, args):
    state["done"] += 1
    if state["done"] % args.report_every == 0 or state["done"] == state["total"]:
        dt = time.time() - state["t0"]
        eta = dt / state["done"] * (state["total"] - state["done"])
        print(f"  {state['done']}/{state['total']}  "
              f"{dt/60:.1f} min elapsed  ETA {eta/60:.0f} min", flush=True)


def _write_summary(args, t0):
    out = pd.read_json(f"{args.out}/results.jsonl", lines=True)
    out.to_csv(f"{args.out}/solo_pass_rate.csv", index=False)
    print(f"\n  wrote {args.out}/solo_pass_rate.csv  "
          f"({time.time()-t0:.0f}s this run)", flush=True)
    print(f"  {'data_source':28} {'n':>5} {'solo':>8} {'guru':>8} {'corr':>6} {'err':>5}", flush=True)
    for src, g in out.groupby("data_source"):
        gp = pd.to_numeric(g["guru_pass_rate"], errors="coerce")
        c = g["solo_pass_rate"].corr(gp) if len(g) > 2 else float("nan")
        print(f"  {src:28} {len(g):>5} {g.solo_pass_rate.mean():>8.3f} "
              f"{gp.mean():>8.3f} {c:>6.2f} {int(g.n_err.sum()):>5}", flush=True)


async def grade_async(args) -> int:
    """--mode grade: grade <out>/gens.jsonl with the full process pool (no servers)."""
    d = pd.read_parquet(args.parquet)
    rows = {str(dict(r["extra_info"])["index"]): r for _, r in d.iterrows()}
    res_path = f"{args.out}/results.jsonl"
    done_idx = set()
    if os.path.exists(res_path):
        with open(res_path) as fh:
            done_idx = {str(json.loads(l)["index"]) for l in fh if l.strip()}
    gens = [json.loads(l) for l in open(f"{args.out}/gens.jsonl") if l.strip()]
    todo = [g for g in gens if str(g["index"]) not in done_idx]
    print(f"  grade-only: {len(gens)} generated, {len(done_idx)} already graded, "
          f"{len(todo)} to grade, grade workers {args.grade_workers}", flush=True)
    pool = ProcessPoolExecutor(max_workers=args.grade_workers,
                               mp_context=multiprocessing.get_context("spawn"),
                               initializer=_warm_worker)
    loop = asyncio.get_running_loop()
    sem = asyncio.Semaphore(args.grade_workers * 4)   # bounds pickled args in flight
    state = {"done": 0, "total": len(todo), "t0": time.time()}

    async def one(g):
        async with sem:
            r = rows[str(g["index"])]
            ei = dict(r["extra_info"])
            ds = ei.get("guru_data_source") or r["data_source"]
            gt = dict(r["reward_model"]).get("ground_truth")
            scores = await asyncio.gather(*[
                loop.run_in_executor(pool, domain_score, ds, t, gt, ei)
                for t in g["texts"]])
            rate = (sum(1 for s in scores if s >= 1.0 - 1e-9) / len(scores)) if scores else float("nan")
            rec = {"index": g["index"], "domain": g.get("domain"), "data_source": ds,
                   "solo_pass_rate": rate, "guru_pass_rate": g.get("guru_pass_rate"),
                   "n_ok": len(scores), "n_err": g.get("n_err", 0),
                   "mean_score": (sum(scores) / len(scores)) if scores else float("nan"),
                   "system_prompt": g.get("system_prompt", args.system_prompt)}
            out_fh.write(json.dumps(rec) + "\n")
            out_fh.flush()
            _tick(state, args)

    with open(res_path, "a") as out_fh:
        await asyncio.gather(*[one(g) for g in todo])
    pool.shutdown(wait=True)
    _write_summary(args, state["t0"])
    return 0


async def main_async(args) -> int:
    if args.mode == "grade":
        return await grade_async(args)
    eps = Endpoints(os.environ["SUBSOLVER_STUDENT_ENDPOINT"])
    model = os.environ.get("SUBSOLVER_STUDENT_MODEL", "student")
    d = pd.read_parquet(args.parquet)
    if args.domains:
        d = d[d["ability"].isin(args.domains)]
    if args.per_domain:
        d = pd.concat([g.head(args.per_domain) for _, g in d.groupby("ability")])

    os.makedirs(args.out, exist_ok=True)
    res_path = f"{args.out}/results.jsonl"
    out_path = f"{args.out}/gens.jsonl" if args.mode == "generate" else res_path
    done_idx = set()
    # rows already graded are done for every mode; generate also skips rows
    # already generated
    for p in {res_path, out_path}:
        if os.path.exists(p):
            with open(p) as fh:
                done_idx |= {str(json.loads(l)["index"]) for l in fh if l.strip()}
    if done_idx:
        print(f"  resume: skipping {len(done_idx)} problems", flush=True)
    todo = d[~d["extra_info"].map(lambda e: str(e["index"])).isin(done_idx)]
    print(f"  {len(todo)} problems x n={args.n}  system_prompt={args.system_prompt}  "
          f"({', '.join(f'{k}:{len(g)}' for k, g in todo.groupby('ability'))})  "
          f"concurrency {args.max_concurrent}, grade workers {args.grade_workers}", flush=True)

    sem = asyncio.Semaphore(args.max_concurrent)
    # PROCESS pool (spawn): the math grader's SIGALRM timeout only works in a main
    # thread, and code execution / sympy are CPU-bound — threads would thrash the
    # GIL and starve the event loop that pumps generations. Not created at all in
    # generate mode (that is the point: GPUs run free of grader CPU contention).
    pool = None
    if args.mode != "generate":
        pool = ProcessPoolExecutor(max_workers=args.grade_workers,
                                   mp_context=multiprocessing.get_context("spawn"),
                                   initializer=_warm_worker)
    loop = asyncio.get_running_loop()
    limits = httpx.Limits(max_connections=args.max_concurrent * 2)
    state = {"done": 0, "total": len(todo), "t0": time.time()}
    with open(out_path, "a") as out_fh:
        async with httpx.AsyncClient(timeout=httpx.Timeout(args.timeout, connect=30.0),
                                     limits=limits) as client:
            await asyncio.gather(*[
                run_problem(r, sem, client, eps, model, pool, loop, args, out_fh, state)
                for _, r in todo.iterrows()])
    if pool is not None:
        pool.shutdown(wait=True)
    if args.mode == "generate":
        print(f"\n  wrote {out_path} — grade with --mode grade", flush=True)
        return 0
    _write_summary(args, state["t0"])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--per_domain", type=int, default=0, help="0 = every row")
    ap.add_argument("--domains", nargs="*", default=None)
    ap.add_argument("--system_prompt", choices=["row", "code", "cpsolver", "default", "none"], default="row")
    ap.add_argument("--disable_thinking", action="store_true",
                    help="add chat_template_kwargs enable_thinking=False (Qwen3.5 LLM solo)")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_concurrent", type=int, default=256)
    ap.add_argument("--grade_workers", type=int, default=32)
    ap.add_argument("--report_every", type=int, default=100)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--mode", choices=["pipeline", "generate", "grade"], default="pipeline",
                    help="pipeline = original concurrent behaviour; generate = dump "
                         "gens.jsonl only; grade = grade gens.jsonl (no servers)")
    ap.add_argument("--out", required=True)
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
