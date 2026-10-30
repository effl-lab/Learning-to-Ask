# LearningToAsk — training code map

This checkout is upstream verl `main` @ 081c76bb plus the subsolver patches, reduced to the code
that trains the three stages of the method (evaluation harnesses, analysis scripts and ablation
lines are not included).

Run everything from the repo root with `.venv` active (`source subsolver/gemma_env.sh` for the
Gemma 4 line). Datasets live under `data/` and checkpoints under `checkpoints/` (both
git-ignored); every script resolves paths from the repo root (`VERL_ROOT`, derived from its own location).

## Stage 1 — call gate (whether-to-call)

| step | files |
|---|---|
| solo pass-rate labels (`need = 1[avg@n <= t/n]`) | `measure_train_avg8.py` (DAPO-17k / CF pools, driven by `run_measure_7b_avg16.sh`), `measure_solo_fast.py` (Guru pools; codegen graded by execution, stem_web by the `TIGER-Lab/general-verifier` judge) |
| gate parquet — DAPO / CF line | `examples/data_preprocess/subsolver_callgate_preprocess.py` (labels → balanced train / natural val), `build_llama_cf_parquet.py` (CF-style pool, also the harvest pool for stage 2), `build_warmup_gate_parquet.py` (call warm-up variant) |
| gate parquet — Guru math + stem + code | `build_guru_callgate_parquet.py` (24k pool + 600-row val from the Guru train parquets) → `build_guru_mine_candidates.py` (extra easy math to measure) → `build_guru_gate_labels.py` (measured relabel: balanced train / natural val) → `build_guru_gate_labels_eqmix.py` (equal-domain train, stem oversampled) → `build_guru_gate_codeprompt_parquet.py` (per-domain student prompt; the `_eqmix_v2` parquets the gates train on) |
| gate parquet — Guru code only | `build_guru_code_pool.py` (primeintellect + leetcode2k + taco) → `build_guru_code_gate_parquet.py` |
| training | `run_callgate_grpo_17k.sh` with `callgate_reward.py` (reward = `1[called == need]`), tool exposed through `student_tool_config.yaml` but `max_assistant_turns=1`, so the tool never executes and no frozen servers are needed |
| Guru chain | `chains/guru/p1_labels.sh` (pool → solo sweeps → label parquets), `p2_gate.sh` (t12 / t4 gates → `checkpoints/guru_gates/t{T}_eqmix_v2_best`, the stage-2 student), `code_gate.sh` (code-only line → `checkpoints/guru_gates/code_t{T}_best`, the stage-3 student) |
| Gemma 4 chain | `chains/gemma_e2b/p2_gate.sh` (train), `p2_merge.sh` (pick + merge the best checkpoint) |

Guru data: the `train/` split of the Reasoning360 release (`LLM360/guru-RL-92k` on Hugging Face) at
`data/reasoning360/train/` or `$GURU_TRAIN_DIR`. Gate recipe: batch 32/GPU, mini 16/GPU, n 8,
lr 1e-6, KL 1e-3, `data.max_prompt_length=4096`; t12 10 epochs, t4 3 epochs; best checkpoint by
val decision score.

Call detection per model family: `llama_toolcall.py`, `qwen3_toolcall.py`, `gemma4_toolcall.py`
(shared by the reward, the verl tool parsers in `verl/experimental/agent_loop/tool_parser.py`, and
the vLLM plugin `vllm_llama_reasoning_parser.py` + `chat_templates/` for Llama-3.1).

## Stage 2 — question maker (how-to-call)

| step | files |
|---|---|
| serve the merged gate | `serve_student.sh` (hermes parser; Gemma: `chains/gemma_e2b/gemma_common.sh` serve flags) |
| harvest prefixes up to the `ask_advisor` call | `gen_guru_prefixes.py` (wraps `generate_student_traces.py`) |
| qmaker parquet | `build_guru_qmaker_parquet.py` (Guru / Gemma line), `examples/data_preprocess/subsolver_preprocess.py` (DAPO line), `build_costq_parquet.py` (cost recipe), `sanitize_prefix_parquet.py` |
| frozen fleet | `serve_qmaker_frozen.sh` (advisor xN client-side LB + student continuation) or `serve_frozen.sh`; endpoints read by `frozen_clients.py` |
| training | `run_qmaker_costq.sh` → `run_subsolver_grpo.sh` with `cost_reward.py` (`SUBSOLVER_COST_SCORE_MODE=cut`: reward = `1[correct] · 1[advisor reply within cap]`, incomplete-query judge on, whole gate off). Guru math+code: `REWARD_PATH=pkg://subsolver.cost_reward_guru` (same is-cut reward, domain graders through `domain_reward.py` → `verl/utils/reward_score/{naive_dapo,coder1,stem_llm_judge}`) |
| Gemma 4 chain | `chains/gemma_e2b/p3_harvest.sh`, `p4_qmaker.sh <CAP>`, orchestration `after_gate.sh` / `after_qmaker.sh` |

`reward.py` holds the base qmaker reward and the incomplete / whole judges that `cost_reward*.py`
build on. `prompts.py` (system prompts, tool schema, `QMAKER_COST_SYSTEM_PROMPT`) and `pipeline.py`
(message manipulation, boxed-answer parsing) are shared by every stage.

## Stage 3 — integration (student trained end-to-end with the frozen qmaker + advisor)

Fleet: advisor xN + merged qmaker (`serve_phase2_frozen.sh` layout). The tool now runs the qmaker
(`integration_tool_config*.yaml`, `enable_qmaker: true`; `_code`, `_code_t12`, `_q35` variants —
`tokenizer_path` points at the gate checkpoint).

Segmented GRPO (the paper recipe) reuses `run_callgate_grpo_17k.sh` with these overrides:

```bash
export SUBSOLVER_ADVISOR_ENDPOINT=... SUBSOLVER_QMAKER_ENDPOINT=... \
       SUBSOLVER_ADVISOR_MODEL=advisor SUBSOLVER_QMAKER_MODEL=qmaker \
       SUBSOLVER_ADVISOR_MAX_TOKENS=<cap> SUBSOLVER_ADVISOR_DISABLE_THINKING=1 \
       SUBSOLVER_SEG_PRE_CORRECT_W=0.3 SUBSOLVER_INTEG_CAP_GATE=0
TRAIN_FILE=<gate train.parquet> VAL_FILE="[<gate val.parquet>,<lcb_integ_val/val.parquet>]" \
TOOL_CONFIG=subsolver/integration_tool_config_code_t12.yaml \
  bash subsolver/run_callgate_grpo_17k.sh \
    actor_rollout_ref.model.path=<merged gate> \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=2 \
    reward.custom_reward_function.path=pkg://subsolver.integration_reward_seg \   # math: integration_reward_seg_math
    algorithm.adv_estimator=grpo_segmented
```

Per trajectory `integration_reward_seg.py` / `integration_reward_seg_math.py` emit
`seg_pre = 1[called == need] + 0.3 · 1[correct]` (pre-call tokens) and `seg_post = 1[correct]`
(continuation); `compute_grpo_segmented_advantage` in `verl/trainer/ppo/core_algos.py` splits the
response at the masked tool turn and group-normalises each part. The V1 trainer exposes the
components through `PPOTrainer._fetch_reward_extra_info` (`verl/trainer/ppo/v1/trainer_base.py`,
test: `tests/trainer/ppo/v1/test_reward_extra_info_on_cpu.py`).

Alternative product reward (`integration_reward.py` / `integration_reward_code.py`, decision ×
correctness × cap gate) runs through `run_student_grpo.sh` with
`REWARD_PATH=pkg://subsolver.integration_reward`; `student_reward.py` is plain final-answer
correctness. `build_lcb_integ_val_parquet.py` builds the LiveCodeBench auxiliary validation set.

## verl patches carried in this checkout

- `verl/experimental/agent_loop/tool_parser.py`: `llama3_json_reasoning`, `qwen3_xml_think`,
  `gemma4_subsolver` parsers; LaTeX-tolerant Hermes JSON recovery.
- `verl/trainer/ppo/core_algos.py`, `ray_trainer.py`, `v1/trainer_base.py`, `v1/utils.py`:
  `grpo_segmented` estimator and `reward_extra/*/mean` train metrics.
- `verl/utils/reward_score/__init__.py` + `naive_dapo.py`, `coder1/`, `stem_llm_judge/`: Guru
  data-source dispatch (`math__*`, `codegen*`, `stem_web*`).
- `verl/workers/config/rollout.py`, `agent_loop.py`: `rollout.presence_penalty`.
- `verl/utils/net_utils.py`, `single_controller/{base/worker,ray/base}.py`,
  `vllm_rollout/vllm_async_server.py`: `VERL_PORT_RANGE`, per-replica `VLLM_PORT` stride, port
  reservation release (shared-node EADDRINUSE fixes).

## Environment

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements-frozen.txt     # exact source venv (torch 2.13.0+cu130, vLLM 0.32.0, transformers 5.16.1, flashinfer 0.7.0.post1)
uv pip install -e . --no-deps
```

`requirements-subsolver.txt` lists the key pins only (same versions) for a fresh resolve. Gemma 4
runs additionally source `subsolver/gemma_env.sh`.
