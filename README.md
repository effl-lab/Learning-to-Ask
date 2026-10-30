# LearningToAsk

Training code for the three-stage *learning to ask* pipeline — a student model learns **whether** to
call an advisor (stage 1, call gate), **how** to ask (stage 2, question maker) and finally trains
end-to-end with both (stage 3, integration) — built on [verl](https://github.com/verl-project/verl).

The repository is a verl checkout (upstream `main` @ 081c76bb) plus:

- `subsolver/` — the pipeline: rewards, the `ask_advisor` tool, frozen-server clients, data
  builders, run scripts and chain drivers. Start at [`subsolver/README.md`](subsolver/README.md).
- verl patches: the `grpo_segmented` advantage estimator and `reward_extra/*` metrics in the V1
  trainer, the `llama3_json_reasoning` / `qwen3_xml_think` / `gemma4_subsolver` tool parsers, the
  Guru (Reasoning360) reward dispatch, `rollout.presence_penalty`, and shared-node port isolation
  (file list in `subsolver/README.md`).

## Setup

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements-frozen.txt     # exact source environment (torch 2.13.0+cu130, vLLM 0.32.0, transformers 5.16.1)
uv pip install -e . --no-deps
```

`requirements-subsolver.txt` holds the key pins only, for a fresh resolve on another CUDA stack.
Datasets go under `data/`, checkpoints under `checkpoints/`, run logs under `logs/` (all
git-ignored); every script resolves the repo root from its own location (`VERL_ROOT`).

## License

Apache 2.0 (`LICENSE`, `Notice.txt`). verl is Copyright 2024 Bytedance Ltd. and/or its affiliates;
the subsolver additions are Copyright 2025 Subsolver Authors.
