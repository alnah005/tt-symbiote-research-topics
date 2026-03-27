# Workflow Commands

This page covers how `run.py` uses the `ModelSpec` to drive the serving stack and other automation workflows, the four workflow types and their output directories, and how to invoke the `tt-vllm-plugin` directly without Docker for local development and debugging.

---

## How `run.py` uses the `ModelSpec`

`run.py` is the single entry point for all automation workflows. Its required arguments are `--model`, `--device`, and `--workflow`:

```bash
python3 run.py --model <model_name> --device <device> --workflow server
```

The `--model` value is matched against the `model_name` field of every `ModelSpec` in `MODEL_SPECS`. `model_name` is the human-readable slug derived from the HuggingFace repo name (e.g., `my-symbiote-model-8b`) and is the identifier you use on the CLI.

> **`model_name` vs. `model_id`:** These are two distinct fields. `model_name` is the human-readable CLI name used by `run.py --model`. `model_id` is the internal programmatic key used as the dict key in `MODEL_SPECS` (e.g., `id_tt_symbiote_my-symbiote-model-8b_n300`). You never pass `model_id` on the command line — `run.py` constructs it internally after resolving `--model` + `--device`.

The `--device` value (case-insensitive) selects among the specs registered for that model. Accepted values match the full `DeviceTypes` enum: `n150`, `n300`, `t3k`, `galaxy`, `p100`, `p150`, `p150x4`, `p150x8`, `p300`. Together `--model` and `--device` uniquely identify one `ModelSpec` instance.

Once resolved, `run.py`:

1. Calls `apply_runtime_args()` to merge any `--override-tt-config` or `--vllm-override-args` overrides supplied on the command line.
2. Serializes the resulting `ModelSpec` to a timestamped JSON file on the host.
3. Sets `TT_MODEL_SPEC_JSON_PATH` to the path of that file.
4. Builds the `docker run` command, injecting `TT_MODEL_SPEC_JSON_PATH` and the rest of `env_vars` into the container environment.
5. Launches the container, which runs `run_vllm_api_server.py` and starts the vLLM HTTP server.

Optional flags that affect server behavior:

| Flag | Effect |
|------|--------|
| `--service-port <N>` | Overrides `SERVICE_PORT`; defaults to `8000` |
| `--dev-mode` | Mounts the local source tree into the container, enabling live code changes without rebuilding the image |
| `--override-tt-config '<json>'` | Merges a JSON string into `device_model_spec.override_tt_config` at runtime |
| `--vllm-override-args '<json>'` | Merges a JSON string into `device_model_spec.vllm_args` at runtime |
| `--disable-trace-capture` | Skips TTNN trace compilation during warmup; faster start but lower steady-state throughput |

---

## Workflow types

`run.py` supports four workflow types passed via `--workflow`:

### `server`

Starts the inference server only. The vLLM HTTP server binds to `SERVICE_PORT` and waits for requests. No benchmarking or evaluation is performed. This is the workflow to use when you want a live endpoint.

```bash
python3 run.py \
  --model my-symbiote-model-8b \
  --device n300 \
  --workflow server
```

### `benchmarks`

Runs a performance sweep over input/output sequence length combinations and concurrency levels defined in the model's `BENCHMARK_CONFIGS`. Results are written as JSON under `benchmarks_output/`. Use this to generate throughput and latency numbers for a given `ModelSpec` configuration.

```bash
python3 run.py \
  --model my-symbiote-model-8b \
  --device t3k \
  --workflow benchmarks
```

### `evals`

Runs accuracy evaluation tasks (e.g. via `lm-eval` or `lmms-eval`) defined in the model's `EVAL_CONFIGS`. Results are written under `evals_output/`. Use this to validate that the model produces correct outputs on standard benchmarks.

```bash
python3 run.py \
  --model my-symbiote-model-8b \
  --device t3k \
  --workflow evals
```

### `reports`

Post-processes the raw JSON files from previous `benchmarks` and `evals` runs and generates a Markdown summary with pass/fail validation against the `perf_reference` targets defined in the `DeviceModelSpec`. Reports are written under `reports_output/`.

```bash
python3 run.py \
  --model my-symbiote-model-8b \
  --device t3k \
  --workflow reports
```

---

## Output directories

All output is written relative to the repository root. Directory names include a timestamp and a config tag derived from the model name and device, so successive runs do not overwrite each other.

| Directory | Contents |
|-----------|---------|
| `docker_server/` | Stdout and stderr logs from the vLLM container; one subdirectory per server run |
| `benchmarks_output/` | JSON files with per-request latency and throughput measurements from benchmark sweeps |
| `evals_output/` | Evaluation framework output files and sampled prompt-response pairs |
| `reports_output/` | Markdown summary documents with pass/fail status against performance targets |
| `run_logs/` | Stdout and stderr from `run.py` itself, separate from the Docker container logs |

---

## Direct plugin invocation without Docker

For development and debugging it is often faster to run the vLLM server directly on the host, bypassing Docker entirely. This requires the `tt-vllm-plugin` package to be installed in the active Python environment.

### Install the plugin

```bash
cd tt-vllm-plugin
pip install -e .
```

### Set required environment variables

```bash
export VLLM_USE_V1=1
export HF_MODEL=myorg/my-symbiote-model-8b
export VLLM_TARGET_DEVICE=tt
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
export MESH_DEVICE=N300
export ARCH_NAME=wormhole_b0
```

`VLLM_USE_V1=1` is mandatory — the TT plugin only supports the vLLM V1 execution path. `HF_MODEL` tells the plugin which weights to load.

### Start the vLLM server

```bash
vllm serve myorg/my-symbiote-model-8b \
  --max-model-len 32768 \
  --max-num-seqs 1 \
  --block-size 64 \
  --max-num-batched-tokens 32768 \
  --enable-chunked-prefill false
```

The flags here mirror the `vllm_args` values from the `DeviceModelSpec`. When iterating on the model implementation this approach eliminates container build and push round-trips and gives direct access to Python tracebacks.

To pass `override_tt_config` keys without a JSON file:

```bash
vllm serve myorg/my-symbiote-model-8b \
  --max-model-len 32768 \
  --max-num-seqs 1 \
  --block-size 64 \
  --max-num-batched-tokens 32768 \
  --enable-chunked-prefill false \
  --override-tt-config '{"optimizations": "accuracy", "trace_region_size": 80000000}'
```

---

**Next:** [Chapter 3 — The Model Implementation Contract](../ch3_model_contract/index.md)
