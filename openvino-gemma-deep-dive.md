# OpenVINO + Gemma deep dive

Use this guide when the user asks for Gemma-specific OpenVINO tuning, training, compiling,
serving, or edge optimization.

## Key finding

OpenVINO is primarily an **inference compilation/optimization/serving path**, not the main
place to train Gemma weights. For Gemma customization:

1. Tune/train adapters with Unsloth, Hugging Face PEFT/QLoRA, Keras, or Tunix.
2. Validate adapter/base behavior before export.
3. Export/convert/compress to OpenVINO IR or serve through OpenVINO GenAI / OVMS.
4. Use OpenVINO LoRA adapter support where applicable, but verify exact adapter format and
   target runtime before assuming drop-in compatibility.

## Gemma-specific OpenVINO sources to consult

Primary Gemma notebooks:

- Gemma 4 OpenVINO notebook: `https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/gemma4`
- Gemma 4 notebook file: `https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/gemma4/gemma4.ipynb`
- Gemma 3 OpenVINO notebook: `https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/gemma3`
- Gemma 3 notebook file: `https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/gemma3/gemma3.ipynb`

OpenVINO runtime/serving docs:

- Optimum Intel inference/export: `https://docs.openvino.ai/2025/openvino-workflow-generative/inference-with-optimum-intel.html`
- OpenVINO GenAI model prep: `https://docs.openvino.ai/2026/openvino-workflow-generative/genai-model-preparation.html`
- OpenVINO GenAI on NPU: `https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html`
- OVMS efficient LLM serving: `https://docs.openvino.ai/2026/model-server/ovms_docs_llm_reference.html`
- OVMS export GenAI models: `https://docs.openvino.ai/2026/model-server/ovms_demos_common_export.html`
- OVMS long-context optimization: `https://docs.openvino.ai/2026/model-server/ovms_demo_long_context.html`
- OVMS speculative decoding: `https://docs.openvino.ai/2026/model-server/ovms_demos_continuous_batching_speculative_decoding.html`
- OVMS RAG demo: `https://docs.openvino.ai/2026/model-server/ovms_demos_continuous_batching_rag.html`
- OVMS agentic demo: `https://docs.openvino.ai/2026/model-server/ovms_demos_continuous_batching_agent.html`

Optimization/API docs:

- LLM weight compression: `https://docs.openvino.ai/2026/openvino-workflow/model-optimization-guide/weight-compression.html`
- 4-bit weight quantization: `https://docs.openvino.ai/2026/openvino-workflow/model-optimization-guide/weight-compression/4-bit-weight-quantization.html`
- `openvino_genai.LLMPipeline`: `https://docs.openvino.ai/2026/api/genai_api/_autosummary/openvino_genai.LLMPipeline.html`
- `openvino_genai.VLMPipeline`: `https://docs.openvino.ai/2026/api/genai_api/_autosummary/openvino_genai.VLMPipeline.html`
- `openvino_genai.Adapter`: `https://docs.openvino.ai/2026/api/genai_api/_autosummary/openvino_genai.Adapter.html`
- `openvino_genai.AdapterConfig`: `https://docs.openvino.ai/2026/api/genai_api/_autosummary/openvino_genai.AdapterConfig.html`
- `openvino_genai.draft_model`: `https://docs.openvino.ai/2026/api/genai_api/_autosummary/openvino_genai.draft_model.html`

## Gemma 4 OpenVINO notebook facts

The OpenVINO Gemma 4 notebook is specifically for visual-language assistant workflows and
lists these model IDs:

```python
model_ids = [
    "google/gemma-4-E2B-it",
    "google/gemma-4-E4B-it",
    "google/gemma-4-12b-it",
    "google/gemma-4-26B-A4B-it",
    "google/gemma-4-31B-it",
]
```

Notebook summary:

- Gemma 4 E2B: 5.1B total / 2.3B effective, text/image/audio, 128K context, Dense + PLE.
- Gemma 4 E4B: 8B total / 4.5B effective, text/image/audio, 128K context, Dense + PLE.
- Gemma 4 12B: supported by notebook; requires newer Transformers per notebook comments.
- Gemma 4 26B-A4B: 25.2B total / 3.8B active, text/image, 256K context.
- Gemma 4 31B: listed by notebook.

The notebook installs nightly/pre-release OpenVINO stack pieces and pins Transformers because
`gemma-4-12b-it` uses the Gemma 4 Unified architecture:

```bash
pip install "git+https://github.com/huggingface/optimum-intel.git"
pip install -U --pre openvino-genai openvino openvino-tokenizers nncf
pip install "transformers==5.10.0"
pip install "torch>=2.10" torchvision Pillow
```

Do not blindly copy versions into production. Treat these as notebook-tested versions and
re-check current compatibility first.

## Convert / compile / compress path

Preferred OpenVINO conversion route for Gemma 4 notebook:

```bash
optimum-cli export openvino \
  --model <model_id_or_path> \
  --task image-text-to-text \
  --weight-format <fp16|int8|int4> \
  <output_dir>
```

Notebook behavior:

- It prefers preconverted OpenVINO Hub models if present:
  `OpenVINO/<pt_model_name>-<weight_format>-ov`.
- Otherwise it converts locally with Optimum Intel.
- Weight format choices in the notebook: `FP16`, `INT8`, `INT4`; default is `INT4`.
- Inference path uses `openvino_genai.VLMPipeline(str(model_export_dir), device=<device>)`.

For text-only Gemma exports use the relevant text-generation task from Optimum/OpenVINO docs
instead of copying the VLM `image-text-to-text` task.

## Compilation/runtime path

For direct Python/OpenVINO GenAI:

```python
import openvino_genai as ov_genai
pipe = ov_genai.VLMPipeline(str(model_export_dir), device="CPU")  # or GPU/NPU where supported
```

For text-only models, use `LLMPipeline` when applicable. For multimodal Gemma, use
`VLMPipeline`.

Runtime checks before declaring success:

- model directory contains OpenVINO IR files and tokenizer/chat template assets;
- tokenizer and chat template load without modification errors;
- system role works if the workload depends on it;
- streaming output is valid;
- images/audio/video inputs survive preprocessing and serialization;
- target device actually compiles the model (`CPU`, `GPU`, `NPU`, `AUTO`, etc.);
- first token and steady-state throughput measured on target hardware.

## Prompt formatting and thinking

Gemma 4 OpenVINO notebook explicitly uses native system role support via `ChatHistory`:

```python
history = ov_genai.ChatHistory()
history.append({"role": "system", "content": "..."})
history.append({"role": "user", "content": "..."})
pipe.generate(history, image=image_to_tensor(image), max_new_tokens=200, streamer=streamer)
```

The notebook also has a **Thinking mode** section. It says Gemma 4 supports built-in
chain-of-thought reasoning and uses control tokens like:

```text
<|channel>thought
[internal reasoning]
<channel|>
[final answer]
```

In that notebook, thinking is enabled by modifying the chat template to set
`enable_thinking = true`, injecting the `<|think|>` trigger token. Treat that as a
notebook implementation detail: verify the current chat template, OpenVINO GenAI tokenizer,
and model-specific recommended thinking controls before using in production.

## LoRA / adapter support

OpenVINO GenAI exposes LoRA adapter objects:

- `openvino_genai.Adapter`: immutable LoRA adapter carrying adaptation matrices and acting as
  a unique adapter identifier.
- `openvino_genai.AdapterConfig`: defines a combination of LoRA adapters with blending
  parameters.

This does **not** mean OpenVINO is the training framework. Recommended workflow:

1. Train LoRA/QLoRA with Unsloth/HF/PEFT/Tunix/Keras.
2. Validate adapter against the base model in the training framework.
3. Confirm OpenVINO GenAI adapter format support and conversion path.
4. Test base-only versus adapter-enabled OpenVINO outputs on the same eval set.
5. Only then deploy adapter config in OpenVINO/OVMS.

If adapter conversion is uncertain, merge adapter into the base model in the source framework
first, then export/quantize the merged model to OpenVINO as a fallback.

## Weight compression and quantization

OpenVINO LLM weight compression is designed to reduce memory footprint and improve inference
latency by quantizing weights while leaving activations floating point.

Important OpenVINO note from docs:

- 4-bit weight quantization is mixed precision, primarily INT4 with backup asymmetric INT8.
- It can significantly reduce size and latency, but may cause noticeable accuracy degradation.
- Smaller models can suffer more accuracy loss from low-bit compression than larger models.

Gemma-specific implications:

- Do not assume INT4 is acceptable for extraction/RAG/agentic correctness without eval.
- For e2b/e4b, compare FP16, INT8, INT4 on task quality because smaller/effective smaller
  models may be more sensitive.
- For 12B/26B/31B, INT4 may be necessary for edge hardware, but still require held-out evals.
- Always record quantization, export command, OpenVINO version, tokenizer, chat template, and
  device.

## MTP / speculative decoding with OpenVINO

Gemma 4 includes dedicated draft models for MTP/speculative decoding in Google docs. For
OpenVINO, do not assume Gemma 4 MTP is automatically active. Verify support in the chosen path:

- OpenVINO GenAI has `draft_model` API docs.
- OVMS has a speculative decoding demo using `draft_models_path`.
- OVMS docs describe speculative decoding as reducing generation latency without changing the
  output distribution by using a lightweight draft model and validating candidates with the
  main model in parallel.

Gemma-specific MTP plan:

1. Export/obtain both target and draft model in compatible OpenVINO format.
2. Verify tokenizer/chat template compatibility between target and draft.
3. Enable speculative decoding through OpenVINO GenAI or OVMS config.
4. Measure against non-MTP baseline: TTFT, output tok/s, quality delta, memory, streaming.
5. Test with real agent prompts, not just short synthetic prompts.

## OVMS serving path

Use OpenVINO Model Server for production/API serving when you need OpenAI-compatible-ish HTTP,
continuous batching, long-context features, RAG/agent demos, or NPU/CPU serving. Consult:

- `ovms_docs_llm_reference.html`
- `ovms_demos_common_export.html`
- `ovms_demo_long_context.html`
- `ovms_demos_continuous_batching_speculative_decoding.html`
- `ovms_demos_continuous_batching_rag.html`
- `ovms_demos_continuous_batching_agent.html`

OVMS checks:

- export model repository layout correctly;
- set `kv_cache_precision` only after eval;
- validate OpenAI API compatibility with the actual client/harness;
- test streaming and cancellation;
- test continuous batching under expected concurrency;
- monitor first-token latency separately from throughput.

## RAG / agentic / Hermes-specific OpenVINO testing

For RAG:

- Keep retrieval/embedding/reranking outside Gemma eval first.
- Test OpenVINO Gemma answer quality with fixed retrieved contexts.
- Then test full RAG pipeline with OVMS/OpenVINO serving.
- Evaluate citation faithfulness, abstention, long-context behavior, and prompt packing.

For agents/Hermes:

- Verify prompt formatting and system/tool messages survive the OpenVINO runtime.
- Test function/tool call formatting if using Gemma for tool-use tasks.
- Replay real Hermes tasks; measure tool-call JSON validity and recovery after tool errors.
- Check streaming behavior, partial token handling, cancellation, timeout, and retries.
- Compare OpenVINO output to current llama.cpp/MLX/llama-swap baseline before switching.

## Troubleshooting tree: OpenVINO Gemma

1. **Conversion fails**
   - Check model ID, HF access/license, Transformers version, Optimum Intel version, task name.
   - For Gemma 4 12B/unified architecture, verify current Transformers requirement.
   - Try preconverted `OpenVINO/<model>-<precision>-ov` if available.

2. **Compiled model fails on target device**
   - Try CPU first to isolate device support.
   - Check OpenVINO/OpenVINO GenAI install extras for GPU/NPU.
   - Reduce precision or model size only after confirming device plugin support.

3. **Bad prompt behavior**
   - Verify Gemma 4 prompt template and `ChatHistory` roles.
   - Confirm system role is preserved.
   - Compare output against HF/Transformers baseline with same prompt.

4. **Thinking does not work or leaks internal text**
   - Inspect current chat template.
   - Confirm trigger token and parsing behavior.
   - Evaluate whether final-answer extraction/filtering is needed in the harness.

5. **INT4 quality regression**
   - Retest FP16 and INT8.
   - Evaluate smaller e2b/e4b especially carefully.
   - If using agent/RAG/extraction, prefer correctness over memory savings.

6. **MTP/speculative decoding no speedup**
   - Verify draft model actually loaded.
   - Check concurrency: OVMS docs note speedups are most pronounced at concurrency 1.
   - Test longer generations; short outputs may hide benefits.
   - Compare memory and token acceptance rate.

7. **Multimodal input failures**
   - Confirm exact model supports image/audio.
   - Validate preprocessing and tensor conversion.
   - Test text-only, then single image, then full multimodal prompt.

8. **OVMS client incompatibility**
   - Test raw curl first.
   - Then test OpenAI client.
   - Then test Hermes/LiteLLM integration.
   - Validate streaming chunks and error formats.

## Minimum benchmark matrix

For any Gemma OpenVINO candidate, benchmark:

| Dimension | Values |
|---|---|
| Model | e2b, e4b, 12b, and target larger model if needed |
| Precision | FP16, INT8, INT4 |
| Runtime | OpenVINO GenAI Python, OVMS if serving, current baseline runtime |
| Device | CPU/GPU/NPU as available |
| Prompt mode | normal, system-role, thinking, tool/RAG prompt |
| MTP | off/on if supported |
| Metrics | TTFT, tok/s, RAM/VRAM, compile/load time, quality, task success |

Never claim OpenVINO is better until this matrix has at least one apples-to-apples baseline
against the current runtime.
