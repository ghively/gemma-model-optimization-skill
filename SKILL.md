---
name: gemma-model-optimization
description: >
  Use when training, fine-tuning, evaluating, prompting, serving, quantizing, or
  optimizing Google Gemma models, especially Gemma 4 e4b, e2b, 12b, and any other
  Gemma 4 size/variant. Covers LoRA/QLoRA/full fine-tuning, prompt formatting,
  system instructions, thinking, agent-performance tuning, efficient edge inference,
  llama.cpp, MLX, Unsloth, inference runtimes, serving stacks, MTP, context sizing,
  model cards, and variant-specific docs. Triggers: Gemma, gemma4, e4b, e2b, 12b,
  LoRA, QLoRA, fine tune, finetune, prompt formatting, thinking, agent performance,
  Gemma optimization, Gemma serving, llama.cpp Gemma, MLX Gemma, edge Gemma,
  Unsloth Gemma, Ollama Gemma, vLLM Gemma, Tunix, Keras Gemma, Hugging Face Gemma.
---

# Gemma Model Optimization Skill

This skill is the local operating guide and link database for developing with, prompting,
tuning, training, serving, and optimizing Gemma models. It is built from the Google Gemma
docs sidebar plus Gemma library fine-tuning docs. Treat prompt formatting, efficient
runtime choice, and agent-behavior tuning as first-class optimization levers, not as
secondary concerns.

## First rule: retrieve before advising

Before giving a concrete tuning/training/serving plan, query the bundled docs database:

```bash
python ~/.claude/skills/gemma-model-optimization/scripts/query_links.py --main gemma4 qlora --limit 20
python ~/.claude/skills/gemma-model-optimization/scripts/query_links.py --main prompt formatting gemma4 --limit 20
python ~/.claude/skills/gemma-model-optimization/scripts/query_links.py --main thinking gemma --limit 20
python ~/.claude/skills/gemma-model-optimization/scripts/query_links.py --main llama.cpp mlx edge --limit 20
python ~/.claude/skills/gemma-model-optimization/scripts/query_links.py --main unsloth gemma --limit 20
python ~/.claude/skills/gemma-model-optimization/scripts/query_links.py --main e4b --limit 20
```

Use this data as evidence. Do not invent support for a model/runtime combo if the docs do
not show it. If the user asks about current capabilities, refresh against upstream docs.

## Bundled database

Files:

- `data/gemma_docs_links.sqlite` — authoritative local database.
- `data/gemma_docs_links.md` — human-readable grouped link index.
- `data/gemma_source_pages.txt` — every crawled source page.
- `data/curated_views.json` — prebuilt fine-tuning / Gemma 4 / source-page views.
- `scripts/query_links.py` — quick CLI search.

SQLite objects:

- `source_pages` — one row per crawled source page.
- `links` — all link instances with source page, section, context, category, area.
- `unique_links` — de-duplicated outbound URL view.
- `gemma_workbench_links` — tuning/training/optimization/development focused view.

Useful raw SQL examples:

```bash
sqlite3 ~/.claude/skills/gemma-model-optimization/data/gemma_docs_links.sqlite \
  "select source_url, source_title, main_content_links from source_pages order by source_url;"

sqlite3 ~/.claude/skills/gemma-model-optimization/data/gemma_docs_links.sqlite \
  "select text,url,context from links where document_area='main_content' and lower(context||url||text) like '%qlora%' limit 25;"
```

## Covered sources

The crawler includes every sidebar page from `https://ai.google.dev/gemma/docs`, including:

- Core Gemma overview, setup, prompt formatting, system instructions, model cards.
- Gemma 4 run/serve docs and integrations: LM Studio, Ollama, llama.cpp, MLX,
  Hugging Face Transformers, Keras, Gemini API, Google Cloud, GKE.
- Edge/local efficiency paths: llama.cpp, MLX, LiteRT-LM, Ollama, quantized formats,
  context/KV-cache tradeoffs, and platform-specific runtime choices.
- Unsloth and other efficient fine-tuning paths.
- MTP / Multi-Token Prediction docs and runtime implications.
- Capabilities: text, function calling, vision, video, audio, thinking.
- Applied optimization playbooks: RAG, agentic/tool-use work, custom harnesses,
  Hermes agents, multimodal workflows, and OpenVINO/edge acceleration research.
- Tuning overview and Hugging Face QLoRA/full fine-tune pages.
- Specialized variants: Gemma 3n, DiffusionGemma, FunctionGemma, EmbeddingGemma,
  PaliGemma, ShieldGemma, RecurrentGemma, DataGemma, Gemma Scope, Gemma-APS.
- External Gemma library Colab fine-tuning page:
  `https://gemma-llm.readthedocs.io/en/latest/colab_finetuning.html`.
- Unsloth Gemma 4, Gemma 3 tutorial, and general fine-tuning guide pages.
- OpenVINO / OpenVINO Model Server / OpenVINO GenAI pages relevant to LLM serving,
  model export, NPU inference, Optimum Intel, and generative AI performance.
- Deep-dive runbook: `openvino-gemma-deep-dive.md` for Gemma-specific OpenVINO
  tuning, conversion, compiling, LoRA adapter handling, MTP/speculative decoding,
  OVMS serving, RAG/agent/Hermes tests, and troubleshooting.

## Decision workflow

When asked to tune or optimize a Gemma model:

1. **Identify target model and task**
   - Model/size: prioritize e4b, e2b, 12b when named, but support any Gemma 4 model.
   - Task: chat/instruction, extraction, classification, embeddings, vision, function calling,
     long-context generation, edge/mobile, production serving.
   - Hardware: VRAM/RAM, Apple Silicon/Metal/MLX, NVIDIA/CUDA, CPU-only, cloud.

2. **Choose method**
   - Prompt formatting/system-instruction changes first for behavior problems.
   - Thinking-mode and tool/agent workflow changes before fine-tuning if the goal is
     better agent performance rather than new knowledge.
   - LoRA/QLoRA for most task adaptation and consumer GPUs.
   - Unsloth when the goal is efficient fine-tuning with less memory or faster iteration.
   - Full fine-tune only if explicitly justified by data volume, budget, and need.
   - Specialized Gemma variants may need their own pages and formats; query docs first.

3. **Choose runtime**
   - Local desktop/edge efficiency: llama.cpp, MLX, LiteRT-LM, Ollama, LM Studio.
   - Python research/fine-tuning: Hugging Face Transformers, Keras, Tunix/JAX, Unsloth.
   - Production/cloud: vLLM, SGLang, GKE, Gemini Enterprise Agent Platform.
   - Apple Silicon: prioritize MLX or documented Metal-capable path.
   - NVIDIA: prefer CUDA-backed HF/Unsloth/vLLM/llama.cpp path depending on goal.
   - Edge deployments: optimize quantization, batch/concurrency, context/KV cache,
     prompt length, tokenizer compatibility, and first-token latency.

4. **Prepare data**
   - Start small and task-specific; docs note useful behavior changes can happen with small
     prompt/response sets, then scale only after measuring.
   - Preserve Gemma prompt formatting for the target generation mode.
   - Keep train/validation/test splits and an untouched regression/eval set.

5. **Run measurable experiments**
   - Baseline the untuned model and baseline the current prompt/agent workflow.
   - Change one variable at a time: prompt format, thinking setting, system prompt,
     dataset, rank, LR, quantization, context, runtime, concurrency, or tool policy.
   - Track task quality, agent success rate, latency, TTFT, throughput, VRAM/RAM,
     context length, cost/energy if relevant, and failure cases.
   - Verify tuned model against safety/boundary tests before deployment.

6. **Deploy cautiously**
   - Match adapter/base model/version exactly.
   - Document runtime, quantization, context, prompt format, tokenizer, and eval results.
   - Keep rollback path to base model or prior adapter.

## Gemma 4 e4b/e2b/12b emphasis

Treat e4b, e2b, and 12b as the priority sizes for planning, but do not assume they tune
identically:

- Smaller models usually need tighter, higher-quality task data and more careful evals.
- Larger models generally preserve broad capability better but cost more memory and time.
- Runtime support can differ by exact model release/format; query model-card and runtime docs.
- Quantization can change behavior and adapter compatibility; record quantization type.
- Context length and KV cache can dominate memory. Do not cut context without measuring real
  requests and usage.

For Gene's current local stack, remember from prior ops context: Gemma4-e4b has served a
72,014-token prompt, so reducing its context can break real workloads unless the user
explicitly chooses that tradeoff after measurement.

## MTP / Multi-Token Prediction

Do not drop MTP from Gemma 4 optimization work. Treat it as a first-class performance
path whenever the target model/runtime supports it.

Use MTP when the goal is generation speed or serving efficiency, especially for local
Gemma 4 deployments. Before recommending it, query the bundled docs and verify runtime
support for the exact serving path.

MTP query recipes:

```bash
python ~/.claude/skills/gemma-model-optimization/scripts/query_links.py --main mtp gemma --limit 30
python ~/.claude/skills/gemma-model-optimization/scripts/query_links.py --main multi-token prediction --limit 30
python ~/.claude/skills/gemma-model-optimization/scripts/query_links.py --main hugging face mtp --limit 30
```

When testing MTP, measure at minimum:

- output tokens/sec
- TTFT
- quality/regression delta versus non-MTP
- VRAM/RAM change
- runtime compatibility with quantization and context length
- streaming behavior for agent use

## Workload-specific optimization playbooks

Keep these areas in active memory. Do not treat them as afterthoughts.

### RAG / retrieval-augmented generation

Optimize the whole pipeline, not just Gemma weights:

- Validate chunking, metadata, reranking, embedding model, retrieval depth, and prompt layout.
- Preserve citations/source IDs in the prompt so the model can ground answers.
- Measure retrieval recall separately from answer quality.
- Test hallucination, citation faithfulness, stale-context handling, and abstention behavior.
- For multimodal RAG, separately evaluate text, image, audio transcript, and video-frame paths.
- Fine-tune only after retrieval quality and prompt packing are measured.

### Agentic work / tool use / custom harnesses / Hermes agents

For agents, optimize the harness and prompt contract before changing weights:

- Check prompt formatting, system prompt placement, tool schema clarity, function-calling docs,
  and multi-turn chat behavior.
- Evaluate tool-call correctness, argument validity, recovery after tool failure, context retention,
  refusal/over-eagerness, and end-to-end task completion.
- Use small regression suites that replay real Hermes/agent tasks.
- Track latency by phase: planning, tool calls, model generation, retrieval, and final synthesis.
- Tune thinking settings and examples before LoRA if the issue is reasoning or workflow policy.
- If fine-tuning for tool use, keep a held-out set of malformed/edge tool calls and negative cases.

### Thinking optimization

Thinking is a runtime/behavior dimension, not just a model-size question:

- Query thinking docs before advising: `query_links.py --main thinking gemma --limit 30`.
- Compare no/low/high thinking or equivalent runtime settings with the same eval set.
- Measure quality gain versus latency/throughput/context cost.
- Use harder reasoning, planning, and tool-recovery tasks for evaluation, not generic chat only.

### Image / video / audio capabilities

For multimodal Gemma work:

- Query capability pages first: image, video, audio, vision, function calling, prompt formatting.
- Verify the exact model variant supports the requested modality; do not assume all Gemma models do.
- For image/video, test resolution, frame sampling, prompt placement, latency, memory, and failure modes.
- For audio, distinguish direct audio support from ASR-transcript pipelines.
- For agents, evaluate whether multimodal content survives tool/harness serialization intact.

### OpenVINO / edge acceleration

For Gemma-specific OpenVINO work, read `openvino-gemma-deep-dive.md` first.

OpenVINO and OpenVINO Model Server docs are now bundled in the database. If the user asks
for OpenVINO, query these sources before implementation and still verify the exact target
model/version because support can vary by release. Check:

- whether the target Gemma 4 size/variant is supported;
- conversion/export path and tokenizer/chat-template compatibility;
- OpenVINO GenAI versus Optimum Intel versus OpenVINO Model Server path;
- quantization type and accuracy delta;
- CPU/iGPU/NPU target hardware;
- streaming, batching, context length, paged attention/KV behavior, and agent prompt compatibility.

OpenVINO queries:

```bash
python ~/.claude/skills/gemma-model-optimization/scripts/query_links.py --main openvino gemma --limit 30
python ~/.claude/skills/gemma-model-optimization/scripts/query_links.py --main openvino genai npu --limit 30
python ~/.claude/skills/gemma-model-optimization/scripts/query_links.py --main openvino model server llm --limit 30
python ~/.claude/skills/gemma-model-optimization/scripts/query_links.py --main optimum intel --limit 30
python ~/.claude/skills/gemma-model-optimization/scripts/query_links.py --main gemma4 openvino --limit 30
python ~/.claude/skills/gemma-model-optimization/scripts/query_links.py --main adapter lora openvino --limit 30
python ~/.claude/skills/gemma-model-optimization/scripts/query_links.py --main speculative decoding openvino --limit 30
```

## Troubleshooting tree

Use this decision tree for common Gemma optimization failures.

1. **Bad answers / wrong behavior**
   - First: verify prompt formatting/chat template/system instructions.
   - Then: test few-shot examples and thinking settings.
   - Then: inspect data/retrieval/tool context.
   - Only then: LoRA/QLoRA/full fine-tune.

2. **RAG hallucinations**
   - Check retrieval recall and reranker quality before blaming Gemma.
   - Verify source snippets are present, relevant, deduplicated, and not overpacked.
   - Add citation/abstention instructions and evaluate faithfulness.

3. **Agent tool-call failures**
   - Validate function-calling format and tool schema.
   - Check malformed JSON/arguments, missing required fields, and recovery path.
   - Add harness-level validators before fine-tuning.

4. **Slow generation / poor edge performance**
   - Check runtime choice: llama.cpp, MLX, LiteRT-LM, Ollama, vLLM/SGLang.
   - Check quantization, context length, KV cache, prompt length, batch/concurrency.
   - Check MTP support for Gemma 4 and measure with/without it.

5. **High VRAM/RAM**
   - Measure model weights versus KV cache separately.
   - Reduce prompt/context only with evidence; do not break known long-context workloads.
   - Try quantization/runtime changes before cutting required context.

6. **Fine-tune overfits or regresses**
   - Shrink/clean data, add validation, reduce epochs/LR/rank, restore base prompt format.
   - Compare adapter against base on general and task-specific held-out sets.

7. **Adapter/runtime mismatch**
   - Confirm exact base model, tokenizer, chat template, quantization, and adapter target modules.
   - Reproduce in the training framework before exporting to serving runtime.

8. **Multimodal failures**
   - Confirm the variant supports the modality.
   - Validate preprocessing, serialization, resolution/frame sampling/transcription.
   - Test simple single-modal cases before combined agent workflows.

## Prompting, thinking, and agent performance

Use this section whenever the user wants better real-world agent behavior using Gemma,
even if they say "tune" casually. Many improvements should happen before weight tuning:

- Validate Gemma 4 prompt formatting and chat template for the exact runtime.
- Check system-instruction placement and whether the runtime preserves it.
- Evaluate thinking-capability docs/settings when reasoning quality is the bottleneck.
- Shorten or structure prompts for edge inference to reduce TTFT and KV-cache pressure.
- Prefer examples/few-shot prompts for narrow behavior changes before LoRA.
- For agents, measure end-to-end task success, tool-call correctness, refusal rate,
  hallucinated tool arguments, context-retention failures, and recovery behavior.
- For llama.cpp/MLX edge deployments, test quantization, context size, prompt length,
  cache reuse, batch/concurrency, and streaming behavior with the actual agent prompt.

Prompting / thinking / agent queries:

```bash
python ~/.claude/skills/gemma-model-optimization/scripts/query_links.py --main prompt formatting gemma4 --limit 30
python ~/.claude/skills/gemma-model-optimization/scripts/query_links.py --main prompt system instructions --limit 30
python ~/.claude/skills/gemma-model-optimization/scripts/query_links.py --main thinking gemma --limit 30
python ~/.claude/skills/gemma-model-optimization/scripts/query_links.py --main function calling gemma4 --limit 30
python ~/.claude/skills/gemma-model-optimization/scripts/query_links.py --main basic multi-turn chat --limit 30
python ~/.claude/skills/gemma-model-optimization/scripts/query_links.py --main rag retrieval --limit 30
python ~/.claude/skills/gemma-model-optimization/scripts/query_links.py --main image understanding --limit 30
python ~/.claude/skills/gemma-model-optimization/scripts/query_links.py --main video understanding --limit 30
python ~/.claude/skills/gemma-model-optimization/scripts/query_links.py --main audio data --limit 30
```

Edge / efficient local runtime queries:

```bash
python ~/.claude/skills/gemma-model-optimization/scripts/query_links.py --main llama.cpp gemma --limit 30
python ~/.claude/skills/gemma-model-optimization/scripts/query_links.py --main mlx gemma --limit 30
python ~/.claude/skills/gemma-model-optimization/scripts/query_links.py --main litert-lm gemma --limit 30
python ~/.claude/skills/gemma-model-optimization/scripts/query_links.py --main ollama gemma --limit 30
python ~/.claude/skills/gemma-model-optimization/scripts/query_links.py --main unsloth gemma --limit 30
python ~/.claude/skills/gemma-model-optimization/scripts/query_links.py --main unsloth lora --limit 30
python ~/.claude/skills/gemma-model-optimization/scripts/query_links.py --main unsloth qlora --limit 30
```

## Common query recipes

Fine-tuning / QLoRA:

```bash
python ~/.claude/skills/gemma-model-optimization/scripts/query_links.py --main qlora gemma --limit 30
python ~/.claude/skills/gemma-model-optimization/scripts/query_links.py --main lora keras --limit 30
python ~/.claude/skills/gemma-model-optimization/scripts/query_links.py --main full fine tune --limit 30
```

Gemma 4 serving / local inference:

```bash
python ~/.claude/skills/gemma-model-optimization/scripts/query_links.py --main gemma4 transformers --limit 25
python ~/.claude/skills/gemma-model-optimization/scripts/query_links.py --main llama.cpp gemma --limit 25
python ~/.claude/skills/gemma-model-optimization/scripts/query_links.py --main mlx gemma --limit 25
python ~/.claude/skills/gemma-model-optimization/scripts/query_links.py --main vllm gemma4 --limit 25
```

Variants:

```bash
python ~/.claude/skills/gemma-model-optimization/scripts/query_links.py --main embeddinggemma fine tune --limit 25
python ~/.claude/skills/gemma-model-optimization/scripts/query_links.py --main functiongemma fine tune --limit 25
python ~/.claude/skills/gemma-model-optimization/scripts/query_links.py --main paligemma fine tune --limit 25
python ~/.claude/skills/gemma-model-optimization/scripts/query_links.py --main datagemma --limit 25
```

## Output style for Claude Code

For plans, produce:

- Target model + task.
- Workload type: RAG, agentic/tool-use, custom harness/Hermes, multimodal, edge, serving,
  fine-tune, or prompt-only optimization.
- Recommended tuning/serving/prompting path.
- Source links consulted from the database, including Unsloth/OpenVINO sources when relevant.
- Exact experiment matrix.
- Troubleshooting branch used or ruled out.
- Verification/eval checklist.
- Risks and rollback.

Avoid declaring a model optimized without benchmark/eval evidence.
