# Gemma Model Optimization Skill

Claude Code skill and supporting documentation database for Gemma model optimization.

Covers:

- Gemma 4 prompt formatting, thinking, MTP, and function/tool use
- Fine-tuning paths including Unsloth, LoRA/QLoRA, HF/PEFT, Keras, and Tunix/JAX
- RAG, agentic workflows, custom harnesses, and Hermes agent validation
- Edge/local/cloud serving paths including llama.cpp, MLX, LiteRT-LM, Ollama, vLLM, SGLang, Vertex/GKE
- Gemma-specific OpenVINO conversion, compilation, compression, adapter handling, OVMS serving, and troubleshooting
- Multimodal image/video/audio capability checks

## Contents

- `SKILL.md` — main Claude Code skill operating guide
- `openvino-gemma-deep-dive.md` — Gemma-specific OpenVINO tuning/compilation/troubleshooting runbook
- `scripts/query_links.py` — query tool for the bundled docs database
- `data/gemma_docs_links.sqlite` — searchable SQLite link/source database
- `data/curated_views.json` — curated result groups for common workflows
- `build_gemma_tune_link_db.py` — crawler/database rebuild script
- `research/openvino-gemma/` — captured OpenVINO Gemma notebook research artifacts

## Query examples

```bash
python scripts/query_links.py --main gemma4 openvino --limit 30
python scripts/query_links.py --main unsloth qlora --limit 30
python scripts/query_links.py --main thinking gemma --limit 30
python scripts/query_links.py --main rag retrieval --limit 30
python scripts/query_links.py --main adapter lora openvino --limit 30
```

## Install as a Claude Code skill

```bash
mkdir -p ~/.claude/skills/gemma-model-optimization
cp SKILL.md openvino-gemma-deep-dive.md ~/.claude/skills/gemma-model-optimization/
cp -r scripts data ~/.claude/skills/gemma-model-optimization/
```

## Rebuild database

```bash
python build_gemma_tune_link_db.py
```

The generated database is intentionally checked in so the skill is usable offline after clone.
