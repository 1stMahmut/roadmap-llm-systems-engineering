# Layer 01: LLM Foundations

## What you need to produce before committing

| # | Deliverable | File |
|---|---|---|
| 1 | Tokenizer comparison notebook | `tokenizer_comparison.ipynb` |
| 2 | Tiny transformer (trains on property descriptions) | `tiny_transformer.py` |
| 3 | Completion record with gate answers | `COMPLETION.md` |

---

## Gate questions (answer these in COMPLETION.md)

1. Why does an Arabic property description cost more tokens than the same text in English? Name one specific reason related to how BPE tokenizers are trained.

2. You have a property chatbot that processes 10,000 listings per day. A user pastes a 5-page PDF floor plan (≈3,000 tokens) into the chat. What happens to TTFT and why?

3. You run the same prompt with `temperature=0` ten times. Do you always get the same answer? Why or why not?

---

## Exercises

See `PERSONALIZED_ROADMAP.md` → Layer 1 for full exercise descriptions.

### Exercise 1.1 — Tokenizer cost analysis
- Download: https://www.kaggle.com/datasets/kanchana1990/uae-real-estate-2024-dataset
- Compare Arabic vs English token counts across 4 tokenizers
- Build a cost projection table

### Exercise 1.2 — Decode behavior
- Install Ollama: https://ollama.com
- Pull model: `ollama pull qwen2.5:7b-instruct-q4_K_M`
- Run the same property prompt at 4 temperature settings

### Exercise 1.3 — KV cache intuition
- Send prompts of 200 / 2000 / 10000 tokens to Ollama
- Record TTFT for each

### Tiny transformer
- Train on 1000 rows from the Kaggle dataset (English descriptions only first)
- Reference: https://github.com/karpathy/minGPT

---

## Setup

```bash
pip install transformers tiktoken datasets torch jupyter
ollama pull qwen2.5:7b-instruct-q4_K_M
```

---

## When you are done

Copy `templates/layer_completion_template.md` to this folder as `COMPLETION.md`, fill it in, then update `PROGRESS.md`.
