# Layer 1: LLM Foundations — Completion Record

## Status
- [x] Artifact built and runs without errors
- [x] Gate questions answered below
- [x] Failure notes written below

---

## Artifact

What you built (one sentence): A tokenizer comparison notebook analyzing Arabic vs English token costs across 4 tokenizers on UAE property listings, and a tiny decoder-only transformer trained on property descriptions from scratch.

How to run it:
```bash
cd layers/layer_01
jupyter notebook tokenizer_comparison.ipynb
# tiny transformer:
python transformer-c.py
```

---

## Gate Answers

Answer each gate question in your own words.
Do not copy from the roadmap. Write what you actually understand.

**Q1: Why does an Arabic property description cost more tokens than English?**

**Your answer:**

Arabic tokenizers are squeezed into a fixed vocabulary that is heavily allocated toward English, leaving fewer merge slots for Arabic. On top of that, Arabic is morphologically rich — prefixes, suffixes, and roots fuse into single words that English tokenizers can't recognize as a unit, so they split them into many small pieces, multiplying the token count.

**Q2: Which tokenizer would you choose for a MENA property chatbot and why?**

**Your answer:**

GPT-4o's tokenizer (tiktoken cl100k_base) because the MENA property data is code-switched — listings mix Arabic and English in the same text. Compared to the other tokenizers tested, it handled both languages efficiently without blowing up the token count on Arabic segments, making it the best cost/quality tradeoff for a bilingual workload.

**Q3: What happens to KV cache when a user pastes a 5-page PDF floor plan?**

**Your answer:**

A 5-page PDF could be 5,000–10,000 tokens. The KV cache stores key/value pairs for every token in context, and its memory grows quadratically — each new token attends to all previous ones. TTFT gets very slow as the cache fills, and once the context limit is hit, the model starts dropping the beginning of the document, losing critical information like room dimensions or property specs at the top of the PDF.

---

## Failure Notes

What broke or surprised you while doing this layer?
(At least 2 entries required — if nothing broke, you skipped something.)

1. Short prompt TTFT was slower than medium — the short prompt took ~30s but medium took ~12s. The model was cold-starting on the first request, so the first run was loading the model into memory, not just processing tokens.

2. Temperature hallucinations — at temperature 1.2, deepseek confidently invented places like "Al Mawga market" that don't exist. High temperature doesn't just make output more creative, it makes the model fabricate specific-sounding facts.

---

## What I would do differently

Slow down before moving to the next step. A few times I moved on without fully understanding the current concept (like how the Linear layer and embeddings actually work). Asking "why" more before writing code, and spreading the layer over 2 sessions instead of one sitting, would make the concepts stick better.

---

## Time spent

Approximate hours on this layer: ~4 hours
