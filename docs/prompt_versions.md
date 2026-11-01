# Prompt versions (Phase 8)

The prompt is frozen before the final Phase 9 experiments and tagged
`prompt-v-final`. Record every wording change here.

| Version | Date | Change | Smoke-test status |
|---------|------|--------|-------------------|
| v1-draft | 2026-10-31 | Initial structured prompt: CSI mean/var/slope per class, queues, dQ_emg, U_t, trigger reason, active mask, constraints, schema, 3 few-shot examples. Few-shot bandwidth floors corrected to sum <= 0.95 so they match the validator. | MockLLM: 100% valid. Real Groq: pending (local network 403). |
| **v2-final** | 2026-11-01 | Added CSI scale hints (<0.1 = deep fading), queue-trend diagnosis rules, an explicit "queue growth is not a surge under deep fading" rule, JSON-formatted few-shot responses, and 2-significant-figure rounding of all numeric features for cache stability. | Real Groq `qwen/qwen3.8-27b`, n=50, cache disabled: **100% schema-valid**, 92% strict diagnosis accuracy, median 1159 ms. Frozen for Phase 9. |

Rules:
- The prompt must never contain `ood_flag` or ground-truth scenario labels.
- Numeric features are rounded to 2 significant figures for cache stability.
- Any prompt change invalidates the LLM cache; re-run the smoke test after it.
