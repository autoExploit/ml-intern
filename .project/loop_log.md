# Engram-Micro Loop Log

## Loop 0 — Bootstrap

### OBSERVE
- Repo `ml-intern` exists with unrelated agent scaffolding (backend/frontend/agent dirs, HF Pro tooling). Not relevant to Engram-Micro mission. Will treat root as workspace; new code will go under `engram_micro/`.
- Remote: `https://github.com/autoExploit/ml-intern` configured.
- No `.project/` dir existed → created `research/`, `report/`, `results/`.
- No prior Engram code, no research notes, no results.

### ORIENT
- Highest leverage right now: **understand Engram precisely** before writing a single line. Briefing in the prompt is detailed but must be verified vs the paper + repo. If the briefing is wrong on a key detail (e.g., what the "memory table" actually stores per slot, how multi-head hashing partitions are organized, what gating signal is), every downstream decision compounds error.
- Not yet meaningful to refactor or pick a baseline backbone — those decisions depend on memory-table mechanics.

### DECIDE
**Action:** Research the Engram paper (arXiv 2601.07372) and the deepseek-ai/Engram GitHub repo. Produce `.project/research/engram_spec.md` capturing the *exact* mechanics of (1) tokenizer compression, (2) hashed N-gram lookup table shape & per-slot contents, (3) gating, (4) layer integration, (5) sparsity allocation. Also a short companion file `.project/research/small_scale_questions.md` listing what we expect changes at <1B.

**Falsifiable prediction:** After this loop I should be able to answer, with citations to specific files/equations: "what is the dtype, shape, and update rule of one Engram memory table entry?" If I can't, the research is incomplete.

### DEVIL'S ADVOCATE
- *Technical attack:* Paper might describe the table differently from the code (papers often lag). Mitigation: read code as ground truth, paper as motivation.
- *Experimental attack:* Just reading docs doesn't reduce risk unless I write down the *small-scale-relevant* invariants. Mitigation: explicit `small_scale_questions.md`.
- *Priority attack:* Could I skip research and just clone the official repo + scale it down? No — the mission is to understand mechanism well enough to deviate justifiably. Cloning would hide the assumptions that break at <1B.

Rebuttals stand. Proceeding.

### DO
(spawn research sub-agent — large, parallel-ish, benefits from isolated context: reading paper + repo end-to-end is exactly the use case)

