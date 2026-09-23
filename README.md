# Causal Skill Library

> AI Agent 的因果知识 + 技能库 — 知道 WHY，也知道 HOW。

**11,807 causal atoms** across **93 domains** · **53 physics models** · **15 feedback loops** · skill extraction pipeline

---

## What is this

A structured knowledge library that combines:
- **Causal knowledge** (WHY: "A causes B, confidence 0.90") — for reasoning and explanation
- **Skills** (HOW: "To achieve X, do steps 1-2-3") — for execution
- **Physics models** (VERIFY: 53 simulation engines) — for grounding

Built for AI agents that need to not just **do things**, but **understand why things work**.

## Quick Start

```python
from causal_knowledge_base import CausalKnowledgeBase
kb = CausalKnowledgeBase.load("/path/to/data")
result = kb.reason("央行降准", goal="经济影响")
# → causal chain with confidence scores
```

## Data Format

Each causal atom in `data/causal_atoms.jsonl`:

```json
{
  "id": "a1b2c3d4e5f6",
  "cause": "央行降低存款准备金率",
  "effect": "银行可贷资金增加",
  "confidence": 0.9,
  "mechanism": "货币政策工具",
  "domain": "经济学",
  "tier": "curated",
  "evidence_type": "stated_mechanism",
  "verification_status": "verified",
  "quality_tier": "A",
  "source": "PMID:12345 | Nature 2020 | ..."
}
```

## License

- Data: CC-BY-4.0
- Code: MIT
