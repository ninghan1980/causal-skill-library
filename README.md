# Causal Skill Library

> AI Agent 的因果知识 + 技能库 — 知道 WHY，也知道 HOW。

**三层因果库**：

| 层 | 条数 | 来源 | 置信度 |
|---|---|---|---|
| **curated** (策展) | 11,807 | 文献/专家/物理引擎验证 | 0.5-0.95 |
| **measured** (实测) | 20,359 | 果蝇全脑连接组 FlyWire v630 | 0.55-0.85 |
| **generated** (生成) | 2,636 | LLM 生成（待验证） | 0.5-0.70 |

**合计 ~34,800 条因果原子** · 93+ 领域 · 53 物理引擎 · 15 反馈回路

---

## 三层结构

```
curated (策展)
  人类策展 + 文献溯源 + 物理引擎交叉验证
  每条有: cause → effect + confidence + mechanism + source(PMID/DOI)

measured (实测)
  来自果蝇全脑连接组 FAFB v630 快照
  124,891 神经元 → 聚合为 1,341 细胞型 → 20,359 有向边
  evidence_type=physical_measurement, direction_source=measured
  MFAS = 34.76% → 真实生物系统 88.6% 节点在递归核中

generated (生成)
  LLM 批量生成（领域模板+交叉组合）
  confidence ≤ 0.70, quality_tier=C, 未经验证
  用于扩大假设空间，使用时需人工确认
```

## 关键校准发现

真实生物因果系统 vs 我们策展库的结构差异：

| | 果蝇脑 (真值) | 策展库 |
|---|---|---|
| 巨型 SCC | 88.6% 节点 | 0.1% |
| 互惠率 | 8.9% | 0.1% |
| direction_source | measured 100% | assumed 92.8% |

→ 我们的库不像真实因果系统。后续扩容方向：增加反馈环。

## 文件结构

```
data/
├── causal_atoms.jsonl.gz       ← 策展层 11,807 条 (gzip)
├── measured/
│   └── connectome_celltype.jsonl.gz  ← 果蝇实测 20,359 条 (gzip)
├── generated/
│   └── generated.jsonl.gz      ← LLM 生成 2,636 条 (gzip)
├── by-domain/                  ← 20 个领域分片
└── loops.json                  ← 反馈回路
engine/physics/                  ← 53 个物理模型 (27 个 Python 文件)
skills/skos_registry.json        ← 75 个技能注册表
schema/atom.schema.json          ← JSON Schema
```

## 用法

```python
import gzip, json

# 读取策展层
with gzip.open("data/causal_atoms.jsonl.gz", "rt") as f:
    atoms = [json.loads(line) for line in f]

# 读取果蝇实测层
with gzip.open("data/measured/connectome_celltype.jsonl.gz", "rt") as f:
    connectome = [json.loads(line) for line in f]
```

## License

- Data: CC-BY-4.0
- Code: MIT
