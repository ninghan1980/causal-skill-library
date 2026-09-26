#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""给每条原子分级，并导出干净子集。

分级不靠猜测，靠字段：
  synthetic_flip    mechanism 含 [反向]           ← 脚本按规则翻转出来的边
  synthetic_hub     mechanism 含 [枢纽互联]/跨域连接
  generator_output  source 含「生成器」
  llm_gapfill       source 含 llm / gapfill
  claim_testable    其余中，成立条件具体且 != domain
  claim_generic     其余中，条件泛化或缺失

输出
  data/measured/../tiers.jsonl            每条原子带 _tier
  data/causal_atoms_clean.jsonl.gz        仅 claim_*（供 causal_dynamics.py 的 clean 图）
"""
import json, gzip, os, collections, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "causal_atoms.jsonl.gz")
TIERS = os.path.join(ROOT, "data", "tiers.jsonl.gz")
CLEAN = os.path.join(ROOT, "data", "causal_atoms_clean.jsonl.gz")

def tier(d):
    m = str(d.get("mechanism") or "")
    s = str(d.get("source") or "")
    if "[反向]" in m: return "synthetic_flip"
    if "枢纽互联" in m or "跨域连接" in m: return "synthetic_hub"
    if "生成器" in s: return "generator_output"
    if "llm" in s.lower() or "gapfill" in s.lower(): return "llm_gapfill"
    c = str(d.get("condition") or "").strip()
    dm = str(d.get("domain") or "")
    if c and c != "一般成立" and c != dm: return "claim_testable"
    return "claim_generic"

def main():
    if not os.path.exists(SRC):
        sys.exit(f"找不到 {SRC}")
    buckets = collections.defaultdict(list)
    n = 0
    with gzip.open(SRC, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: d = json.loads(line)
            except Exception: continue
            n += 1
            d["_tier"] = tier(d)
            buckets[d["_tier"]].append(d)

    order = ["synthetic_flip","synthetic_hub","generator_output","llm_gapfill",
             "claim_testable","claim_generic"]
    print(f"总条数 {n:,}\n═══ 分级 ═══")
    for k in order:
        v = len(buckets.get(k, []))
        print(f"  {k:<18} {v:>6,}  {v/n*100:5.1f}%")
    clean = buckets["claim_testable"] + buckets["claim_generic"]
    synth = buckets["synthetic_flip"] + buckets["synthetic_hub"]
    print(f"\n  可推理主张（claim_*）{len(clean):>6,}  {len(clean)/n*100:5.1f}%")
    print(f"  拓扑填充物（synthetic）{len(synth):>6,}  {len(synth)/n*100:5.1f}%")

    with gzip.open(TIERS, "wt", encoding="utf-8") as f:
        for k in order:
            for d in buckets.get(k, []):
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with gzip.open(CLEAN, "wt", encoding="utf-8") as f:
        for d in clean:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"\n→ {os.path.relpath(TIERS, ROOT)}")
    print(f"→ {os.path.relpath(CLEAN, ROOT)}  ({len(clean):,} 条)")

if __name__ == "__main__":
    main()
