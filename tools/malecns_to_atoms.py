#!/usr/bin/env python3
"""MaleCNS v1.0 → 因果库 measured/ 层原子格式。

v1 的 bug（已修）：
  · 用了 type+side 做键，而库的 FAFB 层 0% 带 _L/_R 后缀 → 改为裸细胞型名
  · 没剔除空 type（2,194 个神经元 / 影响 170,545 条连接）
    空串在 np.unique 里排第一，使索引 0 变成垃圾桶，首条原子成了它的自环
  · 未校验输出即宣称成功 —— 是格式对照才发现
"""
import json, gzip, hashlib, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy import sparse

sys.path.insert(0, "/home/ubuntu/FlyDrones/src")
from flydrones.brain.connectome import Connectome

MIN_SYN = int(sys.argv[1]) if len(sys.argv) > 1 else 5
OUT = "/home/ubuntu/causal-net/integrate/malecns_connectome_atoms.jsonl.gz"

c = Connectome.load("/home/ubuntu/FlyDrones/data/malecns_brain.npz")
T = c.types.astype(str)
n_all = len(T)
good = (T != "") & (T != "nan")
print(f"神经元 {n_all:,}，其中 type 有效 {int(good.sum()):,}，剔除空 type {int((~good).sum()):,}", flush=True)

coo = c.weights.tocoo()
mask = good[coo.row] & good[coo.col]
rows_n, cols_n, vals_n = coo.row[mask], coo.col[mask], coo.data[mask]
print(f"连接 {coo.nnz:,} → 保留两端都有 type 的 {mask.sum():,} ({mask.sum()/coo.nnz*100:.1f}%)", flush=True)

Tg = T[good]
uniq, inv_g = np.unique(Tg, return_inverse=True)
n_types = len(uniq)
assert uniq[0] != "", "索引 0 仍是空串 —— 拒绝写出"
# 全局索引 → 类型索引
gidx = np.full(n_all, -1, dtype=np.int64)
gidx[np.where(good)[0]] = inv_g
a = gidx[rows_n]; b = gidx[cols_n]
print(f"细胞型 {n_types:,}（裸名，与库 FAFB 层命名一致）", flush=True)

P = sparse.coo_matrix((np.abs(vals_n), (a, b)), shape=(n_types, n_types)).tocsr()
P.sum_duplicates(); P.eliminate_zeros()
print(f"细胞型层级边 {P.nnz:,}", flush=True)

cnt = np.bincount(inv_g, minlength=n_types)
rr, cc = P.nonzero()
ww = np.asarray(P[rr, cc]).ravel()
keep = ww >= MIN_SYN
rr, cc, ww = rr[keep], cc[keep], ww[keep]
selfloop = int((rr == cc).sum())
print(f"≥{MIN_SYN} 突触保留 {len(rr):,}（其中细胞型自环 {selfloop:,}）", flush=True)

SRC = "MaleCNS v1.0 (HHMI Janelia / Cambridge / MRC LMB / Google Research)"
n = 0
with gzip.open(OUT, "wt", encoding="utf-8") as f:
    for i, j, w in zip(rr, cc, ww):
        ca, cb = str(uniq[i]), str(uniq[j])
        assert ca and cb, "空名字漏出"
        w = int(round(float(w)))
        f.write(json.dumps({
            "id": hashlib.sha1(f"malecns|{ca}|{cb}".encode()).hexdigest()[:12],
            "cause": f"{ca} 细胞型激活",
            "effect": f"{cb} 细胞型激活",
            "confidence": 0.9 if w >= 50 else (0.8 if w >= 20 else 0.7),
            "condition": (f"MaleCNS v1.0 结构连接（{w} 突触；"
                          f"{ca} {int(cnt[i])} 个神经元 → {cb} {int(cnt[j])} 个）"),
            "mechanism": "突触传递（结构连接，方向由突触前/后定义）",
            "formula": "", "variables": {}, "units": "",
            "source": SRC,
            "evidence": [f"electron microscopy, {w} synapses"],
            "counterexamples": [],
            "domain": "神经连接组",
            "tags": ["connectome", "malecns", "measured"],
            "evidence_type": "physical_measurement",
            "verification_status": "verified",
            "quality_tier": "A",
            "tier": "measured", "kb_layer": "connectome", "entity_type": "cell_type",
            "dataset": "malecns_v1", "weight_synapses": w, "direction": "forward",
            "reversal_condition": None, "direction_source": "measured",
            "identifiability": "identifiable",
        }, ensure_ascii=False) + "\n")
        n += 1
print(f"\n→ {OUT}  ({n:,} 条)", flush=True)
