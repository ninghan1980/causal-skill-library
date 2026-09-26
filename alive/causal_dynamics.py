#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""因果库动力学引擎 —— 传播激活 + 三条活性判据

用法
----
    echo '{"graph":"clean","start":"全球变暖","compare":"海平面上升"}' \
      | python3 alive/causal_dynamics.py

stdin（JSON）
    graph    clean | full | measured | fafb     默认 clean
    start    起点标签（可模糊匹配）；省略则自动取出度最高节点
    compare  可选。给第二个起点，用于测 L2 输入依赖
    mode     nonlinear（默认） | linear
    steps    传播步数，默认 40
    decay    衰减，默认 0.2
    top_k    返回前 k 个激活节点，默认 10

活性判据（三条全满足才算「计算」）
    L1 非平凡    稳态标准差 > 1e-4
    L2 输入依赖  两个起点的稳态余弦相似度 < 0.9
    L3 有选择性  活跃节点占比 < 5%

    ★ 只满足 L1 的叫「震荡」；L1+L2 叫「回声」；三条全满足才是「计算」。

依赖
    numpy, scipy   （apt install python3-numpy python3-scipy）

开发记录（两个已修的 bug，留档以免重蹈）
----------------------------------------
bug-1  方向写反：用 Mn.T@a 而不是 Mn@a —— 从「全球变暖」走到了它的【原因】。
       靠经验 sanity check 发现（看激活集内容合不合理），不是读代码发现的。
bug-2  绝对阈值：θ=0.02 是绝对量，起点出边多时每条分不到 θ 就被全杀，
       传播完全断掉、只剩起点，**而三条判据仍全报 True、输出「判定=计算」**。
       已改为相对阈值 θ*max。
       ⇒ 教训：三条判据全满足时，先看激活集是不是只剩起点。
"""
from __future__ import annotations
import json
import gzip
import os
import sys

import numpy as np
from scipy import sparse

# ── 定位仓库根：从本文件往上找，直到看见 data/causal_atoms.jsonl.gz ──
def _repo_root() -> str:
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        if os.path.exists(os.path.join(d, "data", "causal_atoms.jsonl.gz")):
            return d
        d = os.path.dirname(d)
    return os.getcwd()

ROOT = _repo_root()

GRAPHS = {
    "full":     ("data/causal_atoms.jsonl.gz",                 "全量主张 24,589 条"),
    "clean":    ("data/causal_atoms_clean.jsonl.gz",           "干净子集（已剔合成物）"),
    "measured": ("data/measured/malecns_connectome_atoms.jsonl.gz", "MaleCNS 连接组 1,059,501 条"),
    "fafb":     ("data/measured/connectome_celltype.jsonl.gz", "FAFB v630 连接组 20,359 条"),
}


def load(path: str):
    idx, E, C, W = {}, [], [], []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            c = str(d.get("cause"))
            e = str(d.get("effect"))
            w = d.get("weight_synapses")
            if not isinstance(w, (int, float)) or w <= 0:
                w = float(d.get("confidence") or 0.5)
            for x in (c, e):
                if x not in idx:
                    idx[x] = len(idx)
            C.append(idx[c]); E.append(idx[e]); W.append(float(w))
    n = len(idx)
    # M[effect, cause] —— 正向：cause 驱动 effect
    return sparse.csr_matrix((W, (E, C)), shape=(n, n)), list(idx)


def propagate(M, names, start, mode="nonlinear", steps=40, decay=0.2, theta=0.02):
    n = M.shape[0]
    col = np.asarray(M.sum(0)).ravel()
    col[col == 0] = 1.0
    Mn = M.multiply(1.0 / col).tocsr()        # 每列（每个 cause）的出权重归一
    a = np.zeros(n)
    a[names.index(start)] = 1.0
    for _ in range(steps):
        # ★ 正向传播：Mn@a（不是 Mn.T@a —— 见文件头 bug-1）
        a = (1 - decay) * a + decay * np.asarray(Mn @ a).ravel()
        if mode == "nonlinear":
            mx = a.max()
            if mx > 0:
                a = np.where(a < theta * mx, 0.0, a)      # ★ 相对阈值（bug-2）
            if a.sum() > 0:
                k = max(1, int(0.05 * n))                 # 侧抑制：只留最强 5%
                thr = np.partition(a, -k)[-k]
                a = np.where(a >= thr, a, 0.0)
        s = a.sum()
        if s > 0:
            a = a / s
    return a


def resolve(names, s):
    if s in names:
        return s, None
    hits = [x for x in names if s in x]
    if not hits:
        return None, f"起点未找到: {s}"
    hits.sort(key=len)
    return hits[0], f"起点模糊匹配为: {hits[0]}"


def main():
    try:
        req = json.load(sys.stdin)
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"bad stdin: {e}"}, ensure_ascii=False))
        return

    gk = req.get("graph") or "clean"
    if gk not in GRAPHS:
        print(json.dumps({"ok": False, "error": f"unknown graph {gk}; 可选 {list(GRAPHS)}"}, ensure_ascii=False))
        return
    rel, label = GRAPHS[gk]
    path = os.path.join(ROOT, rel)
    if not os.path.exists(path):
        hint = ""
        if gk == "clean":
            hint = "（先运行 tools/classify_tiers.py 生成）"
        print(json.dumps({"ok": False, "error": f"数据文件不存在: {rel} {hint}"}, ensure_ascii=False))
        return

    M, names = load(path)
    mode = req.get("mode") or "nonlinear"
    steps = int(req.get("steps") or 40)
    decay = float(req.get("decay") or 0.2)
    topk = int(req.get("top_k") or 10)

    outd = np.asarray(M.sum(0)).ravel()
    indeg = np.asarray(M.sum(1)).ravel()
    s_req = req.get("start") or names[int(outd.argmax())]
    start, note = resolve(names, s_req)
    if start is None:
        print(json.dumps({"ok": False, "error": note}, ensure_ascii=False))
        return

    a = propagate(M, names, start, mode, steps, decay)
    order = np.argsort(-a)[:topk]
    active = int((a > 0).sum())

    res = {
        "ok": True, "graph": gk, "graph_label": label,
        "nodes": int(M.shape[0]), "edges": int(M.nnz),
        "start": start, "start_note": note, "mode": mode, "steps": steps,
        "L1_std": round(float(a.std()), 8),
        "L3_active_nodes": active,
        "L3_active_ratio": round(active / M.shape[0], 5),
        "activated": [{
            "node": names[i], "weight": round(float(a[i]), 6),
            "out_degree": int(outd[i]), "in_degree": int(indeg[i]),
        } for i in order if a[i] > 0],
    }

    cmp_req = req.get("compare")
    if cmp_req:
        s2, _ = resolve(names, cmp_req)
        if s2:
            b = propagate(M, names, s2, mode, steps, decay)
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            cos = float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0
            res["L2_compare_start"] = s2
            res["L2_cosine_similarity"] = round(cos, 5)
            res["L2_input_dependent"] = bool(cos < 0.9)

    L1 = res["L1_std"] > 1e-4
    L3 = res["L3_active_ratio"] < 0.05
    L2 = res.get("L2_input_dependent")
    # ★ 若激活集只剩起点，任何判据都不算数（见文件头 bug-2）
    degenerate = active <= 1
    res["verdict"] = {
        "L1_nontrivial": L1, "L3_structured": L3, "L2_input_dependent": L2,
        "degenerate": degenerate,
        "is_computation": bool(L1 and L3 and L2 is True and not degenerate),
        "note": ("★ 退化了：激活集只剩起点，判据一律不算数" if degenerate else
                 "三条全满足=计算；缺 L2 未测" if L2 is None else
                 "计算" if (L1 and L3 and L2) else
                 "震荡（非平凡但塌陷/无选择性）" if L1 else "平凡（塌成常数）"),
    }
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
