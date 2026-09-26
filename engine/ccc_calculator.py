#!/usr/bin/env python3
"""
CCC 因果热力学计算器 v1.0
C_total = S_BC + Ω₀·CD = constant（复杂度守恒律）
S_BC: 玻尔兹曼-因果熵 | CD: 因果密度 | 数学同构热力学(S_BC↔S, Ω₀CD↔-F/kT)
"""
import json, math, hashlib
from collections import defaultdict

KB = '/var/minis/shared/causal_knowledge'

class CCCCalculator:
    def __init__(self, kb_path=None):
        d = json.load(open(kb_path or f'{KB}/causal_knowledge.json'))
        self.atoms = d['atoms']
        self.n = len(self.atoms)
        self._build()

    def _build(self):
        self.edges = []
        self.nodes = set()
        self.out_w = defaultdict(list)
        for a in self.atoms:
            c, e = a['cause'], a['effect']
            w = a.get('confidence', 0.5)
            self.edges.append((c, e, w))
            self.nodes.add(c); self.nodes.add(e)
            self.out_w[c].append((e, w))
        self.N = len(self.nodes)
        self.E = len(self.edges)

    # ===== CD 因果密度 =====
    def causal_density(self, atoms_subset=None, label='全局'):
        """CD = 加权因果连接密度（单位节点的有效因果流量）"""
        edges = self.edges if atoms_subset is None else atoms_subset
        nodes = set()
        w_sum = 0.0
        for c, e, w in edges:
            nodes.add(c); nodes.add(e); w_sum += w
        n = len(nodes)
        if n < 2:
            return 0.0, {'nodes': n, 'edges': len(edges), 'w_sum': w_sum}
        max_edges = n * (n - 1)  # 有向图最大边
        density = len(edges) / max_edges
        avg_w = w_sum / max(len(edges), 1)
        connectivity = len(edges) / n  # 每节点平均边
        CD = density * avg_w * math.log(1 + connectivity)
        return CD, {'nodes': n, 'edges': len(edges), 'density': round(density, 6),
                    'avg_w': round(avg_w, 4), 'connectivity': round(connectivity, 3)}

    # ===== S_BC 因果熵 =====
    def causal_entropy(self, atoms_subset=None):
        """S_BC = 边权重分布的 Shannon 熵 + 出边分布熵（两层）"""
        edges = self.edges if atoms_subset is None else atoms_subset
        # 层1: 权重分布熵
        W = sum(w for _, _, w in edges)
        if W <= 0:
            return 0.0, {}
        S1 = -sum((w/W) * math.log((w/W) + 1e-12) for _, _, w in edges)
        # 层2: 节点出边分布熵（因果流量的组织度）
        out_count = defaultdict(float)
        for c, e, w in edges:
            out_count[c] += w
        total = sum(out_count.values())
        S2 = -sum((v/total) * math.log(v/total + 1e-12) for v in out_count.values()) if total > 0 else 0
        S_max = math.log(max(len(edges), 2))
        S_norm = (0.5*S1 + 0.5*S2) / S_max  # 归一化 [0,1]
        return S_norm, {'S_weight': round(S1, 4), 'S_flow': round(S2, 4), 'S_max': round(S_max, 4)}


    def _fast_CD_S(self, edges):
        """单次遍历同时算 CD 和 S(优化)"""
        nodes = set()
        W = 0.0
        out_count = defaultdict(float)
        for c, e, w in edges:
            nodes.add(c); nodes.add(e); W += w; out_count[c] += w
        n = len(nodes)
        if n < 2 or W <= 0:
            return 0.0, 0.0
        density = len(edges) / (n*(n-1))
        avg_w = W / len(edges)
        conn = len(edges) / n
        CD = density * avg_w * math.log(1 + conn)
        S1 = -sum((w/W)*math.log(w/W + 1e-12) for _, _, w in edges)
        total = sum(out_count.values())
        S2 = -sum((v/total)*math.log(v/total + 1e-12) for v in out_count.values())
        S_max = math.log(max(len(edges), 2))
        return CD, (0.5*S1 + 0.5*S2) / S_max

    # ===== 守恒律数值检验 =====
    def conservation_test(self, n_trials=10, perturb_frac=0.05):
        """CCC 核心检验: C_total 守恒性(快速版)"""
        import random
        CD0, S0 = self._fast_CD_S(self.edges)
        deltas = []
        for seed in range(5):
            rng = random.Random(seed)
            k = max(1, int(len(self.edges) * perturb_frac))
            idx = set(rng.sample(range(len(self.edges)), k))
            pert = [(c, e, w*rng.uniform(0.5, 1.5)) for i, (c, e, w) in enumerate(self.edges) if i not in idx]
            CD1, S1 = self._fast_CD_S(pert)
            if abs(CD1-CD0) > 1e-10:
                deltas.append(abs(S1-S0)/abs(CD1-CD0))
        deltas.sort()
        omega0 = deltas[len(deltas)//2] if deltas else 1.0
        C0 = S0 + omega0*CD0
        print(f'Ω₀={omega0:.4f} | 基准: CD={CD0:.6f} S={S0:.4f} C_total={C0:.4f}')
        drifts = []
        for seed in range(n_trials):
            rng = random.Random(1000+seed)
            k = max(1, int(len(self.edges)*perturb_frac))
            idx = set(rng.sample(range(len(self.edges)), k))
            pert = [(c, e, w*rng.uniform(0.5, 1.5)) for i, (c, e, w) in enumerate(self.edges) if i not in idx]
            CD1, S1 = self._fast_CD_S(pert)
            drifts.append(abs((S1+omega0*CD1) - C0)/C0)
        drifts.sort()
        med = drifts[len(drifts)//2]
        verdict = '守恒近似成立(慢变量)' if med < 0.15 else '守恒不成立(强变量)'
        print(f'守恒检验({n_trials}次{perturb_frac*100:.0f}%扰动): 中位漂移 {med*100:.2f}% | 最大 {drifts[-1]*100:.2f}%')
        print(f'判定: {verdict}')
        return {'omega0': omega0, 'drift_median': med, 'verdict': verdict}

    # ===== 分域热力学 + 相变检测 =====
    def domain_thermodynamics(self):
        """每个域的 CD/S_BC → 相变边界检测"""
        bydom = defaultdict(list)
        for a in self.atoms:
            bydom[a.get('domain', '')].append(a)
        stats = []
        for dom, atoms in bydom.items():
            sub = [(a['cause'], a['effect'], a.get('confidence', 0.5)) for a in atoms]
            CD, _ = self.causal_density(sub)
            S, _ = self.causal_entropy(sub)
            if CD > 0:
                stats.append({'domain': dom, 'CD': round(CD, 4), 'S_BC': round(S, 4),
                              'ratio': round(S/CD, 3) if CD > 0 else 0, 'n': len(atoms)})
        stats.sort(key=lambda x: -x['ratio'])
        print(f'\n═══ 分域热力学状态 (S_BC/CD = "因果温度"代理) ═══')
        print(f'  {"域":<14}{"CD":>9}{"S_BC":>8}{"S/CD":>8}{"n":>6}')
        for s in stats[:15]:
            print(f'  {s["domain"][:14]:<14}{s["CD"]:>9.4f}{s["S_BC"]:>8.4f}{s["ratio"]:>8.3f}{s["n"]:>6}')
        # 相变边界: S/CD 分布的离群检测
        ratios = [s['ratio'] for s in stats if s['n'] >= 20]
        if len(ratios) > 5:
            ratios.sort()
            q1, q3 = ratios[len(ratios)//4], ratios[3*len(ratios)//4]
            iqr = q3 - q1
            outliers = [s for s in stats if s['n'] >= 20 and (s['ratio'] > q3 + 1.5*iqr or s['ratio'] < q1 - 1.5*iqr)]
            print(f'\n  相变边界候选(S/CD 离群, IQR法): {len(outliers)} 个域')
            for s in outliers[:5]:
                print(f'    {s["domain"]}: S/CD={s["ratio"]:.3f} (n={s["n"]})')
        return stats

    # ===== 因果涌现: 宏观 vs 微观 CD =====
    def causal_emergence(self):
        """宏观(域级粗粒化)CD - 微观CD > 0 → 涌现"""
        # 微观: 全图
        CD_micro, _ = self.causal_density()
        # 宏观: 域级粗粒化(域=超节点)
        dom_edges = defaultdict(float)
        dom_nodes = set()
        for a in self.atoms:
            dc = a.get('domain', '')
            dom_edges[(dc, dc)] += a.get('confidence', 0.5)
            dom_nodes.add(dc)
        sub = [(c, e, w) for (c, e), w in dom_edges.items()]
        CD_macro, _ = self.causal_density(sub)
        emergence = CD_macro - CD_micro
        print(f'\n═══ 因果涌现检验 ═══')
        print(f'  微观 CD(原子级): {CD_micro:.4f}')
        print(f'  宏观 CD(域级粗粒化): {CD_macro:.4f}')
        print(f'  涌现量 Δ = {emergence:.4f} ({">0 涌现" if emergence > 0 else "≤0 未涌现"})')
        return {'micro': CD_micro, 'macro': CD_macro, 'emergence': emergence}

if __name__ == '__main__':
    import time
    t0 = time.time()
    c = CCCCalculator()
    print(f'加载: {c.n} 原子 | {c.N} 节点 | {c.E} 边')
    c.conservation_test()
    c.domain_thermodynamics()
    c.causal_emergence()
    print(f'\n耗时 {time.time()-t0:.1f}s')
