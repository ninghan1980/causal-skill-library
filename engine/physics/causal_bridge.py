#!/usr/bin/env python3
"""physics_engine/causal_bridge.py — 因果推断体系 × 物理引擎 桥接层
========================================================
把完整的因果推断体系接入物理引擎:
  - 因果发现引擎 v2.0 (causal_discovery_engine)
  - Pearl 因果框架 (pearl_causal)

三位一体:
  物理规律 (先验) + 因果发现 (学习) + Pearl推断 (反事实)

用法:
  from physics_engine.causal_bridge import CausalPhysicsBridge
  bridge = CausalPhysicsBridge()
  bridge.analyze(data, X='台风风速', Y='海洋温度')
"""
import sys
import numpy as np

# 因果体系路径
sys.path.insert(0, '/var/minis/skills/multi-search-engine/assets')

from causal_discovery_engine import CausalDiscoveryEngine
from pearl_causal import CausalGraph, PearlInference


class CausalPhysicsBridge:
    """因果推断体系 × 物理引擎 统一桥接"""

    def __init__(self):
        self.discovery = CausalDiscoveryEngine()
        self.pearl = PearlInference()
        self.graph = CausalGraph()

    # ─── 完整分析流程 ───
    def analyze(self, data: dict, X: str, Y: str, confounders: list = None,
                intervene_value: float = None) -> dict:
        """一体化因果分析: 发现 → 建模 → 推断
        data: {变量名: 时间序列数组}
        X, Y: 关注的两个变量
        confounders: 已知混杂变量 (可选)
        """
        result = {'X': X, 'Y': Y}

        # 1. 关联发现
        result['pearson'] = round(self.discovery.pearson(data[X], data[Y]), 4)

        # 2. Granger 时序因果
        gc = self.discovery.granger_matrix(data, max_lag=3, significance=0.01)
        relevant = [g for g in gc if g['cause'] == X and g['effect'] == Y]
        result['granger'] = relevant[0] if relevant else None

        # 3. 混杂识别 (如果给了混杂候选)
        if confounders:
            conf_result = []
            for z in confounders:
                c = self.discovery.is_confounder(data[X], data[Y], data[z])
                conf_result.append({'confounder': z, **c})
            result['confounder_analysis'] = conf_result

        # 4. Pearl 因果图建模
        self.graph = CausalGraph()
        self.graph.add_edge(X, Y)  # 假设 X→Y
        if confounders:
            for z in confounders:
                self.graph.add_edges([(z, X), (z, Y)])  # 混杂: Z→X, Z→Y

        # 5. do-演算 (观测关联 vs 因果效应)
        do_result = self.pearl.do_calculus(
            X, Y, data,
            adjustment_set=(confounders if confounders else
                            self.graph.find_confounders(X, Y)))
        result['do_calculus'] = do_result

        # 6. 反事实 (若给了干预值)
        if intervene_value is not None:
            cf = self.discovery.counterfactual(data[X], data[Y], intervene_value)
            result['counterfactual'] = cf

        return result

    # ─── 从物理引擎时序数据直接分析 ───
    def analyze_physical(self, engine_results: dict, X: str, Y: str, **kwargs):
        """直接分析物理引擎产生的时序数据
        engine_results: 物理引擎输出的 {变量: 数组}
        """
        return self.analyze(engine_results, X, Y, **kwargs)

    # ─── 工具变量分析 ───
    def instrumental_variable(self, data, Z, X, Y):
        """工具变量法: Z 工具 → X → Y"""
        return self.pearl.instrumental_variable(Z, X, Y, data)

    # ─── 中介分析 ───
    def mediation(self, data, X, M, Y):
        """中介分析: X → M → Y"""
        return self.pearl.frontdoor_adjustment(X, Y, M, data)

    # ─── 总结报告 ───
    def summarize(self, analysis: dict) -> str:
        """生成分析总结"""
        parts = []
        parts.append(f"{analysis['X']} 与 {analysis['Y']} 因果分析:")
        parts.append(f"  关联 r={analysis['pearson']}")

        if analysis.get('granger'):
            g = analysis['granger']
            parts.append(f"  Granger: {g['cause']}→{g['effect']} F={g['F_stat']} (时序因果)")
        else:
            parts.append(f"  Granger: 未检测到 {analysis['X']}→{analysis['Y']} 时序因果")

        do = analysis.get('do_calculus', {})
        if do:
            parts.append(f"  观测关联={do.get('observational_association')}, "
                         f"因果效应={do.get('causal_effect_do')}")
            parts.append(f"  混杂控制: {do.get('confounders_controlled')}")

        if analysis.get('counterfactual'):
            parts.append(f"  {analysis['counterfactual']['note']}")

        return "\n".join(parts)


# ─── CLI 演示 ───
if __name__ == "__main__":
    print("="*70)
    print("因果推断体系 × 物理引擎 桥接演示")
    print("="*70)

    bridge = CausalPhysicsBridge()

    # 场景: 台风-海洋 (含混杂)
    np.random.seed(42)
    n = 200
    # 海温(混杂) → 台风风速 和 海温 → 海表面蒸发
    sst = np.random.normal(28, 1, n)
    wind = 1.2 * sst + np.random.normal(0, 0.5, n)
    evaporation = 0.9 * sst + np.random.normal(0, 0.5, n)

    data = {'海温': sst, '台风风速': wind, '蒸发量': evaporation}

    print("\n▶ 场景: 台风-海洋物理系统")
    print("  变量: 海温(混杂), 台风风速, 蒸发量")

    r = bridge.analyze(data, X='台风风速', Y='蒸发量',
                       confounders=['海温'], intervene_value=30)
    print("\n" + bridge.summarize(r))

    print("\n▶ 正确结论:")
    print("  台风风速 ↔ 蒸发量 是假因果 (都是海温驱动的)")
    print("  控制海温后, 两者的'虚假'相关应消失")

    print("\n" + "="*70)
