#!/usr/bin/env python3
"""
δPT 导航器 v1.0 — 微分-投影理论的工程化
区分树 (Distinction Tree) 在 (d,s,θ) 处的投影 = 物理理论
d=区分深度 s=尺度 θ=视角 | 变换算子 T₁/T₂/T₃ | 矛盾消解
"""
import re

# d 轴: 区分深度 → 现象层次
D_LEVELS = [
    (0, '存在本身', ['存在', '本体', '区分本身', '为什么有', '宇宙开端', '本源']),
    (1, '时空/物质/能量', ['时空', '时间', '空间', '物质', '能量', '宇宙学', '大爆炸', '引力', '膨胀', '暗能量']),
    (2, '粒子/场/力', ['粒子', '量子场', '夸克', '电子', '光子', '核', '强力', '弱力', '电磁', '标准模型', '希格斯', '量子力学', '波函数']),
    (3, '原子/分子/化学', ['原子', '分子', '化学', '化学键', '反应', '晶体', '材料', '催化']),
    (4, '细胞/生物', ['细胞', 'DNA', '基因', '蛋白质', '生物', '进化', '神经元', '免疫', '代谢', '酶']),
    (5, '器官/个体/认知', ['器官', '大脑', '心脏', '个体', '认知', '感知', '意识', '学习', '记忆', '行为', '智能', 'AI', '神经网络', '大模型']),
    (6, '社会/文明', ['社会', '经济', '文明', '国家', '市场', '政治', '文化', '战争', '就业', '人口', '制度', '历史']),
]

# s 轴: 尺度
S_LEVELS = [
    ('普朗克', ['普朗克', '量子引力', '弦', '10^-35']),
    ('核/粒子', ['核', '粒子', '夸克', '费米']),
    ('原子/分子', ['原子', '分子', '化学', '纳米']),
    ('宏观连续', ['流体', '连续', '经典', '材料', '工程', '日常']),
    ('天体/宇宙', ['天体', '行星', '恒星', '宇宙', '星系', '黑洞']),
    ('个体/社会', ['个体', '人', '社会', '经济', '国家']),
]

# θ 轴: 视角
THETA_LEVELS = [
    (0, '本体', ['是什么', '本体', '存在', '定义', '本质', '实在']),
    (45, '动力学', ['如何变化', '演化', '动力', '过程', '机制', '方程', '预测', '发展']),
    (90, '对称性', ['对称', '守恒', '不变', '诺特', '守恒律', '对称性破缺']),
    (180, '自指', ['自指', '意识', '自我', '观察者', '怪圈', '反思', '元认知', '涌现']),
]

# 已知理论定位表
THEORY_LOCATIONS = {
    '量子力学': (2, '核/粒子', 45), '量子场论': (2, '普朗克', 45),
    '广义相对论': (1, '天体/宇宙', 45), '狭义相对论': (1, '宏观连续', 45),
    '热力学': (3, '宏观连续', 90), '统计力学': (3, '原子/分子', 45),
    '牛顿力学': (3, '宏观连续', 45), '电磁学': (2, '宏观连续', 45),
    '进化论': (4, '个体/社会', 45), '分子生物学': (4, '原子/分子', 0),
    '神经科学': (5, '宏观连续', 45), '经济学': (6, '个体/社会', 45),
    '大语言模型': (5, '宏观连续', 45), '信息论': (2, '宏观连续', 90),
    '计算理论': (0, '个体/社会', 180), '意识科学': (5, '宏观连续', 180),
}

class DPTNavigator:
    def assign_d(self, text):
        scores = []
        for d, name, kws in D_LEVELS:
            s = sum(1 for kw in kws if kw.lower() in text.lower())
            scores.append((s, d, name))
        scores.sort(reverse=True)
        return scores[0] if scores[0][0] > 0 else (1, 1, '时空/物质/能量')

    def assign_s(self, text):
        scores = []
        for name, kws in S_LEVELS:
            s = sum(1 for kw in kws if kw.lower() in text.lower())
            scores.append((s, name))
        scores.sort(reverse=True)
        return scores[0] if scores[0][0] > 0 else (1, '宏观连续')

    def assign_theta(self, text):
        best = max(THETA_LEVELS, key=lambda t: sum(1 for kw in t[2] if kw in text.lower()))
        return best[0], best[1]

    def locate(self, phenomenon):
        """坐标分配: 现象 → (d,s,θ)"""
        d_score, d, d_name = self.assign_d(phenomenon)
        s_score, s = self.assign_s(phenomenon)
        theta, theta_name = self.assign_theta(phenomenon)
        return {'d': d, 'd_name': d_name, 's': s, 'theta': theta,
                'theta_name': theta_name, 'confidence': min(d_score, s_score)}

    def locate_theory(self, theory_name):
        """已知理论定位"""
        for name, loc in THEORY_LOCATIONS.items():
            if name in theory_name or theory_name in name:
                d, s, th = loc
                theta_name = next(t[1] for t in THETA_LEVELS if t[0] == th)
                return {'theory': name, 'd': d, 's': s, 'theta': th, 'theta_name': theta_name}
        return None

    def dissolve_contradiction(self, theory_a, theory_b):
        """矛盾消解: 两个理论的坐标比较"""
        la, lb = self.locate_theory(theory_a), self.locate_theory(theory_b)
        if not la or not lb:
            return {'error': '理论未在定位表中', 'hint': '先用 locate 分配坐标'}
        diffs = []
        if la['d'] != lb['d']: diffs.append(f"d 轴不同({la['d']} vs {lb['d']}): 区分深度不同——T₁ 深度变换可导出")
        if la['s'] != lb['s']: diffs.append(f"s 轴不同({la['s']} vs {lb['s']}): 尺度不同——T₂ 重整化变换可导出")
        if la['theta'] != lb['theta']: diffs.append(f"θ 轴不同({la['theta_name']} vs {lb['theta_name']}): 视角不同——T₃ 视角变换")
        return {
            'theories': [theory_a, theory_b],
            'coordinates': [la, lb],
            'contradiction': '表面矛盾' if diffs else '同一坐标(应直接可比)',
            'resolution': diffs if diffs else ['坐标相同, 若仍矛盾则需检查理论内部自洽性'],
            'verdict': '不矛盾——同一棵区分树在不同(d,s,θ)处的投影' if diffs else '需进一步分析'
        }

    def recommend_equations(self, phenomenon):
        """坐标 → 方程族推荐"""
        loc = self.locate(phenomenon)
        eq_map = {
            (1, '天体/宇宙'): '广义相对论场方程 / FLRW 度规 / ΛCDM',
            (2, '普朗克'): '量子场论 / 标准模型 / 路径积分',
            (2, '核/粒子'): '薛定谔方程 / 狄拉克方程 / QED',
            (3, '原子/分子'): '统计力学 / 玻尔兹曼方程 / 量子化学',
            (3, '宏观连续'): '热力学 / 流体力学 / 连续介质力学',
            (4, '原子/分子'): '分子动力学 / 系统生物学',
            (4, '个体/社会'): '进化动力学 / 种群方程',
            (5, '宏观连续'): '认知架构 / 连接主义模型 / 信息论',
            (6, '个体/社会'): '宏观经济模型 / 博弈论 / 系统动力学',
            (5, '个体/社会'): 'Agent 模型 / 强化学习 / 决策论',
        }
        key = (loc['d'], loc['s'])
        eq = eq_map.get(key, '需人工定位方程族')
        return {'phenomenon': phenomenon, 'coordinates': loc, 'equation_family': eq}

    def full_navigation(self, phenomenon):
        """完整导航: 坐标 + 方程族 + 变换建议"""
        # 优先: 查询中包含已知理论名 → 用理论定位
        for name, loc3 in THEORY_LOCATIONS.items():
            if name in phenomenon:
                th_name = next(t[1] for t in THETA_LEVELS if t[0] == loc3[2])
                loc = {'d': loc3[0], 'd_name': ['存在','时空/物质/能量','粒子/场/力','原子/分子/化学','细胞/生物','器官/个体/认知','社会/文明'][loc3[0]],
                       's': loc3[1], 'theta': loc3[2], 'theta_name': th_name,
                       'confidence': 3, 'via_theory': name}
                break
        else:
            loc = self.locate(phenomenon)
        eq_map = {
            (1, '天体/宇宙'): '广义相对论场方程 / FLRW 度规 / ΛCDM',
            (2, '普朗克'): '量子场论 / 标准模型 / 路径积分',
            (2, '核/粒子'): '薛定谔方程 / 狄拉克方程 / QED',
            (3, '原子/分子'): '统计力学 / 玻尔兹曼方程 / 量子化学',
            (3, '宏观连续'): '热力学 / 流体力学 / 连续介质力学',
            (4, '原子/分子'): '分子动力学 / 系统生物学',
            (4, '个体/社会'): '进化动力学 / 种群方程',
            (5, '宏观连续'): '认知架构 / 连接主义模型 / 信息论',
            (6, '个体/社会'): '宏观经济模型 / 博弈论 / 系统动力学',
            (5, '个体/社会'): 'Agent 模型 / 强化学习 / 决策论',
            (2, '宏观连续'): '量子-经典界面理论 / 对应原理 / 退相干理论',
        }
        key = (loc['d'], loc['s'])
        eq = eq_map.get(key, '需人工定位方程族')
        print(f"═══ δPT 导航: {phenomenon[:40]} ═══")
        print(f"  d = {loc['d']} ({loc['d_name']})  ← 区分深度")
        print(f"  s = {loc['s']}        ← 尺度")
        print(f"  θ = {loc['theta']}° ({loc['theta_name']})  ← 视角")
        if loc.get('via_theory'):
            print(f"  (经由已知理论定位: {loc['via_theory']})")
        print(f"  → 方程族: {eq}")
        near = []
        for name, (d, s, th) in THEORY_LOCATIONS.items():
            dist = abs(d-loc['d']) + (0 if s == loc['s'] else 1) + abs(th-loc['theta'])/45
            if dist <= 1.5 and name not in phenomenon:
                near.append((dist, name))
        near.sort()
        if near:
            print(f"  → 邻近理论(投影变换可达): {', '.join(n[1] for n in near[:4])}")
            print(f"  → 变换建议: T₁(d)/T₂(s)/T₃(θ) 到达邻近理论——矛盾消解为投影变换")
        return {'phenomenon': phenomenon, 'coordinates': loc, 'equation_family': eq}

if __name__ == '__main__':
    import sys
    nav = DPTNavigator()
    qs = sys.argv[1:] or ['量子力学和广义相对论为什么矛盾', '意识是什么', '经济衰退如何演化']
    for q in qs:
        nav.full_navigation(q)
        print()
    # 矛盾消解演示
    print('═══ 矛盾消解演示 ═══')
    r = nav.dissolve_contradiction('量子力学', '广义相对论')
    print(f"  {r['theories']}")
    for x in r['resolution']:
        print(f"  - {x}")
    print(f"  判定: {r['verdict']}")
