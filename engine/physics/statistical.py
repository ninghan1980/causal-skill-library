"""physics_engine/statistical.py — 统计力学 (2引擎: 熵/热机 + 相变临界)"""
import math
from .core import PhysicsEngine, PhysicsState


class EntropyEngine(PhysicsEngine):
    """熵与热力学第二定律: 克劳修斯/玻尔兹曼/卡诺/热机效率"""
    def __init__(self, k_B=1.380649e-23, R=8.314):
        super().__init__(name="entropy")
        self.k_B = k_B
        self.R = R

    def clausius_entropy(self, dQ, T):
        """克劳修斯熵变: dS = dQ_rev/T (J/K)"""
        return dQ / T

    def boltzmann_entropy(self, W):
        """玻尔兹曼熵: S = k_B·ln(W)"""
        return self.k_B * math.log(W)

    def mixing_entropy(self, n1, n2, x1, x2):
        """混合熵: ΔS_mix = -R(n1·ln x1 + n2·ln x2)"""
        if x1 <= 0 or x2 <= 0: return 0
        return -self.R * (n1 * math.log(x1) + n2 * math.log(x2))

    def carnot_efficiency(self, T_hot, T_cold):
        """卡诺效率: η = 1 - T_cold/T_hot (必须用开尔文)"""
        if T_hot <= 0: return None
        eta = 1 - T_cold / T_hot
        return {'T_hot_K': T_hot, 'T_cold_K': T_cold,
                'efficiency': round(eta, 4),
                'efficiency_pct': round(eta*100, 2),
                'note': '卡诺热机最高效率(可逆循环)'}

    def otto_efficiency(self, compression_ratio, gamma=1.4):
        """奥托循环(汽油机): η = 1 - 1/r^(γ-1)"""
        eta = 1 - 1 / compression_ratio**(gamma - 1)
        return {'compression_ratio': compression_ratio, 'gamma': gamma,
                'efficiency': round(eta, 4), 'efficiency_pct': round(eta*100, 2),
                'note': '汽油机理想循环'}

    def diesel_efficiency(self, compression_ratio, cutoff_ratio, gamma=1.4):
        """狄塞尔循环(柴油机): η = 1 - (1/r^(γ-1))·[(α^γ-1)/(γ(α-1))]"""
        r, alpha = compression_ratio, cutoff_ratio
        eta = 1 - (1/r**(gamma-1)) * (alpha**gamma - 1) / (gamma*(alpha-1))
        return {'compression_ratio': r, 'cutoff_ratio': alpha,
                'efficiency': round(eta, 4), 'efficiency_pct': round(eta*100, 2),
                'note': '柴油机理想循环'}

    def entropy_change_ideal_gas(self, n, T1, T2, V1, V2, Cv=12.47):
        """理想气体熵变: ΔS = n·Cv·ln(T2/T1) + n·R·ln(V2/V1)"""
        dS = n * Cv * math.log(T2/T1) + n * self.R * math.log(V2/V1)
        return {'delta_S_J_K': round(dS, 4), 'n_mol': n,
                'process': f'{T1}K→{T2}K, {V1}L→{V2}L',
                'spontaneous': dS > 0}

    def step(self, state, dt):
        return state


class PhaseTransitionEngine(PhysicsEngine):
    """相变与临界现象: 克拉珀龙方程 + 相平衡"""
    def __init__(self):
        super().__init__(name="phase_critical")

    def clausius_clapeyron(self, T, L, dV, T0=373.15, P0=101325):
        """克拉珀龙方程: dP/dT = L/(T·dV) → 积分"""
        # 简化: 固-液或液-气相变
        dPdT = L / (T * dV)
        P = P0 + dPdT * (T - T0)
        return {'T_K': T, 'dP_dT': round(dPdT, 4),
                'P_Pa': round(P, 1), 'L_J_kg': L}

    def water_boiling_vs_altitude(self, altitude_m):
        """海拔 vs 沸点: 气压降低 → 沸点降低"""
        # 简化气压模型
        P = 101325 * (1 - 2.25577e-5 * altitude_m)**5.25588
        # 安托万方程求沸点
        if P > 0:
            T_boil = 373.15 / (1 - 8.314/4.07e6 * math.log(P/101325))
        else:
            T_boil = None
        return {'altitude_m': altitude_m,
                'pressure_kPa': round(P/1000, 2),
                'boiling_point_K': round(T_boil, 2) if T_boil else None,
                'boiling_point_C': round(T_boil-273.15, 2) if T_boil else None,
                'note': '高原煮不熟鸡蛋就因为这'}

    def step(self, state, dt):
        return state
