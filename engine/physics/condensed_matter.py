"""physics_engine/condensed_matter.py — 凝聚态物理 (2引擎: 能带/半导体/超导)"""
import math
from .core import PhysicsEngine, PhysicsState

K_B = 1.381e-23
H_BAR = 1.055e-34
M_E = 9.11e-31
Q_E = 1.602e-19


class BandTheoryEngine(PhysicsEngine):
    """能带理论: 导体/半导体/绝缘体 / PN结"""
    def __init__(self):
        super().__init__(name="band_theory")
        # 带隙 (eV)
        self.band_gaps = {
            'Si': 1.12, 'Ge': 0.67, 'GaAs': 1.42, 'GaN': 3.4,
            'SiC': 3.3, 'Diamond': 5.5, 'SiO2': 9.0,
            'Cu': 0, 'Ag': 0, 'Al': 0,  # 导体
        }

    def classify(self, Eg):
        """按带隙分类"""
        if Eg < 0.1:
            return '导体(能带重叠)'
        elif Eg < 3.0:
            return '半导体'
        elif Eg < 5.0:
            return '宽带隙半导体'
        else:
            return '绝缘体'

    def intrinsic_carrier(self, Eg, T, m_eff=1.0):
        """本征载流子浓度: n_i = 2(2πkT/h²)^(3/2) · (m_e*m_h)^(3/4) · exp(-Eg/2kT)"""
        factor = 2 * (2 * math.pi * K_B * T / H_BAR**2)**1.5
        eff = (m_eff * M_E)**0.75
        n_i = factor * eff * math.exp(-Eg * Q_E / (2 * K_B * T))
        return {'n_i': f'{n_i:.2e}', 'Eg_eV': Eg, 'T_K': T}

    def pn_junction(self, Na, Nd, T=300, material='Si'):
        """PN结: 内建电势 + 耗尽层宽度"""
        Eg = self.band_gaps.get(material, 1.12)
        # 本征载流子浓度 (Si@300K ~ 1.5e16 m^-3)
        n_i = 1.5e16 if material == 'Si' else 2.4e19 if material == 'Ge' else 1e10
        # 内建电势
        V_bi = K_B * T / Q_E * math.log(Na * Nd / n_i**2)
        # 耗尽层宽度
        eps = 11.7 * 8.854e-12  # Si介电常数
        W = math.sqrt(2 * eps * V_bi / Q_E * (1/Na + 1/Nd))
        # 耗尽层两侧宽度
        x_n = W * Na / (Na + Nd)
        x_p = W * Nd / (Na + Nd)
        return {'material': material,
                'Na': f'{Na:.2e}', 'Nd': f'{Nd:.2e}',
                'V_bi_V': round(V_bi, 4),
                'W_um': round(W * 1e6, 3),
                'x_n_um': round(x_n * 1e6, 3),
                'x_p_um': round(x_p * 1e6, 3),
                'note': 'PN结内建电势 + 耗尽层'}

    def solar_cell(self, I_sc, V_oc, FF=0.75):
        """太阳能电池: 效率 = FF * I_sc * V_oc / P_in"""
        P_in = 1000  # 标准入射功率 W/m²
        P_max = I_sc * V_oc * FF
        eta = P_max / P_in * 100
        return {'I_sc_A': I_sc, 'V_oc_V': V_oc, 'FF': FF,
                'P_max_W': round(P_max, 2),
                'efficiency_pct': round(eta, 2),
                'note': '标准测试条件(1000W/m², 25°C)'}

    def step(self, state, dt):
        return state


class SuperconductivityEngine(PhysicsEngine):
    """超导: BCS理论/临界磁场/迈斯纳效应"""
    def __init__(self):
        super().__init__(name="superconductivity")
        # 超导体参数
        self.materials = {
            'Al': {'Tc': 1.2, 'Hc0': 0.01},
            'Nb': {'Tc': 9.3, 'Hc0': 0.20},
            'NbTi': {'Tc': 10.0, 'Hc0': 15.0},
            'Nb3Sn': {'Tc': 18.3, 'Hc0': 30.0},
            'YBCO': {'Tc': 93.0, 'Hc0': 120.0},
            'MgB2': {'Tc': 39.0, 'Hc0': 16.0},
            'HgBaCuO': {'Tc': 134.0, 'Hc0': 200.0},
        }

    def critical_field(self, T, Tc, Hc0):
        """临界磁场: Hc(T) = Hc0 * [1 - (T/Tc)²]"""
        if T >= Tc:
            return 0
        return Hc0 * (1 - (T / Tc)**2)

    def energy_gap(self, T, Tc):
        """BCS能隙: Δ(T) ≈ Δ0 * tanh(1.74 * sqrt(Tc/T - 1))"""
        if T >= Tc:
            return 0
        delta0 = 1.764 * K_B * Tc  # T=0时能隙
        delta = delta0 * math.tanh(1.74 * math.sqrt(Tc / T - 1))
        return {'delta_J': f'{delta:.4e}',
                'delta_meV': round(delta / Q_E * 1000, 3),
                'delta0_meV': round(delta0 / Q_E * 1000, 3)}

    def london_penetration(self, T, Tc, lambda_0=50e-9):
        """伦敦穿透深度: λ(T) = λ0 / sqrt(1 - (T/Tc)⁴)"""
        if T >= Tc:
            return float('inf')
        return lambda_0 / math.sqrt(1 - (T / Tc)**4)

    def critical_current(self, T, Tc, Ic0):
        """临界电流: Ic(T) = Ic0 * [1 - (T/Tc)²]^(3/2)"""
        if T >= Tc:
            return 0
        return Ic0 * (1 - (T / Tc)**2)**1.5

    def step(self, state, dt):
        return state
