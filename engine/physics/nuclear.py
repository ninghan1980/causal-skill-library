"""physics_engine/nuclear.py — 核物理 (2引擎: 衰变链/裂变聚变)"""
import math
from .core import PhysicsEngine, PhysicsState

# 物理常量
K_B = 1.381e-23      # 玻尔兹曼常数
H_BAR = 1.055e-34    # 约化普朗克常数
M_E = 9.11e-31       # 电子质量
M_P = 1.673e-27      # 质子质量
M_N = 1.675e-27      # 中子质量
C = 2.998e8          # 光速
MEV_TO_J = 1.602e-13 # MeV → J


class RadioactiveDecayEngine(PhysicsEngine):
    """放射性衰变: 半衰期/衰变链/年龄测定"""
    def __init__(self):
        super().__init__(name="radioactive_decay")
        # 常见核素半衰期 (秒)
        self.half_lives = {
            'U-238': 4.468e9 * 365.25 * 24 * 3600,
            'U-235': 7.04e8 * 365.25 * 24 * 3600,
            'Th-232': 1.405e10 * 365.25 * 24 * 3600,
            'K-40': 1.277e9 * 365.25 * 24 * 3600,
            'C-14': 5730 * 365.25 * 24 * 3600,
            'Cs-137': 30.17 * 365.25 * 24 * 3600,
            'Co-60': 5.27 * 365.25 * 24 * 3600,
            'I-131': 8.02 * 24 * 3600,
            'Pu-239': 24110 * 365.25 * 24 * 3600,
            'Ra-226': 1600 * 365.25 * 24 * 3600,
        }

    def decay_constant(self, half_life):
        """衰变常数 λ = ln(2) / T_1/2"""
        return math.log(2) / half_life

    def remaining_fraction(self, t, half_life):
        """剩余比例 N/N0 = exp(-λt) = (1/2)^(t/T)"""
        return 0.5 ** (t / half_life)

    def decay_chain(self, N0, half_life, times):
        """计算多个时间点的剩余量"""
        results = []
        for t in times:
            N = N0 * self.remaining_fraction(t, half_life)
            results.append({'time': t, 'remaining': round(N, 4), 'decayed': round(N0 - N, 4)})
        return results

    def carbon_dating(self, remaining_ratio):
        """碳14测年: t = -8033 * ln(N/N0)"""
        if remaining_ratio <= 0 or remaining_ratio > 1:
            return None
        age = -8033 * math.log(remaining_ratio)
        return {'remaining_ratio': remaining_ratio,
                'age_years': round(age, 0),
                'age_bp': f'{round(age, 0)}年前(1950年基准)'}

    def activity(self, N, half_life):
        """活度 A = λN (Bq)"""
        lam = self.decay_constant(half_life)
        return {'activity_Bq': round(lam * N, 4),
                'activity_Ci': round(lam * N / 3.7e10, 6)}

    def step(self, state, dt):
        return state


class NuclearFissionEngine(PhysicsEngine):
    """核裂变/聚变: 结合能/链式反应/反应堆"""
    def __init__(self):
        super().__init__(name="fission_fusion")
        # 比结合能 (MeV/nucleon)
        self.binding_energy = {
            'H-2': 1.11, 'He-4': 7.07, 'Li-6': 5.33,
            'C-12': 7.68, 'O-16': 7.98, 'Fe-56': 8.79,
            'U-235': 7.59, 'Pu-239': 7.56,
        }
        # 每次裂变释放能量 (MeV)
        energy_per_fission = 200  # U-235 约 200 MeV

    def q_value(self, mass_reactants, mass_products):
        """Q值 = (反应物质量 - 生成物质量)c² (MeV)"""
        dm = mass_reactants - mass_products  # 质量亏损 (u)
        return dm * 931.5  # 1 u = 931.5 MeV/c²

    def fission_energy(self, n_fissions):
        """n次裂变释放能量"""
        E_mev = n_fissions * 200
        E_joule = n_fissions * 200 * MEV_TO_J
        return {'n_fissions': n_fissions,
                'energy_MeV': E_mev,
                'energy_J': f'{E_joule:.4e}',
                'equivalent_tnt_ton': round(E_joule / 4.184e9, 2)}

    def critical_mass(self, density, nu=2.5, sigma_f=1.2e-28, sigma_a=1.5e-28):
        """临界质量估算 (简化球几何)"""
        # 临界半径: R_c ≈ π / (Σ_f * (ν-1))^0.5
        N = density / 235 * 6.022e23  # 原子数密度
        Sigma_f = N * sigma_f
        Sigma_a = N * sigma_a
        if Sigma_f * (nu - 1) <= Sigma_a:
            return {'critical': False, 'note': '无法达到临界'}
        R_c = math.pi / math.sqrt(Sigma_f * (nu - 1) - Sigma_a)
        V_c = 4/3 * math.pi * R_c**3
        M_c = V_c * density / 1000  # kg
        return {'critical_radius_m': round(R_c, 4),
                'critical_mass_kg': round(M_c, 2),
                'critical': True,
                'note': '简化估算, 实际需中子输运计算'}

    def fusion_energy(self, reaction='D-T'):
        """聚变能量"""
        reactions = {
            'D-T': {'energy_MeV': 17.6, 'products': 'He-4 + n'},
            'D-D': {'energy_MeV': 3.65, 'products': 'He-3 + n / T + p'},
            'D-He3': {'energy_MeV': 18.3, 'products': 'He-4 + p'},
        }
        r = reactions.get(reaction, {})
        E_joule = r.get('energy_MeV', 0) * MEV_TO_J
        return {'reaction': reaction,
                'energy_MeV': r.get('energy_MeV'),
                'energy_per_reaction_J': f'{E_joule:.4e}',
                'products': r.get('products'),
                'note': '聚变需10keV以上温度(约1亿度)'}

    def step(self, state, dt):
        return state
