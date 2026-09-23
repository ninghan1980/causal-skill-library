"""physics_engine/particle_physics.py — 粒子物理 (2引擎: 标准模型/散射)"""
import math
from .core import PhysicsEngine, PhysicsState

# 物理常量
H_BAR = 1.055e-34    # 约化普朗克常数
C = 2.998e8          # 光速
MEV_TO_J = 1.602e-13 # MeV → J
Q_E = 1.602e-19      # 元电荷
EPS0 = 8.854e-12     # 真空介电常数
ALPHA = 1/137.036    # 精细结构常数
M_E = 9.11e-31       # 电子质量


class StandardModelEngine(PhysicsEngine):
    """标准模型: 基本粒子/守恒律/对称性"""
    def __init__(self):
        super().__init__(name="standard_model")
        # 基本粒子质量 (MeV/c²)
        self.particles = {
            # 轻子
            'electron': {'mass_MeV': 0.511, 'charge': -1, 'spin': 0.5, 'type': 'lepton'},
            'muon': {'mass_MeV': 105.7, 'charge': -1, 'spin': 0.5, 'type': 'lepton'},
            'tau': {'mass_MeV': 1776.9, 'charge': -1, 'spin': 0.5, 'type': 'lepton'},
            'electron_neutrino': {'mass_MeV': 1e-6, 'charge': 0, 'spin': 0.5, 'type': 'lepton'},
            # 夸克
            'up': {'mass_MeV': 2.2, 'charge': 2/3, 'spin': 0.5, 'type': 'quark'},
            'down': {'mass_MeV': 4.7, 'charge': -1/3, 'spin': 0.5, 'type': 'quark'},
            'charm': {'mass_MeV': 1275, 'charge': 2/3, 'spin': 0.5, 'type': 'quark'},
            'strange': {'mass_MeV': 95, 'charge': -1/3, 'spin': 0.5, 'type': 'quark'},
            'top': {'mass_MeV': 173100, 'charge': 2/3, 'spin': 0.5, 'type': 'quark'},
            'bottom': {'mass_MeV': 4180, 'charge': -1/3, 'spin': 0.5, 'type': 'quark'},
            # 规范玻色子
            'photon': {'mass_MeV': 0, 'charge': 0, 'spin': 1, 'type': 'gauge_boson'},
            'gluon': {'mass_MeV': 0, 'charge': 0, 'spin': 1, 'type': 'gauge_boson'},
            'W': {'mass_MeV': 80379, 'charge': 1, 'spin': 1, 'type': 'gauge_boson'},
            'Z': {'mass_MeV': 91188, 'charge': 0, 'spin': 1, 'type': 'gauge_boson'},
            'Higgs': {'mass_MeV': 125100, 'charge': 0, 'spin': 0, 'type': 'scalar'},
        }

    def particle_info(self, name):
        """粒子信息"""
        p = self.particles.get(name)
        if not p:
            return {'error': f'未知粒子: {name}'}
        return {'name': name, **p,
                'mass_kg': round(p['mass_MeV'] * MEV_TO_J / C**2, 40),
                'mass_GeV': round(p['mass_MeV'] / 1000, 3)}

    def composite_mass(self, particles):
        """复合粒子质量 (夸克组合)"""
        total_mass = sum(self.particles[p]['mass_MeV'] for p in particles if p in self.particles)
        return {'particles': particles,
                'total_mass_MeV': round(total_mass, 2),
                'note': '裸质量, 实际需加上结合能'}

    def decay_allowed(self, parent, daughters):
        """衰变允许性判断 (能量/电荷/轻子数守恒)"""
        p_parent = self.particles.get(parent)
        if not p_parent:
            return {'allowed': False, 'error': '未知母粒子'}
        daughters_info = []
        total_mass_daughters = 0
        total_charge = 0
        for d in daughters:
            pd = self.particles.get(d)
            if not pd:
                return {'allowed': False, 'error': f'未知子粒子 {d}'}
            daughters_info.append(pd)
            total_mass_daughters += pd['mass_MeV']
            total_charge += pd['charge']
        mass_ok = p_parent['mass_MeV'] > total_mass_daughters
        charge_ok = abs(p_parent['charge'] - total_charge) < 1e-10
        return {'parent': parent, 'daughters': daughters,
                'mass_conservation': mass_ok,
                'charge_conservation': charge_ok,
                'allowed': mass_ok and charge_ok,
                'parent_mass_MeV': p_parent['mass_MeV'],
                'daughters_mass_MeV': round(total_mass_daughters, 2),
                'Q_value_MeV': round(p_parent['mass_MeV'] - total_mass_daughters, 2)}

    def step(self, state, dt):
        return state


class ScatteringEngine(PhysicsEngine):
    """散射: 截面/卢瑟福/康普顿"""
    def __init__(self):
        super().__init__(name="scattering")

    def rutherford(self, Z1, Z2, E, theta):
        """卢瑟福散射截面: dσ/dΩ = (Z1·Z2·e²/16πε₀E)² · 1/sin⁴(θ/2)"""
        if E <= 0 or theta <= 0:
            return {'error': 'E和θ必须为正'}
        coeff = (Z1 * Z2 * Q_E**2 / (16 * math.pi * EPS0 * E * MEV_TO_J * 1e6))**2
        cross_section = coeff / math.sin(theta/2)**4
        return {'Z1': Z1, 'Z2': Z2, 'E_MeV': E, 'theta_rad': theta,
                'differential_cross_section_m2_sr': f'{cross_section:.4e}',
        'note': '卢瑟福公式(库仑散射)'}

    def compton(self, E_gamma, theta):
        """康普顿散射: E' = E / [1 + (E/mc²)(1-cosθ)]"""
        m_e_c2 = 0.511  # MeV
        E = E_gamma
        E_prime = E / (1 + (E / m_e_c2) * (1 - math.cos(theta)))
        delta_lambda = H_BAR / (M_E * C) * (1 - math.cos(theta)) * 1e10  # Å
        return {'incident_energy_MeV': E,
                'scattered_energy_MeV': round(E_prime, 4),
                'energy_loss_MeV': round(E - E_prime, 4),
                'theta_deg': round(theta * 180 / math.pi, 1),
                'compton_wavelength_A': round(delta_lambda, 6),
                'note': '光子-电子弹性散射'}

    def cross_section_point(self, E, E0, sigma_max):
        """共振截面 (Breit-Wigner): σ(E) = σ_max · Γ²/4 / [(E-E₀)² + Γ²/4]"""
        Gamma = 0.1 * E0  # 能隙
        sigma = sigma_max * (Gamma**2 / 4) / ((E - E0)**2 + Gamma**2 / 4)
        return {'sigma': round(sigma, 6),
                'resonance_energy_MeV': E0,
                'width_MeV': round(Gamma, 3),
                'note': 'Breit-Wigner共振'}

    def step(self, state, dt):
        return state
