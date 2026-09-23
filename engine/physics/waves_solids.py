"""physics_engine/waves_solids.py — 第三层: 波动与固体 (6引擎)
StressStrain / BeamBending / SoundWave / Doppler / GeometricOptics / WaveOptics
"""
import math
import numpy as np
from .core import PhysicsEngine, PhysicsState


class StressStrainEngine(PhysicsEngine):
    """材料应力-应变: σ = E·ε (胡克定律)"""
    def __init__(self, E=200e9, yield_strength=250e6):
        """E: 弹性模量Pa, yield: 屈服强度Pa"""
        super().__init__(name="stress_strain")
        self.E, self.yield_strength = E, yield_strength

    def stress(self, strain):
        """应变→应力"""
        return self.E * strain

    def strain(self, stress):
        return stress / self.E

    def elongation(self, L0, strain):
        return L0 * strain

    def safety_factor(self, applied_stress):
        return self.yield_strength / applied_stress

    def edge_condition(self, strain):
        """判断是否超过弹性极限"""
        s = self.stress(strain)
        return {'stress': s, 'elastic': s < self.yield_strength,
                'plastic': s >= self.yield_strength,
                'safety': self.yield_strength / s if s > 0 else None}


class BeamBendingEngine(PhysicsEngine):
    """梁弯曲: δ = PL³/(48EI) 悬臂 / 简支"""
    def __init__(self, E=200e9, I=1e-6, L=3.0):
        super().__init__(name="beam_bending")
        self.E, self.I, self.L = E, I, L   # I: 截面惯性矩 m⁴

    def simply_supported_deflection(self, P, L=None, E=None, I=None):
        """简支梁中点挠度 δ = PL³/(48EI)"""
        L = L or self.L; E = E or self.E; I = I or self.I
        return P * L**3 / (48 * E * I)

    def cantilever_deflection(self, P, L=None, E=None, I=None):
        """悬臂梁端部挠度 δ = PL³/(3EI)"""
        L = L or self.L; E = E or self.E; I = I or self.I
        return P * L**3 / (3 * E * I)

    def max_bending_stress(self, P, L=None, c=0.05, I=None):
        """最大弯曲应力 σ = M·c/I, M=PL/4"""
        L = L or self.L; I = I or self.I
        M = P * L / 4
        return M * c / I


class SoundWaveEngine(PhysicsEngine):
    """声波: p = p₀·sin(kx-ωt), 分贝→压强"""
    def __init__(self, rho=1.225, c_sound=343.0):
        super().__init__(name="sound_wave")
        self.rho, self.c = rho, c_sound

    def pressure_amplitude_from_dB(self, dB, p_ref=2e-5):
        """分贝→声压幅值 p = p_ref·10^(dB/20)"""
        return p_ref * 10**(dB / 20)

    def intensity_from_dB(self, dB, I0=1e-12):
        """分贝→声强 W/m²"""
        return I0 * 10**(dB / 10)

    def wavelength(self, f):
        return self.c / f

    def doppler_frequency(self, f0, vs, vo=0, toward=True):
        """多普勒: 声源运动. toward=True 接近 (f升高); False 远离"""
        return f0 * (self.c + vo) / (self.c - vs) if toward else f0 * (self.c - vo) / (self.c + vs)


class DopplerEngine(PhysicsEngine):
    """多普勒效应: f' = f₀·c/(c∓vs)"""
    def __init__(self, c=343.0):
        super().__init__(name="doppler")
        self.c = c

    def observed_freq(self, f0, vs, vo=0, source_receding=False):
        """"声源运动. 接近: f'=f0·c/(c-vs); 远离: f'=f0·c/(c+vs)"""
        if source_receding:
            return f0 * self.c / (self.c + vs)
        return f0 * self.c / (self.c - vs)

    def relative_velocity(self, f0, f1, c=343.0):
        """由频移反推相对速度(近似)"""
        return c * abs(f1 - f0) / f0


class GeometricOpticsEngine(PhysicsEngine):
    """几何光学: Snell n₁sinθ₁ = n₂sinθ₂"""
    def __init__(self):
        super().__init__(name="geometric_optics")

    def snell_angle(self, n1, n2, theta1):
        """折射角(度)"""
        s2 = n1 * math.sin(math.radians(theta1)) / n2
        if abs(s2) > 1: return None  # 全反射
        return math.degrees(math.asin(s2))

    def critical_angle(self, n1, n2):
        """全反射临界角(光密→光疏)"""
        if n1 <= n2: return None
        return math.degrees(math.asin(n2 / n1))

    def thin_lens(self, f, u):
        """薄透镜成像 1/f=1/u+1/v"""
        v = 1.0 / (1.0/f - 1.0/u)
        return v


class WaveOpticsEngine(PhysicsEngine):
    """波动光学: I = I₀(sinβ/β)² 单缝衍射"""
    def __init__(self):
        super().__init__(name="wave_optics")

    def single_slit_intensity(self, I0, a, lam, theta):
        """单缝衍射强度分布"""
        if abs(theta) < 1e-9: return I0
        beta = math.pi * a * math.sin(math.radians(theta)) / lam
        return I0 * (math.sin(beta) / beta)**2

    def diffraction_minima(self, a, lam, m=1):
        """暗纹条件 a·sinθ = mλ → θ"""
        return math.degrees(math.asin(m * lam / a))
