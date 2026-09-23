"""physics_engine/extremes.py — 第四层: 极端物理 (6引擎)
NBodyMD / Plasma / Schwarzschild / Lensing / Precession / BlackHole
"""
import math
import numpy as np
from .core import PhysicsEngine, PhysicsState

G = 6.674e-11   # 万有引力常数
C_LIGHT = 3e8   # 光速 m/s
M_SUN = 1.989e30 # 太阳质量 kg
EPS0 = 8.854e-12 # 真空介电常数
K_B = 1.381e-23 # 玻尔兹曼常数
E_CHARGE = 1.602e-19 # 元电荷


class NBodyMDFngine(PhysicsEngine):
    """分子动力学: Lennard-Jones + Verlet 积分"""
    def __init__(self, sigma=3.4e-10, epsilon=1.65e-21, m=6.63e-26):
        super().__init__(name="nbody_md")
        self.sigma, self.epsilon, self.m = sigma, epsilon, m

    def lennard_jones(self, r):
        """LJ势能 U = 4ε[(σ/r)¹² - (σ/r)⁶]"""
        return 4*self.epsilon*((self.sigma/r)**12 - (self.sigma/r)**6)

    def lj_force(self, r):
        """LJ力 F = -dU/dr"""
        return 4*self.epsilon/r*(12*(self.sigma/r)**12 - 6*(self.sigma/r)**6)

    def verlet_step(self, pos, vel, force, dt):
        """单步 Verlet 积分"""
        accel = force / self.m
        new_pos = pos + vel*dt + 0.5*accel*dt**2
        return new_pos


class PlasmaEngine(PhysicsEngine):
    """等离子体: 德拜长度 λ_D = √(ε₀kT/ne²)"""
    def __init__(self):
        super().__init__(name="plasma")

    def debye_length(self, T, n):
        """德拜长度 m. T:K, n:m⁻³"""
        return math.sqrt(EPS0 * K_B * T / (n * E_CHARGE**2))

    def plasma_frequency(self, n):
        """等离子体频率 ω_p = √(ne²/(ε₀m_e))"""
        return math.sqrt(n * E_CHARGE**2 / (EPS0 * 9.11e-31))


class SchwarzschildEngine(PhysicsEngine):
    """史瓦西黑洞: ds² = -(1-rs/r)c²dt² + ..."""
    def __init__(self):
        super().__init__(name="schwarzschild")

    def schwarzschild_radius(self, M):
        """史瓦西半径 rs = 2GM/c²"""
        return 2*G*M / C_LIGHT**2

    def event_horizon(self, M):  # 别名
        return self.schwarzschild_radius(M)

    def time_dilation_factor(self, r, M):
        """远处观测者时距因子 √(1-rs/r)"""
        rs = self.schwarzschild_radius(M)
        if r <= rs: return 0.0
        return math.sqrt(1 - rs/r)


class LensingEngine(PhysicsEngine):
    """引力透镜: α = 4GM/(c²·b) 偏折角"""
    def __init__(self):
        super().__init__(name="lensing")

    def deflection_angle(self, M, b):
        """偏折角 rad. b: 撞击参数m"""
        return 4*G*M / (C_LIGHT**2 * b)


class PrecessionEngine(PhysicsEngine):
    """轨道进动: Δφ = 6πGM/(c²·a(1-e²)) 每圈"""
    def __init__(self):
        super().__init__(name="precession")

    def perihelion_precession(self, M, a, e):
        """水星进动 rad/rev"""
        return 6*math.pi*G*M / (C_LIGHT**2 * a * (1-e**2))


class BlackHoleEngine(PhysicsEngine):
    """黑洞辐射: T = ℏc³/(8πGMk_B) 霍金温度"""
    def __init__(self):
        super().__init__(name="black_hole")

    def hawking_temperature(self, M):
        """霍金温度 K"""
        return 1.055e-34 * C_LIGHT**3 / (8*math.pi*G*M*K_B)

    def blackbody_peak(self, T):
        """维恩位移 λpeak = b/T"""
        b = 2.898e-3
        return b / T
