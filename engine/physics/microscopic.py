"""physics_engine/microscopic.py — 第二层: 微观物理 (6引擎)
ElectricField / MagneticField / Schrodinger / Tunneling / Arrhenius / Equilibrium
"""
import math
import numpy as np
from .core import PhysicsEngine, PhysicsState

K = 8.99e9        # 库仑常数 N·m²/C²
MU0 = 4e-7*math.pi # 真空磁导率
H_BAR = 1.055e-34 # 约化普朗克常数
M_E = 9.11e-31    # 电子质量
KB = 1.381e-23    # 玻尔兹曼常数


class ElectricFieldEngine(PhysicsEngine):
    """电场: E = kq/r² (库仑)"""
    def __init__(self, q=1e-6):
        super().__init__(name="electric_field")
        self.q = q

    def field_strength(self, r, q=None):
        q = self.q if q is None else q
        return K * abs(q) / r**2

    def potential(self, r, q=None):
        q = self.q if q is None else q
        return K * q / r

    def force_between(self, q1, q2, r):
        return K * q1 * q2 / r**2


class MagneticFieldEngine(PhysicsEngine):
    """磁场: dB = mu0·I·dl×r̂/(4πr²) (Biot-Savart, 直导线简化 B=mu0·I/2πr)"""
    def __init__(self, I=1.0):
        super().__init__(name="magnetic_field")
        self.I = I

    def straight_wire_field(self, r, I=None):
        I = self.I if I is None else I
        return MU0 * I / (2*math.pi*r)

    def lorentz_force(self, q, v, B):
        """洛伦兹力 F=qvB (垂直)"""
        return q * v * B


class SchrodingerEngine(PhysicsEngine):
    """薛定谔: iℏdψ/dt = -ℏ²/2m·d²ψ/dx² + Vψ (有限方阱能级)"""
    def __init__(self, m=M_E, a=1e-9):
        super().__init__(name="schrodinger")
        self.m, self.a = m, a

    def particle_in_box(self, n):
        """无限方阱能级 E_n = n²·π²·ℏ²/(2ma²)"""
        return n**2 * math.pi**2 * H_BAR**2 / (2 * self.m * self.a**2)

    def energy_levels(self, n_max=5):
        return {n: round(self.particle_in_box(n) / 1.6e-19, 3) for n in range(1, n_max+1)}  # eV


class TunnelingEngine(PhysicsEngine):
    """隧穿: T = f(E, V0, a) 矩形势垒透射系数"""
    def __init__(self, m=M_E):
        super().__init__(name="tunneling")
        self.m = m

    def transmission(self, E, V0, a):
        """E,V0单位eV, a单位m → 透射系数(近似, E<V0)"""
        if E >= V0:
            return 1.0
        kappa = math.sqrt(2 * self.m * (V0-E) * 1.6e-19) / H_BAR
        return math.exp(-2 * kappa * a)


class ArrheniusEngine(PhysicsEngine):
    """Arrhenius: k = A·exp(-Ea/RT) 反应速率"""
    def __init__(self, A=1e10, Ea=50000.0):
        super().__init__(name="arrhenius")
        self.A, self.Ea = A, Ea

    def rate(self, T):
        return self.A * math.exp(-self.Ea / (8.314 * T))

    def activation_energy(self, T1, k1, T2, k2):
        """已知两温速率推导活化能 Ea = -R·ln(k2/k1)/(1/T2 - 1/T1)"""
        return -8.314 * math.log(k2/k1) / (1/T2 - 1/T1)


class EquilibriumEngine(PhysicsEngine):
    """化学平衡: ΔG° = -RT·ln K"""
    def __init__(self):
        super().__init__(name="equilibrium")

    def deltaG(self, T, K):
        return -8.314 * T * math.log(K)

    def equilibrium_constant(self, T, deltaG):
        return math.exp(-deltaG / (8.314 * T))

    def van_t_hoff(self, T1, K1, T2, deltaH):
        """Van't Hoff: 温度对平衡常数影响 ln(K2/K1) = -ΔH/R·(1/T2-1/T1)"""
        K2 = K1 * math.exp(-deltaH / 8.314 * (1/T2 - 1/T1))
        return K2


class FaradayInductionEngine(PhysicsEngine):
    """法拉第电磁感应: ε = -dΦ/dt (磁通变化→感应电动势)"""
    def __init__(self):
        super().__init__(name="faraday_induction")

    def emf(self, dPhi, dt):
        """感应电动势 ε = -dΦ/dt"""
        return -dPhi / dt

    def coil_emf(self, N, A, dB_dt, theta=0):
        """N匝线圈: ε = -N·A·(dB/dt)·cosθ"""
        return -N * A * dB_dt * math.cos(theta)

    def motional_emf(self, B, L, v):
        """动生电动势: ε = B·L·v (导体切割磁感线)"""
        return B * L * v

    def self_inductance(self, mu, N, A, l):
        """自感: L = μ·N²·A/l"""
        return mu * N**2 * A / l

    def inductor_energy(self, L, I):
        """电感储能: E = ½LI²"""
        return 0.5 * L * I**2

    def step(self, state, dt):
        return state


class MaxwellStressEngine(PhysicsEngine):
    """麦克斯韦应力张量: 电磁场对物体的力"""
    def __init__(self):
        super().__init__(name="maxwell_stress")
        self.eps0 = 8.854e-12

    def electric_pressure(self, E):
        """电场压力: P = ½ε₀E²"""
        return 0.5 * self.eps0 * E**2

    def magnetic_pressure(self, B):
        """磁场压力: P = B²/(2μ₀)"""
        return B**2 / (2 * MU0)

    def radiation_pressure(self, I, c=3e8):
        """光压: P = I/c (完全吸收)"""
        return I / c

    def step(self, state, dt):
        return state


class QuantumSpinEngine(PhysicsEngine):
    """量子自旋: S=1/2,1,3/2... 泡利矩阵 + 斯特恩-盖拉赫"""
    def __init__(self):
        super().__init__(name="spin")
        self.h_bar = H_BAR
        self.g_e = 2.0023  # 电子g因子
        self.mu_B = 9.274e-24  # 玻尔磁子 J/T

    def spin_magnitude(self, s):
        """自旋大小: |S| = √(s(s+1))·ℏ"""
        S = math.sqrt(s*(s+1)) * self.h_bar
        Sz_max = s * self.h_bar
        return {'s': s, 'magnitude': f'{S:.4e}',
                'Sz_max': f'{Sz_max:.4e}', 'm_s_values': [f'{m}' for m in [s, s-1, -s]]}

    def magnetic_moment(self, s, l=0, j=None):
        """磁矩: μ = -g_J·μ_B·J/ℏ (Landé g因子)"""
        if j is None:
            j = s + l
        # Landé g因子
        if s == 0:
            g_J = 1  # 纯轨道
        elif l == 0:
            g_J = self.g_e  # 纯自旋
        else:
            g_J = 1 + (j*(j+1) + s*(s+1) - l*(l+1)) / (2*j*(j+1))
        mu = g_J * self.mu_B * j
        return {'s': s, 'l': l, 'j': j, 'g_J': round(g_J, 4),
                'mu_J_T': f'{mu:.4e}', 'mu_mu_B': round(mu/self.mu_B, 3)}

    def zeeman_energy(self, m_j, B, g_J=2):
        """塞曼能级分裂: E = g_J·μ_B·m_j·B"""
        E = g_J * self.mu_B * m_j * B
        return {'m_j': m_j, 'B_T': B, 'E_J': f'{E:.4e}',
                'E_eV': round(E/1.6e-19, 6)}

    def stern_gerlach(self, s, B_gradient):
        """斯特恩-盖拉赫: 自旋量子化分裂"""
        # 力: F = μ·(dB/dz), μ_z = -g·μ_B·m_s
        forces = []
        for m_s in [s - i for i in range(int(2*s)+1)]:
            mu_z = -self.g_e * self.mu_B * m_s
            F = mu_z * B_gradient
            forces.append({'m_s': m_s, 'mu_z': f'{mu_z:.4e}', 'force_N': f'{F:.4e}'})
        return {'s': s, 'n_spots': int(2*s+1), 'forces': forces,
                'note': f'分裂为{int(2*s)+1}条, 证明自旋量子化'}

    def step(self, state, dt):
        return state


class IdenticalParticlesEngine(PhysicsEngine):
    """全同粒子: 玻色-爱因斯坦/费米-狄拉克统计"""
    def __init__(self):
        super().__init__(name="identical_particles")
        self.k_B = KB
        self.h = 6.626e-34

    def boson(self, T, E, mu=0):
        """玻色-爱因斯坦分布: n(E) = 1/(exp((E-μ)/kT) - 1)"""
        if E <= mu: return {'n': float('inf'), 'note': 'E≤μ, 发生BEC'}
        n = 1 / (math.exp((E - mu) / (self.k_B * T)) - 1)
        return {'statistic': 'Bose-Einstein', 'T_K': T, 'E_eV': round(E/1.6e-19, 6),
                'n': round(n, 4), 'note': '整数自旋, 可凝聚(BEC)'}

    def fermion(self, T, E, mu):
        """费米-狄拉克分布: n(E) = 1/(exp((E-μ)/kT) + 1)"""
        n = 1 / (math.exp((E - mu) / (self.k_B * T)) + 1)
        return {'statistic': 'Fermi-Dirac', 'T_K': T, 'E_eV': round(E/1.6e-19, 6),
                'n': round(n, 4), 'note': '半整数自旋, 泡利不相容'}

    def fermi_energy(self, n_density):
        """费米能: E_F = (ℏ²/2m)(3π²n)^(2/3)"""
        E_F = (self.h_bar**2 / (2*M_E)) * (3*math.pi**2 * n_density)**(2/3)
        return {'n_density': f'{n_density:.2e}',
                'E_F_J': f'{E_F:.4e}', 'E_F_eV': round(E_F/1.6e-19, 3),
                'note': 'T=0K最高占据态'}

    def degeneracy_pressure(self, n_density):
        """简并压(白矮星): P = (ℏ²/5m)(3π²)^(2/3)·n^(5/3)"""
        P = (self.h_bar**2 / (5*M_E)) * (3*math.pi**2)**(2/3) * n_density**(5/3)
        return {'n_density': f'{n_density:.2e}',
                'P_Pa': f'{P:.4e}', 'P_GPa': round(P/1e9, 2),
                'note': '费米子简并压支撑白矮星'}

    def step(self, state, dt):
        return state
