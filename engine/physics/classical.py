"""physics_engine/classical.py — 第一层: 经典物理 (5引擎)
NewtonCooling / HeatConduction / PhaseTransition / Cosmology / Typhoon
"""
import math
import numpy as np
from .core import PhysicsEngine, PhysicsState


class NewtonCoolingEngine(PhysicsEngine):
    """牛顿冷却: dT/dt = -k(T-T_env) (含辐射修正项)"""
    def __init__(self, k=0.015, T_env=298.0, sigma=5.67e-8, eps=0.3, m=0.5, c=4186.0):
        super().__init__(name="newton_cooling")
        self.k, self.T_env = k, T_env
        self.sigma, self.eps, self.m, self.c = sigma, eps, m, c

    def step(self, state, dt):
        T = state.temperature
        # 线性: -k*dT ; 辐射: -eps*sigma*(T^4-T_env^4)/(m*c)
        dT = -self.k * (T - self.T_env) - self.eps * self.sigma * (T**4 - self.T_env**4) / (self.m * self.c)
        s = PhysicsState(temperature=T + dT * dt)
        s.time = state.time + dt
        return s

    def analytical_solution(self, T0, t):
        if abs(self.k) < 1e-12: return self.T_env
        return self.T_env + (T0 - self.T_env) * math.exp(-self.k * t)


class HeatConductionEngine(PhysicsEngine):
    """热传导: dT/dt = alpha*d^2T/dx^2 (Crank-Nicolson)"""
    def __init__(self, alpha=1e-6, dx=0.01):
        super().__init__(name="heat_conduction")
        self.alpha, self.dx = alpha, dx

    def solve_1d(self, T_init, nx, dt, steps, T_left=None, T_right=None):
        """Crank-Nicolson 隐式格式求解一维导热"""
        r = self.alpha * dt / (self.dx**2)
        T = np.array(T_init, float)
        A = np.zeros((nx, nx))
        for i in range(nx):
            A[i, i] = 1 + r
            if i > 0: A[i, i-1] = -0.5 * r
            if i < nx-1: A[i, i+1] = -0.5 * r
        hist = [T.copy()]
        for _ in range(steps):
            # 右端项(显式部分)
            b = T.copy()
            for i in range(1, nx-1):
                b[i] += 0.5 * r * (T[i-1] - 2*T[i] + T[i+1])
            T = np.linalg.solve(A, b)
            if T_left is not None: T[0] = T_left
            if T_right is not None: T[-1] = T_right
            hist.append(T.copy())
        return hist


class PhaseTransitionEngine(PhysicsEngine):
    """相变: Clausius-Clapeyron + 比热/潜热能量"""
    def __init__(self):
        super().__init__(name="phase_transition")

    def get_phase(self, T):
        if T < 273.15: return "solid"
        if T < 373.15: return "liquid"
        return "gas"

    def clausius_clapeyron_pressure(self, T, P0=101325.0, L=2.26e6, R=8.314, T0=373.15):
        """高压锅/沸腾压力随温度"""
        return P0 * math.exp(-L / R * (1/T - 1/T0))

    def boiling_temp(self, P, P0=101325.0, L=2.26e6, R=8.314, T0=373.15):
        """给定气压, 水的沸点"""
        return 1.0 / (1.0/T0 - R / L * math.log(P / P0))

    def energy_to_phase_change(self, mass, T, target_T, c=4186.0, L=334e3):
        """从T加热到目标(T<熔点:显热; 跨熔点:显热+潜热)"""
        if target_T <= 273.15:
            return mass * c * (target_T - T)
        # 液固同温 0°C 熔化潜热 + 升温显热
        return mass * c * (273.15 - T) + mass * L + mass * c * (target_T - 273.15)


class CosmologyEngine(PhysicsEngine):
    """宇宙膨胀: FLRW 数值积分 (简化一维, 取Hubble参数)"""
    def __init__(self, H0=67.4e-3, Omega_m=0.315, Omega_Lambda=0.685, H_now=70.0):
        """H0单位: km/s/Mpc → 1/s; H_now km/s/Mpc"""
        super().__init__(name="cosmology")
        # 缩放因子a演化 d(a)/dt = H0 * sqrt(Omega_m/a^3 + Omega_Lambda)
        self.H0 = H0 * 1e-3  # 简化: km/s/Mpc, 数值用相对
        self.Om = Omega_m
        self.Ol = Omega_Lambda

    def scale_factor_history(self, a0=1e-8, steps=200):
        """从大爆炸后积分缩放因子"""
        a = a0
        H = 70.0  # km/s/Mpc (数值)
        hist = [a]
        t = 0.0
        dt = 0.02
        for _ in range(steps):
            # H(a) = H0*sqrt(Om/a^3 + Ol)
            Ha = H * math.sqrt(self.Om / a**3 + self.Ol)
            t += dt / Ha
            a += Ha * dt * 1e-2  # 无量纲推进
            hist.append(a)
        return hist


class TyphoonTrackEngine(PhysicsEngine):
    """台风路径: β漂移(科氏) + 环境引导气流 (精简)"""
    def __init__(self, beta_coef=1.5e-5, env_flow=(0.0, 0.0)):
        super().__init__(name="typhoon_track")
        self.beta = beta_coef
        self.env = env_flow

    def classify(self, ws):
        if ws < 17.2: return "TD/热带风暴"
        if ws < 32.7: return "TS/强热带风暴"
        if ws < 41.5: return "TY/台风"
        return "SuperTY/超强台风"

    def distance_to(self, lon1, lat1, lon2, lat2):
        R = 6371.0
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dp = math.radians(lat2-lat1); dl = math.radians(lon2-lon1)
        a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
        return 2*R*math.asin(math.sqrt(a))

    def step(self, state, dt):
        lon, lat = state.position
        # β漂移 向北(约3-5 km/h 量级, 用经度体现)
        dlat = self.beta * dt
        dlon = self.env[0] * dt + math.cos(math.radians(lat)) * 0
        # 环境引导
        dlon = self.env[0] * dt
        dlat = self.beta * dt + self.env[1] * dt
        s = PhysicsState(temperature=state.temperature, pressure=state.pressure - 0.2)
        s.position = (lon + dlon, lat + dlat)
        s.time = state.time + dt
        return s
