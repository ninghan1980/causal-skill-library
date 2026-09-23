"""physics_engine/thermodynamics.py — 热力学引擎"""
import numpy as np
from .core import PhysicsEngine, PhysicsState


class NewtonCoolingEngine(PhysicsEngine):
    """牛顿冷却定律: dT/dt = -k(T - T_env)"""
    def __init__(self, k=0.015, T_env=298.0):
        super().__init__("newton_cooling")
        self.k = k
        self.T_env = T_env

    def step(self, state, dt):
        dT = -self.k * (state.temperature - self.T_env) * dt
        new_T = state.temperature + dT
        # 理想气体: ρ ∝ T(等压)
        new_rho = state.density * state.temperature / (new_T + 1e-10)
        return PhysicsState(
            temperature=new_T,
            pressure=state.pressure,
            density=new_rho,
            velocity=state.velocity,
            position=state.position,
            time=state.time + dt,
        )

    def analytical_solution(self, T0, t):
        """解析解: T(t) = T_env + (T0 - T_env)·e^(-kt)"""
        return self.T_env + (T0 - self.T_env) * np.exp(-self.k * t)


class HeatConductionEngine(PhysicsEngine):
    """一维热传导: ∂T/∂t = α·∂²T/∂x² (有限差分)"""
    def __init__(self, alpha=1e-6, dx=0.01):
        super().__init__("heat_conduction")
        self.alpha = alpha
        self.dx = dx

    def solve_1d(self, T_init, nx, dt, steps, T_left=None, T_right=None):
        T = T_init.copy()
        history = [T.copy()]
        r = self.alpha * dt / (self.dx ** 2)
        if r > 0.5:
            print(f"⚠ CFL 不满足 r={r:.3f} > 0.5,结果可能不稳定")
        for _ in range(steps):
            T_new = T.copy()
            T_new[1:-1] = T[1:-1] + r * (T[2:] - 2 * T[1:-1] + T[:-2])
            T_new[0] = T_left if T_left is not None else T_new[1]
            T_new[-1] = T_right if T_right is not None else T_new[-2]
            T = T_new.copy()
            history.append(T.copy())
        return np.array(history)


class PhaseTransitionEngine(PhysicsEngine):
    """相变引擎 — 水 固/液/气 状态变化"""
    def __init__(self):
        super().__init__("phase_transition")
        self.melting_point = 273.15  # K
        self.boiling_point = 373.15  # K
        self.L_fusion = 3.34e5       # J/kg 熔化潜热
        self.L_vap = 2.26e6          # J/kg 汽化潜热
        self.c = {'ice': 2100, 'water': 4186, 'steam': 1996}

    def get_phase(self, T):
        if T < self.melting_point: return 'ice'
        if T < self.boiling_point: return 'water'
        return 'steam'

    def energy_to_phase_change(self, mass, T, target_T):
        cur = self.get_phase(T)
        tar = self.get_phase(target_T)
        E = 0.0
        if cur == 'ice':
            E += mass * self.c['ice'] * (self.melting_point - T)
            if tar != 'ice':
                E += mass * self.L_fusion
                E += mass * self.c['water'] * (target_T - self.melting_point)
            if tar == 'steam':
                E += mass * self.L_vap + mass * self.c['steam'] * (target_T - self.boiling_point)
        elif cur == 'water':
            if tar == 'ice':
                E -= mass * self.L_fusion + mass * self.c['ice'] * (target_T - self.melting_point)
            else:
                E += mass * self.c['water'] * (target_T - T)
            if tar == 'steam':
                E += mass * self.L_vap + mass * self.c['steam'] * (target_T - self.boiling_point)
        else:  # steam
            E += mass * self.c['steam'] * (target_T - T)
        return {
            'phase_from': cur,
            'phase_to': tar,
            'energy_required_J': E,
            'energy_required_kcal': E / 4184,
        }