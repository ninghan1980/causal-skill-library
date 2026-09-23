"""physics_engine/fluid_dynamics.py — 流体力学引擎"""
import numpy as np
import math
from .core import PhysicsEngine, PhysicsState


class GradientWindEngine(PhysicsEngine):
    """梯度风平衡 — 台风强度估算"""
    def __init__(self):
        super().__init__("gradient_wind")
        self.rho = 1.2            # 空气密度 kg/m³
        self.omega = 7.2921e-5    # 地球自转角速度 rad/s
        self.R_earth = 6371.0e3   # 地球半径 m

    def coriolis_param(self, lat):
        return 2 * self.omega * math.sin(math.radians(lat))

    def gradient_wind(self, dP, lat, R=50000.0):
        """Vg = √(ΔP / (ρ·f·R))"""
        f = self.coriolis_param(lat)
        if f <= 0: return 0.0
        return math.sqrt(dP / (self.rho * f * R))

    def sydney_velocity(self, Vg, lat, R):
        """科氏力修正最大风速"""
        f = self.coriolis_param(lat)
        term = f * R / 2
        return Vg - term + math.sqrt(term ** 2 + Vg ** 2)


class TyphoonTrackEngine(PhysicsEngine):
    """台风路径引擎 — 引导气流 + 科氏偏转 + 副高转向"""
    def __init__(self):
        super().__init__("typhoon_track")
        self.R_earth = 6371.0e3
        self.omega = 7.2921e-5

    def step(self, state, dt):
        lat, lon = state.position[1], state.position[0]
        # 西北引导气流
        base_u, base_v = 5.0, 3.0
        # 副高转向
        if lon > 126:    turn_factor = 0.0
        elif lon > 124:  turn_factor = 0.3
        elif lon > 122:  turn_factor = 0.6
        else:            turn_factor = 0.9
        coriolis_turn = 0.15 * (lat - 18) if lat > 18 else 0
        turn_rad = math.radians(turn_factor * 45 + coriolis_turn * 10)
        u_eff = base_u * math.cos(turn_rad) - base_v * math.sin(turn_rad)
        v_eff = base_u * math.sin(turn_rad) + base_v * math.cos(turn_rad)
        # 位移
        dlon = u_eff * dt / (self.R_earth * math.cos(math.radians(lat))) * 180 / math.pi
        dlat = v_eff * dt / self.R_earth * 180 / math.pi
        # 海温 → 强度
        sst = 28.0 - 0.04 * (lon - 124) - (0.3 * (lat - 22) if lat > 22 else 0)
        if sst < 26: sst = 26
        press_change = 2 - (sst - 26.5) * 0.5
        new_p = max(975, min(1005, state.pressure + press_change * (dt / 21600)))
        return PhysicsState(
            temperature=state.temperature,
            pressure=new_p,
            density=state.density,
            velocity=(u_eff, v_eff, 0.0),
            position=(lon + dlon, lat + dlat),
            time=state.time + dt,
        )

    def classify(self, ws):
        if ws < 17.2: return "TD (热带低压)"
        if ws < 24.5: return "TS (热带风暴)"
        if ws < 32.7: return "STS (强热带风暴)"
        if ws < 41.5: return "TY (台风)"
        return "STY (强台风)"

    def distance_to(self, lon1, lat1, lon2, lat2):
        """大圆距离 km"""
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * \
            math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
        return self.R_earth * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)) / 1000


class BernoulliEngine(PhysicsEngine):
    """伯努利方程 — 风速与气压关系"""
    def dynamic_pressure(self, v, rho=1.225):
        return 0.5 * rho * v ** 2

    def wind_from_pressure_diff(self, dP, rho=1.225):
        return math.sqrt(2 * abs(dP) / rho)

    def pressure_from_wind(self, v, rho=1.225):
        return 0.5 * rho * v ** 2


class NavierStokes1DEngine(PhysicsEngine):
    """一维 Navier-Stokes 方程: 粘性流体管流 (简化 N-S)
    ρ(∂u/∂t + u∂u/∂x) = -∂P/∂x + μ∂²u/∂x² + ρg
    用迎风格式 + 隐式粘性项 数值求解
    """
    def __init__(self, mu=1.81e-5, rho=1.225):
        super().__init__(name="navier_stokes")
        self.mu = mu    # 动力粘度 (空气: 1.81e-5 Pa·s)
        self.rho = rho  # 密度

    def solve_pipe_flow(self, nx=50, dt=0.001, steps=500, u_inlet=1.0,
                        pressure_gradient=-0.1, g=0.0):
        """一维管流: 入口速度固定 + 压力梯度驱动"""
        dx = 1.0 / nx
        u = np.zeros(nx)
        u[0] = u_inlet  # 入口狄利克雷
        nu = self.mu / self.rho  # 运动粘度
        r = nu * dt / dx**2
        hist = [u.copy()]
        for _ in range(steps):
            un = u.copy()
            for i in range(1, nx-1):
                # 对流项: 迎风
                conv = -un[i] * (un[i] - un[i-1]) / dx if un[i] > 0 else -un[i] * (un[i+1] - un[i]) / dx
                # 粘性项: 中心扩散
                diff = nu * (un[i+1] - 2*un[i] + un[i-1]) / dx**2
                # 压力梯度 + 重力
                force = -pressure_gradient / self.rho + g
                u[i] = un[i] + dt * (conv + diff + force)
            u[0] = u_inlet  # 固定入口
            u[-1] = u[-2]   # 出口零梯度
            hist.append(u.copy())
        return {'u_profile': u.tolist(),
                'u_max': round(float(np.max(u)), 5),
                'u_mean': round(float(np.mean(u)), 5),
                'Re': round(self.rho * u_inlet * 1.0 / self.mu, 1),
                'nu': nu, 'r_stability': round(r, 4),
                'steps': steps}

    def reynolds_number(self, v, L, rho=None, mu=None):
        """雷诺数 Re = ρvL/μ (判断层流/湍流)"""
        rho = rho or self.rho
        mu = mu or self.mu
        Re = rho * v * L / mu
        flow = '层流' if Re < 2300 else ('过渡' if Re < 4000 else '湍流')
        return {'Re': round(Re, 1), 'flow_regime': flow,
                'v': v, 'L': L, 'rho': rho, 'mu': mu}

    def step(self, state, dt):
        return state