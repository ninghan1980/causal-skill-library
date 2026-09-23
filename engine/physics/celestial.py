"""physics_engine/celestial.py — 天体力学 (3引擎: 开普勒/霍曼转移/拉格朗日点)"""
import math
from .core import PhysicsEngine, PhysicsState

# 标准引力参数 μ = GM (km³/s²)
MU_SUN = 1.32712440018e11
MU_EARTH = 3.986004418e5
MU_MARS = 4.282837e4
MU_JUPITER = 1.26686534e8

# 天体半径 (km)
R_EARTH = 6371.0
R_MARS = 3389.5


class KeplerOrbitEngine(PhysicsEngine):
    """开普勒轨道: 二体问题 + 轨道六根数 → 位置/速度"""
    def __init__(self, mu=MU_SUN):
        super().__init__(name="kepler_orbit")
        self.mu = mu

    def classify(self, a, e):
        if e >= 1.0: return "双曲线/逃逸" if e > 1 else "抛物线"
        if abs(e) < 1e-6: return "正圆"
        if e < 0.3: return "近圆"
        return "椭圆"

    def orbital_period(self, a):
        """开普勒第三定律: T = 2π√(a³/μ)"""
        return 2 * math.pi * math.sqrt(a**3 / self.mu)

    def vis_viva(self, r, a):
        """活力公式: v² = μ(2/r - 1/a)"""
        return math.sqrt(self.mu * (2/r - 1/a))

    def orbital_elements_to_state(self, a, e, i, Omega, omega, theta):
        """轨道六根数 → 位置(r) + 速度(v)"""
        p = a * (1 - e**2)  # 半通径
        r = p / (1 + e * math.cos(theta))
        v = math.sqrt(self.mu * (2/r - 1/a))
        return {'r_km': round(r, 2), 'v_km_s': round(v, 4),
                'a_km': a, 'e': e, 'period_s': round(self.orbital_period(a), 1),
                'period_h': round(self.orbital_period(a)/3600, 2),
                'type': self.classify(a, e)}

    def circular_orbit(self, r):
        """圆轨道: 给定半径 → 速度+周期"""
        v = math.sqrt(self.mu / r)
        T = 2 * math.pi * r / v
        return {'r_km': r, 'v_km_s': round(v, 4),
                'period_s': round(T, 1), 'period_min': round(T/60, 2),
                'type': '正圆'}

    def step(self, state, dt):
        """简化: 推进真近点角"""
        if state.position is None:
            return state
        theta = state.extra.get('theta', 0) + dt * self.mu**0.5 / state.position[0]**1.5
        a = state.extra.get('a', state.position[0])
        e = state.extra.get('e', 0)
        r = a*(1-e**2)/(1+e*math.cos(theta))
        v = math.sqrt(self.mu*(2/r-1/a))
        s = PhysicsState()
        s.position = (r, theta)
        s.velocity = v
        s.extra = {'theta': theta, 'a': a, 'e': e}
        s.time = state.time + dt
        return s


class HohmannTransferEngine(PhysicsEngine):
    """霍曼转移: 两个共面圆轨道间最省燃料转移"""
    def __init__(self, mu=MU_EARTH):
        super().__init__(name="hohmann_transfer")
        self.mu = mu

    def transfer(self, r1, r2):
        """r1, r2: 初始/目标轨道半径 (km)"""
        a_transfer = (r1 + r2) / 2  # 转移椭圆半长轴
        # 活力公式
        v1_circ = math.sqrt(self.mu / r1)
        v2_circ = math.sqrt(self.mu / r2)
        v1_transfer = math.sqrt(self.mu * (2/r1 - 1/a_transfer))
        v2_transfer = math.sqrt(self.mu * (2/r2 - 1/a_transfer))
        # 两次脉冲增量
        dv1 = abs(v1_transfer - v1_circ)  # 第一次加速
        dv2 = abs(v2_circ - v2_transfer)  # 第二次加速
        # 转移时间 = 半个周期
        T_transfer = math.pi * math.sqrt(a_transfer**3 / self.mu)
        return {
            'r1_km': r1, 'r2_km': r2,
            'a_transfer_km': a_transfer,
            'dv1_km_s': round(dv1, 4),
            'dv2_km_s': round(dv2, 4),
            'dv_total_km_s': round(dv1 + dv2, 4),
            'transfer_time_s': round(T_transfer, 1),
            'transfer_time_min': round(T_transfer/60, 2),
            'transfer_time_h': round(T_transfer/3600, 3),
            'v1_circ': round(v1_circ, 4), 'v2_circ': round(v2_circ, 4),
        }

    def step(self, state, dt):
        return state


class LagrangePointEngine(PhysicsEngine):
    """拉格朗日点: 圆型三体问题的5个平动点 (简化: L1/L2/L3/L4/L5)"""
    def __init__(self, mu_primary=MU_SUN, mu_secondary=MU_EARTH,
                 distance=1.496e8):  # 日地距离 km
        super().__init__(name="lagrange_point")
        self.mu_p = mu_primary
        self.mu_s = mu_secondary
        self.D = distance

    def l1(self):
        """L1: 日月连线之间 (航天常用, 如SOHO/嫦娥)"""
        # 近似: r ≈ D * (1 - (μs/3μp)^(1/3))
        mu_ratio = self.mu_s / self.mu_p
        r = self.D * (1 - (mu_ratio/3)**(1/3))
        return {'name': 'L1', 'distance_from_primary_km': round(r, 0),
                'distance_from_secondary_km': round(self.D - r, 0),
                'note': '日地之间, 太阳观测/深空探测'}

    def l2(self):
        """L2: 日月连线外侧 (韦伯/嫦娥)"""
        mu_ratio = self.mu_s / self.mu_p
        r = self.D * (1 + (mu_ratio/3)**(1/3))
        return {'name': 'L2', 'distance_from_primary_km': round(r, 0),
                'distance_from_secondary_km': round(r - self.D, 0),
                'note': '地球外侧, 詹姆斯·韦伯/鹊桥'}

    def l3(self):
        """L3: 太阳背面"""
        return {'name': 'L3', 'distance_from_primary_km': round(self.D*2, 0),
                'note': '太阳背面, 难以观测'}

    def l4_l5(self):
        """L4/L5: 等边三角形点 (稳定)"""
        return {'name': 'L4/L5', 'distance_from_primary_km': round(self.D, 0),
                'angle_deg': 60, 'stability': '稳定',
                'note': '特洛伊小行星群所在'}

    def all_points(self):
        return [self.l1(), self.l2(), self.l3(), self.l4_l5()]

    def step(self, state, dt):
        return state
