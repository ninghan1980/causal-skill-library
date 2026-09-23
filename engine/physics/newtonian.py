"""physics_engine/newtonian.py — 牛顿力学 (3引擎: 刚体/碰撞/角动量)"""
import math
from .core import PhysicsEngine, PhysicsState


class RigidBodyRotationEngine(PhysicsEngine):
    """刚体转动: 角动量守恒 + 转动惯量 + 陀螺进动"""
    def __init__(self):
        super().__init__(name="rigid_body")

    def moment_of_inertia(self, mass, shape, **kwargs):
        """转动惯量 I = Σmr² (多种规则形状)"""
        if shape == 'sphere':    # 实心球: 2/5 mr²
            return 0.4 * mass * kwargs['radius']**2
        if shape == 'hollow_sphere':  # 空心球: 2/3 mr²
            return 2/3 * mass * kwargs['radius']**2
        if shape == 'disk':      # 圆盘绕中心轴: 1/2 mr²
            return 0.5 * mass * kwargs['radius']**2
        if shape == 'rod_center':  # 细杆绕中心: 1/12 mL²
            return 1/12 * mass * kwargs['length']**2
        if shape == 'rod_end':   # 细杆绕端点: 1/3 mL²
            return 1/3 * mass * kwargs['length']**2
        if shape == 'cylinder':  # 圆柱绕轴: 1/2 mr²
            return 0.5 * mass * kwargs['radius']**2
        if shape == 'ring':      # 圆环: mr²
            return mass * kwargs['radius']**2
        return None

    def angular_momentum(self, I, omega):
        """角动量 L = Iω"""
        return I * omega

    def rotational_kinetic_energy(self, I, omega):
        """转动动能 E = ½Iω²"""
        return 0.5 * I * omega**2

    def conservation_demo(self, I1, omega1, I2):
        """角动量守恒: I1ω1 = I2ω2 (花样滑冰收臂)"""
        L = I1 * omega1
        omega2 = L / I2
        E1 = 0.5 * I1 * omega1**2
        E2 = 0.5 * I2 * omega2**2
        return {'L': round(L, 4), 'omega1': omega1, 'omega2': round(omega2, 4),
                'E1_J': round(E1, 4), 'E2_J': round(E2, 4),
                'delta_E_J': round(E2-E1, 4),
                'note': '角动量守恒, 收臂转速↑, 能量↑(内力做功)'}

    def gyroscopic_precession(self, I, omega_spin, torque):
        """陀螺进动: Ω = τ / (I·ω_spin)"""
        Omega = torque / (I * omega_spin)
        return {'precession_rate_rad_s': round(Omega, 6),
                'period_s': round(2*math.pi/Omega, 2) if Omega>0 else None,
                'note': '角动量守恒的表现'}

    def step(self, state, dt):
        omega = state.extra.get('omega', 0)
        I = state.extra.get('I', 1.0)
        tau = state.extra.get('tau', 0.0)
        omega += tau / I * dt
        s = PhysicsState()
        s.extra = {'omega': omega, 'I': I, 'tau': tau, 'L': I*omega}
        s.time = state.time + dt
        return s


class CollisionEngine(PhysicsEngine):
    """碰撞: 动量守恒 + 动能(弹性/非弹性/完全非弹性)"""
    def __init__(self):
        super().__init__(name="collision")

    def elastic_1d(self, m1, v1, m2, v2):
        """一维弹性碰撞: 动量守恒 + 动能守恒"""
        v1f = ((m1-m2)*v1 + 2*m2*v2) / (m1+m2)
        v2f = ((m2-m1)*v2 + 2*m1*v1) / (m1+m2)
        p_before = m1*v1 + m2*v2
        p_after = m1*v1f + m2*v2f
        K_before = 0.5*m1*v1**2 + 0.5*m2*v2**2
        K_after = 0.5*m1*v1f**2 + 0.5*m2*v2f**2
        return {'v1_final': round(v1f, 4), 'v2_final': round(v2f, 4),
                'p_before': round(p_before, 4), 'p_after': round(p_after, 4),
                'K_before': round(K_before, 4), 'K_after': round(K_after, 4),
                'type': '弹性碰撞(动能守恒)'}

    def perfectly_inelastic(self, m1, v1, m2, v2):
        """完全非弹性碰撞: 碰后共速"""
        vf = (m1*v1 + m2*v2) / (m1+m2)
        K_before = 0.5*m1*v1**2 + 0.5*m2*v2**2
        K_after = 0.5*(m1+m2)*vf**2
        return {'v_final': round(vf, 4),
                'K_before': round(K_before, 4), 'K_after': round(K_after, 4),
                'energy_loss_J': round(K_before-K_after, 4),
                'energy_loss_pct': round((K_before-K_after)/K_before*100, 2) if K_before>0 else 0,
                'type': '完全非弹性(共速, 动能损失最大)'}

    def coefficient_of_restitution(self, m1, v1, m2, v2, e):
        """通用碰撞: e=1弹性, e=0完全非弹性, 0<e<1一般"""
        v1f = (m1*v1 + m2*v2 + m2*e*(v2-v1)) / (m1+m2)
        v2f = (m1*v1 + m2*v2 + m1*e*(v1-v2)) / (m1+m2)
        return {'v1_final': round(v1f, 4), 'v2_final': round(v2f, 4),
                'e': e,
                'type': f'e={e}碰撞'}

    def step(self, state, dt):
        return state


class ProjectileEngine(PhysicsEngine):
    """抛体运动: 斜抛/最高点/射程 + 空气阻力(可选)"""
    def __init__(self, g=9.81):
        super().__init__(name="projectile")
        self.g = g

    def ideal(self, v0, angle_deg, y0=0):
        """理想斜抛(无阻力)"""
        ang = math.radians(angle_deg)
        vx0 = v0 * math.cos(ang)
        vy0 = v0 * math.sin(ang)
        t_peak = vy0 / self.g
        h_max = y0 + vy0**2 / (2*self.g)
        t_total = 2 * t_peak if y0 == 0 else (vy0 + math.sqrt(vy0**2 + 2*self.g*y0)) / self.g
        R = vx0 * t_total
        return {'v0': v0, 'angle': angle_deg,
                'vx0': round(vx0, 3), 'vy0': round(vy0, 3),
                't_peak': round(t_peak, 3), 'h_max': round(h_max, 3),
                't_total': round(t_total, 3), 'range': round(R, 3),
                'note': '理想无阻力'}

    def with_drag(self, v0, angle_deg, mass=0.5, Cd=0.47, area=0.01, rho=1.225, dt=0.01):
        """有空气阻力: 数值积分(球体, Cd≈0.47)"""
        ang = math.radians(angle_deg)
        vx = v0 * math.cos(ang)
        vy = v0 * math.sin(ang)
        x, y, t = 0, 0, 0
        traj = [(0, 0)]
        while y >= 0 or t == 0:
            v = math.sqrt(vx**2 + vy**2)
            Fd = 0.5 * rho * v**2 * Cd * area
            ax = -Fd * vx / (mass * v) if v > 0 else 0
            ay = -self.g - Fd * vy / (mass * v) if v > 0 else -self.g
            vx += ax * dt
            vy += ay * dt
            x += vx * dt
            y += vy * dt
            t += dt
            if t < 5: traj.append((round(x, 1), round(y, 1)))
            if t > 30: break
        return {'v0': v0, 'angle': angle_deg, 'range': round(x, 2),
                'h_max': round(max(p[1] for p in traj), 2),
                't_flight': round(t, 3),
                'trajectory': traj[:20]}

    def step(self, state, dt):
        return state
