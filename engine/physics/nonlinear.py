"""physics_engine/nonlinear.py — 非线性动力学 (2引擎: 混沌/分形)"""
import math
from .core import PhysicsEngine, PhysicsState


class ChaosEngine(PhysicsEngine):
    """混沌: 洛伦兹吸引子/李雅普诺夫指数/分岔"""
    def __init__(self):
        super().__init__(name="chaos")

    def lorenz(self, sigma=10.0, rho=28.0, beta=8/3, dt=0.01, nsteps=1000, x0=1.0, y0=1.0, z0=1.0):
        """洛伦兹吸引子: dx/dt=σ(y-x), dy/dt=x(ρ-z)-y, dz/dt=xy-βz"""
        x, y, z = x0, y0, z0
        traj = []
        for i in range(nsteps):
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            x += dx * dt
            y += dy * dt
            z += dz * dt
            if i % 10 == 0:
                traj.append({'step': i, 'x': round(x, 5), 'y': round(y, 5), 'z': round(z, 5)})
        return {'sigma': sigma, 'rho': rho, 'beta': beta,
                'x_final': round(x, 4), 'y_final': round(y, 4), 'z_final': round(z, 4),
                'trajectory_sample': traj[:10],
                'note': '蝴蝶效应: 初值微小变化导致轨迹完全改变'}

    def logistic_map(self, r, x0=0.5, nsteps=100):
        """逻辑斯蒂映射: x_{n+1} = r·x_n·(1-x_n)"""
        x = x0
        traj = []
        for i in range(nsteps):
            x = r * x * (1 - x)
            if i >= nsteps - 20:
                traj.append({'step': i, 'x': round(x, 6)})
        # 判断状态
        if r < 3.0:
            status = '不动点收敛'
        elif r < 3.449:
            status = '周期2振荡'
        elif r < 3.544:
            status = '周期4振荡'
        elif r < 3.57:
            status = '周期倍增级联'
        else:
            status = '混沌'
        return {'r': r, 'x0': x0,
                'final_x': round(x, 6),
                'status': status,
                'trajectory': traj,
                'note': '最简单的一维混沌系统'}

    def lyapunov_exponent(self, r, x0=0.5, nsteps=1000):
        """李雅普诺夫指数: λ = lim(1/n)·Σ ln|r·(1-2x_n)|"""
        x = x0
        lyap_sum = 0
        for i in range(nsteps):
            x = r * x * (1 - x)
            lyap_sum += math.log(abs(r * (1 - 2 * x)) + 1e-10)
        lam = lyap_sum / nsteps
        return {'r': r, 'lyapunov_exponent': round(lam, 4),
                'chaotic': lam > 0,
                'note': 'λ>0表示混沌(对初值敏感)'}

    def step(self, state, dt):
        return state


class FractalEngine(PhysicsEngine):
    """分形: 曼德博/朱利亚/科赫雪花/维数"""
    def __init__(self):
        super().__init__(name="fractal")

    def mandelbrot(self, c_real, c_imag, max_iter=100):
        """曼德博集: z_{n+1} = z_n² + c"""
        z_real, z_imag = 0, 0
        for i in range(max_iter):
            z_real_new = z_real**2 - z_imag**2 + c_real
            z_imag_new = 2 * z_real * z_imag + c_imag
            z_real, z_imag = z_real_new, z_imag_new
            if z_real**2 + z_imag**2 > 4:
                return {'c': f'{c_real}+{c_imag}i', 'escaped': True,
                        'iterations': i, 'in_set': False}
        return {'c': f'{c_real}+{c_imag}i', 'escaped': False,
                'iterations': max_iter, 'in_set': True}

    def julia(self, z_real, z_imag, c_real=-0.7, c_imag=0.27015, max_iter=100):
        """朱利亚集: z_{n+1} = z_n² + c"""
        for i in range(max_iter):
            z_real_new = z_real**2 - z_imag**2 + c_real
            z_imag_new = 2 * z_real * z_imag + c_imag
            z_real, z_imag = z_real_new, z_imag_new
            if z_real**2 + z_imag**2 > 4:
                return {'z0': f'{z_real}+{z_imag}i', 'c': f'{c_real}+{c_imag}i',
                        'escaped': True, 'iterations': i}
        return {'z0': f'{z_real}+{z_imag}i', 'c': f'{c_real}+{c_imag}i',
                'escaped': False, 'iterations': max_iter}

    def koch_snowflake(self, side=1.0, n_iter=4):
        """科赫雪花: 边长 = (4/3)^n · 原边长, 面积有限, 周长无穷"""
        perimeter = 3 * side * (4/3)**n_iter
        area = math.sqrt(3)/4 * side**2 * (1 + 3/9 * (1 - (4/9)**(n_iter-1)) / (1 - 4/9)) if n_iter > 0 else math.sqrt(3)/4 * side**2
        return {'side': side, 'iterations': n_iter,
                'perimeter': round(perimeter, 4),
                'area': round(area, 4),
                'dimension': round(math.log(4)/math.log(3), 4),
                'note': f'分形维数={math.log(4)/math.log(3):.3f}(>1, <2)'}

    def self_similarity_dimension(self, N, r):
        """自相似维数: D = log(N)/log(1/r)"""
        if N <= 0 or r <= 0 or r >= 1:
            return None
        D = math.log(N) / math.log(1/r)
        return {'N': N, 'r': r, 'dimension': round(D, 4),
                'note': '科赫雪花: N=4, r=1/3 → D≈1.262'}

    def step(self, state, dt):
        return state
