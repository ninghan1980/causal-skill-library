"""physics_engine/fluid_dynamics_2d.py — 二维流体动力学 (P1)
========================================================
二维 Navier-Stokes 方程 + 涡量-流函数法
用于: 台风风场剖面 / 大气环流 / 圆柱绕流 / 方腔流
"""
import numpy as np
import math
from .core import PhysicsEngine, PhysicsState


class NavierStokes2DEngine(PhysicsEngine):
    """二维 Navier-Stokes: 涡量-流函数法 (稳定)
    ∂ω/∂t + u·∂ω/∂x + v·∂ω/∂y = ν(∂²ω/∂x² + ∂²ω/∂y²)
    ∇²ψ = -ω, u=∂ψ/∂y, v=-∂ψ/∂x
    """
    def __init__(self, nx=50, ny=50, Lx=1.0, Ly=1.0, nu=1e-3):
        super().__init__(name="navier_stokes_2d")
        self.nx, self.ny = nx, ny
        self.Lx, self.Ly = Lx, Ly
        self.nu = nu  # 运动粘度
        self.dx, self.dy = Lx/nx, Ly/ny

    def solve_cavity_flow(self, Re=100, nsteps=1000, dt=0.001):
        """顶盖驱动方腔流 (经典验证案例)"""
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy
        nu = self.nu
        # 初始化
        omega = np.zeros((ny, nx))
        psi = np.zeros((ny, nx))
        # 顶盖速度 (U=1)
        U_lid = 1.0
        for step in range(nsteps):
            old_omega = omega.copy()
            # 内部点: 涡量输运方程 (FTCS + 迎风格式)
            for i in range(1, ny-1):
                for j in range(1, nx-1):
                    u = (psi[i+1,j] - psi[i-1,j]) / (2*dy)
                    v = -(psi[i,j+1] - psi[i,j-1]) / (2*dx)
                    # 一阶迎风格式
                    if u > 0:
                        dom_dx = (old_omega[i,j] - old_omega[i,j-1]) / dx
                    else:
                        dom_dx = (old_omega[i,j+1] - old_omega[i,j]) / dx
                    if v > 0:
                        dom_dy = (old_omega[i,j] - old_omega[i-1,j]) / dy
                    else:
                        dom_dy = (old_omega[i+1,j] - old_omega[i,j]) / dy
                    diff = nu * ((old_omega[i,j+1] - 2*old_omega[i,j] + old_omega[i,j-1]) / dx**2 +
                                 (old_omega[i+1,j] - 2*old_omega[i,j] + old_omega[i-1,j]) / dy**2)
                    omega[i,j] = old_omega[i,j] + dt * (-u*dom_dx - v*dom_dy + diff)
            # 边界条件: 壁面涡量
            omega[0,:] = -2*psi[1,:] / dy**2  # 底壁
            omega[-1,:] = -2*psi[-2,:] / dy**2 - 2*U_lid/dy  # 顶盖
            omega[:,0] = -2*omega[:,1]  # 左壁
            omega[:,-1] = -2*omega[:,-2]  # 右壁
            # 泊松方程求解流函数 (Jacobi 迭代)
            for _ in range(50):
                psi_old = psi.copy()
                psi[1:-1,1:-1] = (
                    dy**2*(psi_old[1:-1,2:] + psi_old[1:-1,:-2]) +
                    dx**2*(psi_old[2:,1:-1] + psi_old[:-2,1:-1]) +
                    dx**2*dy**2*omega[1:-1,1:-1]
                ) / (2*(dx**2 + dy**2))
                psi[0,:] = psi[-1,:] = psi[:,0] = psi[:,-1] = 0
        # 速度场
        u = np.zeros((ny, nx))
        v = np.zeros((ny, nx))
        u[1:-1,1:-1] = (psi[2:,1:-1] - psi[:-2,1:-1]) / (2*dy)
        v[1:-1,1:-1] = -(psi[1:-1,2:] - psi[1:-1,:-2]) / (2*dx)
        return {
            'Re': Re,
            'u_max': round(float(np.max(np.abs(u))), 5),
            'v_max': round(float(np.max(np.abs(v))), 5),
            'omega_center': round(float(omega[ny//2, nx//2]), 5),
            'psi_center': round(float(psi[ny//2, nx//2]), 5),
            'u_center_line': u[ny//2, :].tolist(),
            'summary': f'方腔流 Re={Re}: u_max={np.max(np.abs(u)):.4f}, ψ_center={psi[ny//2,nx//2]:.4f}',
        }

    def typhoon_wind_field(self, R_max, V_max, r_out, ngrid=50):
        """台风径向风场剖面 (Rankine 涡 + 指数衰减)
        R_max: 最大风速半径(km), V_max: 最大风速(m/s)
        """
        # 径向网格
        r = np.linspace(1, r_out, ngrid)
        # 径向风速 (Rankine 简化)
        V = np.where(r <= R_max, V_max * (r / R_max), V_max * (R_max / r) ** 0.5)
        # 切向风速
        Vt = V * 0.8  # 简化
        # 径向风速 (流入/流出)
        Vr = np.where(r <= R_max, -0.5 * V_max * (r / R_max), 0)
        # 风压 (梯度风平衡: dP/dr = ρ(V²/r + fV))
        rho = 1.225
        f = 2 * 7.2921e-5 * math.sin(math.radians(15))  # 科氏参数 @15°N
        P = np.zeros_like(r)
        P[0] = 93500  # 中心气压 Pa (超强台风)
        for i in range(1, len(r)):
            r_m = r[i] * 1000  # km → m
            dP = rho * (V[i]**2 / r_m + f * V[i]) * (r[i] - r[i-1]) * 1000
            P[i] = P[i-1] + dP
        return {
            'r_km': r.tolist(),
            'V_tangential': V.tolist(),
            'V_radial': Vr.tolist(),
            'pressure': (P/100).tolist(),  # hPa
            'R_max_km': R_max,
            'V_max_m_s': V_max,
            'summary': f'台风风场: V_max={V_max}m/s @ R={R_max}km, 外围{r_out}km',
        }

    def step(self, state, dt):
        return state
