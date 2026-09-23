"""physics_engine/numerical_kernel.py — 数值内核 (向量化 + scipy加速)
========================================================
物理引擎的数值加速层, 用两个手段替代 numba (musl限制):
  1. numpy 向量化: 消除 Python 循环
  2. scipy 稀疏矩阵: 泊松方程直接求解 (快迭代法一个数量级)

基准 (100x100 网格):
  纯Python循环:    超时 (>120s)
  numpy向量化+迭代: 5.8s
  scipy稀疏直接解:  0.99s  ← 当前最优

用法:
  from physics_engine.numerical_kernel import PoissonSolver, CavityFlowVectorized
"""
import numpy as np

class PoissonSolver:
    """泊松方程求解器 (scipy稀疏LU分解复用)"""
    def __init__(self, nx, ny):
        self.nx, self.ny = nx, ny
        self.N = nx * ny
        self.h = 1.0 / nx
        self._build_laplacian()

    def _build_laplacian(self):
        """构造二维拉普拉斯算子稀疏矩阵 + 预分解 LU (复用)"""
        from scipy.sparse import diags
        from scipy.sparse.linalg import splu
        nx, ny = self.nx, self.ny
        N = nx * ny
        main = np.ones(N) * -4
        side = np.ones(N - 1)
        # 消除行边界的横向连接
        side[nx-1::nx] = 0
        updown = np.ones(N - nx)
        A = diags([updown, side, main, side, updown],
                  [-nx, -1, 0, 1, nx], format='csc') / self.h**2
        # 关键优化: LU分解一次, 之后每次 solve 只做回代
        self.lu = splu(A)

    def solve(self, rhs):
        """求解 ∇²φ = rhs (rhs: (ny, nx) 数组), 复用预分解的 LU"""
        b = rhs.reshape(-1)
        phi = self.lu.solve(b)
        return phi.reshape(self.ny, self.nx)


class CavityFlowVectorized:
    """向量化方腔流求解器 (引入 scipy 直接解)"""
    def __init__(self, nx, ny, Re=100, dt=0.0005):
        self.nx, self.ny = nx, ny
        self.Re = Re
        self.nu = 1.0 / Re
        self.dt = dt
        self.dx = self.dy = 1.0 / nx
        self.poisson = PoissonSolver(nx, ny)

    def solve(self, nsteps=300):
        """求解顶盖驱动方腔流"""
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy
        nu, dt = self.nu, self.dt
        U = 1.0

        omega = np.zeros((ny, nx))
        psi = np.zeros((ny, nx))

        for _ in range(nsteps):
            o = omega
            # 速度场 (向量化)
            u = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2*dy)
            v = -(psi[1:-1, 2:] - psi[1:-1, :-2]) / (2*dx)

            # 涡量输运 (向量化)
            lap = ((o[1:-1, 2:] - 2*o[1:-1, 1:-1] + o[1:-1, :-2]) / dx**2 +
                   (o[2:, 1:-1] - 2*o[1:-1, 1:-1] + o[:-2, 1:-1]) / dy**2)
            dcdx = (o[1:-1, 2:] - o[1:-1, :-2]) / (2*dx)
            dcdy = (o[2:, 1:-1] - o[:-2, 1:-1]) / (2*dy)

            omega[1:-1, 1:-1] = o[1:-1, 1:-1] + dt*(-u*dcdx - v*dcdy + nu*lap)

            # 边界涡量
            omega[0, :] = -2*psi[1, :] / dy**2
            omega[-1, :] = -2*psi[-2, :] / dy**2 - 2*U/dy
            omega[:, 0] = -2*psi[:, 1] / dx**2
            omega[:, -1] = -2*psi[:, -2] / dx**2

            # 泊松方程 (scipy直接解, 替代Jacobi迭代)
            psi = self.poisson.solve(-omega)

        # 结果
        u_full = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2*dy)
        v_full = -(psi[1:-1, 2:] - psi[1:-1, :-2]) / (2*dx)
        return {
            'psi': psi,
            'u': u_full,
            'v': v_full,
            'u_max': float(np.max(np.abs(u_full))),
            'v_max': float(np.max(np.abs(v_full))),
            'psi_center': float(psi[ny//2, nx//2]),
            'Re': self.Re,
            'grid': f'{nx}x{ny}',
        }


def benchmark():
    """性能基准测试"""
    import time
    print("="*60)
    print("数值内核加速基准 (scipy直接解)")
    print("="*60)
    for grid in [40, 60, 80, 100, 128, 160, 200]:
        t0 = time.time()
        solver = CavityFlowVectorized(grid, grid, Re=100)
        r = solver.solve(nsteps=100)
        t1 = time.time()
        print(f"  {grid}x{grid}: {t1-t0:.3f}s (u_max={r['u_max']:.4f})")


if __name__ == '__main__':
    benchmark()
