"""physics_engine/core.py — 核心框架
世界物理引擎核心框架
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Callable
import json


@dataclass
class PhysicsState:
    """物理状态基类"""
    temperature: float = 298.0      # K
    pressure: float = 1013.25      # hPa
    density: float = 1.225         # kg/m³
    velocity: tuple = (0.0, 0.0, 0.0)
    position: tuple = (120.0, 30.0) # (lon, lat)
    time: float = 0.0
    extra: Dict = field(default_factory=dict)

    def to_dict(self):
        return {
            'temperature': self.temperature, 'pressure': self.pressure,
            'density': self.density, 'velocity': self.velocity,
            'position': self.position, 'time': self.time, **self.extra,
        }


@dataclass
class PredictionResult:
    """预测结果容器"""
    states: list
    uncertainties: list
    time_points: list
    causal_edges: list = field(default_factory=list)
    discovered_equations: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class PhysicsEngine:
    """物理引擎基类 - 所有具体引擎的父类"""
    def __init__(self, name="base"):
        self.name = name
        self.laws: List[Callable] = []

    def step(self, state, dt):
        raise NotImplementedError

    def predict(self, state, dt, steps):
        """多步预测 - 自动累积不确定性"""
        current = state
        states, uncertainties, times = [current], [0.0], [current.time]
        for i in range(steps):
            nxt = self.step(current, dt)
            nxt.time = current.time + dt
            states.append(nxt)
            uncertainties.append(uncertainties[-1] + 0.01 * (i + 1) ** 0.5)
            times.append(nxt.time)
            current = nxt
        return PredictionResult(states=states, uncertainties=uncertainties, time_points=times)


class MonteCarloEngine:
    """蒙特卡洛不确定性量化"""
    def __init__(self, engine, n_samples=500):
        self.engine = engine
        self.n_samples = n_samples

    def predict_with_uncertainty(self, state, dt, steps, param_noise=None):
        param_noise = param_noise or {}
        trajectories = []
        for _ in range(self.n_samples):
            s = PhysicsState(
                temperature=state.temperature + np.random.normal(0, param_noise.get('temperature', 0)),
                pressure=state.pressure + np.random.normal(0, param_noise.get('pressure', 0)),
                density=state.density,
                velocity=state.velocity,
                position=state.position,
                time=state.time,
            )
            traj = [s]
            cur = s
            for _ in range(steps):
                ns = self.engine.step(cur, dt)
                ns.time = cur.time + dt
                ns.temperature += np.random.normal(0, param_noise.get('process_temp', 0.1))
                ns.pressure += np.random.normal(0, param_noise.get('process_press', 0.1))
                traj.append(ns)
                cur = ns
            trajectories.append(traj)

        out = {}
        for v in ['temperature', 'pressure', 'density']:
            arr = np.array([[getattr(t, v) for t in tr] for tr in trajectories])
            out[v] = {
                'mean': np.mean(arr, axis=0).tolist(),
                'std': np.std(arr, axis=0).tolist(),
                'p5': np.percentile(arr, 5, axis=0).tolist(),
                'p95': np.percentile(arr, 95, axis=0).tolist(),
            }
        return out


class CausalDiscovery:
    """因果发现 — 从时间序列学习因果关系"""

    @staticmethod
    def _linear_regression(X, y):
        """纯 numpy 最小二乘 (替代 sklearn.LinearRegression)"""
        # 加偏置列
        Xb = np.hstack([np.ones((X.shape[0], 1)), X])
        # lstsq 解 Xb β = y
        beta, _, _, _ = np.linalg.lstsq(Xb, y, rcond=None)
        y_pred = Xb @ beta
        return y_pred, beta

    @staticmethod
    def pc_algorithm(data, alpha=0.05, max_lag=3):
        """简化PC算法: 条件独立性测试 → 因果边"""
        n_vars = data.shape[1]
        edges = []
        n = data.shape[0]
        for lag in range(1, max_lag + 1):
            for src in range(n_vars):
                for tgt in range(n_vars):
                    if src == tgt: continue
                    corr = np.corrcoef(data[lag:, src], data[:n - lag, tgt])[0, 1]
                    if abs(corr) > 0.3:
                        edges.append({
                            'source': src, 'target': tgt, 'lag': lag,
                            'strength': round(abs(corr), 3),
                            'sign': 'positive' if corr > 0 else 'negative',
                        })
        return edges

    @staticmethod
    def granger_causality(data, max_lag=3):
        """Granger 因果检验: 有 x vs 无 x 的 F 检验"""
        edges = []
        n, n_vars = data.shape[0], data.shape[1]
        for lag in range(1, max_lag + 1):
            for src in range(n_vars):
                for tgt in range(n_vars):
                    if src == tgt: continue
                    y = data[lag:, tgt]
                    y_lag = data[:n - lag, tgt].reshape(-1, 1)
                    x_lag = data[:n - lag, src].reshape(-1, 1)
                    # 有 x 模型
                    X_with = np.hstack([np.ones((n - lag, 1)), y_lag, x_lag])
                    pred_with, _ = CausalDiscovery._linear_regression(X_with, y)
                    resid_with = y - pred_with
                    # 无 x 模型
                    X_without = np.hstack([np.ones((n - lag, 1)), y_lag])
                    pred_without, _ = CausalDiscovery._linear_regression(X_without, y)
                    resid_without = y - pred_without
                    rss1, rss0 = np.sum(resid_with ** 2), np.sum(resid_without ** 2)
                    if rss1 < rss0 and rss0 > 0:
                        f_stat = ((rss0 - rss1) / 1) / (rss1 / max(n - lag - 3, 1))
                        if f_stat > 2.0:
                            edges.append({
                                'source': src, 'target': tgt, 'lag': lag,
                                'f_stat': round(f_stat, 2),
                                'strength': round(1 - rss1 / rss0, 3),
                            })
        return edges

    @staticmethod
    def discover_equation(x, y):
        """从数据自动发现最佳物理方程形式"""
        from scipy.optimize import curve_fit
        cand = {
            'linear':     lambda x, a, b: a * x + b,
            'exponential':lambda x, a, b: a * np.exp(b * x),
            'power_law':  lambda x, a, b: a * np.abs(x) ** b,
            'log':        lambda x, a, b: a * np.log(np.abs(x) + 1e-10) + b,
            'quadratic':  lambda x, a, b, c: a * x ** 2 + b * x + c,
        }
        best = {'form': 'none', 'score': 0, 'params': []}
        for name, func in cand.items():
            try:
                n_params = func.__code__.co_varnames.__len__() - 1
                params, _ = curve_fit(func, x, y, p0=[1.0] * n_params, maxfev=5000)
                y_pred = func(x, *params)
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                if r2 > best['score']:
                    best = {
                        'form': name, 'score': round(r2, 4),
                        'params': [round(float(p), 4) for p in params],
                    }
            except Exception:
                continue
        return best