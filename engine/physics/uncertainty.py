"""physics_engine/uncertainty.py — 不确定性量化"""
import numpy as np
from typing import Dict, Callable, Tuple
from .core import PhysicsEngine, PhysicsState, MonteCarloEngine


class UncertaintyQuantifier:
    """不确定性量化器"""
    def __init__(self, engine):
        self.engine = engine
        self.mc = MonteCarloEngine(engine)

    def ensemble_prediction(self, state, dt, steps, param_ensemble=None):
        """多参数组合集成预测"""
        param_ensemble = param_ensemble or [
            {'k': 0.010, 'T_env': 296},
            {'k': 0.015, 'T_env': 298},
            {'k': 0.020, 'T_env': 300},
        ]
        all_trajs = []
        for params in param_ensemble:
            eng = self.engine.__class__(**params)
            all_trajs.append([s.temperature for s in eng.predict(state, dt, steps).states])
        arr = np.array(all_trajs)
        return {
            'mean': np.mean(arr, axis=0).tolist(),
            'std': np.std(arr, axis=0).tolist(),
            'min': np.min(arr, axis=0).tolist(),
            'max': np.max(arr, axis=0).tolist(),
            'ensemble_size': len(param_ensemble),
        }

    def sensitivity_analysis(self, state, param_name, param_range, n_points=5,
                             dt=1.0, steps=10):
        """参数敏感性分析"""
        values = np.linspace(*param_range, n_points)
        finals = []
        for val in values:
            if hasattr(self.engine, param_name):
                setattr(self.engine, param_name, val)
            finals.append(self.engine.predict(state, dt, steps).states[-1].temperature)
        if values[-1] != values[0]:
            sens = (max(finals) - min(finals)) / (values[-1] - values[0])
        else:
            sens = 0
        return {
            'param_values': values.tolist(),
            'final_temperatures': finals,
            'sensitivity': round(float(sens), 4),
            'interpretation': f"{param_name} 变化 {values[-1]-values[0]:.2f} → 结果变化 {max(finals)-min(finals):.2f} (敏感度 {sens:.4f})",
        }

    def bootstrap_uncertainty(self, data, n_bootstrap=1000, statistic=None):
        """Bootstrap 不确定性估计"""
        if statistic is None:
            statistic = np.mean
        estimates = [statistic(np.random.choice(data, len(data), replace=True)) for _ in range(n_bootstrap)]
        return {
            'estimate': float(np.mean(estimates)),
            'ci_90': {'lower': float(np.percentile(estimates, 5)), 'upper': float(np.percentile(estimates, 95))},
            'ci_95': {'lower': float(np.percentile(estimates, 2.5)), 'upper': float(np.percentile(estimates, 97.5))},
            'std': float(np.std(estimates)),
        }