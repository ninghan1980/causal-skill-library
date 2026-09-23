"""physics_engine/control.py — 控制论 (2引擎: PID/反馈系统)"""
import math
from .core import PhysicsEngine, PhysicsState


class PIDControllerEngine(PhysicsEngine):
    """PID控制器: 比例-积分-微分控制"""
    def __init__(self, Kp=1.0, Ki=0.1, Kd=0.05):
        super().__init__(name="pid_controller")
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd

    def tune_zn(self, Ku, Tu):
        """Ziegler-Nichols 整定法"""
        # P: Kp=0.5Ku
        # PI: Kp=0.45Ku, Ti=Tu/1.2
        # PID: Kp=0.6Ku, Ti=Tu/2, Td=Tu/8
        return {
            'P': {'Kp': 0.5 * Ku},
            'PI': {'Kp': 0.45 * Ku, 'Ki': 0.45 * Ku / (Tu / 1.2)},
            'PID': {'Kp': 0.6 * Ku, 'Ki': 0.6 * Ku / (Tu / 2), 'Kd': 0.6 * Ku * (Tu / 8)},
            'note': 'Ziegler-Nichols临界比例带法',
        }

    def simulate(self, setpoint, initial, dt=0.1, nsteps=100, plant_type='first_order'):
        """模拟PID控制响应"""
        # plant_type: 'first_order' (一阶系统), 'second_order' (二阶), 'integrator' (积分)
        y = initial
        integral = 0
        prev_error = setpoint - initial
        history = []
        for step in range(nsteps):
            error = setpoint - y
            integral += error * dt
            derivative = (error - prev_error) / dt
            u = self.Kp * error + self.Ki * integral + self.Kd * derivative
            # 简单一阶系统: dy/dt = (u - y) / tau
            tau = 1.0
            dy = (u - y) / tau * dt
            y += dy
            prev_error = error
            history.append({'step': step, 'time': round(step*dt, 2),
                           'output': round(y, 4), 'error': round(error, 4),
                           'control': round(u, 4)})
        # 性能指标
        overshoot = max(0, (max(h['output'] for h in history) - setpoint) / setpoint * 100)
        settling_time = next((h['time'] for h in history if abs(h['output'] - setpoint) < 0.02 * setpoint), None)
        return {'setpoint': setpoint, 'initial': initial,
                'overshoot_pct': round(overshoot, 2),
                'settling_time': settling_time,
                'final_output': round(history[-1]['output'], 4),
                'history': history[:20],
                'summary': f'PID: 超调{overshoot:.1f}%, 调节时间{settling_time}s, 终值{history[-1]["output"]:.3f}'}

    def step(self, state, dt):
        return state


class FeedbackSystemEngine(PhysicsEngine):
    """反馈系统: 传递函数/稳定性/根轨迹"""
    def __init__(self):
        super().__init__(name="feedback_system")

    def closed_loop(self, G, H=1):
        """闭环传递函数: T = G / (1 + GH)"""
        # G, H 为 (num, den) 元组
        # 简化: 只返回特征方程
        return {'open_loop': G, 'feedback': H,
                'characteristic_eq': f'1 + G(s)H(s) = 0',
                'note': '稳定性由特征根实部决定'}

    def routh_hurwitz(self, coeffs):
        """劳斯-赫尔维茨稳定性判据"""
        n = len(coeffs) - 1
        if n < 2:
            return {'stable': True, 'note': '低阶系统'}
        # 构建劳斯表 (确保两行长度一致)
        routh = [[], []]
        for i in range(len(coeffs)):
            if i % 2 == 0:
                routh[0].append(coeffs[i])
            else:
                routh[1].append(coeffs[i])
        # 补齐较短行
        while len(routh[0]) > len(routh[1]):
            routh[1].append(0)
        while len(routh[1]) > len(routh[0]):
            routh[0].append(0)
        # 计算后续行
        for i in range(2, n + 1):
            row = []
            for j in range(len(routh[0]) - 1):
                if abs(routh[i-1][0]) < 1e-12:
                    return {'stable': False, 'note': '首列为零, 系统临界稳定'}
                val = (routh[i-1][0] * routh[i-2][j+1] - routh[i-2][0] * routh[i-1][j+1]) / routh[i-1][0]
                row.append(val)
            routh.append(row)
            # 补齐后续行
            while len(row) < len(routh[0]):
                row.append(0)
        # 检查首列符号
        first_col = [row[0] for row in routh if row]
        sign_changes = sum(1 for i in range(1, len(first_col)) if first_col[i] * first_col[i-1] < 0)
        return {'stable': sign_changes == 0,
                'sign_changes': sign_changes,
                'first_col': [round(x, 4) for x in first_col],
                'note': f'劳斯表首列变号{sign_changes}次'}

    def bode_plot(self, G_type='first_order', params=None):
        """波特图: 幅频/相频特性"""
        if G_type == 'first_order':
            tau = params.get('tau', 1.0)
            freqs = [0.01, 0.1, 1, 10, 100]
            response = []
            for w in freqs:
                mag = 1 / math.sqrt(1 + (w * tau)**2)
                phase = -math.atan(w * tau) * 180 / math.pi
                response.append({'freq': w, 'magnitude_dB': round(20*math.log10(mag), 2),
                                'phase_deg': round(phase, 1)})
            return {'type': '一阶惯性', 'tau': tau, 'response': response}
        return {'type': G_type, 'note': '未实现'}

    def step(self, state, dt):
        return state
