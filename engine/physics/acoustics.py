"""physics_engine/acoustics.py — 声学 (2引擎: 声场/噪声/建筑声学)"""
import math
from .core import PhysicsEngine, PhysicsState

# 常量
K_B = 1.381e-23
RHO_AIR = 1.225       # 空气密度 kg/m³
C_AIR = 343.0         # 声速 m/s (20°C)
P_REF = 20e-6         # 参考声压 Pa (人耳阈值)
I_REF = 1e-12         # 参考声强 W/m²


class SoundFieldEngine(PhysicsEngine):
    """声场: 声压级/声强/传播/多普勒"""
    def __init__(self):
        super().__init__(name="sound_field")

    def spl(self, p_rms):
        """声压级: Lp = 20·log10(p/p_ref) dB"""
        if p_rms <= 0:
            return None
        return 20 * math.log10(p_rms / P_REF)

    def sil(self, I):
        """声强级: LI = 10·log10(I/I_ref) dB"""
        if I <= 0:
            return None
        return 10 * math.log10(I / I_REF)

    def intensity_from_spl(self, Lp):
        """从声压级求声强: I = I_ref · 10^(L/10)"""
        return I_REF * 10**(Lp / 10)

    def spherical_spreading(self, Lp1, r1, r2):
        """球面扩展衰减: Lp2 = Lp1 - 20·log10(r2/r1)"""
        return Lp1 - 20 * math.log10(r2 / r1)

    def inverse_square(self, I1, r1, r2):
        """声强随距离平方反比: I2 = I1 · (r1/r2)²"""
        return I1 * (r1 / r2)**2

    def doppler(self, f0, v_source, v_observer, v_sound=343):
        """多普勒效应: f' = f₀·(v+v_o)/(v-v_s)"""
        f_observed = f0 * (v_sound + v_observer) / (v_sound - v_source)
        return {'f0': f0, 'f_observed': round(f_observed, 2),
                'shift': round(f_observed - f0, 2),
                'v_source': v_source, 'v_observer': v_observer}

    def standing_wave(self, L, n, v=343):
        """驻波: f_n = n·v/2L (两端固定)"""
        f = n * v / (2 * L)
        wavelength = 2 * L / n
        return {'harmonic': n, 'frequency': round(f, 2),
                'wavelength': round(wavelength, 4),
                'length': L}

    def step(self, state, dt):
        return state


class NoiseEngine(PhysicsEngine):
    """噪声: 评价/控制/建筑声学"""
    def __init__(self):
        super().__init__(name="noise_control")

    def equivalent_level(self, levels, times):
        """等效连续声级: Leq = 10·log10(1/T · Σ t_i·10^(L_i/10))"""
        total_time = sum(times)
        weighted_sum = sum(t * 10**(L/10) for L, t in zip(levels, times))
        Leq = 10 * math.log10(weighted_sum / total_time)
        return {'Leq_dB': round(Leq, 2),
                'total_time': total_time,
                'note': '能量平均声级'}

    def noise_rating(self, Leq):
        """噪声评价 (NR/NC)"""
        if Leq < 30:
            return '极安静(录音棚/病房)'
        elif Leq < 40:
            return '安静(图书馆/卧室)'
        elif Leq < 50:
            return '一般(办公室/教室)'
        elif Leq < 60:
            return '较吵(商场/餐厅)'
        elif Leq < 70:
            return '吵闹(工厂/交通)'
        elif Leq < 80:
            return '很吵(摇滚乐/机场)'
        else:
            return '极吵(损伤听力)'

    def sound_absorption(self, S, alpha):
        """吸声量: A = S·α (m²)"""
        return {'area': S, 'absorption_coeff': alpha,
                'absorption_m2': round(S * alpha, 2),
                'note': '赛宾吸声量'}

    def reverberation_time(self, V, A):
        """混响时间 (赛宾公式): T60 = 0.161·V/A"""
        if A <= 0:
            return None
        T60 = 0.161 * V / A
        return {'volume_m3': V, 'absorption_m2': round(A, 2),
                'T60_s': round(T60, 3),
                'note': '声能衰减60dB所需时间'}

    def transmission_loss(self, m, f):
        """隔声量 (质量定律): TL ≈ 20·log10(m·f) - 47"""
        if m <= 0 or f <= 0:
            return None
        TL = 20 * math.log10(m * f) - 47
        return {'surface_density': m, 'frequency': f,
                'TL_dB': round(TL, 2),
                'note': '单层匀质墙隔声量'}

    def step(self, state, dt):
        return state
