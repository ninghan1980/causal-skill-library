"""physics_engine/geophysics.py — 地球物理 (2引擎: 地震波/大气环流)"""
import math
from .core import PhysicsEngine, PhysicsState


class SeismicWaveEngine(PhysicsEngine):
    """地震波: P波/S波速度 + 走时计算 + 震级能量"""
    def __init__(self):
        super().__init__(name="seismic_waves")
        # 典型地壳弹性参数
        self.vp_crust = 6.5   # P波速度 km/s (地壳)
        self.vs_crust = 3.8   # S波速度 km/s
        self.vp_mantle = 8.0  # 地幔P波
        self.vs_mantle = 4.5  # 地幔S波

    def wave_velocity(self, K, mu, rho):
        """P波 vp = √((K+4μ/3)/ρ), S波 vs = √(μ/ρ)"""
        vp = math.sqrt((K + 4*mu/3) / rho)
        vs = math.sqrt(mu / rho)
        return {'vp_km_s': round(vp/1000, 2), 'vs_km_s': round(vs/1000, 2),
                'vp_vs_ratio': round(vp/vs, 3),
                'K_GPa': round(K/1e9, 2), 'mu_GPa': round(mu/1e9, 2),
                'rho_kg_m3': rho}

    def travel_time(self, distance_km, depth_km=0, wave='P'):
        """直达波走时: t = d / v"""
        v = self.vp_crust if wave == 'P' else self.vs_crust
        t = distance_km / v
        return {'wave': wave, 'distance_km': distance_km,
                'depth_km': depth_km, 'velocity_km_s': v,
                'travel_time_s': round(t, 2), 'travel_time_min': round(t/60, 3)}

    def epicenter_distance(self, delta_t_s, v_p=None, v_s=None):
        """P-S时差 → 震中距: d = Δt / (1/vs - 1/vp)"""
        vp = v_p or self.vp_crust
        vs = v_s or self.vs_crust
        d = delta_t_s / (1/vs - 1/vp)
        return {'delta_t_s': delta_t_s, 'epicenter_distance_km': round(d, 2),
                'vp': vp, 'vs': vs}

    def moment_magnitude(self, M0):
        """矩震级: Mw = 2/3·log₁₀(M₀) - 10.7 (M₀单位N·m)"""
        Mw = 2/3 * math.log10(M0) - 10.7
        return {'M0_N_m': f'{M0:.2e}', 'Mw': round(Mw, 1)}

    def energy_from_magnitude(self, Mw):
        """Gutenberg-Richter: log₁₀E = 4.8 + 1.5·Mw (E单位J)"""
        logE = 4.8 + 1.5 * Mw
        E = 10**logE
        # 换算TNT当量 (1吨TNT = 4.184e9 J)
        tnt_ton = E / 4.184e9
        return {'Mw': Mw, 'energy_J': f'{E:.2e}',
                'tnt_equivalent_ton': f'{tnt_ton:.2e}',
                'tnt_equivalent_kt': round(tnt_ton/1e6, 2)}

    def intensity_at_distance(self, Mw, distance_km):
        """简化衰减: I = I₀·e^(-βr) / √r"""
        I0 = 10  # 极震区烈度基准
        beta = 0.005
        I = I0 * math.exp(-beta * distance_km) / math.sqrt(distance_km/10 + 1)
        # 用中国烈度表对应
        if I > 8: c = 'XI-XII度(毁灭)'
        elif I > 6: c = 'IX-X度(严重)'
        elif I > 4: c = 'VII-VIX度(破坏)'
        elif I > 2: c = 'V-VI度(有感/轻微)'
        else: c = 'IV度以下(基本无损)'
        return {'Mw': Mw, 'distance_km': distance_km,
                'intensity': round(I, 2), 'description': c}

    def step(self, state, dt):
        return state


class AtmosphericCirculationEngine(PhysicsEngine):
    """大气环流: 地转风 + 梯度风 + 三圈环流"""
    def __init__(self):
        super().__init__(name="atmospheric_circulation")
        self.omega = 7.2921e-5  # 地球自转角速度
        self.g = 9.81

    def coriolis(self, lat):
        """科里奥利参数 f = 2Ω·sin(φ)"""
        return 2 * self.omega * math.sin(math.radians(lat))

    def geostrophic_wind(self, dP_dx, lat, rho=1.225):
        """地转风: Vg = -(1/ρf)·∂P/∂x"""
        f = self.coriolis(lat)
        if abs(f) < 1e-10:
            return {'v_geostrophic': 0, 'note': '赤道f=0, 地转风不适用'}
        Vg = -1/(rho * f) * dP_dx
        return {'v_geostrophic_m_s': round(Vg, 2),
                'v_geostrophic_kmh': round(abs(Vg)*3.6, 1),
                'lat': lat, 'f': f, 'direction': '平行等压线(北半球右偏)'}

    def gradient_wind(self, dP_dx, lat, R, rho=1.225):
        """梯度风(弯曲等压线): V²/(R) + fV + (1/ρ)·∂P/∂x = 0"""
        f = self.coriolis(lat)
        if abs(f) < 1e-10:
            return {'v_gradient': 0, 'note': '赤道不适用'}
        # 解二次方程: V²/R + fV + dP/(rho*dx) = 0
        a = 1/R
        b = f
        c = dP_dx/rho
        disc = b**2 - 4*a*c
        if disc < 0:
            return {'v_gradient': None, 'note': '无实解(气压梯度太强)'}
        V1 = (-b + math.sqrt(disc)) / (2*a)
        V2 = (-b - math.sqrt(disc)) / (2*a)
        # 反气旋取正根, 气旋取合适的根
        V = V1 if V1 > 0 else V2
        return {'v_gradient_m_s': round(V, 2),
                'v_gradient_kmh': round(V*3.6, 1),
                'cyclone_type': '气旋' if R > 0 else '反气旋'}

    def hadley_cell(self, lat):
        """哈德来环流: 赤道上升 → 30°下沉"""
        if abs(lat) < 5:
            return {'cell': 'Hadley (equator)', 'motion': '上升(赤道辐合带)',
                    'rain': 'ITCZ热带雨带', 'pressure': '低压'}
        elif abs(lat) < 30:
            return {'cell': 'Hadley', 'motion': '下沉(副热带高压)',
                    'rain': '少雨(沙漠带)', 'pressure': '高压'}
        elif abs(lat) < 60:
            return {'cell': 'Ferrel', 'motion': '极锋带, 温带气旋活跃',
                    'rain': '中纬降水', 'pressure': '波动'}
        else:
            return {'cell': 'Polar', 'motion': '极地下沉(极地高压)',
                    'rain': '极少(极地沙漠)', 'pressure': '高压'}

    def sea_breeze(self, T_land, T_sea, c_p=1004, g=9.81):
        """海风环流: 海陆温差驱动"""
        dT = T_land - T_sea
        # 简化海风速度: v ≈ √(g·h·ΔT/T) (h~1000m)
        h = 1000
        v = math.sqrt(g * h * abs(dT) / 300)
        return {'T_land_C': T_land, 'T_sea_C': T_sea, 'delta_T': dT,
                'sea_breeze_speed_m_s': round(v, 2),
                'sea_breeze_speed_kmh': round(v*3.6, 1),
                'direction': '海→陆(白天)' if dT > 0 else '陆→海(夜晚)'}

    def step(self, state, dt):
        return state
