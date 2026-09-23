"""physics_engine/coupled.py — L3 多物理场耦合引擎
① TyphoonOceanCoupling: 台风(流体)↔海洋(热) 双向耦合
   暖海面供能增强台风; 强风搅拌+蒸发 冷却海表 削弱台风(风-海负反馈)
② ThermalStressEngine: 热(场)↔固体(力) 耦合
   温度变化→热应变→约束热应力→是否屈服/变形
"""
import math
import numpy as np
from .core import PhysicsEngine, PhysicsState
from .physical_constants import Atmosphere, Water, get_material, THERMAL


# ============ ① 台风-海洋热交换耦合 ============
class TyphoonOceanCoupling(PhysicsEngine):
    """台风-海洋双向耦合 (简化风-海热力反馈)
    正向: 海温越暖(O>26.5°C) → 蒸发供能 → 台风强度↑
    逆向: 台风风应力搅拌海洋 / 冷涌(上升流) → 海表降温 → 强度↓
    耦合方程: dI/dp ∝ (SST-T_26.5)·A_warm · 供能 - 风搅拌冷却
    """
    SST_CRIT = 26.5  # °C, 台风生成海温阈值

    def __init__(self, lat=20.0, lon=128.0, mixing_coef=0.15, energy_coef=4.0):
        super().__init__(name="typhoon_ocean_coupling")
        self.lat, self.lon = lat, lon
        self.mixing_coef = mixing_coef    # 风搅拌冷却强度
        self.energy_coef = energy_coef    # 暖海供能系数

    def set_sea_temps(self, sst_grid):
        """设置真实海温场(°C网格或单点函数), 可接入卫星SST"""
        self.sst_grid = sst_grid  # 若为函数则实时计算

    def sst_at(self, lon, lat, t):
        """当前海温: 优先用场/函数, 否则用典型纬度公式伪真实"""
        if hasattr(self, 'sst_grid'):
            if callable(self.sst_grid):
                return self.sst_grid(lon, lat, t)
            return self.sst_grid
        # 简化真实: 西北太平洋副热带海温(纬度+季节)
        base = 28.0 - 0.03*(lat-20) - 0.02*(lon-125)
        seasonal = 2.0 * math.sin(2*math.pi*(t/365.25/86400 - 0.12))  # 8月暖
        return max(24, min(31, base + seasonal))

    def step(self, state, dt):
        """state需含: wind_speed(最大风速m/s), sst_initial(初始海温), time"""
        ws = state.extra.get('wind_speed', 30.0)
        t = state.time
        sst = self.sst_at(self.lon, self.lat, t)
        # 正向供能: 海温超过临界越多, 蒸发供能越强 → 台风增强
        warm_energy = max(0.0, sst - self.SST_CRIT) * self.energy_coef
        # 逆向冷却: 风应力搅拌+蒸发吸热, 风速是主驱动(与海温弱依赖)
        if sst > self.SST_CRIT:
            cooling_rate = self.mixing_coef * (ws/35.0)**2  # 每天降温幅度
        else:
            cooling_rate = 0  # 低于临界不再台风驱动冷却
        sst_new = sst - cooling_rate * dt/86400
        sst_new = max(sst_new, self.SST_CRIT - 0.5)  # 下限(不会无限降)
        # 台风强度变化: 供能与冷却平衡
        d_ws = (warm_energy - cooling_rate*5.0) * dt/86400
        ws_new = max(15.0, min(55.0, ws + d_ws))
        s_new = PhysicsState(temperature=sst_new, pressure=state.pressure,
                             position=state.position)
        s_new.extra = {'wind_speed': ws_new, 'sst': sst_new, 'intensity_change_per_day': d_ws*86400}
        s_new.time = t + dt
        return s_new

    def classify(self, ws):
        if ws < 17.2: return "TD 热带低压"
        if ws < 24.5: return "TS 热带风暴"
        if ws < 32.7: return "STS 强热带风暴"
        if ws < 41.5: return "TY 台风"
        return "STY 超强台风"

    def predict(self, state, dt, steps):
        current = state
        states, times = [current], [current.time]
        for _ in range(steps):
            nxt = self.step(current, dt)
            nxt.time = current.time + dt
            states.append(nxt); times.append(nxt.time)
            current = nxt
        from .core import PredictionResult
        return PredictionResult(states=states, uncertainties=[0]*len(states), time_points=times)


# ============ ② 热应力-变形耦合 ============
class ThermalStressEngine(PhysicsEngine):
    """热应力耦合: 温度变化 → 热应变 → 约束热应力 → 屈服判断
    自由膨胀: ε = α·ΔT (无应力)
    约束(固定两端/嵌入): σ = E·α·ΔT (热应力)
    若σ≥屈服 → 屈服/永久变形
    """
    def __init__(self):
        super().__init__(name="thermal_stress")

    def thermal_strain(self, mat, dT):
        """自由热应变 ε = α·ΔT"""
        m = get_material(mat)
        alpha = m.get('cte')
        if alpha is None: return {'dT':dT, 'strain':None, 'note':'材料无热膨胀数据'}
        return {'dT':dT, 'alpha':alpha, 'strain':alpha*dT, 'material':m['name'], 'E':m.get('E_GPa')}

    def constrained_stress(self, mat, dT):
        """约束热应力 σ = E·α·ΔT (MPa)
        正值=膨胀受压(约束阻碍膨胀), 负=收缩受拉"""
        m = get_material(mat)
        alpha = m.get('cte'); E = m.get('E_GPa')
        if alpha is None or E is None: return {'stress_MPa':None,'note':'缺材料数据'}
        stress_MPa = E*1e3 * alpha * dT   # alpha(1/K), E(GPa), dT(K) → MPa
        yield_MPa = m.get('yield_MPa')
        if yield_MPa:
            elastic = abs(stress_MPa) < yield_MPa
        else:
            elastic = None
        return {
            'material':m['name'], 'dT_K':dT, 'stress_MPa':round(stress_MPa,1),
            'yield_MPa':yield_MPa, 'elastic':elastic,
            'verdict': '弹性' if elastic else ('屈服/塑性变形' if elastic is False else '未知'),
            'sign':'膨胀受压' if dT>0 else '收缩受拉',
        }

    def elongation(self, mat, dT, L0):
        """自由膨胀伸长量 ΔL = α·ΔT·L0"""
        m = get_material(mat); alpha=m.get('cte')
        if alpha is None: return None
        return round(alpha*dT*L0*1000, 2)  # mm

    def temperature_change_for_yield(self, mat):
        """材料热应力达到屈服所需温变 ΔT_yield = σy/(E·α)"""
        m=get_material(mat); alpha=m.get('cte'); E=m.get('E_GPa'); y=m.get('yield_MPa')
        if not (alpha and E and y): return None
        return round(y/(E*1e3*alpha), 1)  # K


if __name__ == '__main__':
    import json
    # 测试1: 台风-海洋耦合
    print("="*64)
    print("L3 耦合引擎 ① 台风-海洋热交换")
    print("="*64)
    te = TyphoonOceanCoupling(lat=22, lon=128)
    import datetime
    # 8月中旬
    s0 = PhysicsState(temperature=301, pressure=965, time=0)
    s0.extra = {'wind_speed':35.0, 'sst':28.5}
    r = te.predict(s0, dt=3600*6, steps=16)  # 4天
    print(" 时间(h) | 最大风速(m/s) | 海温(°C) | 等级")
    for i, st in enumerate(r.states):
        if i % 4 == 0 or i==len(r.states)-1:
            ws=st.extra['wind_speed']; sst=st.extra['sst']
            print(f"  {i*6:>6} | {ws:>8.1f} | {sst:>7.1f} | {te.classify(ws)}")

    # 测试2: 热应力
    print("\n"+"="*64)
    print("L3 耦合引擎 ② 热应力-变形 (钢轨/混凝土梁)")
    print("="*64)
    tse = ThermalStressEngine()
    for mat in ['steel_Q235','concrete_C40','glass','aluminum_6061']:
        res = tse.constrained_stress(mat, 30)  # 温升30K
        yield_dT = tse.temperature_change_for_yield(mat)
        print(f"  {mat:>14}: ΔT=30K → 应力{res['stress_MPa']:.0f}MPa / 屈服{res['yield_MPa']}MPa → {res['verdict']}  (屈服临界ΔT={yield_dT}K)")


# ============ ③ 爆轰-结构耦合 (冲击波超压 → 建筑破坏响应) ============
class BlastStructureCoupling(PhysicsEngine):
    """爆炸冲击波 ↔ 建筑结构 耦合
    输入: 爆炸TNT当量W(kg), 目标距爆心R(m), 结构类型, 标定方式
    物理: 超压 P(R) 沿爆炸冲击波衰减 → 结构响应 → 与L2真实破坏判据对比
    标定(calib):
      'kb'         → Kingery-Bulmash 标准爆炸(自由空气/开放地形)
      'constraint' → 84/P 约束/近地表爆炸(建筑群/平房内), 衰减慢传播远(工程常用)
    这是 流体(冲击波场) + 固体(结构) 两域耦合
    """
    def __init__(self, k_ground=1.8, calib='constraint'):
        super().__init__(name="blast_structure_coupling")
        self.k_ground = k_ground    # 地面反射
        self.calib = calib

    def overpressure_at(self, W, R):
        """给定当量W(kg)与距离R(m), 峰值超压kPa
        calib='kb': 标准KB表; 'constraint': 84/P标定(约束/近地表爆炸)"""
        if self.calib == 'constraint':
            # 84/P 约束爆炸: P = 84·W^(1/3)/R (教授工程标定, 冲击波传播比自由空气远)
            return min(3000, 84.0 * W**(1/3) / R * self.k_ground**0)  
        # 默认 KB 表
        import numpy as np
        Weff = W * self.k_ground
        Z = R / Weff**(1/3)
        KB = np.array([
            [0.5,3000],[0.6,2100],[0.7,1500],[0.8,1100],[0.9,850],[1.0,680],
            [1.1,560],[1.2,470],[1.3,400],[1.4,340],[1.5,290],[1.6,250],
            [1.7,220],[1.8,190],[1.9,170],[2.0,150],[2.2,120],[2.4,100],
            [2.6,85],[2.8,72],[3.0,62],[3.2,54],[3.4,47],[3.6,42],
            [3.8,37],[4.0,33],[4.5,26],[5.0,21],[5.5,17],[6.0,14],
            [7.0,10.5],[8.0,8.3],[9.0,6.8],[10.0,5.7],[12.0,4.3],
            [14.0,3.4],[16.0,2.8],[20.0,2.0],[24.0,1.5],[30.0,1.1]])
        zt=KB[:,0]; pt=KB[:,1]
        if Z <= zt[0]: return pt[0]
        if Z >= zt[-1]: return pt[-1]
        return float(np.exp(np.interp(np.log(Z), np.log(zt), np.log(pt))))

    def struct_verdict(self, P_kPa, structure):
        """超压P(kPa) → 该结构的破坏等级(用L2真实判据)"""
        from .physical_constants import BUILDING
        if structure in ('glass','window','fenestration'):
            if P_kPa >= BUILDING['fenestration_glass_breaking_kPa']:
                return '玻璃破碎'; 
            return '完好'
        if structure in ('brick','brick_wall','wall'):
            if P_kPa >= BUILDING['brick_wall_collapse_kPa']:
                return '砖墙坍塌'
            if P_kPa >= BUILDING['concrete_wall_crack_kPa']:
                return '墙体开裂'
            if P_kPa >= BUILDING['window_frame_damage_kPa']:
                return '窗框损坏'
            if P_kPa >= BUILDING['fenestration_glass_breaking_kPa']:
                return '玻璃破碎'
            return '基本完好'
        if structure in ('concrete','rc','frame','beam','column'):
            if P_kPa >= BUILDING['brick_wall_collapse_kPa']:
                return '承重结构严重损坏(坍塌风险)'
            if P_kPa >= BUILDING['concrete_wall_crack_kPa']:
                return '混凝土结构开裂'
            return '结构基本完好'
        # 人(若structure='human')
        if structure in ('human','person','ear'):
            if P_kPa >= BUILDING['human_50killed_kPa']: return '致死(~50%)'
            if P_kPa >= BUILDING['human_ear_damage_kPa']: return '耳膜损伤'
            return '安全'
        return '未知结构'

    def predict(self, state, dt, steps):
        # state.extra: {W_kg, R_m, structure}
        from .core import PredictionResult
        W = state.extra.get('W_kg', 100.0); R = state.extra.get('R_m', 30.0)
        structure = state.extra.get('structure','brick')
        # 超压随略变R(模拟冲击波穿过建筑群衰减)
        results = []
        for i in range(steps+1):
            Ri = R + i*dt*0.5   # 冲击波传播
            P = self.overpressure_at(W, Ri)
            verdict = self.struct_verdict(P, structure)
            results.append({'R':round(Ri,1),'P_kPa':round(P,1),'verdict':verdict})
        s = PhysicsState(temperature=300, extra=results[-1])
        return PredictionResult(states=[s]* (steps+1), uncertainties=[0]*(steps+1),
                                time_points=list(range(steps+1)))


# ============ ④ 风-建筑耦合 (风荷载 → 结构应力) ============
class WindStructureCoupling(PhysicsEngine):
    """风(流) ↔ 建筑(固体) 耦合 — 工程抗风设计工具 (v2)
    承载能力设计法: 阵风风压(考虑阵风系数/体型系数) vs 各部位允许承载
    输出: 全部位安全评估 + 薄弱点 + 抗风结论
    用L2真实: 空气密度(海拔/温度)/阵风系数/体型系数/各部位允许风压
    """
    def __init__(self, alt=0.0, gust_factor=None):
        super().__init__(name="wind_structure_coupling")
        self.alt = alt
        from .physical_constants import WIND_DESIGN
        self.gust_factor = gust_factor or WIND_DESIGN['gust_factor']

    def real_air_density(self, v_avg):
        """真实空气密度(海拔+温度), 温度用真实感知"""
        from .physical_constants import Atmosphere
        from .real_world import sensor_env
        try:
            env = sensor_env()
            T_c = env['T_env_C']
        except Exception:
            T_c = 25.0
        return Atmosphere.density(self.alt, T_c)

    def design_wind_pressure(self, v, part='wall', rho=None):
        """设计风压 q = ½·ρ·(阵风V)²·μs (kPa), 用承载设计法"""
        from .physical_constants import get_shape_coef
        rho = rho or self.real_air_density(v)
        V_gust = v * self.gust_factor   # 阵风风速
        mu_s = get_shape_coef(part)
        q = 0.5 * rho * V_gust**2 * abs(mu_s) / 1000   # kPa
        return {'q_kPa':round(q,3), 'V_gust':round(V_gust,1),
                'mu_s':mu_s, 'rho':round(rho,3), 'part':part}

    def assess(self, v, parts=None, rho=None):
        """多部位抗风评估: 设计(阵)风压 vs 允许承载
        返回各部位 verdict + 全局薄弱点"""
        from .physical_constants import get_shape_coef, get_allowable_wind, WIND_DESIGN
        rho = rho or self.real_air_density(v)
        if parts is None:
            parts = list(WIND_DESIGN['allowable_wind_pressure_kPa'].keys())
        results = []
        weak = []
        for part in parts:
            allow = get_allowable_wind(part)
            q = self.design_wind_pressure(v, part, rho)['q_kPa']
            # 安全判断: 设计风压 vs 允许
            margin = allow / q if q>0 else float('inf')
            if q > allow:          verdict = '危险(超承载)'
            elif margin < 1.5:     verdict = '临界(注意)'
            elif margin < 3.0:     verdict = '基本安全'
            else:                  verdict = '安全(富余)'
            results.append({'part':part,'q_kPa':q,'allow_kPa':allow,
                            'safety_factor':round(margin,2),'verdict':verdict})
            if '危险' in verdict or '临界' in verdict:
                weak.append(part)
        return {'results':results, 'weak_points':weak,
                'V_gust':round(v*self.gust_factor,1), 'rho':round(rho,3),
                'gust_factor':self.gust_factor}

    def wind_pressure(self, v, part='wall', rho=None):
        """(兼容) 平均风压(非设计)"""
        rho = rho or self.real_air_density(v)
        from .physical_constants import get_shape_coef
        return 0.5*rho*v**2*abs(get_shape_coef(part))

    def wall_stress_from_wind(self, v, height=3.0, thickness=0.24, material='brick'):
        """(保留) 墙体弯曲应力(参考)"""
        from .physical_constants import get_material
        q = self.wind_pressure(v)
        M = q * height**2 / 2
        Z = 1.0 * thickness**2 / 6
        sigma = M / Z
        mat = get_material(material)
        y = mat.get('yield_MPa')
        sf = y*1e6/sigma if (y and sigma>0) else None
        return {'q_Pa':q,'q_kPa':round(q/1000,2),'sigma_MPa':round(sigma/1e6,3),
                'yield_MPa':y,'safety_factor':round(sf,2) if sf else None,
                'verdict':'安全' if (sf and sf>1) else ('屈服' if sf is not None else '未知')}

    def wind_force_on_building(self, v, area, part='wall'):
        """整面风荷载 F = q·A"""
        q = self.design_wind_pressure(v, part)['q_kPa']*1000
        return {'q_design_kPa':round(q/1000,3),'force_kN':round(q*area/1000,1)}

    def weakest_point(self, v):
        """找最薄弱部位 + 抗风结论"""
        a = self.assess(v)
        wp = a['weak_points']
        verdict = ''
        if 'light' in wp:
            verdict = '台风下需加固临时/轻型结构'
        elif any(x in wp for x in ['window', 'frame']):
            verdict = '主要风险在门窗玻璃'
        elif wp:
            verdict = '薄弱部位:' + ','.join(wp)
        else:
            verdict = '整体抗风安全(富余)'
        return {'wind_speed_avg': v, 'gust_speed': a['V_gust'], 'rho': a['rho'],
                'weak_points': a['weak_points'], 'verdict': verdict, 'gust_factor': self.gust_factor}

# ============ ⑤ 燃烧-热-固耦合 (火 → 温升 → 结构热应力) ============
class FireThermalCoupling(PhysicsEngine):
    """燃烧(化学/热) ↔ 固体结构 耦合
    输入: 火灾热释放率HRR(W/m²), 构件材料, 时间
    物理: 燃烧释热 → 高斯火源温度场 → 构件温升 → 热膨胀/热应力 → 结构损伤
    用L2真实: 材料比热/导热/热膨胀/屈服
    """
    def __init__(self):
        super().__init__(name="fire_thermal_coupling")

    def fire_temp_field(self, HRR_kW_m2, max_T=1000):
        """火灾热释放率 → 稳态温升(简化热释放-温度关系)"""
        import math
        # 简化: 温度与热释放率对数关系(火灾工程常用)
        return min(max_T, 293 + 60*math.log1p(HRR_kW_m2*100))

    def temp_rise_time(self, HRR, material, target_dT, L=0.2):
        """构件中心温升到target_dT需要的时间: 半无限体热扩散近似"""
        from .physical_constants import THERMAL, get_material
        mat = get_material(material)
        k = mat.get('k'); rho = mat.get('density'); c_ = THERMAL.get('c_steel',500)
        if material in ('concrete_C25','concrete_C40','brick'): c_ = THERMAL.get('c_concrete',880)
        alpha = k/(rho*c_) if (k and rho) else 1e-6     # 热扩散系数
        # 一维热进入: x被加热深度, 粗略 t = L²/(π·α)
        return L**2 / (math.pi * alpha)   # 秒

    def heat_stress_at(self, dT, material):
        """温升dT → 约束热应力 MPa (复用ThermalStressEngine)"""
        from .physical_constants import get_material
        mat = get_material(material)
        alpha = mat.get('cte'); E = mat.get('E_GPa'); y = mat.get('yield_MPa')
        if not (alpha and E): return {'stress_MPa':None,'note':'缺材料数据'}
        stress = E*1e3*alpha*dT
        return {'stress_MPa':round(stress,1),'yield_MPa':y,
                'elastic':abs(stress)<y if y else None,
                'verdict':'弹性' if (y and abs(stress)<y) else ('屈服' if y else '未知')}


# ============ ⑥ 质量-动量-热耦合 (流体输运 + 热) ============
class AdvectionDiffusionConvection(PhysicsEngine):
    """热(对流/扩散) 在运动中(流)输运: 对流-扩散耦合
    如一维传热管道中同时有: 热扩散(∂²T/∂x²) + 流体平流(-u·∂T/∂x)
    加外部散热(牛顿冷却) → 输运+散热耦合
    """
    def __init__(self, u=0.02, alpha=1e-4, dx=0.1, h_loss=0.0, T_amb=298.0):
        super().__init__(name="advection_diffusion")
        self.u, self.alpha, self.dx = u, alpha, dx
        self.h, self.T_amb = h_loss, T_amb

    def cn_step(self, T, nx, dt):
        """Crank-Nicolson + 迎风差分 求解 对流-扩散 方程 (稳定)
        ∂T/∂t = α∂²T/∂x² - u∂T/∂x - h(T-T_amb)
        平流用迎风(upwind), 保证任何Péclet数稳定
        """
        import numpy as np
        r = self.alpha*dt/self.dx**2
        cfl = self.u*dt/self.dx     # 对流CFL数
        A = np.zeros((nx,nx)); b = T.copy()
        for i in range(nx):
            A[i,i] = 1 + 2*r + self.h*dt + (cfl if self.u>0 else -cfl)
            if i>0:    A[i,i-1] = -r - (cfl if self.u>0 else 0)
            if i<nx-1: A[i,i+1] = -r + (0 if self.u>0 else cfl)
            b[i] += self.h*dt*self.T_amb
        sol = np.linalg.solve(A,b)
        if self.u>0 and sol[0]>sol[1]:
            sol[0]=sol[1]  # 入口边界(防震荡)
        return sol


# ============ ⑦ 多物理耦合器 (通用框架) ============
class MultiPhysicsCoupler:
    """多物理耦合器 — 让任意引擎按物理规律交换数据
    核心: 每个引擎声明"输出量纲", 耦合器自动匹配"输入量纲"建立数据流
    支持: 热-流-固-电磁 任意两两/链式耦合
    """
    # 物理量纲 → 单位 (量纲匹配表)
    QUANTITIES = {
        'temperature': {'unit': 'K', 'aliases': ['T', 'temp', 'temperature']},
        'pressure':    {'unit': 'Pa', 'aliases': ['P', 'pressure', 'stress', 'sigma']},
        'velocity':    {'unit': 'm/s', 'aliases': ['v', 'velocity', 'u', 'wind_speed']},
        'force':       {'unit': 'N', 'aliases': ['F', 'force', 'load']},
        'displacement':{'unit': 'm', 'aliases': ['x', 'displacement', 'deformation']},
        'heat_flux':   {'unit': 'W/m²', 'aliases': ['q', 'heat_flux', 'HRR']},
        'electric_field':{'unit': 'V/m', 'aliases': ['E', 'electric_field']},
        'magnetic_field':{'unit': 'T', 'aliases': ['B', 'magnetic_field']},
        'density':     {'unit': 'kg/m³', 'aliases': ['rho', 'density']},
    }

    def __init__(self):
        self.engines = {}
        self.couplings = []
        self.history = []

    def register(self, name, engine, outputs=None, inputs=None):
        """注册引擎 + 声明其输出/输入量纲"""
        self.engines[name] = {
            'engine': engine,
            'outputs': outputs or {},
            'inputs': inputs or {},
            'state': None,
        }

    def couple(self, from_eng, to_eng, from_qty, to_qty, transfer_func=None):
        """建立耦合: from引擎的输出 → to引擎的输入
        transfer_func: 传递函数 (默认直接赋值), 可自定义 (如热→力: 温度→热应力)
        """
        self.couplings.append({
            'from': from_eng, 'to': to_eng,
            'from_qty': from_qty, 'to_qty': to_qty,
            'transfer': transfer_func or (lambda x: x),
        })

    def auto_couple(self):
        """自动耦合: 扫描所有引擎的输出/输入, 按量纲匹配"""
        for n1, e1 in self.engines.items():
            for n2, e2 in self.engines.items():
                if n1 == n2:
                    continue
                for oq in e1['outputs'].values():
                    for iq in e2['inputs'].values():
                        if self._match(oq, iq):
                            self.couples.append({'from': n1, 'to': n2,
                                'from_qty': oq, 'to_qty': iq,
                                'transfer': lambda x: x, 'auto': True})

    def _match(self, qty1, qty2):
        """量纲匹配"""
        q1 = self.QUANTITIES.get(qty1, {})
        q2 = self.QUANTITIES.get(qty2, {})
        return q1.get('unit') == q2.get('unit')

    def step(self, dt):
        """一步耦合推进: 各引擎独立步 → 交换耦合数据 → 更新边界"""
        results = {}
        # 1) 各引擎独立推进
        for name, eng in self.engines.items():
            e = eng['engine']
            if hasattr(e, 'step') and eng['state'] is not None:
                eng['state'] = e.step(eng['state'], dt)
            results[name] = eng['state']
        # 2) 按耦合关系交换数据
        for c in self.couplings:
            src = self.engines[c['from']]
            dst = self.engines[c['to']]
            if src['state'] is None or dst['state'] is None:
                continue
            val = self._extract(src['state'], c['from_qty'])
            if val is not None:
                transferred = c['transfer'](val)
                self._inject(dst['state'], c['to_qty'], transferred)
        self.history.append(results)
        return results

    def _extract(self, state, qty):
        """从状态提取物理量"""
        m = state.extra or {}
        # 先查 extra
        if qty in m: return m[qty]
        # 再查常用字段
        qty_map = {'temperature': 'temperature', 'pressure': 'pressure',
                   'velocity': 'velocity', 'density': 'density'}
        if qty in qty_map and hasattr(state, qty_map[qty]):
            return getattr(state, qty_map[qty])
        return None

    def _inject(self, state, qty, val):
        """注入物理量到状态"""
        if state.extra is None:
            state.extra = {}
        state.extra[qty] = val

    def run(self, n_steps, dt):
        """运行多步"""
        for _ in range(n_steps):
            self.step(dt)
        return self.history

    def status(self):
        """耦合状态"""
        return {
            'n_engines': len(self.engines),
            'n_couplings': len(self.couplings),
            'engines': list(self.engines.keys()),
            'couplings': [f"{c['from']}.{c['from_qty']} → {c['to']}.{c['to_qty']}" for c in self.couplings],
        }
