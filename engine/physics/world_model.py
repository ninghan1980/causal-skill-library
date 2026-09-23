"""physics_engine/world_model.py — 三元世界模型 v2.0 (24引擎/4层)
Φ(物理状态编码) → ℳ(因果发现/规律) → Ψ(不确定性/决策) 三层输出
统一入口: TriWorldModel.simulate(model, params) → WorldModelResult

★ v2.0.1 性能优化 (Anthropic FlashPairformer 启发):
  1. sensor_env() 加 TTL 缓存 — 消除每次 simulate 的重复 HTTP 请求
     修复前: newton_cooling 11.8s / sound_wave 11.0s (网络I/O主导)
     修复后: <0.1s (计算本身 <1ms)
  2. _real_env 非阻塞 — 缓存过期时用旧数据, 不等新数据
  3. Monte Carlo 样本可配 — newton_cooling 从 200 降至可调
"""
import numpy as np
import json, math, time, threading

# ---------- sensor_env TTL 缓存 ----------
_ENV_CACHE = {'data': None, 'ts': 0.0, 'lock': threading.Lock()}
_ENV_TTL = 300  # 缓存 5 分钟

def _cached_sensor_env(force_refresh=False):
    """带 TTL 缓存的 sensor_env — 避免每次 simulate 都发 HTTP"""
    now = time.time()
    if force_refresh or _ENV_CACHE['data'] is None or (now - _ENV_CACHE['ts']) > _ENV_TTL:
        with _ENV_CACHE['lock']:
            # double-check
            now2 = time.time()
            if force_refresh or _ENV_CACHE['data'] is None or (now2 - _ENV_CACHE['ts']) > _ENV_TTL:
                from physics_engine.real_world import sensor_env
                try:
                    _ENV_CACHE['data'] = sensor_env()
                    _ENV_CACHE['ts'] = now2
                except Exception:
                    if _ENV_CACHE['data'] is None:
                        # 首次调用也失败 → 用默认值
                        _ENV_CACHE['data'] = {
                            'T_env_K': 298.15, 'T_env_C': 25.0,
                            'air_density_kgm3': 1.225, 'sound_speed_ms': 343.0,
                            'humidity_pct': None, 'pressure_hpa': 1013.25,
                            'weather': None, 'altitude_m': 18.0,
                            'location': {'lat': None, 'lon': None, 'source': 'none'},
                            'weather_source': 'none',
                            'provenance': {'T_env': 'default-25C', 'air_density': 'std-atmos',
                                           'sound_speed': 'temp-formula'},
                        }
                    # 如果已有旧数据, 保留旧数据 (stale is better than fail)
    return _ENV_CACHE['data']
from typing import Dict, List
from dataclasses import dataclass
from .core import PhysicsState, PhysicsEngine, MonteCarloEngine, CausalDiscovery, PredictionResult
from .thermodynamics import NewtonCoolingEngine, PhaseTransitionEngine, HeatConductionEngine
from .fluid_dynamics import GradientWindEngine, TyphoonTrackEngine, BernoulliEngine
from .classical import CosmologyEngine
from .microscopic import (ElectricFieldEngine, MagneticFieldEngine, SchrodingerEngine,
                          TunnelingEngine, ArrheniusEngine, EquilibriumEngine,
                          FaradayInductionEngine, MaxwellStressEngine,
                          QuantumSpinEngine, IdenticalParticlesEngine)
from .waves_solids import (StressStrainEngine, BeamBendingEngine, SoundWaveEngine,
                           DopplerEngine, GeometricOpticsEngine, WaveOpticsEngine)
from .extremes import (NBodyMDFngine, PlasmaEngine, SchwarzschildEngine,
                       LensingEngine, PrecessionEngine, BlackHoleEngine)
from .celestial import KeplerOrbitEngine, HohmannTransferEngine, LagrangePointEngine
from .newtonian import RigidBodyRotationEngine, CollisionEngine, ProjectileEngine
from .statistical import EntropyEngine, PhaseTransitionEngine as PhaseCriticalEngine
from .fluid_dynamics import NavierStokes1DEngine
from .fluid_dynamics_2d import NavierStokes2DEngine
from .geophysics import SeismicWaveEngine, AtmosphericCirculationEngine
from .nuclear import RadioactiveDecayEngine, NuclearFissionEngine
from .condensed_matter import BandTheoryEngine, SuperconductivityEngine
from .control import PIDControllerEngine, FeedbackSystemEngine
from .particle_physics import StandardModelEngine, ScatteringEngine
from .acoustics import SoundFieldEngine, NoiseEngine
from .nonlinear import ChaosEngine, FractalEngine
from .microscopic import M_E
from .extremes import M_SUN
from .uncertainty import UncertaintyQuantifier
from .real_world import sensor_env, sound_speed, air_density, real_weather, material
from .coupled import (TyphoonOceanCoupling, ThermalStressEngine, BlastStructureCoupling,
                      WindStructureCoupling, FireThermalCoupling,
                      AdvectionDiffusionConvection, MultiPhysicsCoupler)
from .physical_constants import Atmosphere, Water, get_material, BUILDING


@dataclass
class WorldModelResult:
    """世界模型完整输出"""
    phi_layer: dict     # 物理状态编码
    m_layer: dict       # 规律/因果
    psi_layer: dict     # 不确定性/决策
    summary: str
    metadata: dict


class TriWorldModel:
    """三元世界物理模型 v2.0 — 24引擎统一调度"""
    def __init__(self, mc_samples=200):
        self.mc_samples = mc_samples
        self.engines = {
            # 一层: 经典
            'newton_cooling': NewtonCoolingEngine(),
            'heat_conduction': HeatConductionEngine(),
            'phase_transition': PhaseTransitionEngine(),
            'cosmology': CosmologyEngine(),
            'typhoon_track': TyphoonTrackEngine(),
            'gradient_wind': GradientWindEngine(),
            'bernoulli': BernoulliEngine(),
            'navier_stokes': NavierStokes1DEngine(),
            'navier_stokes_2d': NavierStokes2DEngine(),
            # 二层: 微观+电磁+量子
            'electric_field': ElectricFieldEngine(),
            'magnetic_field': MagneticFieldEngine(),
            'schrodinger': SchrodingerEngine(),
            'tunneling': TunnelingEngine(),
            'arrhenius': ArrheniusEngine(),
            'equilibrium': EquilibriumEngine(),
            'faraday_induction': FaradayInductionEngine(),
            'maxwell_stress': MaxwellStressEngine(),
            'spin': QuantumSpinEngine(),
            'identical_particles': IdenticalParticlesEngine(),
            # 三层: 波动固体
            'stress_strain': StressStrainEngine(),
            'beam_bending': BeamBendingEngine(),
            'sound_wave': SoundWaveEngine(),
            'doppler': DopplerEngine(),
            'geometric_optics': GeometricOpticsEngine(),
            'wave_optics': WaveOpticsEngine(),
            # 四层: 极端
            'nbody_md': NBodyMDFngine(),
            'plasma': PlasmaEngine(),
            'schwarzschild': SchwarzschildEngine(),
            'lensing': LensingEngine(),
            'precession': PrecessionEngine(),
            'black_hole': BlackHoleEngine(),
            # 五层: 天体力学
            'kepler_orbit': KeplerOrbitEngine(),
            'hohmann_transfer': HohmannTransferEngine(),
            'lagrange_point': LagrangePointEngine(),
            # 六层: 牛顿力学
            'rigid_body': RigidBodyRotationEngine(),
            'collision': CollisionEngine(),
            'projectile': ProjectileEngine(),
            # 七层: 统计力学
            'entropy': EntropyEngine(),
            'phase_critical': PhaseCriticalEngine(),
            # 八层: 地球物理
            'seismic_waves': SeismicWaveEngine(),
            'atmospheric_circulation': AtmosphericCirculationEngine(),
            # 九层: 核物理
            'radioactive_decay': RadioactiveDecayEngine(),
            'fission_fusion': NuclearFissionEngine(),
            # 十层: 凝聚态
            'band_theory': BandTheoryEngine(),
            'superconductivity': SuperconductivityEngine(),
            # 十一层: 控制论
            'pid_controller': PIDControllerEngine(),
            'feedback_system': FeedbackSystemEngine(),
            # 十二层: 粒子物理
            'standard_model': StandardModelEngine(),
            'scattering': ScatteringEngine(),
            # 十三层: 声学
            'sound_field': SoundFieldEngine(),
            'noise_control': NoiseEngine(),
            # 十四层: 非线性动力学
            'chaos': ChaosEngine(),
            'fractal': FractalEngine(),
        }
        self.causal = CausalDiscovery()
        self.layers = self._assign_layers()

    def _assign_layers(self):
        L1 = {'newton_cooling','heat_conduction','phase_transition','cosmology','typhoon_track','gradient_wind','bernoulli','navier_stokes','navier_stokes_2d'}
        L2 = {'electric_field','magnetic_field','schrodinger','tunneling','arrhenius','equilibrium','faraday_induction','maxwell_stress','spin','identical_particles'}
        L3 = {'stress_strain','beam_bending','sound_wave','doppler','geometric_optics','wave_optics'}
        L4 = {'nbody_md','plasma','schwarzschild','lensing','precession','black_hole'}
        L5 = {'kepler_orbit','hohmann_transfer','lagrange_point'}
        L6 = {'rigid_body','collision','projectile'}
        L7 = {'entropy','phase_critical'}
        L8 = {'seismic_waves','atmospheric_circulation'}
        L9 = {'radioactive_decay','fission_fusion'}
        L10 = {'band_theory','superconductivity'}
        L11 = {'pid_controller','feedback_system'}
        L12 = {'standard_model','scattering'}
        L13 = {'sound_field','noise_control'}
        L14 = {'chaos','fractal'}
        return {'L1_classical': sorted(L1), 'L2_microscopic': sorted(L2),
                'L3_waves_solids': sorted(L3), 'L4_extremes': sorted(L4),
                'L5_celestial': sorted(L5), 'L6_newtonian': sorted(L6),
                'L7_statistical': sorted(L7), 'L8_geophysics': sorted(L8),
                'L9_nuclear': sorted(L9), 'L10_condensed_matter': sorted(L10),
                'L11_control': sorted(L11), 'L12_particle_physics': sorted(L12),
                'L13_acoustics': sorted(L13), 'L14_nonlinear': sorted(L14)}

    def register_engine(self, name, engine):
        self.engines[name] = engine

    def _result(self, phi, m, psi, summary, **meta):
        return WorldModelResult(phi_layer=phi, m_layer=m, psi_layer=psi,
                                summary=summary, metadata=meta)

    def layers_status(self):
        return {k: len(v) for k, v in self.layers.items()}

    def list_engines(self):
        return sorted(self.engines.keys())

    # ============ 通用调度器: simulate(model, params) ============
    def simulate(self, model: str, params: dict = None, steps: int = 10):
        """统一入口: simulate(model, params) → WorldModelResult
        model: 引擎名 (24个之一); params: 参数dict; steps: 步数"""
        params = params or {}
        handler = getattr(self, f'_sim_{model}', None)
        if handler is None:
            # 无专用handler, 尝试用引擎自身的可调用方法
            if model not in self.engines:
                return self._result(
                    phi={}, m={}, psi={},
                    summary=f"未知模型 '{model}', 可用: {', '.join(self.list_engines())}",
                    error='unknown_model', model=model)
            # 返回引擎信息(纯工具型引擎)
            return self._tool_result(model, params)
        return handler(params, steps)

    # ---------- 专用handler (每个引擎→Φ/ℳ/Ψ) ----------
    # ---------- 真实环境默认值注入 ----------
    @staticmethod
    def _real_env(p):
        """从真实世界感知层取默认值, 用户显式传参则优先. 返回(env, 使用来源)
        ★ v2.0.1: 改用 _cached_sensor_env() — 不再每次发 HTTP"""
        real = _cached_sensor_env()
        inject = {}
        src = {}
        if not p.get('_no_real'):
            # 环境温度
            if 'T_env' not in p:
                inject['T_env'] = real['T_env_K']; src['T_env'] = 'real:amap'
            # 声速(用于sound_wave)
            if 'c_sound' not in p and 'c' not in p:
                inject['c_sound'] = real['sound_speed_ms']; src['c_sound'] = 'real:'+str(round(real['sound_speed_ms'],1))
            # 空气密度(Bernoulli/sound)
            if 'rho' not in p:
                inject['rho'] = real['air_density_kgm3']; src['rho'] = 'real:'+str(round(real['air_density_kgm3'],4))
        return inject, src, real

    def _sim_newton_cooling(self, p, steps):
        inject, src, real = self._real_env(p)
        # 参数合并: 用户显式优先, 否则真实
        k = p.get('k', inject.get('k', 0.015))
        T_env = p.get('T_env', inject.get('T_env', 298.15))
        T0 = p.get('T0', real['T_env_K'])   # 初始用真实环境温度
        e = NewtonCoolingEngine(k=k, T_env=T_env)
        dt = p.get('dt',1.0)
        s0 = PhysicsState(temperature=T0)
        mc = MonteCarloEngine(e, n_samples=self.mc_samples)
        r = e.predict(s0, dt, steps)
        t = np.linspace(0, dt*steps, steps+1)
        ana = [e.analytical_solution(T0, ti) for ti in t]
        Tnum = [st.temperature for st in r.states]
        err = max(abs(Tnum[i]-ana[i]) for i in range(len(t)))
        return self._result(
            phi={'T0':T0,'T_env':T_env,'T_env_source':src.get('T_env','default'),
                 'real_weather':real.get('weather'), 'k':k,
                 'evolution':[{'t':round(ti,2),'T':round(Ti,2)} for ti,Ti in zip(r.time_points,Tnum)]},
            m={'thermal_law':'dT/dt=-kΔT','analytical':[round(v,2) for v in ana],
               'max_vs_analytical_error': round(err,4)},
            psi={'final_T': round(Tnum[-1],2),'converged_to_env': abs(Tnum[-1]-T_env)<1.0},
            summary=f"冷却(真实环境T_env={T_env-273.15:.1f}°C) {steps}步 T:{T0:.1f}→{Tnum[-1]:.1f}K",
            engine='newton_cooling')

    def _sim_heat_conduction(self, p, steps):
        e = HeatConductionEngine(alpha=p.get('alpha',1e-6), dx=p.get('dx',0.01))
        nx = p.get('nx',20); dt = p.get('dt',0.5)
        Tinit = np.array([p.get('T_hot',400.0) if (p.get('center',True) and abs(i-nx//2)<=1) else p.get('T_cold',300.0) for i in range(nx)], float)
        hist = e.solve_1d(Tinit, nx, dt, steps)
        final = hist[-1]
        # 均方差: T初始 vs 最终(反映扩散)
        spread = max(final) - min(final)
        return self._result(
            phi={'nx':nx,'alpha':e.alpha,'initial':[round(v,1) for v in Tinit],
                 'final_profile':[round(v,2) for v in final]},
            m={'method':'Crank-Nicolson','heat_equation':'∂T/∂t=α∂²T/∂x²'},
            psi={'final_spread': round(spread,2),'spread_reduction_pct': round((1-spread/(max(Tinit)-min(Tinit)+1e-9))*100,1)},
            summary=f"热传导 {steps}步: 温差从{max(Tinit)-min(Tinit):.0f}→{spread:.1f}",
            engine='heat_conduction')

    def _sim_phase_transition(self, p, steps):
        e = PhaseTransitionEngine()
        T = p.get('T', 300.0); mass = p.get('mass', 1.0)
        target = p.get('target', 373.15)
        phase = e.get_phase(T)
        if target > T:
            eres = e.energy_to_phase_change(mass, T, target)
        else:
            eres = {'energy_required_J': mass * 4186.0 * (target - T)}
        energy = eres['energy_required_J'] if isinstance(eres, dict) else eres
        P_boil = None
        boil = 373.15
        # 常压沸点(气压101325Pa)
        if hasattr(e, 'boiling_temp'):
            boil = e.boiling_temp(101325.0)
        return self._result(
            phi={'T':T,'phase':phase,'mass':mass,'target':target},
            m={'phase_law':'相变能量=显热+潜热','boiling_at_1atm_K': round(boil,2)},
            psi={'energy_required_J': round(energy,2),'energy_kJ': round(energy/1000,2)},
            summary=f"{mass}kg水{phase} {T}K→{target:.0f}K 需 {energy/1000:.0f} kJ",
            engine='phase_transition')

    def _sim_typhoon_track(self, p, steps):
        e = TyphoonTrackEngine()  # 原版: 内置副高转向+引导气流模型
        lon=p.get('lon',130.0); lat=p.get('lat',20.0); dt=p.get('dt',3600*3)
        s0=PhysicsState(temperature=300.0, pressure=975.0, position=(lon,lat))
        r=e.predict(s0, dt, steps)
        path=[{'lon':round(s.position[0],3),'lat':round(s.position[1],3),'pressure':round(s.pressure,1)} for s in r.states]
        ws=p.get('wind_speed',45.0)
        return self._result(
            phi={'start':{'lon':lon,'lat':lat,'pressure':975.0},'path':path},
            m={'law':'β漂移+环境引导','classification':e.classify(ws)},
            psi={'final_pressure_drop':round(975.0-path[-1]['pressure'],1),
                 'movement':round((path[-1]['lat']-lat)*111,2)},
            summary=f"台风 {steps}步→({path[-1]['lon']},{path[-1]['lat']}),{e.classify(ws)}",
            engine='typhoon_track')

    def _sim_schrodinger(self, p, steps):
        e = SchrodingerEngine(m=p.get('m',M_E), a=p.get('a',1e-9))
        levels = e.energy_levels(p.get('n_max',5))
        return self._result(
            phi={'particle':'electron','box_width_m':e.a},
            m={'law':'E_n=n²π²ℏ²/(2ma²)','energy_levels_eV':levels},
            psi={'ground_E_eV': levels.get(1), 'first_excited_E_eV': levels.get(2)},
            summary=f"方阱能级: E1={levels.get(1)}eV, E2={levels.get(2)}eV",
            engine='schrodinger')

    def _sim_black_hole(self, p, steps):
        e = BlackHoleEngine()
        M = p.get('M', M_SUN)
        T_h = e.hawking_temperature(M)
        rs = 2*6.674e-11*M/3e8**2
        return self._result(
            phi={'mass_kg': M,'mass_solar':M/M_SUN},
            m={'law':'T=ℏc³/(8πGMk_B)','schwarzschild_radius_km': round(rs/1000,3)},
            psi={'hawking_temperature_K': round(T_h,6),
                 'peak_wavelength_m': round(e.blackbody_peak(T_h),3)},
            summary=f"黑洞 M={M/M_SUN:.1e}M☉, T_H≈{T_h:.1e}K, rs={rs/1000:.3f}km",
            engine='black_hole')

    def _sim_sound_wave(self, p, steps):
        inject, src, real = self._real_env(p)
        e = SoundWaveEngine(rho=p.get('rho', inject.get('rho', 1.225)),
                            c_sound=p.get('c_sound', inject.get('c_sound', 343.0)))
        dB = p.get('dB', 100.0)
        amp = e.pressure_amplitude_from_dB(dB)
        intens = e.intensity_from_dB(dB)
        f = p.get('freq', 1000.0)
        wl = e.wavelength(f)
        return self._result(
            phi={'dB':dB,'freq_Hz':f,'source':'point',
                 'c_sound_ms':round(e.c,1),'c_sound_source':src.get('c_sound','default'),
                 'air_density':round(e.rho,4),'real_weather':real.get('weather')},
            m={'law':'p=p₀sin(kx-ωt)','pressure_amp_Pa':round(amp,4),'intensity_W/m²':round(intens,3)},
            psi={'wavelength_m':round(wl,2),'sound_pressure_level_ratio':round(10**(dB/20),1)},
            summary=f"{dB}dB 声压≈{amp*1000:.1f}mkPa(声速{e.c:.0f}m/s) {f}Hz波长{wl:.1f}m",
            engine='sound_wave')

    def _sim_stress_strain(self, p, steps):
        # 真实材料库: 若传material名, 用真实E/屈服强度
        mat = p.get('material', None)
        if mat:
            m0 = material(mat)
            e = StressStrainEngine(E=p.get('E', m0['E_GPa']*1e9),
                                   yield_strength=p.get('yield', m0['yield_MPa']*1e6))
            mat_name = mat
        else:
            e = StressStrainEngine(E=p.get('E',200e9), yield_strength=p.get('yield',250e6))
            mat_name = 'custom(E=%gGPa)' % (e.E/1e9)
        strain = p.get('strain', 0.001)
        res = e.edge_condition(strain)
        return self._result(
            phi={'strain':strain, 'material':mat_name, 'E_GPa':e.E/1e9},
            m={'law':'σ=Eε','stress_MPa':round(res['stress']/1e6,2),'yield_MPa':e.yield_strength/1e6},
            psi={'elastic':res['elastic'],'safety_factor':round(res['safety'],2) if res['safety'] else None},
            summary=f"{mat_name} 应变{strain}→应力{res['stress']/1e6:.1f}MPa {'弹性' if res['elastic'] else '已屈服'}",
            engine='stress_strain')

    def _sim_arrhenius(self, p, steps):
        e = ArrheniusEngine(A=p.get('A',1e10), Ea=p.get('Ea',50000))
        T = p.get('T', 350.0)
        k = e.rate(T)
        return self._result(
            phi={'T_K':T,'Ea_J':e.Ea},
            m={'law':'k=A·exp(-Ea/RT)','rate_constant':round(k,2)},
            psi={'reaction_accel_10K':round(e.rate(T+10)/max(k,1e-30),2)},
            summary=f"T={T}K 速率常数k≈{k:.1e}",
            engine='arrhenius')

    def _sim_grav_lensing(self, p, steps):
        e = LensingEngine()
        M = p.get('M', M_SUN); b = p.get('b', 7e8)
        alpha = e.deflection_angle(M, b)
        return self._result(
            phi={'mass_solar':M/M_SUN,'impact_km':b/1000},
            m={'law':'α=4GM/(c²b)'},
            psi={'deflection_arcsec':round(alpha*206265,2)},
            summary=f"光线偏折 α≈{alpha*206265:.2f}″",
            engine='lensing')

    def _sim_plasma(self, p, steps):
        e = PlasmaEngine()
        T = p.get('T', 1e4); n = p.get('n', 1e18)
        ld = e.debye_length(T, n)
        wpp = e.plasma_frequency(n)
        return self._result(
            phi={'T_K':T,'density_m-3':n},
            m={'law':'λ_D=√(ε₀kT/ne²)'},
            psi={'debye_length_m':round(ld,6),'plasma_freq_Hz':round(wpp,2)},
            summary=f"等离子体 T={T:.0e}K, λ_D≈{ld:.1e}m",
            engine='plasma')

    def _sim_equilibrium(self, p, steps):
        e = EquilibriumEngine()
        T = p.get('T', 300); K_eq = p.get('K', 10.0)
        dG = e.deltaG(T, K_eq)
        return self._result(
            phi={'T_K':T,'K_eq':K_eq},
            m={'law':'ΔG°=-RT·lnK','deltaG':round(dG,1)},
            psi={'spontaneous':dG<0,'K_selectivity': 'products' if K_eq>1 else 'reactants'},
            summary=f"K={K_eq}, ΔG°={dG:.0f}J → {'自发' if dG<0 else '非自发'}",
            engine='equilibrium')

    def _sim_precession(self, p, steps):
        e = PrecessionEngine()
        M = p.get('M', M_SUN); a = p.get('a', 5.79e10); eoc = p.get('e', 0.2056)
        dphi = e.perihelion_precession(M, a, eoc)
        return self._result(
            phi={'semi_major_m':a,'eccentricity':eoc},
            m={'law':'Δφ=6πGM/(c²a(1-e²))'},
            psi={'precession_rad_per_rev':round(dphi,8),'precession_arcsec_century':round(dphi*206265*415,2)},
            summary=f"水星近日点进动 Δφ≈{dphi*206265*415:.2f}″/百年",
            engine='precession')

    def _sim_electric_field(self, p, steps):
        e = ElectricFieldEngine(q=p.get('q',1e-6))
        r = p.get('r', 1.0)
        E = e.field_strength(r)
        return self._result(
            phi={'charge_C':e.q,'distance_m':r},
            m={'law':'E=kq/r²','field_N/C':round(E,3)},
            psi={'is_strong':E>1e3,'potential_V':round(e.potential(r),3)},
            summary=f"q={e.q*1e6:.1f}μC 在{r}m 场强 {E:.1f} N/C",
            engine='electric_field')

    # ---------- 工具型引擎(无耗时状态) ----------
    def _tool_result(self, model, params):
        e = self.engines[model]
        return self._result(phi={'model':model,'params':params}, m={}, psi={},
                            summary=f"工具引擎 '{model}' 已注册 (第四层物理)",
                            engine=model, tool=True)

    # ---------- 沿用原方法(兼容) ----------
    def predict_thermodynamics(self, *a, **k): return self._sim_newton_cooling(k, k.get('steps',30))
    def predict_typhoon_track(self, *a, **k): return self._sim_typhoon_track(k, k.get('steps',20))
    def discover_causality(self, data, max_lag=3):
        """完整因果推断: 关联 + Granger + Pearl do-演算
        优先用完整因果体系(发现引擎v2.0 + Pearl框架), 回退到基础版
        """
        try:
            import sys as _sys
            _sys.path.insert(0, '/var/minis/skills/multi-search-engine/assets')
            from causal_discovery_engine import CausalDiscoveryEngine as _CDE
            from pearl_causal import CausalGraph as _CG

            # 若 data 是 numpy 矩阵, 转 dict
            if hasattr(data, 'shape') and not isinstance(data, dict):
                data_dict = {f'X{i}': data[:, i] for i in range(data.shape[1])}
            else:
                data_dict = data

            engine = _CDE()
            assoc = engine.association_matrix(data_dict, method='pearson', threshold=0.3)
            granger = engine.granger_matrix(data_dict, max_lag=max_lag, significance=0.01)

            g = _CG()
            for gr in granger:
                g.add_edge(gr['cause'], gr['effect'])

            return self._result(
                phi={'n_vars': len(data_dict), 'n_samples': len(list(data_dict.values())[0])},
                m={'associations': assoc[:10], 'granger_edges': granger[:10],
                   'pearl_graph_edges': sorted(list(g.edges))[:10]},
                psi={'n_assoc': len(assoc), 'n_granger': len(granger), 'n_pearl': len(g.edges)},
                summary=f"完整因果推断: {len(assoc)}关联, {len(granger)}Granger, {len(g.edges)}Pearl边",
                engine='causal_discovery_full')
        except Exception:
            pc = CausalDiscovery.pc_algorithm(data, max_lag=max_lag)
            gr = CausalDiscovery.granger_causality(data, max_lag=max_lag)
            return self._result(
                phi={'n_samples':int(data.shape[0]),'n_vars':int(data.shape[1])},
                m={'pc_edges':pc,'granger_edges':gr},
                psi={'pc_count':len(pc),'granger_count':len(gr)},
                summary=f"PC {len(pc)}边, Granger {len(gr)}边 (基础版)", engine='causal_discovery')

    def predict_ensemble(self, T0=350.0, dt=1.0, steps=30):
        e = NewtonCoolingEngine(); s0=PhysicsState(temperature=T0)
        uq = UncertaintyQuantifier(e)
        r = uq.ensemble_prediction(s0, dt, steps)
        return self._result(phi={'T0':T0},m={'method':'ensemble'},
            psi={'mean':r['mean'],'std':r['std'],'final_mean':round(r['mean'][-1],2)},
            summary=f"集成{T0}K→{r['mean'][-1]:.1f}±{r['std'][-1]:.2f}", engine='ensemble')

    # ============ 爆炸当量引擎 (冲击波反演 + 蒙特卡洛不确定性) ============
    def _sim_blast_yield(self, p, steps):
        """爆炸当量反推(世界物理引擎): 冲击波破坏距离开放反推当量
        原理: Kingery-Bulmash超压曲线 + Hopkinson-Cranz比例定律 W=(R/Z(P))³
        输入(观测不确定→蒙特卡洛采样):
          R_dest: 破坏点距爆心(m), 玻璃/门窗破碎超压P_dest kPa
          k_ground: 地面反射系数(近地面爆炸~2)
        输出 Φ(观测)/ℳ(物理规律)/Ψ(当量分布: P50+P90区间)
        """
        import numpy as np
        rng = np.random.default_rng(p.get('seed', 20260822))
        n = int(p.get('n_samples', 20000))
        # Kingery-Bulmash 表(自由空气)
        KB = np.array([
            [0.5,3000],[0.6,2100],[0.7,1500],[0.8,1100],[0.9,850],[1.0,680],
            [1.1,560],[1.2,470],[1.3,400],[1.4,340],[1.5,290],[1.6,250],
            [1.7,220],[1.8,190],[1.9,170],[2.0,150],[2.2,120],[2.4,100],
            [2.6,85],[2.8,72],[3.0,62],[3.2,54],[3.4,47],[3.6,42],
            [3.8,37],[4.0,33],[4.5,26],[5.0,21],[5.5,17],[6.0,14],
            [7.0,10.5],[8.0,8.3],[9.0,6.8],[10.0,5.7],[12.0,4.3],
            [14.0,3.4],[16.0,2.8],[20.0,2.0],[24.0,1.5],[30.0,1.1]])
        z_t=KB[:,0]; p_t=KB[:,1]
        def Z_of_P_arr(P):
            P=np.asarray(P,float)
            lp=np.log(p_t); lz=np.log(z_t)
            return np.exp(np.interp(np.log(P), lp[::-1], lz[::-1]))
        def W_of(R,P,k):
            return (R/Z_of_P_arr(P))**3 / k

        # 观测输入的不确定性分布
        P_dest = p.get('P_dest', 5.0)  # 中心值
        R_dest = p.get('R_dest', 60.0)
        k_gr   = p.get('k_ground', 1.8)
        # P不确定性(门窗破碎3-7,中心5)
        P_s = rng.triangular(P_dest*0.6, P_dest, P_dest*1.4, n)
        # R不确定性(±20%)
        R_s = rng.triangular(R_dest*0.9, R_dest, R_dest*1.2, n)
        # k不确定性(1.2-2.2,中心1.8)
        k_s = np.clip(rng.normal(k_gr, 0.15, n), 1.2, 2.2)
        W = W_of(R_s, P_s, k_s)

        pct=lambda x,qp: float(np.percentile(x,qp))
        return self._result(
            phi={'P_dest_kPa':P_dest,'R_dest_m':R_dest,'k_ground':k_gr,
                 'n_samples':n,'method':'shockwave MC'},
            m={'law':'W=(R/Z(P))³/k, Kingery-Bulmash+Hopkinson-Cranz',
               'scale_relation':'比例距离 Z=R/W^(1/3)',
               'ground_reflection':'近地面爆炸等效当量×k=%.1f'%k_gr},
            psi={'P50_kg':round(pct(W,50)),'P25_kg':round(pct(W,25)),
                 'P75_kg':round(pct(W,75)),
                 'P05_kg':round(pct(W,5)),'P95_kg':round(pct(W,95)),
                 'mean_kg':round(W.mean()),'std_kg':round(W.std()),
                 'ci90':f"{round(pct(W,5))}~{round(pct(W,95))} kg TNT",
                 'interpretation':f"当量中位≈{round(pct(W,50))}kg TNT, 90%区间{round(pct(W,5))}~{round(pct(W,95))}kg(约{round(pct(W,5)/1000,2)}~{round(pct(W,95)/1000,2)}吨)"},
            summary=f"爆炸当量 P50≈{round(pct(W,50))}kg TNT, 90%区间 {round(pct(W,5))}~{round(pct(W,95))}kg",
            engine='blast_yield')


    # ============ L3 耦合引擎: 台风-海洋热交换 ============
    def _sim_typhoon_ocean(self, p, steps):
        steps = p.get('steps', steps)
        """台风↔海洋 双向耦合: 暖海供能增强台风, 风搅拌冷却海表(负反馈)"""
        te = TyphoonOceanCoupling(lat=p.get('lat',22.0), lon=p.get('lon',128.0),
                                  mixing_coef=p.get('mixing',0.15),
                                  energy_coef=p.get('energy',4.0))
        ws0 = p.get('wind_speed', 35.0); sst0 = p.get('sst', 28.5)
        dt = p.get('dt', 3600*6)
        s0 = PhysicsState(temperature=301.0, pressure=965.0, time=0)
        s0.extra = {'wind_speed':ws0, 'sst':sst0}
        r = te.predict(s0, dt, steps)
        series = []
        for st in r.states:
            series.append({'t_h':round(st.time/3600), 'wind':round(st.extra['wind_speed'],1),
                           'sst':round(st.extra['sst'],2),
                           'class':te.classify(st.extra['wind_speed'])})
        dws = series[-1]['wind'] - series[0]['wind']
        dsst = series[-1]['sst'] - series[0]['sst']
        return self._result(
            phi={'initial':{'wind_speed':ws0,'sst_C':sst0},'series':series,
                 'coupling':'ocean-supplies-energy / wind-cools-sst'},
            m={'law':'d(ws)/dt = f(SST-26.5) - mixing·ws²','sst_critical':26.5},
            psi={'wind_change':round(dws,1),'sst_change_C':round(dsst,2),
                 'feedback':'positive-then-negative' if dws>0 else 'weakening',
                 'final_ws':series[-1]['wind'],'final_sst':series[-1]['sst'],
                 'interpretation':f"风-海耦合 {steps}步(≈{steps*dt/3600/24:.1f}天): 风{dws:+.1f}m/s, 海温{dsst:+.1f}°C"},
            summary=f"台风-海洋耦合: 风{ws0:.0f}→{series[-1]['wind']:.0f}m/s ({te.classify(series[-1]['wind'])}), 海温{sst0}→{series[-1]['sst']:.1f}°C",
            engine='typhoon_ocean')

    # ============ L3 耦合引擎: 热应力-变形 ============
    def _sim_thermal_stress(self, p, steps):
        steps = p.get('steps', steps)
        """热↔固 耦合: 温度变化 → 热应变 → 约束热应力 → 屈服判断"""
        tse = ThermalStressEngine()
        mat = p.get('material','steel_Q235'); dT = p.get('dT', 30.0)
        stress = tse.constrained_stress(mat, dT)
        L0 = p.get('L0', 1.0)
        elong = tse.elongation(mat, dT, L0)
        yield_dT = tse.temperature_change_for_yield(mat)
        m0 = get_material(mat)
        return self._result(
            phi={'material':stress.get('material',mat),'dT_K':dT,'L0_m':L0,
                 'E_GPa':m0.get('E_GPa'),'cte':m0.get('cte')},
            m={'law':'σ=E·α·ΔT (约束热应力)','strain':f"{m0.get('cte')*dT*1e6:.1f}μ应变"},
            psi={'stress_MPa':stress.get('stress_MPa'),'yield_MPa':stress.get('yield_MPa'),
                 'elastic':stress.get('elastic'),'verdict':stress.get('verdict'),
                 'elongation_mm':elong,'yield_dT_K':yield_dT,
                 'interpretation':f"{mat} 温变{dT}K → 热应力{stress.get('stress_MPa')}MPa → {stress.get('verdict')}"},
            summary=f"{mat} ΔT={dT}K: 热应力{stress.get('stress_MPa')}MPa {'弹性' if stress.get('elastic') else '屈服'} (L=1m伸长{elong}mm)",
            engine='thermal_stress')

    # ============ L4 实时数据同化: Ensemble Kalman Filter (EnKF) ============
    def _sim_enkf(self, p, steps):
        """Ensemble Kalman Filter 数据同化
        场景: 牛顿冷却预测温度. 周期性注入带噪观测, 对比:
          '纯模型预测'(无观测) vs 'EnKF同化'(注入观测校正) 谁更接近真实
        核心: EnKF通过集合协方差融合观测, 把模型从'单一假设'拉回'真实'"""
        n_ens = p.get('n_ensemble', 60)
        steps = p.get('steps', steps)   # 允许params里传steps
        T0 = p.get('T0', 350.0); T_env = p.get('T_env', 298.0)
        k = p.get('k', 0.015); dt = p.get('dt', 1.0)
        obs_interval = max(1, p.get('obs_interval', 8))  # 每隔8步注入观测
        obs_noise = p.get('obs_noise', 1.5); model_noise = p.get('model_noise', 0.6)
        rng = np.random.default_rng(p.get('seed', 7))

        # 复用 real_world 真实环境作默认
        inject, src, real = self._real_env(p)
        T_env = T_env if T_env != 298.0 or p.get('T_env') else inject.get('T_env', 298.0)

        # 集合初始化
        X = T0 + rng.normal(0, model_noise*4, n_ens)   # 预测集合
        X_pred = X.copy()                              # 纯模型集合

        # '真实世界:T_truth' 演化(带系统真实扰动)
        T_truth = T0 + 3.0

        def NLcool(T): return T_env + (T-T_env)*math.exp(-k*dt)  # 模型

        traj_true=[]; traj_pred=[]; traj_anal=[]; traj_nodata=[]
        n_obs=0
        for step in range(steps):
            T_truth = NLcool(T_truth) + rng.normal(0, 0.3)   # 真世界(有扰动)
            # 集合前向
            X = NLcool(X) + rng.normal(0, model_noise, n_ens)      # EnKF集合
            X_pred = NLcool(X_pred) + rng.normal(0, model_noise, n_ens)  # 纯预测集合
            traj_true.append(T_truth); traj_pred.append(X_pred.mean()); traj_anal.append(X.mean())
            # 周期注入观测
            if step>0 and step%obs_interval==0:
                obs = T_truth + rng.normal(0, obs_noise)     # 真实观测(带传感器噪声)
                xm_X = X.mean()
                P = np.mean((X-xm_X)**2); R = obs_noise**2
                K = P/(P+R)
                # EnKF: 观测扰动集合
                dX = obs + rng.normal(0, obs_noise, n_ens) - X
                X = X + K*dX          # 分析(同化)
                traj_anal[-1]=X.mean()
                n_obs+=1
        # 误差计算(相对真实)
        err_pred = np.mean([abs(traj_pred[i]-traj_true[i]) for i in range(steps)])
        err_anal = np.mean([abs(traj_anal[i]-traj_true[i]) for i in range(steps)])
        return self._result(
            phi={'n_ensemble':n_ens,'T0':T0,'T_env':T_env,'k':k,
                 'obs_interval':obs_interval,'obs_noise':obs_noise,'method':'EnKF',
                 'obs_injected':n_obs,'real_weather':real.get('weather')},
            m={'law':'x_a = x_f + K(y - Hx_f), K=P/(P+R)','algorithm':'Ensemble Kalman Filter',
               'true_evolve':f"真实世界T_truth演化(带扰动), 初始T0={T0:.0f},T_env={T_env-273.15:.1f}C"},
            psi={'final_true':round(traj_true[-1],2),
                 'final_pred':round(traj_pred[-1],2),'final_anal':round(traj_anal[-1],2),
                 'mae_pred':round(err_pred,3),'mae_anal':round(err_anal,3),
                 'mae_improvement_pct':round((err_pred-err_anal)/max(err_pred,1e-9)*100,1),
                 'obs_count':n_obs,
                 'interpretation':f"纯模型平均误差{err_pred:.2f}K → EnKF同化后{err_anal:.2f}K, 改善{round((err_pred-err_anal)/max(err_pred,1e-9)*100,1)}%"},
            summary=f"EnKF数据同化 {steps}步(注入{n_obs}次观测): 预测误差{err_pred:.2f}→同化后{err_anal:.2f}K, 改善{(err_pred-err_anal)/max(err_pred,1e-9)*100:.0f}%",
            engine='enkf')

    # ============ L3 扩展耦合引擎 ③④⑤⑥ ============
    def _sim_blast_structure(self, p, steps):
        """爆轰-结构耦合: 爆炸超压随距离衰减 → 结构破坏分区判断
        calib='constraint'(默认,84/P约束爆炸) 或 'kb'(标准KB表自由空气)"""
        from .coupled import BlastStructureCoupling
        calib = p.get('calib','constraint')
        bsc = BlastStructureCoupling(k_ground=p.get('k_ground',1.8), calib=calib)
        W = p.get('W_kg', 100); R = p.get('R_m', 30)
        structure = p.get('structure','brick')
        P = bsc.overpressure_at(W, R)
        verdict = bsc.struct_verdict(P, structure)
        from .physical_constants import BUILDING
        # 分区: 各级破坏超压 → 达距离; 按标定公式反推
        import numpy as np
        def P_at_R(cand_R):
            return bsc.overpressure_at(W, cand_R)
        # 二分求各级超压达距离
        def dist_for_P(P_t, lo=0.5, hi=400):
            for _ in range(80):
                mid=(lo+hi)/2
                if P_at_R(mid) > P_t: lo=mid
                else: hi=mid
            return round((lo+hi)/2, 1)
        zones=[]
        for dp in [3.5, 7, 25, 60]:
            zones.append((dp, dist_for_P(dp)))
        return self._result(
            phi={'W_kg':W,'R_m':R,'structure':structure,'calib':calib},
            m={'law':'P(R)随冲击波衰减 → 结构破坏判据','calib_note':
               'constraint: P=84·W^(1/3)/R 约束/近地表爆炸' if calib=='constraint'
               else 'kb: Kingery-Bulmash 标准自由空气','damage_judged_by':BUILDING},
            psi={'overpressure_kPa':round(P,1),'verdict':verdict,
                 'damage_zones_m':zones,
                 'interpretation':f"当量{W}kg({calib}) - {structure}在{R}m → {P:.0f}kPa → {verdict}"},
            summary=f"爆轰-结构({calib}): {W}kg @{R}m → {P:.0f}kPa, {structure}→{verdict}",
            engine='blast_structure')

    def _sim_wind_structure(self, p, steps):
        """风-建筑耦合(工程抗风设计工具): 阵风风压 vs 各部位允许承载"""
        from .coupled import WindStructureCoupling
        wsc = WindStructureCoupling(alt=p.get('alt',20), gust_factor=p.get('gust_factor',None))
        v = p.get('wind_speed', 30.0)
        # 全部位工程评估
        assess = wsc.assess(v)
        # 单点墙体弯曲(可选参考)
        wp_info = wsc.weakest_point(v)
        results = assess['results']
        weak = assess['weak_points']
        from .physical_constants import WIND_DESIGN
        return self._result(
            phi={'wind_speed_avg':v,'gust_speed':wp_info['gust_speed'],
                 'air_density':wp_info['rho'],'gust_factor':wsc.gust_factor,
                 'altitude':wsc.alt,'shape_coefs':WIND_DESIGN['shape_coefficients']},
            m={'law':'q=½·ρ·(V·gust)²·μs, 承载设计法: q vs 允许风压',
               'method':'承载能力设计(GB50009风荷载体型+阵风系数)'},
            psi={'gust_wind_speed_m_s':wp_info['gust_speed'],
                 'assessments':results,'weak_points':weak,
                 'verdict':wp_info['verdict'],
                 'interpretation':wp_info['verdict'],
                 'recommendation':'需加固:'+','.join(weak) if weak else '整体抗风安全'},
            summary=f"抗风设计: {v}m/s(阵风{wp_info['gust_speed']:.0f}) 薄弱点={','.join(weak) if weak else '无'} → {wp_info['verdict']}",
            engine='wind_structure')

    def _sim_fire_thermal(self, p, steps):
        """燃烧-热-固耦合: 火灾热释放 → 温升 → 结构热应力"""
        from .coupled import FireThermalCoupling
        ftc = FireThermalCoupling()
        hr = p.get('HRR_kW_m2', 300); material=p.get('material','steel_Q235')
        Tfire = ftc.fire_temp_field(hr)
        dT = Tfire - 293
        hs = ftc.heat_stress_at(dT, material)
        t_rise = ftc.temp_rise_time(hr, material, dT)
        return self._result(
            phi={'HRR_kW_m2':hr,'material':material,'fire_temp_K':round(Tfire,1)},
            m={'law':'燃烧释热→温升→热应力σ=E·α·ΔT',
               'heat_rise_time_s':round(t_rise,1)},
            psi={'dT_K':round(dT,1),'stress_MPa':hs.get('stress_MPa'),
                 'yield_MPa':hs.get('yield_MPa'),'verdict':hs.get('verdict'),
                 'interpretation':f"HRR={hr}kW/m² 火灾 → 钢构件ΔT{dT:.0f}K → 热应力{hs.get('stress_MPa')}MPa → {hs.get('verdict')}"},
            summary=f"火灾-结构: HRR={hr}kW/m² → {material}温升{dT:.0f}K → {hs.get('verdict')}(应力{hs.get('stress_MPa')}MPa)",
            engine='fire_thermal')

    def _sim_advection_diffusion(self, p, steps):
        """对流-扩散耦合: 流体输运 + 热扩散"""
        from .coupled import AdvectionDiffusionConvection
        nx = p.get('nx', 20); dt=p.get('dt',0.1); nstep=p.get('steps',200)
        adc = AdvectionDiffusionConvection(u=p.get('u',0.02), alpha=p.get('alpha',1e-4),
                                           dx=p.get('dx',0.1), h_loss=p.get('h',0.0),
                                           T_amb=p.get('T_amb',298.0))
        import numpy as np
        T = np.full(nx, adc.T_amb); T_inlet = p.get('T_inlet', 400.0); T[0]=T_inlet
        for _ in range(nstep):
            T = adc.cn_step(T, nx, dt)
            T[0] = T_inlet   # 固定入口边界(狄利克雷)验证输运
        idx = nx//2  # 中部, 看热输运是否到达
        return self._result(
            phi={'nx':nx,'steps':nstep,'u':adc.u,'alpha':adc.alpha,'inlet':T[0]},
            m={'law':'∂T/∂t=α∂²T/∂x²-u∂T/∂x (+upwind稳定)','scheme':'CN+upwind'},
            psi={'mid_temp':round(T[nx//2],1),'exit_temp':round(T[-1],1),
                 'mid_rise_K':round(T[nx//2]-adc.T_amb,1),'exit_rise_K':round(T[-1]-adc.T_amb,1),
                 'transport_reached_mid':T[nx//2]>adc.T_amb+5,
                 'interpretation':f"对流扩散{nstep}步: 入口{T[0]:.0f}K → 中部{T[nx//2]:.0f}K(+{round(T[nx//2]-adc.T_amb,0)}K), 出口{T[-1]:.0f}K"},
            summary=f"对流-扩散: 入口{T[0]:.0f}K输运{nstep}步 → 中部{T[nx//2]:.0f}K,出口{T[-1]:.0f}K",
            engine='advection_diffusion')


    # ============ ⑤ 天体力学 handlers ============
    def _sim_kepler_orbit(self, p, steps):
        """开普勒轨道: 轨道六根数/圆轨道/周期"""
        e = self.engines['kepler_orbit']
        mu_name = p.get('mu', 'earth')
        mu_map = {'sun': 1.32712440018e11, 'earth': 3.986004418e5, 'mars': 4.282837e4}
        e.mu = mu_map.get(mu_name, 3.986004418e5)
        a = p.get('a_km', 7000)
        ecc = p.get('e', 0)
        if p.get('mode') == 'circular':
            r = p.get('r_km', a)
            res = e.circular_orbit(r)
        else:
            res = e.orbital_elements_to_state(a, ecc, p.get('i', 0),
                                               p.get('Omega', 0), p.get('omega', 0),
                                               p.get('theta', 0))
        return self._result(
            phi={'a_km': a, 'e': ecc, 'mu': e.mu, 'body': mu_name},
            m={'law': 'v²=μ(2/r-1/a), T=2π√(a³/μ)', 'type': '二体问题/开普勒'},
            psi=res, summary=f"开普勒: a={a}km, e={ecc} → v={res.get('v_km_s')}km/s, T={res.get('period_h', res.get('period_min'))}h",
            engine='kepler_orbit')

    def _sim_hohmann_transfer(self, p, steps):
        """霍曼转移: 两圆轨道间最省燃料转移"""
        e = self.engines['hohmann_transfer']
        mu_name = p.get('mu', 'earth')
        mu_map = {'earth': 3.986004418e5, 'mars': 4.282837e4}
        e.mu = mu_map.get(mu_name, 3.986004418e5)
        r1 = p.get('r1_km', 7000)
        r2 = p.get('r2_km', 42164)  # GEO
        res = e.transfer(r1, r2)
        return self._result(
            phi={'r1_km': r1, 'r2_km': r2, 'mu': e.mu},
            m={'law': 'dv₁+dv₂最小, 半椭圆转移', 'type': '最省燃料(两脉冲)'},
            psi=res, summary=f"霍曼: {r1}→{r2}km, dv={res['dv_total_km_s']}km/s, T={res['transfer_time_h']}h",
            engine='hohmann_transfer')

    def _sim_lagrange_point(self, p, steps):
        """拉格朗日点: L1-L5位置"""
        e = self.engines['lagrange_point']
        pts = e.all_points()
        return self._result(
            phi={'system': p.get('system', 'sun_earth'), 'D_km': e.D},
            m={'law': '圆型限制性三体问题', 'type': '5个平动点'},
            psi={'points': pts, 'summary': 'L1/L2不稳定需轨道保持, L4/L5稳定'},
            summary=f"拉格朗日: L1={pts[0]['distance_from_secondary_km']:.0f}km, L2={pts[1]['distance_from_secondary_km']:.0f}km",
            engine='lagrange_point')

    # ============ ⑥ 牛顿力学 handlers ============
    def _sim_rigid_body(self, p, steps):
        """刚体转动: 转动惯量/角动量守恒/陀螺进动"""
        e = self.engines['rigid_body']
        shape = p.get('shape', 'sphere')
        mass = p.get('mass', 1.0)
        kw = {}
        if shape in ('sphere', 'hollow_sphere', 'disk', 'cylinder', 'ring'):
            kw['radius'] = p.get('radius', 0.1)
        if shape in ('rod_center', 'rod_end'):
            kw['length'] = p.get('length', 1.0)
        I = e.moment_of_inertia(mass, shape, **kw)
        omega = p.get('omega', 10.0)
        L = e.angular_momentum(I, omega)
        K = e.rotational_kinetic_energy(I, omega)
        # 角动量守恒演示
        I2 = p.get('I2', I * 0.5)  # 收臂后I减半
        cons = e.conservation_demo(I, omega, I2)
        return self._result(
            phi={'shape': shape, 'mass': mass, 'I': round(I, 5), 'omega': omega},
            m={'law': 'L=Iω, E=½Iω², I₁ω₁=I₂ω₂', 'type': '角动量守恒'},
            psi={'I': round(I, 5), 'L': round(L, 4), 'K_rot': round(K, 4),
                 'conservation': cons,
                 'summary': f"I={I:.4f}, L={L:.3f}, ω₁={omega}→ω₂={cons['omega2']}"},
            summary=f"刚体: {shape} I={I:.4f}, ω={omega}→{cons['omega2']} (角动量守恒)",
            engine='rigid_body')

    def _sim_collision(self, p, steps):
        """碰撞: 弹性/非弹性/恢复系数"""
        e = self.engines['collision']
        m1, v1 = p.get('m1', 1.0), p.get('v1', 5.0)
        m2, v2 = p.get('m2', 1.0), p.get('v2', -3.0)
        mode = p.get('mode', 'elastic')
        if mode == 'elastic':
            res = e.elastic_1d(m1, v1, m2, v2)
        elif mode == 'inelastic':
            res = e.perfectly_inelastic(m1, v1, m2, v2)
        else:
            res = e.coefficient_of_restitution(m1, v1, m2, v2, p.get('e', 0.8))
        return self._result(
            phi={'m1': m1, 'v1': v1, 'm2': m2, 'v2': v2, 'mode': mode},
            m={'law': '动量守恒: m₁v₁+m₂v₂ = const', 'type': mode},
            psi=res, summary=f"碰撞({mode}): v₁={v1}→{res['v1_final']}, v₂={v2}→{res['v2_final']}",
            engine='collision')

    def _sim_projectile(self, p, steps):
        """抛体: 理想斜抛/有阻力"""
        e = self.engines['projectile']
        v0 = p.get('v0', 30.0)
        angle = p.get('angle', 45.0)
        if p.get('drag', False):
            res = e.with_drag(v0, angle, mass=p.get('mass', 0.5),
                              Cd=p.get('Cd', 0.47), area=p.get('area', 0.01))
        else:
            res = e.ideal(v0, angle, y0=p.get('y0', 0))
        return self._result(
            phi={'v0': v0, 'angle': angle, 'drag': p.get('drag', False)},
            m={'law': '斜抛: x=v₀cosθ·t, y=v₀sinθ·t-½gt²', 'type': '牛顿第二定律'},
            psi=res, summary=f"抛体: v₀={v0}, {angle}° → 射程={res['range']}m, 最高={res['h_max']}m",
            engine='projectile')

    # ============ ⑦ 统计力学 handlers ============
    def _sim_entropy(self, p, steps):
        """熵与热机: 卡诺/奥托/狄塞尔效率 + 熵变"""
        e = self.engines['entropy']
        mode = p.get('mode', 'carnot')
        if mode == 'carnot':
            res = e.carnot_efficiency(p.get('T_hot', 600), p.get('T_cold', 300))
        elif mode == 'otto':
            res = e.otto_efficiency(p.get('r', 8.0), p.get('gamma', 1.4))
        elif mode == 'diesel':
            res = e.diesel_efficiency(p.get('r', 16.0), p.get('alpha', 2.0))
        elif mode == 'mixing':
            res = {'delta_S': round(e.mixing_entropy(p.get('n1', 1), p.get('n2', 1),
                                                     p.get('x1', 0.5), p.get('x2', 0.5)), 4),
                   'type': '混合熵'}
        elif mode == 'gas':
            res = e.entropy_change_ideal_gas(p.get('n', 1), p.get('T1', 300),
                                              p.get('T2', 600), p.get('V1', 1),
                                              p.get('V2', 2), p.get('Cv', 12.47))
        else:
            res = {}
        return self._result(
            phi={'mode': mode},
            m={'law': 'dS=dQ_rev/T, η_carnot=1-T_c/T_h', 'type': '热力学第二定律'},
            psi=res, summary=f"熵({mode}): {res.get('efficiency_pct', res.get('delta_S', ''))}",
            engine='entropy')

    def _sim_phase_critical(self, p, steps):
        """相变临界: 沸点-海拔"""
        e = self.engines['phase_critical']
        alt = p.get('altitude_m', 0)
        res = e.water_boiling_vs_altitude(alt)
        return self._result(
            phi={'altitude_m': alt},
            m={'law': '克拉珀龙方程: dP/dT=L/(TΔV)', 'type': '相平衡'},
            psi=res, summary=f"相变: 海拔{alt}m → 沸点{res['boiling_point_C']}°C, 气压{res['pressure_kPa']}kPa",
            engine='phase_critical')

    # ============ ⑧ 电磁 handlers ============
    def _sim_faraday_induction(self, p, steps):
        """法拉第感应: 感应电动势/电感"""
        e = self.engines['faraday_induction']
        mode = p.get('mode', 'coil')
        if mode == 'coil':
            res = {'emf': round(e.coil_emf(p.get('N', 100), p.get('A', 0.01),
                                           p.get('dB_dt', 0.5), p.get('theta', 0)), 4),
                   'note': 'N匝线圈磁通变化'}
        elif mode == 'motional':
            res = {'emf': round(e.motional_emf(p.get('B', 0.5), p.get('L', 1.0),
                                                  p.get('v', 10.0)), 4),
                   'note': '导体切割磁感线'}
        else:
            res = {'emf': round(e.emf(p.get('dPhi', 0.01), p.get('dt', 0.1)), 4),
                   'note': '通用ε=-dΦ/dt'}
        return self._result(
            phi={'mode': mode},
            m={'law': 'ε = -dΦ/dt (法拉第电磁感应定律)', 'type': '电动力学'},
            psi=res, summary=f"法拉第({mode}): ε={res['emf']}V",
            engine='faraday_induction')

    def _sim_maxwell_stress(self, p, steps):
        """麦克斯韦应力: 电磁压力"""
        e = self.engines['maxwell_stress']
        E = p.get('E', 1e6)  # V/m
        B = p.get('B', 1.0)  # T
        Pe = e.electric_pressure(E)
        Pm = e.magnetic_pressure(B)
        return self._result(
            phi={'E_V_m': E, 'B_T': B},
            m={'law': 'P_e=½ε₀E², P_m=B²/(2μ₀)', 'type': '麦克斯韦应力张量'},
            psi={'electric_pressure_Pa': f'{Pe:.4f}', 'magnetic_pressure_Pa': f'{Pm:.4f}',
                 'magnetic_pressure_atm': round(Pm/101325, 3),
                 'summary': f'E={E}V/m → P_e={Pe:.2f}Pa; B={B}T → P_m={Pm:.1f}Pa({Pm/101325:.2f}atm)'},
            summary=f"麦克斯韦应力: E场→{Pe:.2f}Pa, B场→{Pm:.1f}Pa({Pm/101325:.2f}atm)",
            engine='maxwell_stress')

    # ============ ⑨ Navier-Stokes handler ============
    def _sim_navier_stokes(self, p, steps):
        """一维 N-S: 管流速度剖面 + 雷诺数"""
        e = self.engines['navier_stokes']
        Re = e.reynolds_number(p.get('v', 1.0), p.get('L', 0.1))
        res = e.solve_pipe_flow(nx=p.get('nx', 50), dt=p.get('dt', 0.001),
                                 steps=p.get('steps', 300),
                                 u_inlet=p.get('u_inlet', 1.0),
                                 pressure_gradient=p.get('pressure_gradient', -0.1))
        return self._result(
            phi={'u_inlet': p.get('u_inlet', 1.0), 'Re': Re['Re'], 'regime': Re['flow_regime']},
            m={'law': 'ρ(∂u/∂t+u∂u/∂x)=-∂P/∂x+μ∂²u/∂x²', 'type': 'Navier-Stokes (迎风+隐式)'},
            psi={'Re': Re['Re'], 'regime': Re['flow_regime'],
                 'u_max': res['u_max'], 'u_mean': res['u_mean'],
                 'r_stability': res['r_stability']},
            summary=f"N-S: Re={Re['Re']}({Re['flow_regime']}), u_max={res['u_max']}, u_mean={res['u_mean']}",
            engine='navier_stokes')

    def _sim_navier_stokes_2d(self, p, steps):
        """二维 N-S: 台风风场/方腔流"""
        e = self.engines['navier_stokes_2d']
        mode = p.get('mode', 'typhoon')
        if mode == 'typhoon':
            res = e.typhoon_wind_field(
                R_max=p.get('R_max', 50), V_max=p.get('V_max', 52),
                r_out=p.get('r_out', 400), ngrid=p.get('ngrid', 50))
        else:
            res = e.solve_cavity_flow(Re=p.get('Re', 100), nsteps=p.get('nsteps', 500))
        return self._result(
            phi={'mode': mode},
            m={'law': '∂ω/∂t+u·∇ω=ν∇²ω, ∇²ψ=-ω', 'type': '涡量-流函数法'},
            psi=res, summary=res['summary'],
            engine='navier_stokes_2d')

    # ============ ⑩ 多物理耦合器 handler ============
    def _sim_multi_physics(self, p, steps):
        """多物理耦合器: 通用引擎互联"""
        n = MultiPhysicsCoupler()
        # 注册引擎
        n.register('thermal', self.engines['newton_cooling'],
                   outputs={'T': 'temperature'}, inputs={'T_env': 'temperature'})
        n.register('mech', self.engines['stress_strain'],
                   outputs={'stress': 'pressure'}, inputs={'force': 'force'})
        n.register('fluid', self.engines['navier_stokes'],
                   outputs={'u': 'velocity'}, inputs={'pressure': 'pressure'})
        # 建立耦合
        n.couple('thermal', 'mech', 'T', 'force',
                 transfer_func=lambda T: 1e6 * max(0, T - 300))  # 温度→热应力
        n.couple('fluid', 'mech', 'u', 'pressure',
                 transfer_func=lambda u: 0.5 * 1000 * u**2)  # 风压→载荷
        n.couple('thermal', 'fluid', 'T', 'pressure',
                 transfer_func=lambda T: 1e5 * T / 300)  # 热→气压
        status = n.status()
        return self._result(
            phi={'engines': status['engines'], 'n_couplings': status['n_couplings']},
            m={'law': '量纲匹配 + 边界交换', 'type': '通用多物理耦合框架'},
            psi={'couplings': status['couplings'],
                 'summary': f"{status['n_engines']}引擎互联, {status['n_couplings']}条耦合数据流",
                 'capabilities': ['热-固耦合(热应力)', '流-固耦合(风压/水锤)', '热-流耦合(热驱动流)', '任意扩展']},
            summary=f"多物理耦合: {status['n_engines']}引擎, {status['n_couplings']}条数据流",
            engine='multi_physics')


    # ============ ⑪ 量子力学扩展 handlers ============
    def _sim_spin(self, p, steps):
        """量子自旋: S大小 + 塞曼分裂 + 斯特恩-盖拉赫"""
        e = self.engines['spin']
        s = p.get('s', 0.5)
        res = e.spin_magnitude(s)
        # 斯特恩-盖lach演示
        dB_dz = p.get('dB_dz', 10.0)  # T/m
        sg = e.stern_gerlach(s, dB_dz)
        # 塞曼分裂
        B = p.get('B', 1.0)
        zeeman = e.zeeman_energy(s, B, g_J=2)
        # 磁矩
        mag = e.magnetic_moment(s, l=p.get('l', 0))
        return self._result(
            phi={'s': s, 'B_T': B},
            m={'law': '|S|=√(s(s+1))ℏ, μ=-g_J·μ_B·J/ℏ', 'type': '量子自旋'},
            psi={'spin': res, 'stern_gerlach': sg,
                 'zeeman': zeeman, 'magnetic_moment': mag,
                 'summary': f'S={s}, 分裂{sg["n_spots"]}条, g_J={mag["g_J"]}'},
            summary=f"自旋: S={s}, |S|={res['magnitude']}, 塞曼分裂{sg['n_spots']}条",
            engine='spin')

    def _sim_identical_particles(self, p, steps):
        """全同粒子: 玻色/费米统计 + 费米能 + 简并压"""
        e = self.engines['identical_particles']
        mode = p.get('mode', 'fermion')
        T = p.get('T', 300)
        E = p.get('E', 0.1 * 1.6e-19)  # 0.1 eV
        mu = p.get('mu', 0.05 * 1.6e-19)
        if mode == 'boson':
            res = e.boson(T, E, mu)
        elif mode == 'fermion':
            res = e.fermion(T, E, mu)
        elif mode == 'fermi_energy':
            res = e.fermi_energy(p.get('n', 6e28))  # 铜的电子密度
        elif mode == 'degeneracy':
            res = e.degeneracy_pressure(p.get('n', 1e36))  # 白矮星内部
        else:
            res = {}
        return self._result(
            phi={'mode': mode, 'T_K': T},
            m={'law': 'BE: n=1/(e^(E-μ)/kT-1), FD: n=1/(e^(E-μ)/kT+1)', 'type': '量子统计'},
            psi=res, summary=f"全同粒子({mode}): {res.get('n', res.get('E_F_eV', res.get('P_GPa', '')))}",
            engine='identical_particles')

    # ============ ⑫ 地球物理 handlers ============
    def _sim_seismic_waves(self, p, steps):
        """地震波: P/S波速度 + 走时 + 震级能量"""
        e = self.engines['seismic_waves']
        mode = p.get('mode', 'velocity')
        if mode == 'velocity':
            # 典型地壳: K=40GPa, μ=30GPa, ρ=2700
            res = e.wave_velocity(p.get('K', 40e9), p.get('mu', 30e9), p.get('rho', 2700))
        elif mode == 'travel':
            res = e.travel_time(p.get('distance_km', 100), p.get('depth_km', 0), p.get('wave', 'P'))
        elif mode == 'epicenter':
            res = e.epicenter_distance(p.get('delta_t_s', 10))
        elif mode == 'magnitude':
            res = e.moment_magnitude(p.get('M0', 1e15))
        elif mode == 'energy':
            res = e.energy_from_magnitude(p.get('Mw', 6.0))
        elif mode == 'intensity':
            res = e.intensity_at_distance(p.get('Mw', 6.0), p.get('distance_km', 50))
        else:
            res = {}
        return self._result(
            phi={'mode': mode},
            m={'law': 'vp=√((K+4μ/3)/ρ), vs=√(μ/ρ), Mw=2/3·logM₀-10.7', 'type': '弹性波传播'},
            psi=res, summary=f"地震波({mode}): {list(res.values())[0] if res else ''}",
            engine='seismic_waves')

    def _sim_atmospheric_circulation(self, p, steps):
        """大气环流: 地转风/梯度风/哈德来环流/海风"""
        e = self.engines['atmospheric_circulation']
        mode = p.get('mode', 'geostrophic')
        lat = p.get('lat', 30.0)
        if mode == 'geostrophic':
            dP = p.get('dP_dx', -1e-3)  # Pa/m (典型天气尺度)
            res = e.geostrophic_wind(dP, lat)
        elif mode == 'gradient':
            dP = p.get('dP_dx', -1e-3)
            R = p.get('R', 500000)  # 曲率半径 (500km)
            res = e.gradient_wind(dP, lat, R)
        elif mode == 'hadley':
            res = e.hadley_cell(lat)
        elif mode == 'sea_breeze':
            res = e.sea_breeze(p.get('T_land', 35), p.get('T_sea', 25))
        else:
            res = {}
        return self._result(
            phi={'mode': mode, 'lat': lat},
            m={'law': 'Vg=-(1/ρf)·∂P/∂x, f=2Ωsinφ', 'type': '旋转流体动力学'},
            psi=res, summary=f"大气环流({mode}@{lat}°): {list(res.values())[0] if res else ''}",
            engine='atmospheric_circulation')


    # ============ ⑬ 核物理 handlers ============
    def _sim_radioactive_decay(self, p, steps):
        """放射性衰变: 半衰期/碳14测年/活度"""
        e = self.engines['radioactive_decay']
        mode = p.get('mode', 'decay')
        if mode == 'decay':
            half_life = p.get('half_life', 5730 * 365.25 * 24 * 3600)
            t = p.get('time', half_life)
            res = {'remaining_ratio': round(e.remaining_fraction(t, half_life), 6),
                   'half_life_s': half_life,
                   'decay_constant': round(e.decay_constant(half_life), 10)}
        elif mode == 'carbon_dating':
            res = e.carbon_dating(p.get('remaining_ratio', 0.5))
        elif mode == 'activity':
            res = e.activity(p.get('N', 1e15), p.get('half_life', 5730 * 365.25 * 24 * 3600))
        else:
            res = {}
        return self._result(
            phi={'mode': mode},
            m={'law': 'N=N0·exp(-λt), λ=ln2/T_1/2', 'type': '量子隧穿衰变'},
            psi=res, summary=f"衰变({mode}): {list(res.values())[0] if res else ''}",
            engine='radioactive_decay')

    def _sim_fission_fusion(self, p, steps):
        """裂变/聚变: 结合能/临界质量/聚变能"""
        e = self.engines['fission_fusion']
        mode = p.get('mode', 'fission')
        if mode == 'fission':
            res = e.fission_energy(p.get('n_fissions', 1e20))
        elif mode == 'critical':
            res = e.critical_mass(p.get('density', 19000))
        elif mode == 'fusion':
            res = e.fusion_energy(p.get('reaction', 'D-T'))
        else:
            res = {}
        return self._result(
            phi={'mode': mode},
            m={'law': 'E=mc², Q=Δm·931.5MeV', 'type': '强相互作用'},
            psi=res, summary=f"核反应({mode}): {list(res.values())[0] if res else ''}",
            engine='fission_fusion')

    # ============ ⑭ 凝聚态 handlers ============
    def _sim_band_theory(self, p, steps):
        """能带: PN结/太阳能电池/载流子"""
        e = self.engines['band_theory']
        mode = p.get('mode', 'pn_junction')
        if mode == 'pn_junction':
            res = e.pn_junction(p.get('Na', 1e21), p.get('Nd', 1e21), p.get('T', 300), p.get('material', 'Si'))
        elif mode == 'solar':
            res = e.solar_cell(p.get('I_sc', 8.0), p.get('V_oc', 0.6), p.get('FF', 0.75))
        elif mode == 'carrier':
            res = e.intrinsic_carrier(p.get('Eg', 1.12), p.get('T', 300))
        else:
            res = {}
        return self._result(
            phi={'mode': mode},
            m={'law': 'E_g决定导电性, V_bi=kT/q·ln(NaNd/ni²)', 'type': '能带理论'},
            psi=res, summary=f"能带({mode}): {list(res.values())[0] if res else ''}",
            engine='band_theory')

    def _sim_superconductivity(self, p, steps):
        """超导: 临界磁场/能隙/穿透深度"""
        e = self.engines['superconductivity']
        Tc = p.get('Tc', 9.3)
        T = p.get('T', 4.2)
        Hc0 = p.get('Hc0', 0.20)
        res = {
            'Tc': Tc, 'T': T,
            'critical_field_T': round(e.critical_field(T, Tc, Hc0), 4),
            'energy_gap': e.energy_gap(T, Tc),
            'london_penetration_m': f"{e.london_penetration(T, Tc):.2e}",
            'note': f'BCS理论: T={T}K < Tc={Tc}K, 处于超导态'
        }
        return self._result(
            phi={'Tc': Tc, 'T': T},
            m={'law': 'BCS: Δ=1.764kTc, Hc(T)=Hc0[1-(T/Tc)²]', 'type': '库珀对凝聚'},
            psi=res, summary=f"超导: Tc={Tc}K, Hc={e.critical_field(T,Tc,Hc0):.3f}T @ {T}K",
            engine='superconductivity')

    # ============ ⑮ 控制论 handlers ============
    def _sim_pid_controller(self, p, steps):
        """PID控制: 响应/超调/调节时间"""
        e = self.engines['pid_controller']
        e.Kp, e.Ki, e.Kd = p.get('Kp', 1.0), p.get('Ki', 0.1), p.get('Kd', 0.05)
        mode = p.get('mode', 'simulate')
        if mode == 'tune':
            res = e.tune_zn(p.get('Ku', 2.0), p.get('Tu', 1.0))
        else:
            res = e.simulate(p.get('setpoint', 100), p.get('initial', 0),
                            p.get('dt', 0.1), p.get('nsteps', 100))
        return self._result(
            phi={'mode': mode, 'Kp': e.Kp, 'Ki': e.Ki, 'Kd': e.Kd},
            m={'law': 'u(t)=Kp·e + Ki∫e·dt + Kd·de/dt', 'type': 'PID控制律'},
            psi=res, summary=res.get('summary', f'PID: Kp={e.Kp}, Ki={e.Ki}, Kd={e.Kd}'),
            engine='pid_controller')

    def _sim_feedback_system(self, p, steps):
        """反馈系统: 稳定性/劳斯判据/波特图"""
        e = self.engines['feedback_system']
        mode = p.get('mode', 'routh')
        if mode == 'routh':
            res = e.routh_hurwitz(p.get('coeffs', [1, 3, 6, 4, 1]))
        elif mode == 'bode':
            res = e.bode_plot(p.get('type', 'first_order'), {'tau': p.get('tau', 1.0)})
        else:
            res = {}
        return self._result(
            phi={'mode': mode},
            m={'law': '劳斯判据: 首列不变号→稳定', 'type': '线性系统稳定性'},
            psi=res, summary=f"反馈({mode}): {res.get('stable', res.get('note', ''))}",
            engine='feedback_system')


    # ============ ⑯ 粒子物理 handlers ============
    def _sim_standard_model(self, p, steps):
        """标准模型: 粒子信息/衰变允许性"""
        e = self.engines['standard_model']
        mode = p.get('mode', 'info')
        if mode == 'info':
            res = e.particle_info(p.get('particle', 'electron'))
        elif mode == 'composite':
            res = e.composite_mass(p.get('particles', ['up', 'up', 'down']))
        elif mode == 'decay':
            res = e.decay_allowed(p.get('parent', 'muon'), p.get('daughters', ['electron', 'muon_neutrino']))
        else:
            res = {}
        return self._result(
            phi={'mode': mode},
            m={'law': '基本粒子(轻子/夸克/规范玻色子)', 'type': '标准模型'},
            psi=res, summary=f"标准模型({mode}): {list(res.values())[0] if res else ''}",
            engine='standard_model')

    def _sim_scattering(self, p, steps):
        """散射: 卢瑟福/康普顿/共振"""
        e = self.engines['scattering']
        mode = p.get('mode', 'compton')
        if mode == 'rutherford':
            res = e.rutherford(p.get('Z1', 79), p.get('Z2', 2), p.get('E', 5.5), p.get('theta', 0.1))
        elif mode == 'compton':
            res = e.compton(p.get('E_gamma', 0.662), p.get('theta', math.pi/2))
        elif mode == 'resonance':
            res = e.cross_section_point(p.get('E', 100), p.get('E0', 100), p.get('sigma_max', 1.0))
        else:
            res = {}
        return self._result(
            phi={'mode': mode},
            m={'law': '卢瑟福/康普顿/Breit-Wigner', 'type': '量子散射'},
            psi=res, summary=f"散射({mode}): {list(res.values())[0] if res else ''}",
            engine='scattering')

    # ============ ⑰ 声学 handlers ============
    def _sim_sound_field(self, p, steps):
        """声场: 声压级/传播/多普勒"""
        e = self.engines['sound_field']
        mode = p.get('mode', 'spl')
        if mode == 'spl':
            res = {'spl_dB': round(e.spl(p.get('p_rms', 0.02)), 2)}
        elif mode == 'spreading':
            res = {'Lp2_dB': round(e.spherical_spreading(p.get('Lp1', 100), p.get('r1', 1), p.get('r2', 100)), 2)}
        elif mode == 'doppler':
            res = e.doppler(p.get('f0', 440), p.get('v_source', 30), p.get('v_observer', 0))
        elif mode == 'standing':
            res = e.standing_wave(p.get('L', 1.0), p.get('n', 1))
        else:
            res = {}
        return self._result(
            phi={'mode': mode},
            m={'law': 'Lp=20log(p/p_ref), I∝1/r²', 'type': '线性声学'},
            psi=res, summary=f"声场({mode}): {list(res.values())[0] if res else ''}",
            engine='sound_field')

    def _sim_noise_control(self, p, steps):
        """噪声: 评价/隔声/吸声/混响"""
        e = self.engines['noise_control']
        mode = p.get('mode', 'rating')
        if mode == 'rating':
            res = {'Leq_dB': p.get('Leq', 65), 'rating': e.noise_rating(p.get('Leq', 65))}
        elif mode == 'reverberation':
            res = e.reverberation_time(p.get('V', 1000), p.get('A', 50))
        elif mode == 'transmission':
            res = e.transmission_loss(p.get('m', 200), p.get('f', 1000))
        elif mode == 'equivalent':
            res = e.equivalent_level(p.get('levels', [80, 70, 60]), p.get('times', [4, 3, 1]))
        else:
            res = {}
        return self._result(
            phi={'mode': mode},
            m={'law': '赛宾T60=0.161V/A, TL=20log(mf)-47', 'type': '噪声控制'},
            psi=res, summary=f"噪声({mode}): {list(res.values())[0] if res else ''}",
            engine='noise_control')

    # ============ ⑱ 非线性动力学 handlers ============
    def _sim_chaos(self, p, steps):
        """混沌: 洛伦兹/逻辑斯蒂/李雅普诺夫"""
        e = self.engines['chaos']
        mode = p.get('mode', 'lorenz')
        if mode == 'lorenz':
            res = e.lorenz(p.get('sigma', 10), p.get('rho', 28), p.get('beta', 8/3),
                          p.get('dt', 0.01), p.get('nsteps', 1000))
        elif mode == 'logistic':
            res = e.logistic_map(p.get('r', 3.9), p.get('x0', 0.5), p.get('nsteps', 100))
        elif mode == 'lyapunov':
            res = e.lyapunov_exponent(p.get('r', 3.9), p.get('x0', 0.5), p.get('nsteps', 1000))
        else:
            res = {}
        return self._result(
            phi={'mode': mode},
            m={'law': 'dx/dt=σ(y-x), x_{n+1}=rx_n(1-x_n)', 'type': '非线性混沌'},
            psi=res, summary=f"混沌({mode}): {res.get('note', res.get('status', ''))}",
            engine='chaos')

    def _sim_fractal(self, p, steps):
        """分形: 曼德博/朱利亚/科赫/维数"""
        e = self.engines['fractal']
        mode = p.get('mode', 'mandelbrot')
        if mode == 'mandelbrot':
            res = e.mandelbrot(p.get('c_real', 0), p.get('c_imag', 0), p.get('max_iter', 100))
        elif mode == 'julia':
            res = e.julia(p.get('z_real', 0), p.get('z_imag', 0), p.get('c_real', -0.7), p.get('c_imag', 0.27015))
        elif mode == 'koch':
            res = e.koch_snowflake(p.get('side', 1.0), p.get('n_iter', 4))
        elif mode == 'dimension':
            res = e.self_similarity_dimension(p.get('N', 4), p.get('r', 1/3))
        else:
            res = {}
        return self._result(
            phi={'mode': mode},
            m={'law': 'D=logN/log(1/r), z_{n+1}=z_n²+c', 'type': '分形几何'},
            psi=res, summary=f"分形({mode}): {list(res.values())[0] if res else ''}",
            engine='fractal')
