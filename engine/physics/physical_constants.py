"""physics_engine/physical_constants.py — 全面真实物理常数库 (L2)
材料 / 大气 / 流体 / 热物理 的真实参数(可查询, 随温度/海拔/材料变化)
取代硬编码占位, 引擎从此查库取真实常数。
"""
import math

# ============ 1. 工程材料库 (E:弹性GPa, 密度, 屈服MPa, 线性热膨胀1e-6/K, 导热W/mK) ============
MATERIALS = {
    # 结构材料
    'concrete_C25': {'E_GPa':30,'density':2400,'yield_MPa':25,'cte':10e-6,'k':1.7,'note':'C25混凝土(民用)','n_poisson':0.2},
    'concrete_C40': {'E_GPa':33,'density':2400,'yield_MPa':40,'cte':10e-6,'k':1.7,'note':'C40混凝土(高层)','n_poisson':0.2},
    'concrete_C60': {'E_GPa':36,'density':2450,'yield_MPa':60,'cte':10e-6,'k':1.8,'note':'C60混凝土(桥梁)','n_poisson':0.2},
    'steel_Q235':   {'E_GPa':200,'density':7850,'yield_MPa':235,'cte':12e-6,'k':50,'note':'Q235普通碳钢','n_poisson':0.3},
    'steel_HRB400': {'E_GPa':200,'density':7850,'yield_MPa':400,'cte':12e-6,'k':50,'note':'HRB400钢筋','n_poisson':0.3},
    'steel_304':    {'E_GPa':193,'density':7930,'yield_MPa':215,'cte':17e-6,'k':16,'note':'304不锈钢','n_poisson':0.29},
    # 玻璃
    'glass':        {'E_GPa':70,'density':2500,'yield_MPa':50,'cte':9e-6,'k':1.0,'note':'钠钙玻璃(门窗)','n_poisson':0.23},
    'toughened_glass':{'E_GPa':72,'density':2500,'yield_MPa':150,'cte':9e-6,'k':1.0,'note':'钢化玻璃','n_poisson':0.23},
    # 砌体
    'brick':        {'E_GPa':8,'density':1800,'yield_MPa':10,'cte':6e-6,'k':0.7,'note':'红砖墙体','n_poisson':0.15},
    'brick_wall':   {'E_GPa':3,'density':1700,'yield_MPa':5,'cte':6e-6,'k':0.6,'note':'砖砌体(灰缝),(抗压~5-10MPa)','n_poisson':0.15},
    # 金属
    'aluminum_6061':{'E_GPa':69,'density':2700,'yield_MPa':275,'cte':23.6e-6,'k':167,'note':'6061铝合金(门窗/车架)','n_poisson':0.33},
    'aluminum_7075':{'E_GPa':72,'density':2800,'yield_MPa':503,'cte':23.6e-6,'k':130,'note':'7075航空铝','n_poisson':0.33},
    'titanium_Ti6Al4V':{'E_GPa':114,'density':4430,'yield_MPa':880,'cte':8.6e-6,'k':6.7,'note':'Ti-6Al-4V钛合金','n_poisson':0.34},
    # 木材/塑料/其他
    'oak_wood':     {'E_GPa':11,'density':700,'yield_MPa':40,'cte':5e-6,'k':0.16,'note':'橡木(顺纹抗)','n_poisson':0.3},
    'pine_wood':    {'E_GPa':10,'density':500,'yield_MPa':35,'cte':5e-6,'k':0.12,'note':'松木','n_poisson':0.3},
    'plywood':      {'E_GPa':9,'density':600,'yield_MPa':30,'cte':6e-6,'k':0.13,'note':'胶合板','n_poisson':0.3},
    'ABS_plastic':  {'E_GPa':2,'density':1050,'yield_MPa':40,'cte':90e-6,'k':0.2,'note':'ABS塑料(家电)','n_poisson':0.35},
    'PVC':          {'E_GPa':3,'density':1400,'yield_MPa':50,'cte':80e-6,'k':0.16,'note':'PVC塑料(管道)','n_poisson':0.4},
    'rubber':       {'E_GPa':0.01,'density':1100,'yield_MPa':5,'cte':200e-6,'k':0.15,'note':'橡胶(密封)','n_poisson':0.49},
    'neoprene':     {'E_GPa':0.005,'density':1250,'yield_MPa':8,'cte':180e-6,'k':0.23,'note':'氯丁橡胶','n_poisson':0.49},
    # 岩石/土壤
    'granite':      {'E_GPa':50,'density':2700,'yield_MPa':140,'cte':8e-6,'k':3.0,'note':'花岗岩','n_poisson':0.25},
    'limestone':    {'E_GPa':40,'density':2600,'yield_MPa':80,'cte':8e-6,'k':2.5,'note':'石灰岩','n_poisson':0.25},
    'sandy_soil':   {'E_GPa':0.02,'density':1600,'yield_MPa':0.2,'cte':None,'k':1.0,'note':'砂质土','n_poisson':0.3},
}

# ============ 2. 标准大气属性 (随海拔/温度, 真实公式) ============
class Atmosphere:
    """标准大气: 密度/气压/温度/声速/粘度 随海拔"""
    R = 287.05          # 空气比气体常数 J/(kg·K)
    g = 9.80665
    gamma = 1.4         # 绝热指数空气
    T0 = 288.15         # 海平面标准温度 K
    
    @staticmethod
    def pressure(altitude):
        """气压随海拔: 对流层(<11000m)指数律, 平流层线性"""
        h = altitude
        if h < 11000:
            return 101325.0 * (1 - 2.25577e-5 * h)**5.2559
        else:
            return 22632.0 * math.exp(-(h-11000)/6341.6)
    
    @staticmethod
    def temperature(altitude):
        """温度随海拔: -6.5K/km(对流层), 平流层~216.65K"""
        if altitude < 11000:
            return 288.15 - 0.0065 * altitude
        return 216.65
    
    @staticmethod
    def density(altitude, T_celsius=None):
        """空气密度: 标准大气 + 可选温度修正"""
        P = Atmosphere.pressure(altitude)
        if T_celsius is None:
            T = Atmosphere.temperature(altitude)
        else:
            T = T_celsius + 273.15
        return P / (Atmosphere.R * T)
    
    @staticmethod
    def sound_speed(altitude=None, T_celsius=None):
        """声速: c = sqrt(γRT), 温度主导"""
        if T_celsius is None:
            T = Atmosphere.temperature(altitude or 0)
        else:
            T = T_celsius + 273.15
        return math.sqrt(Atmosphere.gamma * Atmosphere.R * T)
    
    @staticmethod
    def dynamic_viscosity(T_celsius=20):
        """空气动力粘度(Sutherland公式), Pa·s"""
        Tk = T_celsius + 273.15
        return 1.458e-6 * Tk**1.5 / (Tk + 110.4)
    
    @staticmethod
    def thermal_conductivity(T_celsius=20):
        """空气导热率 W/mK"""
        Tk = T_celsius + 273.15
        return 0.0263 + 0.000074 * (Tk - 300)

# ============ 3. 流体属性 (水/常见流体 随温度) ============
class Water:
    """水的真实物性(温度函数)"""
    @staticmethod
    def density(T_celsius=25):
        """水的密度 kg/m³ (4°C最大, 真实经验式)"""
        T = T_celsius
        return 1000.0 * (1 - (T + 288.9414)/(508929.2*(T + 68.12963))*(T - 3.9863)**2)
    
    @staticmethod
    def dynamic_viscosity(T_celsius=20):
        """水动力粘度 Pa·s (Vogel近似)"""
        return 0.001308 * 10**(247.8/(T_celsius+273.15-140) - 0.777)
    
    @staticmethod
    def thermal_conductivity(T_celsius=20):
        """水导热率 W/mK"""
        return 0.6065 + 0.0017*(T_celsius - 20)
    
    @staticmethod
    def specific_heat(T_celsius=20):
        """水比热 J/(kg·K)"""
        return 4184.0  # 近常数
    
    @staticmethod
    def surface_tension(T_celsius=20):
        """表面张力 N/m"""
        return 0.0757 - 0.00014*(T_celsius - 4)

class CommonFluids:
    """常见流体密度/粘度/导热率 kg/m³, cP, W/(m·K)"""
    AIR        = {'density':1.225,'viscosity_cP':0.018,'k':0.026,'note':'空气(海平面20°C)'}
    WATER      = {'density':998,'viscosity_cP':1.0,'k':0.6,'note':'水(20°C)'}
    SEA_WATER  = {'density':1025,'viscosity_cP':1.1,'k':0.58,'note':'海水(盐度35‰)'}
    GASOLINE   = {'density':750,'viscosity_cP':0.5,'k':0.13,'note':'汽油'}
    DIESEL     = {'density':850,'viscosity_cP':3.0,'k':0.14,'note':'柴油'}
    CRUDE_OIL  = {'density':870,'viscosity_cP':10.0,'k':0.14,'note':'原油(轻质)'}
    BLOOD      = {'density':1060,'viscosity_cP':4.0,'k':0.5,'note':'血液(37°C)'}
    ETHANOL    = {'density':789,'viscosity_cP':1.2,'k':0.17,'note':'乙醇'}
    MERCURY    = {'density':13534,'viscosity_cP':1.5,'k':8.3,'note':'汞'}
    STEAM      = {'density':0.6,'viscosity_cP':0.01,'k':0.025,'note':'水蒸气(100°C)'}

# ============ 4. 热物理通用 ============
THERMAL = {
    'fusion_heat_water': 334e3,      # J/kg 冰熔化
    'vap_heat_water': 2260e3,        # J/kg 水汽化
    'fusion_point_water': 273.15,    # K
    'boil_point_water': 373.15,      # K
    'c_water': 4186,                 # J/(kg·K)
    'c_ice': 2090,
    'c_steam': 2010,
    'c_air_const_p': 1005,           # J/(kg·K) 定压
    'c_air_const_v': 718,
    'c_steel': 500,
    'c_concrete': 880,
    'c_brick': 840,
    'c_aluminum': 900,
    'solar_constant': 1361,          # W/m² 太阳常数
    'stefan_boltzmann': 5.67e-8,     # W/(m²·K⁴)
    'thermal_diffusivity_air': 2.2e-5,  # m²/s 热扩散系数
}

# ============ 5. 结构风荷载/其他常用 ============
BUILDING = {
    'wind_pressure_coef_wall': 0.8,  # 迎风面风荷载体型系数
    'fenestration_glass_breaking_kPa': 3.5,  # 玻璃破碎超压阈值 kPa (工程)
    'brick_wall_collapse_kPa': 60,   # 砖墙坍塌超压 kPa
    'concrete_wall_crack_kPa': 25,   # 混凝土墙开裂超压 kPa
    'window_frame_damage_kPa': 7,    # 窗框损坏超压 kPa
    'human_ear_damage_kPa': 35,      # 人耳损伤超压 kPa
    'human_50killed_kPa': 400,       # 50%致死超压 kPa
}

# ============ 抗风工程设计库 (按承载能力设计法) ============
WIND_DESIGN = {
    'gust_factor': 1.8,       # 阵风系数: 10min平均风→3s阵风(中国荷载规范推荐~1.5-2.0)
    'shape_coefficients': {   # 风荷载体型系数 μs (各部位)
        'wall':       0.8,    # 迎风墙面
        'leeward':   -0.5,    # 背风面(吸力)
        'wall_side': -0.7,    # 侧墙(吸力)
        'window':     1.0,    # 窗户幕墙
        'roof_up':   -0.6,    # 屋面(吸力,平屋顶)
        'roof_slope':-0.9,    # 坡屋顶/彩钢(强吸力)
        'frame':      1.0,    # 门窗框
        'light':      1.3,    # 临时/轻型构件(风振大)
        'concrete':   0.8,    # 混凝土结构
    },
    # 各部位允许风压承载力(kPa, 工程经验/试验值)
    'allowable_wind_pressure_kPa': {
        'wall':       2.0,    # 砖混砌体抗压(安全富余)
        'window':     1.2,    # 玻璃幕墙/窗(整片抗风, 易碎限值)
        'window_glass':1.2,   # 普通玻璃窗抗风
        'toughened':  1.8,    # 钢化玻璃
        'frame':      1.0,    # 铝合金门窗框(薄板易变形)
        'roof_concrete':2.5,  # 混凝土屋面
        'roof_steel': 1.5,    # 彩钢板屋面(需打钉牢固)
        'light':      0.5,    # 临时搭建/简易棚(极低, 严禁)
        'concrete':   5.0,    # 混凝土承重墙
        'brick':      2.0,    # 砖砌体墙
    },
}

def get_shape_coef(part):
    """查部位体型系数"""
    return WIND_DESIGN.get('shape_coefficients', {}).get(part, 0.8)

def get_allowable_wind(part):
    """查部位允许风压承载力 kPa"""
    return WIND_DESIGN.get('allowable_wind_pressure_kPa', {}).get(part, 1.0)

def get_material(name):
    """按名称查材料真实参数"""
    for k, v in MATERIALS.items():
        if name.lower() in k.lower() or k.lower() in name.lower():
            return {'name': k, **v}
    return {'name': name, 'note': 'unknown-material', 'E_GPa': None}

def list_materials():
    return sorted(MATERIALS.keys())

if __name__ == '__main__':
    print("物理常数库 L2 自检:")
    print("材料数:", len(MATERIALS))
    print("材料:", list_materials()[:8], "...")
    print()
    for alt in [0, 1900, 5500, 10000]:
        print(f"海拔{alt:>5}m: ρ={Atmosphere.density(alt):.4f}kg/m³ T={Atmosphere.temperature(alt):.1f}K P={Atmosphere.pressure(alt)/1000:.0f}kPa 声速={Atmosphere.sound_speed(alt):.0f}m/s")
    print()
    for t in [0, 20, 50, 100]:
        print(f"水{t:>3}°C: ρ={Water.density(t):.1f}kg/m³ 粘度={Water.dynamic_viscosity(t)*1000:.2f}cP")
    print()
    print("C40混凝土:", get_material('C40')['E_GPa'], "GPa")
    print("钢化玻璃超压阈值:", BUILDING['fenestration_glass_breaking_kPa'], "kPa")
