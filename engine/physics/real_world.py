"""physics_engine/real_world.py — 真实世界感知层 (数据同化)"""
import json, subprocess, os, math, urllib.request

# L2 完整真实物理常数库
from .physical_constants import (Atmosphere, Water, MATERIALS, THERMAL,
                                 BUILDING, get_material, list_materials, CommonFluids)

def _shell(cmd, timeout=10):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception:
        return ""

def real_location():
    """返回真实定位 (iOS apple-location)"""
    d = {}
    out = _shell("apple-location -q")
    if out:
        try:
            d2 = json.loads(out)
            if isinstance(d2, dict):
                d = d2
        except Exception:
            pass
    if d:
        return {'lat': d.get('latitude') or d.get('lat'),
                'lon': d.get('longitude') or d.get('lon'),
                'altitude_m': d.get('altitude_m') or 18.0,
                'source': d.get('tool', 'apple-location')}
    return {'lat': 22.54, 'lon': 114.07, 'altitude_m': 18.0, 'source': 'fallback-shenzhen'}

def real_weather(lat=None, lon=None):
    """返回真实天气 (高德API, 深圳城市码440300)"""
    key = os.environ.get('AMAP_KEY')
    if not key:
        for p in ['/var/minis/shared/amap_key', '/var/minis/workspace/amap_key']:
            if os.path.exists(p):
                key = open(p).read().strip()
                if key: break
    if not key:
        return None
    for city in ['440300', '440100', '450100']:
        try:
            url = f"https://restapi.amap.com/v3/weather/weatherInfo?key={key}&city={city}&extensions=base"
            req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=8) as r:
                dat = json.loads(r.read().decode('utf-8'))
            if dat.get('status') != '1': continue
            lf = dat['lives'][0]
            return {'T_celsius': float(lf.get('temperature', 25)),
                    'humidity_pct': float(lf.get('humidity', 60)),
                    'pressure_hpa': float(lf.get('pressure', 1013)) if lf.get('pressure') else 1013,
                    'weather': lf.get('weather'), 'wind_dir': lf.get('winddirection'),
                    'wind_power': lf.get('windpower'), 'city': lf.get('city'),
                    'source': 'amap-live-' + city}
        except Exception:
            continue
    return None

def air_density(altitude_m=0, T_celsius=25.0):
    """真实空气密度: 用标准大气类(海拔+温度)"""
    T_ref = Atmosphere.temperature(altitude_m)
    # 用标准大气该海拔的温度, 或用户提供温度
    return Atmosphere.density(altitude_m, T_celsius)

def sound_speed(T_celsius=25.0):
    """真实声速: c = sqrt(γRT)"""
    return Atmosphere.sound_speed(0, T_celsius)

MATERIALS = {
    'concrete_C25': {'E_GPa': 30, 'density': 2400, 'yield_MPa': 25, 'note': 'C25混凝土'},
    'steel_Q235':   {'E_GPa': 200, 'density': 7850, 'yield_MPa': 235, 'note': 'Q235钢'},
    'glass':        {'E_GPa': 70, 'density': 2500, 'yield_MPa': 50, 'note': '钠钙玻璃'},
    'brick':        {'E_GPa': 8, 'density': 1800, 'yield_MPa': 10, 'note': '红砖墙'},
    'aluminum_6061':{'E_GPa': 69, 'density': 2700, 'yield_MPa': 275, 'note': '6081铝'},
}

def material(name):
    return MATERIALS.get(name, {'note': 'unknown-' + str(name)})

def sensor_env(**overrides):
    """整合真实感知 → 环境输入dict(引擎缺省参数从此取真实值)"""
    loc = real_location()
    wx = real_weather(loc.get('lat'), loc.get('lon'))
    alt = overrides.get('altitude_m', loc.get('altitude_m') or 18.0)
    T_c = float(wx['T_celsius']) if wx else 25.0
    env = {
        'T_env_K': T_c + 273.15,
        'T_env_C': T_c,
        'air_density_kgm3': air_density(alt, T_c),
        'sound_speed_ms': sound_speed(T_c),
        'humidity_pct': wx['humidity_pct'] if wx else None,
        'pressure_hpa': wx['pressure_hpa'] if wx else 1013.25,
        'weather': wx['weather'] if wx else None,
        'altitude_m': float(alt),
        'location': {'lat': loc.get('lat'), 'lon': loc.get('lon'), 'source': loc.get('source')},
        'weather_source': wx['source'] if wx else 'none',
        'provenance': {
            'T_env': 'amap-real' if wx else 'default-25C',
            'air_density': 'std-atmos'+str(alt)+'m',
            'sound_speed': 'temp-formula',
        },
    }
    env.update(overrides)
    return env

if __name__ == '__main__':
    import json
    print(json.dumps(sensor_env(), ensure_ascii=False, indent=2))
