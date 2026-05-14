#!/usr/bin/env python3
"""规则引擎 - 从evolution_calibration.json动态加载规则
修改JSON即可生效，无需改代码。JSON路径: data/evolution_calibration.json"""
import os, json

_CAL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'evolution_calibration.json')
_db_cache = None

# ─── 硬编码默认值（仅在JSON加载失败时使用）──────────

_DEFAULT_TIER = {
    '人强1': {'max_odds': 1.35, 'conf': 85, 'hit': 87.5},
    '人强2': {'max_odds': 1.65, 'conf': 65, 'hit': 63.6},
    '普强':  {'max_odds': 2.00, 'conf': 40, 'hit': 37.5, 'warning': '⚠️ 1.90-1.99死亡区间'},
    '准强':  {'max_odds': 2.20, 'conf': 40, 'hit': 37.5},
    '中游':  {'max_odds': 2.80, 'conf': 45, 'hit': 50.0},
    '下游':  {'max_odds': 99.9, 'conf': 45, 'hit': 50.0},
}

_DEFAULT_FINE_GRAIN = {
    (1.10, 1.19): {'w':75,'d':25,'l':0,'n':4,'conf':90},
    (1.20, 1.39): {'w':100,'d':0,'l':0,'n':6,'conf':95},
    (1.40, 1.49): {'w':60,'d':40,'l':0,'n':5,'conf':70,'note':'平局高发'},
    (1.50, 1.59): {'w':50,'d':25,'l':25,'n':4,'conf':55},
    (1.80, 1.89): {'w':75,'d':25,'l':0,'n':4,'conf':60},
    (1.90, 1.99): {'w':0,'d':0,'l':100,'n':3,'conf':30,'warning':'🚨死亡区间'},
    (2.00, 2.09): {'w':25,'d':25,'l':50,'n':4,'conf':35,'note':'客胜率高'},
}

_DEFAULT_DRAW = {
    'low':      {'max': 3.0, 'result': '分胜负', 'draw_rate': 20.0},
    'mid':      {'max': 3.5, 'result': '平局略多', 'draw_rate': 29.4},
    'mid_high': {'max': 4.0, 'result': '分胜负', 'draw_rate': 23.1},
    'high':     {'max': 99.9, 'result': '不防平', 'draw_rate': 7.7},
}

_DEFAULT_LEAGUE = {
    '欧冠': {'adjust': 20, 'note': '✅极高置信'}, '欧罗巴': {'adjust': 10, 'note': '✅中等可信'},
    '英超': {'adjust': -20, 'note': '🚨严重不准'}, '德甲': {'adjust': -10, 'note': '⚠️冷门多'},
}

_DEFAULT_HANDICAP = {
    '-1': {'warning': '⚠️深盘谨慎'}, '-2': {'warning': '🚫避免预测'}, '+1': {'note': '✅受让方向较准'},
}

_DEFAULT_ML = {'high': {'min': 55, 'rate': 100.0}, 'mid': {'min': 45, 'rate': 78.3}, 'low': {'min': 0, 'rate': 53.6}}

_DEFAULT_BAODIAN = [
    {'name':'超低赔','lo':1.00,'hi':1.20,'expected_rate':90},
    {'name':'低赔','lo':1.20,'hi':1.45,'expected_rate':80},
    {'name':'中低赔','lo':1.45,'hi':1.80,'expected_rate':65},
    {'name':'中赔','lo':1.80,'hi':2.30,'expected_rate':55},
    {'name':'中高赔','lo':2.30,'hi':3.00,'expected_rate':45},
    {'name':'高赔','lo':3.00,'hi':99.9,'expected_rate':35},
]

# ─── JSON加载 ───────────────────────────────────────

def _load_cal():
    try:
        with open(_CAL_FILE) as f:
            return json.load(f)
    except Exception:
        return {}

def _cal(key):
    return _load_cal().get(key, {})

def _get_tiers():
    return _cal('tier_rules') or _DEFAULT_TIER

def _get_fine_grain():
    d = _cal('fine_grain_rules')
    if d:
        return {(float(k.split('-')[0]), float(k.split('-')[1])): v for k, v in d.items()}
    return _DEFAULT_FINE_GRAIN

def _get_draw():
    d = _cal('draw_rules')
    return d if d else _DEFAULT_DRAW

def _get_league():
    d = _cal('league_adjust')
    if d:
        # JSON中league_adjust应存整数（如20=+20pp），与_DEFAULT_LEAGUE格式一致
        # 注意：若存小数（如0.2），会因int(0.2*100)=20碰巧正确，但int(20*100)=2000是灾难
        return {lg: {'adjust': int(adj), 'note': '✅正向' if adj > 0 else ('🚨负向' if adj < 0 else '⚠️中性')}
                for lg, adj in d.items()}
    return _DEFAULT_LEAGUE

def _get_handicap():
    return _cal('handicap_rules') or _DEFAULT_HANDICAP

def _get_ml():
    return _cal('ml_rules') or _DEFAULT_ML

def _get_baodian():
    return _cal('baodian_tiers') or _DEFAULT_BAODIAN

# ─── 数据库规则 ───────────────────────────────────────

def _db_conn():
    import psycopg2
    return psycopg2.connect(host=os.environ.get('PGHOST','localhost'),
        dbname=os.environ.get('PGDATABASE','myapp_db'),
        user=os.environ.get('PGUSER','myapp'), password=os.environ.get('PGPASSWORD',''),
        connect_timeout=5)

def load_db_rules():
    global _db_cache
    if _db_cache is not None: return _db_cache
    try:
        c = _db_conn(); cur = c.cursor()
        cur.execute("SELECT category, rule_key, rule_value, description FROM sporttery_rules")
        sr = [{'category':r[0],'rule_key':r[1],'rule_value':r[2],'desc':r[3]} for r in cur.fetchall()]
        cur.execute("SELECT rule_category, rule_name, rule_value, rule_description FROM jc_official_rules")
        jr = [{'category':r[0],'name':r[1],'value':r[2],'desc':r[3]} for r in cur.fetchall()]
        cur.close(); c.close(); _db_cache = (sr, jr)
    except Exception as e:
        print(f"[规则引擎] DB加载失败: {e}"); _db_cache = ([], [])
    return _db_cache

def _add(result, tag, msg):
    result['warnings'].append(msg)
    result['applied_rules'].append(f"{tag}: {msg}")

# ─── 主函数 ───────────────────────────────────────

def apply_calibrated_adjustments(match):
    """应用校准规则，返回调整建议"""
    ho = float(match.get('home_odds') or 0)
    do = float(match.get('draw_odds') or 0)
    ao = float(match.get('away_odds') or 0)
    ml_conf = float(match.get('ml_confidence') or 0)
    league = str(match.get('league') or '')
    rq_hc = str(match.get('rqspf_handicap') or '')
    result = {'warnings': [], 'applied_rules': [], 'sample_total': 53}

    tiers = _get_tiers()
    fine_grain = _get_fine_grain()
    draw_cal = _get_draw()
    league_cal = _get_league()
    hc_cal = _get_handicap()
    ml_cal = _get_ml()
    baodian = _get_baodian()

    # 档位
    tn, ti = '下游', tiers['下游']
    for name, info in tiers.items():
        if ho < info['max_odds']:
            tn, ti = name, info; break
    result['tier'] = tn; result['tier_conf'] = ti['conf']
    if 'warning' in ti:
        _add(result, f"档位{tn}", ti['warning'])

    # 细粒度
    for (lo, hi), fg in fine_grain.items():
        if lo <= ho < hi:
            result['fine_grain'] = fg
            for k in ('warning', 'note'):
                if k in fg: _add(result, "细粒度赔率", f'📊 {fg[k]}')
            break

    # 平赔
    dr = draw_cal['high']
    for k in ('low', 'mid', 'mid_high', 'high'):
        if do < draw_cal[k]['max']:
            dr = draw_cal[k]; break
    result['draw_action'] = dr['result']; result['draw_rate'] = dr['draw_rate']
    if dr['result'] == '不防平':
        _add(result, "平赔规则", f'🔄 高平赔(≥{dr["max"]:.1f})平局率仅{dr["draw_rate"]}%，不防平')
    elif dr['result'] == '平局略多':
        _add(result, "平赔规则", f'📊 中平赔平局率{dr["draw_rate"]}%，关注平局')

    # 让球
    for sign in ('-2', '-1', '+1'):
        if sign in rq_hc and sign in hc_cal:
            for k in ('warning', 'note'):
                if k in hc_cal[sign]: _add(result, "让球规则", hc_cal[sign][k])
            break

    # 联赛
    lg = None
    for key, info in league_cal.items():
        if key in league: lg = info; break
    if lg:
        result['league_info'] = lg; result['league_adjust'] = lg['adjust']
        if lg['adjust'] != 0:
            d = '提升' if lg['adjust'] > 0 else '降低'
            _add(result, "联赛校准", f'📍 {league}{lg["note"]}，{d}置信{abs(lg["adjust"])}%')

    # ML置信度
    if ml_conf >= ml_cal['high']['min']:
        result['ml_band'] = '高'; result['ml_rate'] = ml_cal['high']['rate']
    elif ml_conf >= ml_cal['mid']['min']:
        result['ml_band'] = '中'; result['ml_rate'] = ml_cal['mid']['rate']
    else:
        result['ml_band'] = '低'; result['ml_rate'] = ml_cal['low']['rate']
        if ml_conf > 0: _add(result, "ML校准", '⚠️ ML低置信(<45%)，需其他信号佐证')

    # 综合置信度
    base_conf = ti['conf']
    if lg: base_conf = max(10, min(95, base_conf + lg['adjust']))

    # 宝典分布
    if ho <= 1.45:
        bd_adj = 0; result['baodian_pattern'] = '主胜型'
    elif ao <= 1.45:
        bd_adj = -15; result['baodian_pattern'] = '客胜型'
        _add(result, "宝典格局", '📖 客胜型，降置信15%')
    elif do <= 3.20 and abs(ho - ao) < 0.5:
        bd_adj = -5; result['baodian_pattern'] = '平局型'
        _add(result, "宝典格局", '📖 平局型，关注平局')
    else:
        bd_adj = 0; result['baodian_pattern'] = '均势型'
    base_conf = max(10, min(95, base_conf + bd_adj))

    # 宝典档位对比
    for t in baodian:
        if t['lo'] <= ho < t['hi']:
            result['baodian_tier'] = t['name']; result['baodian_expected_rate'] = t['expected_rate']
            if base_conf > t['expected_rate'] + 10:
                _add(result, "宝典校准", f'📖 {t["name"]}上限{t["expected_rate"]}%，下调')
                base_conf = min(base_conf, t['expected_rate'] + 5)
            break

    # 平赔手法
    if do <= 3.20 and abs(ho - ao) < 0.5:
        _add(result, "平赔手法", '📖 平赔≤3.20且主客赔接近')
        base_conf = max(10, base_conf - 5)

    if ml_conf >= ml_cal['high']['min']:
        base_conf = max(base_conf, ml_conf)
    result['final_confidence'] = base_conf

    # v3策略
    v3 = match.get('v3_max_proba')
    if v3 is not None:
        if v3 < 0.55:
            _add(result, "v3策略", '📊 ML v3低置信(<55%)')
            result['v3_selective'] = False
        else:
            result['v3_selective'] = True; result['v3_confidence'] = round(v3 * 100, 1)

    # ─── 热度异常规则（支持率 + 必发）──────────
    pop = match.get('popularity') or {}
    spf_pop = pop.get('support')
    bf_pop = pop.get('betfair')

    if spf_pop:
        s_vals = [spf_pop.get('home',0), spf_pop.get('draw',0), spf_pop.get('away',0)]
        s_max = max(s_vals)
        s_hot_idx = ['home','draw','away'][s_vals.index(s_max)]
        result['support_hot'] = s_hot_idx
        result['support_max'] = s_max

        # 散户一边倒（>65%）→ 反向指标，降置信
        if s_max >= 65:
            _add(result, "热度异常", f'🔥 支持{s_hot_idx}方向{s_max:.0f}%过热（散户追热），降置信10%')
            base_conf = max(10, base_conf - 10)
        # 支持率与赔率背离（支持率高但赔率也高 = 庄家在诱）
        if s_hot_idx == 'home' and ho > 2.5:
            _add(result, "热度背离", f'⚠️ 主胜支持率最高但赔率{ho:.2f}偏高，庄家可能诱盘')
            base_conf = max(10, base_conf - 5)
        elif s_hot_idx == 'away' and ao > 2.5:
            _add(result, "热度背离", f'⚠️ 客胜支持率最高但赔率{ao:.2f}偏高，庄家可能诱盘')
            base_conf = max(10, base_conf - 5)

    if bf_pop:
        bf_hot = bf_pop.get('hot_dir')
        bf_total = bf_pop.get('total_vol', 0)
        kellys = [bf_pop.get('home_kelly',0), bf_pop.get('draw_kelly',0), bf_pop.get('away_kelly',0)]
        valid_k = [k for k in kellys if k > 0]
        result['betfair_hot'] = bf_hot
        result['betfair_total_vol'] = bf_total

        if valid_k:
            k_min = min(valid_k)
            k_spread = max(valid_k) - min(valid_k)
            result['kelly_min'] = k_min
            result['kelly_spread'] = k_spread
            # 凯利方差极低（<5）= 市场高度一致 → 提升置信
            if k_min <= 3:
                _add(result, "必发共识", f'✅ 凯利方差最低{k_min:.0f}，市场高度一致，升置信5%')
                base_conf = min(95, base_conf + 5)
            # 凯利方差极高（>50）= 市场分歧大 → 降置信
            if k_spread >= 50:
                _add(result, "必发分歧", f'⚠️ 凯利方差极差{k_spread:.0f}，市场分歧大，降置信5%')
                base_conf = max(10, base_conf - 5)

        # 必发vs支持率方向一致 = 强信号
        if spf_pop and bf_hot:
            bf_dir = bf_hot
            if bf_dir == 'home' and s_hot_idx == 'home':
                _add(result, "人气共振", f'✅ 支持+必发均看好主胜，升置信8%')
                base_conf = min(95, base_conf + 8)
            elif bf_dir == 'away' and s_hot_idx == 'away':
                _add(result, "人气共振", f'✅ 支持+必发均看好客胜，升置信8%')
                base_conf = min(95, base_conf + 8)

    result['final_confidence'] = base_conf

    # 数据库规则
    sr, jr = load_db_rules()
    db_rules = []
    for r in jr:
        if 'return_rate' in r['name']:
            result['jc_return_rate'] = float(r['value'])
            db_rules.append(f"竞彩返奖率{r['value']}")
    for r in sr: db_rules.append(f"体彩{r['rule_key']}: {r['rule_value']}")
    if db_rules: result['db_rules'] = db_rules

    return result

# ─── 兼容接口 ───────────────────────────────────────

def load_prediction_rules():
    return {'version':'v4', 'sample_total':53, 'tier_confidence':_get_tiers(),
            'fine_grain':_get_fine_grain(), 'draw_cal':_get_draw(), 'league_cal':_get_league(),
            'handicap_cal':_get_handicap(), 'ml_cal':_get_ml(), 'cal_file':_CAL_FILE}

def apply_rules(match, rules=None):
    return apply_calibrated_adjustments(match)
