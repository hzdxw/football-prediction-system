#!/usr/bin/env python3
"""宝典理论特征提取 - 欧赔核心+亚盘核心"""
import math


# ══════════════════════════════════════════════════════════
# 欧赔特征（10个）
# ══════════════════════════════════════════════════════════

def calc_euro_features(home_odds, draw_odds, away_odds):
    """欧赔基础特征：返奖率/margin/隐含概率/概率不均衡度/平赔相对位置/尾数"""
    try:
        ho, do, ao = float(home_odds), float(draw_odds), float(away_odds)
    except (TypeError, ValueError):
        return {}
    if ho <= 0 or do <= 0 or ao <= 0:
        return {}
    imp_h, imp_d, imp_a = 1/ho, 1/do, 1/ao
    total_imp = imp_h + imp_d + imp_a
    if total_imp == 0:
        return {}
    return_rate = 1 / total_imp
    margin = 1 - return_rate
    prob_h, prob_d, prob_a = imp_h/total_imp, imp_d/total_imp, imp_a/total_imp
    prob_gap = max(prob_h, prob_d, prob_a) - min(prob_h, prob_d, prob_a)
    draw_relative = prob_d / 0.333  # 0.333=三项均分
    # 尾数特征
    def tail_class(o):
        t = round(o * 100) % 100
        if t <= 5: return '0尾'
        if t <= 20: return '低尾'
        if t <= 50: return '中尾'
        if t <= 80: return '高尾'
        return '5尾'
    # 胜负赔率比值
    ha_ratio = ho / ao if ao > 0 else 0
    return {
        'return_rate': round(return_rate, 4),
        'margin': round(margin, 4),
        'prob_h': round(prob_h, 4), 'prob_d': round(prob_d, 4), 'prob_a': round(prob_a, 4),
        'prob_gap': round(prob_gap, 4),
        'draw_relative': round(draw_relative, 4),
        'home_tail': tail_class(ho), 'draw_tail': tail_class(do), 'away_tail': tail_class(ao),
        'ha_ratio': round(ha_ratio, 4),
    }


def calc_distribution_type(rank_diff, recent_form_diff):
    """分布类型：顺/逆/中庸/缓冲"""
    try:
        rd = float(rank_diff)
        rfd = float(recent_form_diff)
    except (TypeError, ValueError):
        return '未知'
    if rd > 3 and rfd > 2: return '顺分布'
    if rd < -3 and rfd < -2: return '逆分布'
    if abs(rd) <= 2 and abs(rfd) <= 2: return '中庸分布'
    return '缓冲分布'


def classify_odds_method(home_odds):
    """开盘手法：实开/韬开/中庸"""
    try:
        ho = float(home_odds)
    except (TypeError, ValueError):
        return '未知'
    if ho < 1.5: return '实开'
    if ho > 2.5: return '韬开'
    return '中庸'


def calc_odds_change(init_home, init_draw, init_away, final_home, final_draw, final_away):
    """变赔方向"""
    try:
        dh = float(final_home) - float(init_home)
        dd = float(final_draw) - float(init_draw)
        da = float(final_away) - float(init_away)
    except (TypeError, ValueError):
        return {}
    def dir_str(d):
        if d < -0.05: return '降'
        if d > 0.05: return '升'
        return '平'
    return {
        'home_change': round(dh, 3), 'draw_change': round(dd, 3), 'away_change': round(da, 3),
        'home_dir': dir_str(dh), 'draw_dir': dir_str(dd), 'away_dir': dir_str(da),
    }


def calc_company_diff(wh_draw, ac_draw):
    """公司差异信号：威廉vs澳彩平赔差"""
    try:
        diff = float(wh_draw) - float(ac_draw)
    except (TypeError, ValueError):
        return {}
    return {'company_draw_diff': round(diff, 3), 'signal': '威廉压平' if diff < -0.1 else '澳彩压平' if diff > 0.1 else '一致'}


# ══════════════════════════════════════════════════════════
# 亚盘特征（8个）
# ══════════════════════════════════════════════════════════

def calc_asian_features(handicap_line, home_water, away_water):
    """亚盘特征：水位分级/生死盘/盘口类型"""
    try:
        hl = float(handicap_line)
        hw = float(home_water) if home_water else 0
        aw = float(away_water) if away_water else 0
    except (TypeError, ValueError):
        return {}
    if not handicap_line:
        return {}
    avg_water = (hw + aw) / 2 if hw and aw else (hw or aw or 0)
    # 水位分级
    if avg_water < 0.75: water_class = '超低水'
    elif avg_water < 0.85: water_class = '低水'
    elif avg_water < 0.95: water_class = '中水'
    elif avg_water < 1.05: water_class = '高水'
    else: water_class = '超高水'
    # 生死盘：半球+高水→下盘82%
    is_life_death = (abs(hl) == 0.5 and avg_water >= 0.95)
    life_death_hint = '下盘(82%)' if is_life_death else None
    # 盘口类型
    ah = abs(hl)
    if ah == 0: ht = '平手'
    elif ah == 0.25: ht = '平手/半球'
    elif ah == 0.5: ht = '半球'
    elif ah == 0.75: ht = '半球/一球'
    elif ah == 1.0: ht = '一球'
    elif ah == 1.25: ht = '一球/球半'
    elif ah == 1.5: ht = '球半'
    elif ah >= 2.0: ht = '深盘(≥2)'
    else: ht = f'其他({hl})'
    # 降盘升水/升盘降水检测
    water_diff = hw - aw if hw and aw else 0
    return {
        'handicap_line': hl, 'handicap_type': ht,
        'home_water': hw, 'away_water': aw, 'avg_water': round(avg_water, 3),
        'water_class': water_class, 'water_diff': round(water_diff, 3),
        'is_life_death': is_life_death, 'life_death_hint': life_death_hint,
    }


def detect_deep_shallow(asian_handicap, euro_home, euro_away):
    """深盘/浅盘检测：亚盘 vs 欧赔折合盘口"""
    try:
        ah = float(asian_handicap)
        ho = float(euro_home)
        ao = float(euro_away)
    except (TypeError, ValueError):
        return '未知'
    if ho <= 0 or ao <= 0:
        return '未知'
    # 欧赔折合盘口：简化公式
    euro_equiv = (1/ho - 1/ao) / (1/ho + 1/ao) * 2.5
    diff = ah - euro_equiv
    if diff > 0.25: return '深盘'
    if diff < -0.25: return '浅盘'
    return '标准盘'


def detect_gap_pan(handicap_change, water_change):
    """缺口盘检测：降盘+高水"""
    try:
        hc = float(handicap_change)
        wc = float(water_change)
    except (TypeError, ValueError):
        return False
    return hc < 0 and wc > 0  # 降盘+升水


# ══════════════════════════════════════════════════════════
# 便捷接口
# ══════════════════════════════════════════════════════════

def extract_all_features(match):
    """一键提取所有宝典特征"""
    f = {}
    ho = match.get('odds_home') or match.get('home_odds')
    do = match.get('odds_draw') or match.get('draw_odds')
    ao = match.get('odds_away') or match.get('away_odds')
    if ho and do and ao:
        f['euro'] = calc_euro_features(ho, do, ao)
        f['odds_method'] = classify_odds_method(ho)
    rank_d = match.get('rank_diff')
    form_d = match.get('form_diff') or match.get('recent_form_diff')
    if rank_d is not None and form_d is not None:
        f['distribution'] = calc_distribution_type(rank_d, form_d)
    ah = match.get('asian_handicap') or match.get('handicap_line')
    hw = match.get('home_water')
    aw = match.get('away_water')
    if ah is not None:
        f['asian'] = calc_asian_features(ah, hw, aw)
        if ho and ao:
            f['deep_shallow'] = detect_deep_shallow(ah, ho, ao)
    return f
