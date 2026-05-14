"""校准泊松矩阵 — 单一概率空间派生所有5玩法"""
import math as _math_mod
from math import exp, factorial, log
from scipy.optimize import minimize


# P4-3 联赛类型映射（避免循环导入）
_LEAGUE_TYPE_MAP = {
    'cup': {'home_rate': 0.55},   # 杯赛主场胜率高
    'league': {'home_rate': 0.45},  # 联赛主场胜率基准
    'other': {'home_rate': 0.46},
}
_CUP_KEYWORDS = {
    '欧冠', '欧罗巴', '欧协联', '欧洲冠军联赛', '欧洲协会联赛',
    '欧冠杯', '欧超杯', '解放者杯', '南美杯', '自由杯', '亚冠',
    '世界杯', '世俱杯', '联合会杯',
}


def _get_league_type_inline(league_name):
    """P4-3 inline league type detection (避免循环导入)"""
    if not league_name:
        return 'other'
    for kw in _CUP_KEYWORDS:
        if kw in str(league_name):
            return 'cup'
    return 'league'


def _is_balanced_odds_inline(home_odds, away_odds):
    """P4-1 inline 均衡赔率检测 (避免循环导入)"""
    try:
        prob_h = 1.0 / home_odds
        prob_a = 1.0 / away_odds
        return abs(prob_h - prob_a) < 0.15
    except:
        return False


def _raw_lambda(home_odds, draw_odds, away_odds, league='', weekday=-1):
    """从赔率反推泊松λ — P0-1星期因子+P4-1均衡+P4-3联赛类型"""
    try:
        imp_h = 1.0 / home_odds
        imp_d = 1.0 / draw_odds
        imp_a = 1.0 / away_odds
        s = imp_h + imp_d + imp_a
        p_h, p_d, p_a = imp_h / s, imp_d / s, imp_a / s
        margin = s
        p_d_adj = p_d * margin
        if p_d_adj < 0.01:
            lam_total = 4.0
        elif p_d_adj > 0.5:
            lam_total = 1.0
        else:
            lam_total = -log(p_d_adj) * 2.0
        lam_total = max(1.0, min(4.5, lam_total))

        # ── P4-1: 均衡赔率调整 ──
        is_balanced = _is_balanced_odds_inline(home_odds, away_odds)
        if is_balanced:
            # 均衡场次：缩小主客差距，向50/50靠拢
            ratio = p_h / (p_a + 1e-9)
            # 原始ratio向1.0靠拢50%
            ratio = ratio ** 0.5  # 开方=向1靠拢

        # ── P4-3: 联赛类型主场基准校准 ──
        lg_type = _get_league_type_inline(league)
        lg_home_rate = _LEAGUE_TYPE_MAP.get(lg_type, {}).get('home_rate', 0.46)

        # 赔率隐含主胜率
        if p_h + p_a > 0:
            implied_home_ratio = p_h / (p_h + p_a)
        else:
            implied_home_ratio = 0.5

        # 基准调整因子：隐含主胜率/联赛基准
        if lg_home_rate > 0:
            # 当隐含主胜率 > 基准时，说明赔率反映的是真实实力差
            # 当隐含主胜率 < 基准时，说明庄家在压低主队（杯赛冷门多）
            adj_factor = implied_home_ratio / lg_home_rate
            # 限制调整幅度，避免过度调整
            adj_factor = max(0.70, min(1.30, adj_factor))
            # 将调整应用到ratio上
            if 'ratio' not in dir():
                ratio = p_h / (p_a + 1e-9)
            ratio *= adj_factor

        if 'ratio' not in dir():
            ratio = p_h / (p_a + 1e-9)

        lam_h = lam_total * ratio / (1 + ratio)
        lam_a = lam_total * 1.0 / (1 + ratio)

        # ── P0-1: 星期因子调整 ──
        # TODO: P1 - Validate weekday factors against last 90 days of actual results
        # Current values are empirical; 周一lam_h*=1.03 may be over-fit
        if 0 <= weekday <= 6:
            # 周一: 04-27数据显示周一主队胜率并不稳定，降低boost
            if weekday == 0:
                lam_h *= 1.01  # 降低: 1.03→1.01 (数据不足时用最小调整)
                lam_a *= 0.99
            # 周五: 主队轻微高估
            elif weekday == 4:
                lam_h *= 0.97
                lam_a *= 1.03
            # 周六: 热门主队(ho<1.5)降权×0.88（额外×0.88叠加），非热门×0.96
            # 根因：周六豪门(ho<1.5)历史命中率<30%，拜仁/巴黎爆冷实际打出平局
            elif weekday == 5:
                if ho < 1.5:
                    factor = 0.88   # 双重降权（原×0.93 → ×0.88）
                else:
                    factor = 0.96
                lam_h *= factor
                lam_a *= factor
            # 周日: 热门主队(ho<1.5)降权×0.90（低于周六但高于工作日）
            # 根因：周日比赛庄家开盘思路与周六不同，豪门优势更小
            elif weekday == 6:
                if ho < 1.5:
                    factor = 0.90
                else:
                    factor = 0.95
                lam_h *= factor
                lam_a *= factor

        # ── P0-2: 亚大联赛专项降权（韩职/日职/澳超/泰超/中超）──
        # 根因：亚大联赛赔率结构与欧洲差异大，历史命中率<35%，需降权避免高估
        _asian_leagues = {'韩职', 'K联赛', '日职', 'J联赛', 'J1', 'J2',
                          '澳超', 'A联赛', '泰超', '中超', '中超联赛'}
        if lg in _asian_leagues:
            lam_h *= 0.90
            lam_a *= 0.95

        return lam_h, lam_a
    except Exception:
        return 1.3, 1.0


LEAGUE_FACTORS = {
    'Serie B': {'draw_boost': 0.20, 'goal_scale': 0.95},
    'Segunda División': {'draw_boost': 0.20, 'goal_scale': 0.95},
    'Ligue 2': {'draw_boost': 0.18, 'goal_scale': 0.95},
    '法乙': {'draw_boost': 0.18, 'goal_scale': 0.95},
    '2. Bundesliga': {'draw_boost': 0.18, 'goal_scale': 0.95},
    'Scottish Prem': {'draw_boost': 0.17, 'goal_scale': 0.95},
    '意乙': {'draw_boost': 0.15, 'goal_scale': 0.95},
    '葡乙': {'draw_boost': 0.12, 'goal_scale': 0.95},
    '西乙': {'draw_boost': 0.15, 'goal_scale': 0.95},
    '英超': {'draw_boost': 0.14, 'goal_scale': 1.0},
    '意甲': {'draw_boost': 0.14, 'goal_scale': 1.0},
    '西甲': {'draw_boost': 0.14, 'goal_scale': 1.0},
    '德甲': {'draw_boost': 0.14, 'goal_scale': 1.0},
    '法甲': {'draw_boost': 0.14, 'goal_scale': 1.0},
    '英甲': {'draw_boost': 0.14, 'goal_scale': 1.0},
    '荷甲': {'draw_boost': 0.14, 'goal_scale': 0.95},
    '挪超': {'draw_boost': 0.10, 'goal_scale': 1.0},
    '苏超': {'draw_boost': 0.10, 'goal_scale': 1.0},
    '比甲': {'draw_boost': 0.10, 'goal_scale': 1.0},
    '瑞典超': {'draw_boost': 0.10, 'goal_scale': 1.0},
    '土超': {'draw_boost': 0.10, 'goal_scale': 1.0},
    '希腊超': {'draw_boost': 0.10, 'goal_scale': 1.0},
    '欧冠': {'draw_boost': 0.10, 'goal_scale': 1.0},
    '欧联': {'draw_boost': 0.10, 'goal_scale': 1.0},
    '澳超': {'draw_boost': 0.04, 'goal_scale': 1.08, 'n_recent': 15, 'cv_shrink': 'extreme'},
    'A-League': {'draw_boost': 0.04, 'goal_scale': 1.08, 'n_recent': 15, 'cv_shrink': 'extreme'},
    '欧罗巴': {'draw_boost': 0.10, 'goal_scale': 1.0},
    '葡超': {'draw_boost': 0.10, 'goal_scale': 1.0},
    '英冠': {'draw_boost': 0.18, 'goal_scale': 0.92},  # P2: 英冠高平局率(40%+)，降低期望进球
    '国际赛': {'draw_boost': 0.16, 'goal_scale': 0.90},
    '友谊赛': {'draw_boost': 0.16, 'goal_scale': 0.90},
    '世预赛': {'draw_boost': 0.12, 'goal_scale': 0.90},
    '_default_': {'draw_boost': 0.08, 'goal_scale': 1.0},
}

# P0: 联赛历史比分先验分布（弥补泊松方差不足）
# key: 联赛名, value: {score: prior_prob} 权重叠加到泊松矩阵
LEAGUE_SCORE_PRIORS = {
    # 英甲: 1-1/0-0/1-0 占29%，小比分极端
    '英甲': {'1-1': 0.11, '0-0': 0.09, '1-0': 0.09, '2-0': 0.08, '0-1': 0.07, 'total_small': 0.44},
    # 英冠: 平局率40%+，1-1/0-0 各9%
    '英冠': {'1-1': 0.09, '0-0': 0.09, '1-0': 0.08, '0-1': 0.07, '2-1': 0.07, '2-0': 0.06, 'total_small': 0.39},
    # 法乙: 次级联赛小比分多
    '法乙': {'1-1': 0.09, '0-0': 0.08, '1-0': 0.09, '2-0': 0.07, 'total_small': 0.33},
    # 荷乙
    '荷乙': {'1-1': 0.08, '0-0': 0.08, '1-0': 0.09, '2-0': 0.07, 'total_small': 0.32},
    # 葡超: 葡超平局率高
    '葡超': {'1-1': 0.09, '0-0': 0.07, '1-0': 0.08, '2-1': 0.07, 'total_small': 0.31},
    # 意甲: 经典1-0/2-1小胜
    '意甲': {'1-0': 0.10, '2-1': 0.09, '1-1': 0.08, '0-1': 0.07, 'total_small': 0.34},
    # 西甲: 进攻型，2-1/1-0
    '西甲': {'1-0': 0.09, '2-1': 0.09, '1-1': 0.07, '2-0': 0.07, 'total_small': 0.32},
    # 英超: 2-1最多，1-0次之
    '英超': {'2-1': 0.09, '1-0': 0.08, '1-1': 0.07, '2-0': 0.06, '3-1': 0.05, 'total_small': 0.35},
}

# P0: 联赛平局率先验（贝叶斯修正用）
LEAGUE_DRAW_RATES = {
    '法乙': 0.59, '日职': 0.35, '英冠': 0.33, '葡超': 0.32,
    '挪超': 0.29, '英超': 0.27, '荷甲': 0.26, '荷乙': 0.25,
    '韩职': 0.25, '沙职': 0.24, '意甲': 0.24, '德甲': 0.23,
    '澳超': 0.22, '德乙': 0.22, '法甲': 0.21, '西甲': 0.19,
    '瑞超': 0.19, '芬超': 0.17, '英甲': 0.12, '美职': 0.00,
    '欧冠': 0.33, '欧罗巴': 0.40, '欧协联': 0.30,
}

# P0: lambda策略（模型vs历史权重）
_DRAW_LAMBDA = {
    'high_draw': 0.3,
    'balanced': 0.5,
    'cup': 0.4,
}
_HIGH_DRAW_LEAGUES = {'法乙', '英冠', '日职', '葡超', '挪超', '欧罗巴'}
_CUP_LEAGUES = {'欧冠', '欧罗巴', '欧协联'}


def apply_draw_prior(spf_probs, league, draw_rate_table=None):
    rates = draw_rate_table or LEAGUE_DRAW_RATES
    prior_draw = rates.get(league, 0.25)
    if prior_draw <= 0:
        return dict(spf_probs)

    if league in _HIGH_DRAW_LEAGUES:
        lam = _DRAW_LAMBDA['high_draw']
    elif league in _CUP_LEAGUES:
        lam = _DRAW_LAMBDA['cup']
    else:
        lam = _DRAW_LAMBDA['balanced']

    p = dict(spf_probs)
    model_draw = p.get('平', 0.15)
    posterior_draw = lam * model_draw + (1 - lam) * prior_draw
    delta = posterior_draw - model_draw
    if delta <= 0:
        return p

    p['平'] = posterior_draw
    others = p.get('胜', 0) + p.get('负', 0)
    if others > 0:
        ratio_w = p.get('胜', 0) / others
        ratio_l = p.get('负', 0) / others
        p['胜'] = max(0.05, p.get('胜', 0) - delta * ratio_w)
        p['负'] = max(0.05, p.get('负', 0) - delta * ratio_l)

    total = sum(p.values())
    if total > 0:
        for k in p:
            p[k] /= total
    return p


# === 博弈论增强函数 ===
def market_equilibrium_deviation(spf_probs, ho=None, do_=None, ao=None):
    """博弈论: 检测模型概率 vs 赔率隐含概率偏离>15%时调整"""
    if ho is None or do_ is None or ao is None:
        return dict(spf_probs), []

    p = dict(spf_probs)
    signals = []

    # 隐含概率（赔率倒数的归一化）
    try:
        imp_h = 1.0 / ho if ho > 1 else 0
        imp_d = 1.0 / do_ if do_ > 1 else 0
        imp_a = 1.0 / ao if ao > 1 else 0
        total_imp = imp_h + imp_d + imp_a
        if total_imp > 0:
            imp_h /= total_imp; imp_d /= total_imp; imp_a /= total_imp
        else:
            return p, []
    except (TypeError, ZeroDivisionError):
        return p, []

    # 偏离检测
    deviations = {'胜': p.get('胜', 0) - imp_h,
                  '平': p.get('平', 0) - imp_d,
                  '负': p.get('负', 0) - imp_a}

    for label, dev in deviations.items():
        if abs(dev) > 0.12:  # 偏离>12%触发
            direction = '↑模型高估' if dev > 0 else '↓模型低估'
            signals.append(f'均衡偏离{label}{direction}({dev:+.1%})')
            # 偏离方向修正：向赔率隐含概率靠拢30%
            implied_map = {'胜': imp_h, '平': imp_d, '负': imp_a}
            adj_strength = min(0.05, abs(dev) * 0.30)
            if dev > 0:  # 模型高估，向下修正
                p[label] = max(0.05, p[label] - adj_strength)
            else:  # 模型低估，向上修正
                p[label] = min(0.85, p[label] + adj_strength)

    # 归一化
    total = sum(p.values())
    if total > 0:
        for k in p: p[k] /= total
    return p, signals


def longshot_bias_adjustment(spf_probs, support=None, ho=None, do_=None, ao=None):
    """博弈论: 冷门偏见修正（Longshot Bias）

    博弈论原理：公众系统性高估低概率事件（赔率>4.0），
    庄家利用此偏见设置陷阱。当支持率异常高但赔率也高时，
    说明散户在追逐冷门，此时跟庄（降低冷门概率）。

    Args:
        spf_probs: 当前SPF概率
        support: {'home': 46, 'draw': 30, 'away': 24}
        ho/do_/ao: 赔率
    Returns:
        (adjusted_probs, signals)
    """
    p = dict(spf_probs)
    signals = []

    if support is None:
        return p, signals

    s_total = support.get('home', 0) + support.get('draw', 0) + support.get('away', 0)
    if s_total <= 0:
        return p, signals

    for label, odds in (('胜', ho), ('平', do_), ('负', ao)):
        if odds is None or odds <= 1:
            continue

        # 长赔率检测：赔率>4.0 = 隐含概率<25%
        if odds < 4.0:
            continue

        support_pct = support.get({'胜': 'home', '平': 'draw', '负': 'away'}.get(label, ''), 0) / s_total

        # 冷门偏见：赔率高 + 支持率也高 = 散户追冷
        if support_pct > 0.25:  # 25%以上支持率
            # 降低冷门概率3-5%
            reduction = min(0.06, support_pct - 0.20)
            old_val = p.get(label, 0)
            p[label] = max(0.03, old_val - reduction)
            # 补偿给主胜
            p['胜'] = min(0.90, p.get('胜', 0) + (old_val - p[label]))
            signals.append(f'冷门偏见修正{label}(赔{odds:.1f},支持{support_pct:.0%}→概率{old_val:.1%}→{p[label]:.1%})')

    # 归一化
    total = sum(p.values())
    if total > 0:
        for k in p: p[k] /= total
    return p, signals

_BQC = {
    (1, 1): '胜-胜', (1, 0): '胜-平', (1, -1): '胜-负',
    (0, 1): '平-胜', (0, 0): '平-平', (0, -1): '平-负',
    (-1, 1): '负-胜', (-1, 0): '负-平', (-1, -1): '负-负',
}


def _build_matrix(lam_h, lam_a, max_g=7):
    matrix = {(i, j): exp(-lam_h) * lam_h**i / factorial(i) *
                   exp(-lam_a) * lam_a**j / factorial(j)
              for i in range(max_g) for j in range(max_g)}
    rho = -0.13
    for (i, j), factor in [((0, 0), 1 - rho * lam_h * lam_a),
                            ((0, 1), 1 + rho * lam_a),
                            ((1, 0), 1 + rho * lam_h),
                            ((1, 1), 1 - rho)]:
        matrix[(i, j)] *= factor
    total = sum(matrix.values())
    for k in matrix:
        matrix[k] /= total
    return matrix


def _spf_from_matrix(m):
    pw = sum(p for (i, j), p in m.items() if i > j)
    pd = sum(p for (i, j), p in m.items() if i == j)
    pa = sum(p for (i, j), p in m.items() if i < j)
    return pw, pd, pa


def calibrate_lambda(home_odds, draw_odds, away_odds,
                     spread_point=None, spread_home=None, spread_away=None,
                     total_line=None, total_over=None, total_under=None,
                     league='', weekday=-1,
                     recent_form=None):
    """联合校准泊松λ

    P0-1: weekday参数影响星期因子调整
    P4-1: 支持联赛类型参数league，用于均衡赔率和联赛主场基准校准
    P4-3: 联赛类型影响主场lambda基准（杯赛55% vs 联赛45%）
    """
    margin = 1 / home_odds + 1 / draw_odds + 1 / away_odds
    impl_h = 1 / home_odds / margin
    impl_d = 1 / draw_odds / margin
    impl_a = 1 / away_odds / margin
    # P0-1+P4-1+P4-3: 传入league和weekday参数
    raw_h, raw_a = _raw_lambda(home_odds, draw_odds, away_odds,
                               league=league, weekday=weekday)
    has_spread = spread_point is not None and spread_home is not None and spread_away is not None
    if has_spread:
        try:
            spread_home = float(spread_home)
            spread_away = float(spread_away)
            spread_point = float(spread_point)
        except (ValueError, TypeError):
            has_spread = False
    if has_spread:
        sp_m = 1 / spread_home + 1 / spread_away
        impl_sh = 1 / spread_home / sp_m
        impl_sa = 1 / spread_away / sp_m
        spread_w = 1.5
    has_ou = total_line is not None and total_over is not None and total_under is not None
    if has_ou:
        ou_m = 1 / total_over + 1 / total_under
        impl_over = 1 / total_over / ou_m
        ou_w = 0.8

    # ════ P0-2新增(2026-05-12): 时序指数衰减权重 ════
    # 当有近期球队状态数据时，用指数衰减加权调整λ
    # 近期进球状态比历史均值更能反映当前实力
    init_h = raw_h
    init_a = raw_a
    if recent_form is not None:
        try:
            import numpy as np
            # decay_half_life=5: 5场前的比赛权重减半
            decay_half_life = 5
            n_home = int(recent_form.get('n_home_matches', 0))
            n_away = int(recent_form.get('n_away_matches', 0))
            
            # 计算时序权重因子（近期权重更高）
            # 范围：[0.85, 1.15]，样本多时趋近1.0
            def time_decay_factor(n):
                if n <= 0:
                    return 1.0
                # 样本量少时：基准1.0 + 微量正偏差（防止样本不足时降权过度）
                if n < 5:
                    return 1.0 + (n / 5) * 0.05  # 0→5场：1.0→1.05
                # 样本量足够时：1.0（不做调整，保持稳定性）
                return 1.0
            
            # 近期得失球与赔率λ的比值作为调整信号
            h_gf = float(recent_form.get('home_recent_goals_for', 0))
            h_ga = float(recent_form.get('home_recent_goals_against', 0))
            a_gf = float(recent_form.get('away_recent_goals_for', 0))
            a_ga = float(recent_form.get('away_recent_goals_against', 0))
            
            if h_gf > 0 and a_ga > 0:
                # 用赔率基础λ计算期望
                base_h = raw_h  # min优化前的初始λ
                base_a = raw_a
                
                # 近期实际得失球率
                actual_h_rate = h_gf / max(h_gf + h_ga, 0.1)
                actual_a_rate = a_gf / max(a_gf + a_ga, 0.1)
                
                # 赔率隐含的主/客队期望进球率
                # 注意：raw_lambda已包含赔率信息，这里取其比率
                expected_h_rate = base_h / max(base_h + base_a, 0.1)
                expected_a_rate = base_a / max(base_h + base_a, 0.1)
                
                # 偏离度：>1说明近期实际比赔率预期更好
                home_dev = actual_h_rate / max(expected_h_rate, 0.01)
                away_dev = actual_a_rate / max(expected_a_rate, 0.01)
                
                # 限制调整幅度在±15%内
                home_dev = min(1.15, max(0.85, home_dev))
                away_dev = min(1.15, max(0.85, away_dev))
                
                # 应用到raw_h/raw_a（优化前的初始值）
                adj_h = raw_h * home_dev
                adj_a = raw_a * away_dev
                
                # 用调整后的值作为optimizer的新起点
                init_h = max(0.5, min(4.0, adj_h))
                init_a = max(0.5, min(4.0, adj_a))
        except Exception:
            init_h = raw_h
            init_a = raw_a

    def loss(params):
        lh, la = params
        if lh <= 0 or la <= 0:
            return 1e6
        m = _build_matrix(lh, la)
        ph, pd, pa = _spf_from_matrix(m)
        loss_val = (ph - impl_h) ** 2 + (pd - impl_d) ** 2 + (pa - impl_a) ** 2
        if has_spread:
            sp = float(spread_point)
            p_home_cover = sum(p for (i, j), p in m.items() if i + sp > j)
            loss_val += spread_w * (p_home_cover - impl_sh) ** 2
        if has_ou:
            tl = float(total_line)
            p_over = sum(p for (i, j), p in m.items() if i + j > tl)
            loss_val += ou_w * (p_over - impl_over) ** 2
        return loss_val

    res = minimize(loss, [init_h, init_a], method='Nelder-Mead',
                   options={'xatol': 1e-4, 'fatol': 1e-8, 'maxiter': 1000})
    return res.x[0], res.x[1], raw_h, raw_a


def apply_league_factor(matrix, league):
    """应用联赛调整因子（限制最大boost倍率为1.5）"""
    f = LEAGUE_FACTORS.get(league, LEAGUE_FACTORS.get(
        '_default_', {'draw_boost': 0.08, 'goal_scale': 1.0}))
    if f['draw_boost'] == 0:
        return matrix
    dt = sum(p for (i, j), p in matrix.items() if i == j)
    nd = sum(p for (i, j), p in matrix.items() if i != j)
    draw_mult = min(1 + f['draw_boost'] / max(dt, 0.01), 1.5)
    nd_mult = max(1 - f['draw_boost'] / max(nd, 0.01), 0.7)
    adj = {}
    for (i, j), p in matrix.items():
        if i == j:
            adj[(i, j)] = max(0, p * draw_mult)
        else:
            adj[(i, j)] = max(0, p * nd_mult)
    t = sum(adj.values())
    return {k: v / t for k, v in adj.items()}


def apply_score_prior(matrix, league, boost=0.05):
    """P0: 应用联赛历史比分先验分布到泊松矩阵"""
    prior = LEAGUE_SCORE_PRIORS.get(league)
    if not prior:
        return matrix
    
    adj = {}
    for (i, j), p in matrix.items():
        score_key = f'{i}-{j}'
        prior_prob = prior.get(score_key, 0)
        if prior_prob > 0:
            # 历史先验叠加：赔率概率 × (1-boost) + 先验 × boost
            adj[(i, j)] = p * (1 - boost) + prior_prob * boost
        else:
            adj[(i, j)] = p * (1 - boost * 0.5)  # 非先验比分略微降权
    
    t = sum(adj.values())
    if t > 0:
        return {k: max(0, v / t) for k, v in adj.items()}
    return matrix


def apply_popularity_prior(matrix, support=None, betfair=None, alpha=0.35):
    """用支持率/必发交易量作为贝叶斯先验修正泊松矩阵

    alpha: 先验权重（0=不修正, 1=完全覆盖, 建议0.3-0.5）
    支持率=反向指标（散户偏见），必发=正向指标（市场共识）
    """
    if not support and not betfair:
        return matrix
    pw = sum(p for (i, j), p in matrix.items() if i > j)
    pd = sum(p for (i, j), p in matrix.items() if i == j)
    pa = sum(p for (i, j), p in matrix.items() if i < j)
    if pw < 0.01 and pd < 0.01 and pa < 0.01:
        return matrix

    prior_h, prior_d, prior_a = pw, pd, pa

    if support:
        s_total = (support.get('home') or 0) + (support.get('draw') or 0) + (support.get('away') or 0)
        if s_total > 0:
            s_h = (support.get('home') or 0) / s_total
            s_d = (support.get('draw') or 0) / s_total
            s_a = (support.get('away') or 0) / s_total

            # 核心修复：使用sigmoid函数使极端支持率信号更强
            dev_h = s_h - 1/3  # >0 = 主队过热
            dev_a = s_a - 1/3  # >0 = 客队过热

            def sig_scale(dev, steep=8):
                return 1 / (1 + _math_mod.exp(-steep * dev))

            scale_h = sig_scale(dev_h)
            scale_a = sig_scale(dev_a)

            if dev_a > 0.05:  # 客支持 > 38%
                contrarian_strength = alpha * (1 + 2 * scale_a)
                prior_a = max(0.05, pa * (1 - contrarian_strength))
                transfer = pa * contrarian_strength
                prior_h += transfer * 0.7
                prior_d += transfer * 0.3

            if dev_h > 0.05:  # 主支持 > 38%
                contrarian_strength = alpha * (1 + 2 * scale_h)
                prior_h = max(0.05, prior_h * (1 - contrarian_strength))
                transfer = prior_h * contrarian_strength
                prior_a += transfer * 0.7
                prior_d += transfer * 0.3

            # 平局过热检测（>38%）
            if s_d > 0.38 and s_d > s_h and s_d > s_a:
                prior_d = max(0.05, pd * 0.75)

    # 必发：正向指标，凯利方差越低权重越高
    if betfair:
        b_total = (betfair.get('home_pct') or 0) + (betfair.get('draw_pct') or 0) + (betfair.get('away_pct') or 0)
        if b_total > 0:
            b_h = (betfair.get('home_pct') or 0) / b_total
            b_d = (betfair.get('draw_pct') or 0) / b_total
            b_a = (betfair.get('away_pct') or 0) / b_total
            kellys = [betfair.get('home_kelly') or 99,
                      betfair.get('draw_kelly') or 99,
                      betfair.get('away_kelly') or 99]
            valid_k = [k for k in kellys if k is not None and k > 0]
            kelly_min = min(valid_k) if valid_k else 50
            kw = max(0.1, 1 - kelly_min / 50)
            bf_alpha = alpha * kw
            prior_h = prior_h * (1 - bf_alpha) + b_h * bf_alpha
            prior_d = prior_d * (1 - bf_alpha) + b_d * bf_alpha
            prior_a = prior_a * (1 - bf_alpha) + b_a * bf_alpha

    pt = prior_h + prior_d + prior_a
    if pt <= 0:
        return matrix
    prior_h /= pt
    prior_d /= pt
    prior_a /= pt

    # 按胜/平/负边际比例调整矩阵
    adj = {}
    for (i, j), p in matrix.items():
        if i > j:
            adj[(i, j)] = p * (prior_h / pw) if pw > 0.01 else p
        elif i == j:
            adj[(i, j)] = p * (prior_d / pd) if pd > 0.01 else p
        else:
            adj[(i, j)] = p * (prior_a / pa) if pa > 0.01 else p
    t = sum(adj.values())
    return {k: v / t for k, v in adj.items()}


def derive_all_plays(matrix, handicap=0, spf_result=None):
    """派生全部5玩法"""
    pw, pd, pa = _spf_from_matrix(matrix)
    ordered = sorted([('胜', pw), ('平', pd), ('负', pa)], key=lambda x: -x[1])
    spf_pred, spf_conf = ordered[0]
    hc = float(handicap) if handicap else 0
    if hc != 0:
        rqw = sum(p for (i, j), p in matrix.items() if i + hc > j)
        rqd = sum(p for (i, j), p in matrix.items() if i + hc == j)
        rql = sum(p for (i, j), p in matrix.items() if i + hc < j)
        rq_ord = sorted([('让胜', rqw), ('让平', rqd), ('让负', rql)], key=lambda x: -x[1])
        rq_pred, rq_conf = rq_ord[0]
        ds = f'主让{abs(hc):.0f}球' if hc < 0 else (f'主受让{abs(hc):.0f}球' if hc > 0 else '平手')
        rq_label = f'{rq_pred}({ds},{rq_conf * 100:.0f}%)'
    else:
        rq_pred, rq_conf = None, 0
        rq_label = '无让球数据'
        rqw = rqd = rql = 0
    td = {}
    for (i, j), p in matrix.items():
        td[i + j] = td.get(i + j, 0) + p
    tg_ord = sorted(td.items(), key=lambda x: -x[1])
    tg_top = tg_ord[0][0]

    rq_group = {'让胜': [], '让平': [], '让负': []}
    for (i, j), p in matrix.items():
        adj = i + hc
        g = '让胜' if adj > j else ('让平' if adj == j else '让负')
        rq_group[g].append((i, j, p))
    top_group = rq_group.get(rq_pred, [])
    if top_group:
        scored = [(i, j, p * (1.5 if (i + j) == tg_top else 1.0))
                  for i, j, p in top_group]
        scored.sort(key=lambda x: -x[2])
        bf_top = scored[0]
        bf5 = [(f'{i}-{j}', f'{p * 100:.1f}%') for i, j, p in scored[:5]]
        top_group.sort(key=lambda x: -x[2])
        bf5_raw = [(f'{i}-{j}', f'{p * 100:.1f}%') for i, j, p in top_group[:5]]
    else:
        cands = [(i, j, p) for (i, j), p in matrix.items()]
        cands.sort(key=lambda x: -x[2])
        bf_top = cands[0]
        bf5 = bf5_raw = [(f'{i}-{j}', f'{p * 100:.1f}%') for i, j, p in cands[:5]]

    total_h = sum(i * p for (i, j), p in matrix.items())
    total_a = sum(j * p for (i, j), p in matrix.items())
    lh2, la2 = total_h * 0.45, total_a * 0.45
    lh2b, la2b = total_h - lh2, total_a - la2
    bqc_dist = {}
    for hi in range(5):
        for hj in range(5):
            ph = exp(-lh2) * lh2**hi / factorial(hi) * exp(-la2) * la2**hj / factorial(hj)
            hr = 1 if hi > hj else (0 if hi == hj else -1)
            for si in range(7 - hi):
                for sj in range(7 - hj):
                    ps = exp(-lh2b) * lh2b**si / factorial(si) * exp(-la2b) * la2b**sj / factorial(sj)
                    fr = 1 if hi+si > hj+sj else (0 if hi+si == hj+sj else -1)
                    bqc_dist[(hr, fr)] = bqc_dist.get((hr, fr), 0) + ph * ps
    bqc3 = [(_BQC.get(k, str(k)), f'{v * 100:.1f}%') for k, v in sorted(bqc_dist.items(), key=lambda x: -x[1])[:3]]

    return {
        'spf': {'pred': spf_pred, 'conf': spf_conf, 'probs': {'胜': pw, '平': pd, '负': pa}},
        'handicap': {'pred': rq_pred, 'conf': rq_conf, 'label': rq_label,
                     'probs': {'让胜': rqw, '让平': rqd, '让负': rql} if hc != 0 else {}},
        'total_goals': {'dist': tg_ord[:5], 'top': tg_ord[0][0], 'top_prob': tg_ord[0][1]},
        'scoreline': {'top': (bf_top[0], bf_top[1]), 'top5': bf5,
                       'top5_raw': bf5_raw, 'top_prob': bf_top[2]},
        'half_full': {'top3': bqc3},
    }


# ═══════════════════════════════════════════════════════════════
# P1: 比分/总进球负二项分布模型（2026-04-05 新增）
# 根因：昨日比分命中率3%，总进球3%——泊松方差不足以描述实际
# 负二项分布：方差 = μ + μ²/r（r越小方差越大，更贴合足球实际）
# ═══════════════════════════════════════════════════════════════

def _nbpmf(k, mu, r):
    """负二项概率质量函数"""
    from math import exp, log, factorial, lgamma
    if mu <= 0 or r <= 0:
        return 0.0
    # log(C(k+r-1, k)) = lgamma(k+r) - lgamma(k+1) - lgamma(r)
    log_prob = (lgamma(k + r) - lgamma(k + 1) - lgamma(r)
                + r * log(r / (r + mu))
                + k * log(mu / (r + mu)))
    return exp(log_prob)


def build_nb_matrix(lam_h, lam_a, r_h=2.0, r_a=2.0, max_g=7, league=''):
    """用负二项分布构建比分矩阵（替代泊松）

    r: 自由度参数，r越小方差越大
    足球实际r≈1.5-2.5（方差显著大于均值）
    默认r=2.0（经验值）

    league: 联赛名称（用于总进球校准）

    Returns:
        {(home_goals, away_goals): prob}
    """
    matrix = {}
    for i in range(max_g):
        for j in range(max_g):
            p_h = _nbpmf(i, lam_h, r_h)
            p_a = _nbpmf(j, lam_a, r_a)
            matrix[(i, j)] = p_h * p_a
    # 相关性修正（沿用泊松的rho=-0.13）
    rho = -0.13
    corrections = {
        (0, 0): 1 - rho * lam_h * lam_a,
        (0, 1): 1 + rho * lam_a,
        (1, 0): 1 + rho * lam_h,
        (1, 1): 1 - rho,
    }
    for (i, j), factor in corrections.items():
        if (i, j) in matrix:
            matrix[(i, j)] *= factor
    total = sum(matrix.values())
    if total > 0:
        for k in matrix:
            matrix[k] = max(0, matrix[k] / total)

    # P2: 总进球校准
    if league:
        matrix = calibrate_by_total_goals(matrix, league)

    return matrix


def derive_all_plays_nb(lam_h, lam_a, handicap=0, league='', r_h=2.0, r_a=2.0, use_score_prior=False):
    """P1: 用负二项分布派生全部5玩法

    对低赔方(r<2.0)和高赔方(r可不同)使用不同方差参数，
    更真实反映足球比赛"爆冷门"的特点。
    
    use_score_prior: 是否应用联赛历史比分先验（P0）
    """
    matrix = build_nb_matrix(lam_h, lam_a, r_h, r_a, league=league)
    if use_score_prior:
        matrix = apply_score_prior(matrix, league)  # P0: 叠加比分先验

    # SPF
    pw = sum(p for (i, j), p in matrix.items() if i > j)
    pd = sum(p for (i, j), p in matrix.items() if i == j)
    pa = sum(p for (i, j), p in matrix.items() if i < j)
    ordered = sorted([('胜', pw), ('平', pd), ('负', pa)], key=lambda x: -x[1])

    # 让球
    hc = float(handicap) if handicap else 0
    if hc != 0:
        rqw = sum(p for (i, j), p in matrix.items() if i + hc > j)
        rqd = sum(p for (i, j), p in matrix.items() if i + hc == j)
        rql = sum(p for (i, j), p in matrix.items() if i + hc < j)
        rq_ord = sorted([('让胜', rqw), ('让平', rqd), ('让负', rql)], key=lambda x: -x[1])
        rq_pred, rq_conf = rq_ord[0]
        ds = f'主让{abs(hc):.0f}球' if hc < 0 else (f'主受让{abs(hc):.0f}球' if hc > 0 else '平手')
        rq_label = f'{rq_pred}({ds},{rq_conf*100:.0f}%)'
    else:
        rq_pred, rq_conf, rq_label = None, 0, '无让球数据'

    # 总进球分布
    tg_dist = {}
    for (i, j), p in matrix.items():
        tg_dist[i + j] = tg_dist.get(i + j, 0) + p
    tg_ord = sorted(tg_dist.items(), key=lambda x: -x[1])

    # 比分（让球约束）
    rq_group = {'让胜': [], '让平': [], '让负': []}
    for (i, j), p in matrix.items():
        adj = i + hc
        g = '让胜' if adj > j else ('让平' if adj == j else '让负')
        rq_group[g].append((i, j, p))
    top_group = rq_group.get(rq_pred, rq_group.get('让胜', []))
    if top_group:
        scored = [(i, j, p * (1.5 if (i + j) == tg_ord[0][0] else 1.0))
                  for i, j, p in top_group]
        scored.sort(key=lambda x: -x[2])
        bf5 = [(f'{i}-{j}', f'{p*100:.1f}%') for i, j, p in scored[:5]]
    else:
        cands = [(i, j, p) for (i, j), p in matrix.items()]
        cands.sort(key=lambda x: -x[2])
        bf5 = [(f'{i}-{j}', f'{p*100:.1f}%') for i, j, p in cands[:5]]

    # 半全场（用半场λ=全局λ×0.45的两阶段NB）
    lh2 = lam_h * 0.45
    la2 = lam_a * 0.45
    bqc_dist = {}
    for hi in range(5):
        for hj in range(5):
            ph = _nbpmf(hi, lh2, r_h) * _nbpmf(hj, la2, r_a)
            hr = 1 if hi > hj else (0 if hi == hj else -1)
            for si in range(7 - hi):
                for sj in range(7 - hj):
                    ps = _nbpmf(si, lam_h * 0.55, r_h) * _nbpmf(sj, lam_a * 0.55, r_a)
                    fr = 1 if hi + si > hj + sj else (0 if hi + si == hj + sj else -1)
                    bqc_dist[(hr, fr)] = bqc_dist.get((hr, fr), 0) + ph * ps
    bqc3 = [(_BQC.get(k, str(k)), f'{v*100:.1f}%')
             for k, v in sorted(bqc_dist.items(), key=lambda x: -x[1])[:3]]

    return {
        'spf': {'pred': ordered[0][0], 'conf': ordered[0][1],
                 'probs': {'胜': pw, '平': pd, '负': pa}},
        'handicap': {'pred': rq_pred, 'conf': rq_conf, 'label': rq_label},
        'total_goals': {'dist': tg_ord[:5], 'top': tg_ord[0][0], 'top_prob': tg_ord[0][1]},
        'scoreline': {'top5': bf5},
        'half_full': {'top3': bqc3},
        'nb_used': True,
    }


# ═══════════════════════════════════════════════════════════════
# P1: 比分赔率融合（scoreline_odds_fusion增强）
# ═══════════════════════════════════════════════════════════════

def nb_scoreline_odds_fusion(nb_plays, league='', market_info=None):
    """P1: 用市场比分赔率修正负二项矩阵

    市场比分赔率（如体彩网、美职平台）反映庄家真实意图，
    用它修正NB矩阵的比分排序。

    market_info: {'score_odds': {'1-0': 8.5, '2-1': 10.0, ...}, ...}
    """
    if not market_info or not nb_plays.get('scoreline', {}).get('top5'):
        return nb_plays

    score_odds = market_info.get('score_odds', {})
    if not score_odds:
        return nb_plays

    try:
        # 归一化市场赔率→概率
        market_probs = {}
        for score, odds in score_odds.items():
            try:
                parts = score.split('-')
                i, j = int(parts[0]), int(parts[1])
                market_probs[(i, j)] = 1.0 / float(odds)
            except:
                continue
        m_total = sum(market_probs.values())
        if m_total > 0:
            for k in market_probs:
                market_probs[k] /= m_total

        # 融合：NB概率×市场概率（的几何平均）
        nb_top = nb_plays['scoreline']['top5']
        fused = []
        for score_str, nb_prob_str in nb_top:
            parts = score_str.split('-')
            i, j = int(parts[0]), int(parts[1])
            m_prob = market_probs.get((i, j), 0.01)
            fused.append((score_str, nb_prob_str, m_prob))

        # 按融合概率重新排序
        fused.sort(key=lambda x: -x[2])
        nb_plays['scoreline']['top5_fused'] = [
            (s, f'{p*100:.1f}%') for s, _, p in fused[:5]
        ]
        nb_plays['scoreline']['top5'] = nb_plays['scoreline']['top5_fused']
    except Exception:
        pass

    return nb_plays


# ============================================================
# 联赛λ分层 + 总进球校准 (2026-04-14 新增)
# 来源: calibrated_poisson.py 联赛分层改造
# ============================================================

# 联赛专属总进球λ参数（主场+客场均值）
LEAGUE_AVG_TOTAL_GOALS = {
    '英超': 2.95, '英冠': 2.75, '英甲': 2.85, '英乙': 2.70,
    '西甲': 2.85, '意甲': 2.90, '德甲': 3.30, '法甲': 2.75,
    '葡超': 2.55, '荷甲': 3.15, '荷乙': 3.20,
    '比甲': 2.85, '瑞士超': 3.05, '奥地利超': 3.10,
    '瑞典超': 2.90, '挪超': 3.20, '丹麦超': 3.00,
    '芬超': 2.60, '希腊超': 2.30, '土超': 2.70,
    '俄超': 2.75, '乌克兰超': 2.65,
    '美职': 2.85, '墨超': 2.70, '巴甲': 2.75, '阿甲': 2.55,
    '智利甲': 2.60, '哥伦甲': 2.55, '秘鲁甲': 2.50,
    'J联赛': 2.60, 'J2联赛': 2.55, 'K1联赛': 2.45, '韩职': 2.50,
    '中超': 2.80, '澳超': 2.80,
    '爱超': 2.95, '苏超': 2.85,
    '欧冠': 2.95, '欧罗巴': 2.85, '欧协联': 2.75,
    '英足总杯': 2.90, '意大利杯': 2.80, '德国杯': 3.10, '法国杯': 2.85,
}

# 联赛主场胜率（用于SPF先验）
LEAGUE_HOME_WIN_RATES = {
    '英超': 0.46, '英冠': 0.44, '英甲': 0.438, '英乙': 0.42,
    '西甲': 0.46, '意甲': 0.46, '德甲': 0.47, '法甲': 0.46,
    '葡超': 0.47, '荷甲': 0.48, '荷乙': 0.47,
    '比甲': 0.45, '瑞士超': 0.48, '奥地利超': 0.49,
    '瑞典超': 0.50, '挪超': 0.51, '丹麦超': 0.49,
    '芬超': 0.46, '希腊超': 0.48, '土超': 0.50,
    '俄超': 0.47, '乌克兰超': 0.45,
    '美职': 0.50, '墨超': 0.48, '巴甲': 0.50, '阿甲': 0.43,
    '智利甲': 0.45, '哥伦甲': 0.46, '秘鲁甲': 0.44,
    'J联赛': 0.47, 'J2联赛': 0.46, 'K1联赛': 0.49, '韩职': 0.50,
    '中超': 0.48, '澳超': 0.45,
    '爱超': 0.40, '苏超': 0.43,
    '欧冠': 0.50, '欧罗巴': 0.48, '欧协联': 0.47,
    '英足总杯': 0.48, '意大利杯': 0.46, '德国杯': 0.50, '法国杯': 0.47,
}


def get_league_avg_total_goals(league):
    """获取联赛总进球λ参数"""
    return LEAGUE_AVG_TOTAL_GOALS.get(league, 2.75)


def get_league_home_win_rate(league):
    """获取联赛主场胜率先验"""
    return LEAGUE_HOME_WIN_RATES.get(league, 0.46)


def calibrate_total_goals(league, default_lam_total=2.75):
    """校准总进球λ（联赛分层版）

    当有联赛参数时，使用 league_avg_goals 替代 default_lam_total。
    用于 derive_all_plays_nb / build_nb_matrix 的 lam_h/lam_a 推导。

    Args:
        league: 联赛名称
        default_lam_total: 默认总进球λ（当联赛不在表中时使用）

    Returns:
        float: 总进球λ参数
    """
    return get_league_avg_total_goals(league)


def calibrate_by_total_goals(matrix, league, expected_total=None):
    """P2: 总进球数校准函数

    根据联赛历史总进球均值校准矩阵，使期望总进球与联赛实际水平一致。

    Args:
        matrix: 泊松/NB比分矩阵 {(home_goals, away_goals): prob}
        league: 联赛名称（用于查找 league_avg_goals）
        expected_total: 期望总进球数（None时使用联赛默认值）

    Returns:
        dict: 校准后的矩阵
    """
    if expected_total is None:
        expected_total = get_league_avg_total_goals(league)

    # 计算当前矩阵的期望总进球
    current_total = sum((i + j) * p for (i, j), p in matrix.items())

    if current_total <= 0 or expected_total <= 0:
        return matrix

    # 缩放因子：目标/当前
    scale = expected_total / current_total
    # 限制缩放幅度，避免过度调整
    scale = max(0.80, min(1.20, scale))

    # 调整矩阵中的期望进球
    adj = {}
    for (i, j), p in matrix.items():
        new_i = max(0, min(7, int(i * scale)))
        new_j = max(0, min(7, int(j * scale)))
        # 合并到对应格子（如果有的话），否则分配到最近的格子
        if (new_i, new_j) in adj:
            adj[(new_i, new_j)] += p
        else:
            adj[(new_i, new_j)] = p

    # 归一化
    total = sum(adj.values())
    if total > 0:
        adj = {k: v / total for k, v in adj.items()}

    return adj


# ═══════════════════════════════════════════════════════════════
# P2-1: 置信度档位重新校准 (2026-04-23)
# 数据来源: 409场回测，置信度分档命中率分析
# ═══════════════════════════════════════════════════════════════

def confidence_tier_recalibrate(confidence, league=''):
    """P2-3: 置信度档位重新校准（方向已反转）

    数据来源: 317场回测，置信度分档命中率分析

    校准逻辑（反转后）：
    - conf<0.4: 实际命中率32.6% → 低估了自己 → 上调
    - conf>=0.6: 实际命中率18.6% → 高估了自己 → 下调20%
    - 弱联赛: 英超36.8%/英冠7.1%/J联赛0% → 降权15%

    注意: 此函数仅调整用于推荐展示的置信度，不修改存储值。
    """
    # P0修复(2026-05-11): 统一弱联赛列表，补全日职/J联赛/韩职
    # 与ml_predict_5play.py WEAK_LEAGUES保持一致
    WEAK_LEAGUES = {'英超', '英冠', '法乙', '沙职', '解放者杯', '欧罗巴', '美职',
                    '日职', '日职联', 'J1', '韩K联', '韩K1联', '韩职', 'K1',
                    '泰超', '马来超', '越南联', '印尼甲', '中超', '中超旧名'}

    if league in WEAK_LEAGUES:
        return confidence * 0.85

    # P2-3: 校准方向反转（实测低conf反而命中率高）
    # P0修复(2026-05-11): conf=0.35-0.40区间4场全中，扩大上调范围
    # 移除0.45上限，允许低conf区间充分膨胀
    if confidence >= 0.65:
        # 高信心严重高估自己(18.6%) → 大幅下调20%
        return confidence * 0.80
    elif confidence >= 0.40:
        # 中等信心轻微高估 → 小幅下调
        return confidence * 0.90
    else:
        # 低信心实际命中率32.6% → 应大幅上调，移除上限
        return confidence * 1.15


# ─────────────────────────────────────────────
# Eloa埃罗预测法 (V7整合)
# ─────────────────────────────────────────────
def eloa_predict(home_pts, away_pts, games=15):
    """埃罗预测法: 主场胜率=44.8%+0.53%×(积分差/场次)
    
    源自《盘口赔率实战大全》
    """
    try:
        diff = home_pts - away_pts
        home_rate = 44.8 + 0.53 * diff / games
        away_rate = 24.5 - 0.39 * diff / games
        draw_rate = max(15, min(45, 100 - home_rate - away_rate))
        return {
            '胜': max(15, min(80, home_rate)),
            '平': draw_rate,
            '负': max(10, min(65, away_rate))
        }
    except:
        return None

def eloa_confidence(home_pts, away_pts, games=15):
    """返回eloa置信度（0-1），积分差越大越可信"""
    try:
        diff = abs(home_pts - away_pts)
        # 积分差>20且比赛>10场 → 高置信
        confidence = min(0.95, 0.40 + (diff / games) * 0.05)
        return confidence
    except:
        return 0.40

# ─────────────────────────────────────────────
# 开盘思维检测 (V7整合)
# ─────────────────────────────────────────────
_ODDS_BAND_DIRECTION = [
    ((0, 1.70), '实盘', 0.08),   # 强队主场，实盘高估
    ((1.70, 2.00), '中庸', 0.00),
    ((2.00, 2.50), '韬盘', -0.06),  # 韬盘：庄家不看好主队
    ((2.50, 100), '高赔诱主', -0.10),
]

def opening_detection(ho, do, ao):
    """开盘思维检测：返回(类型, 方向调整系数)
    
    源自《欧赔核心思维》
    - 实盘: ho<1.7 → 庄家真实看好主队
    - 韬盘: ho>2.0 → 庄家刻意压低主胜，营造信心假象
    """
    try:
        direction = '中庸'
        adj = 0.0
        for (low, high), name, mult in _ODDS_BAND_DIRECTION:
            if low <= ho < high:
                direction = name
                adj = mult
                break
        
        # 平赔手法
        draw_method = '中庸'
        if do < 2.90: draw_method = '超低平'
        elif do < 3.20: draw_method = '单分平'
        
        # 客胜判断
        if ao < 1.80: away_signal = '客胜实盘'
        elif ao > 3.50: away_signal = '客胜诱主'
        else: away_signal = '中性'
        
        return {'direction': direction, 'draw_method': draw_method, 
                'away_signal': away_signal, 'adjustment': adj}
    except:
        return {'direction': '中庸', 'draw_method': '中庸', 'away_signal': '中性', 'adjustment': 0.0}

# ─────────────────────────────────────────────
# 派系关系检测 (V7整合)
# ─────────────────────────────────────────────
_FACTIONS = {
    '巴萨系': ['西班牙人', '塞维利亚', '贝蒂斯', '莱加内斯', '希洪竞技', '马拉加'],
    '皇马系': ['马德里竞技', '皇家社会', '毕尔巴鄂', '巴伦西亚', '西班牙人'],
    '拜仁系': ['法兰克福', '汉堡', '柏林赫塔', '门兴', '沙尔克04'],
    '尤文系': ['都灵', '热那亚', '桑普多利亚', '佛罗伦萨', '卡利亚里'],
    '英超派': ['曼联', '曼城', '利物浦', '阿森纳', '切尔西', '热刺'],
}

_FACTION_BONUS = {
    '巴萨系': 0.12, '皇马系': 0.12, '拜仁系': 0.10, '尤文系': 0.10, '英超派': 0.08
}

def faction_detection(home, away, league=''):
    """检测派系关系，返回主队加成系数
    
    同派系比赛：主队胜率+12%（派系保护）
    """
    for faction, teams in _FACTIONS.items():
        if home in teams and away in teams:
            bonus = _FACTION_BONUS.get(faction, 0.08)
            return {'faction': faction, 'bonus': bonus, 'same_faction': True}
    return {'faction': None, 'bonus': 0.0, 'same_faction': False}

def eloa_faction_adjustment(probs, home_pts, away_pts, games, home, away, league=''):
    """综合调整: Eloa + 开盘思维 + 派系
    
    返回调整后的概率字典和信号列表
    """
    signals = []
    
    # Eloa调整
    eloa = eloa_predict(home_pts, away_pts, games)
    if eloa:
        eo_conf = eloa_confidence(home_pts, away_pts, games)
        # 融合eloa：权重=置信度
        for key in ['胜', '平', '负']:
            if key in probs and key in eloa:
                probs[key] = probs[key] * (1 - eo_conf) + eloa[key] * eo_conf
        signals.append(f"Eloa置信{eo_conf:.0%}")
    
    # 开盘思维调整
    try:
        ho = probs.get('_ho', 2.0)
        do = probs.get('_do', 3.2)
        ao = probs.get('_ao', 3.5)
        opening = opening_detection(ho, do, ao)
        if opening['adjustment'] != 0:
            # 韬盘/高赔诱主：降低主胜概率
            probs['胜'] = max(0.10, probs['胜'] + opening['adjustment'])
            probs['负'] = min(0.85, probs['负'] - opening['adjustment'] * 0.5)
            signals.append(f"开盘:{opening['direction']}({opening['adjustment']:+.0%})")
    except: pass
    
    # 派系调整
    faction_info = faction_detection(home, away, league)
    if faction_info['same_faction']:
        probs['胜'] = min(0.90, probs['胜'] + faction_info['bonus'])
        probs['负'] = max(0.10, probs['负'] - faction_info['bonus'] * 0.5)
        signals.append(f"派系:{faction_info['faction']}(+{faction_info['bonus']:.0%})")
    
    return probs, signals
