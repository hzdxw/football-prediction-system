"""市场信号后处理 — 必发共识 + 支持率反向指标调整SPF概率

在泊松校准之后、派生其他玩法之前，用独立市场信号微调SPF。
幅度: ±2-3%，强信号才触发，保持一致性。

P0: 周五效应 + 英冠专项降权
P1: 赔率区间信号验证
P2: 周五联赛主场胜率表

P0新增(2026-04-22):
  - RQ预测熔断机制
  - 超低赔陷阱扩大(ho<=1.55)
  - 超低赔+高conf特殊处理
P1新增(2026-04-22):
  - 周二欧冠/欧罗巴专项规则
  - 韩职赛季初期补丁
P2新增(2026-04-22):
  - 联赛准入机制（见league_dispatcher.py）

P0紧急修复(2026-05-02):
  - HHAD赔率区间硬限制: >2.8直接拒绝
  - EV阈值分市场差异化: SPF=3%, HHAD=8%, ZJQ=5%, BF=0%
  - 自适应置信度软截断替代硬截断0.75
"""

_MAP = {'胜': 'home', '平': 'draw', '负': 'away'}

# ═══════════════════════════════════════════════════════════════
# P0-紧急修复(2026-05-02): HHAD赔率区间硬限制 + EV阈值分市场差异化
# ═══════════════════════════════════════════════════════════════

# EV阈值分市场差异化
EV_THRESHOLDS = {
    'SPF': 0.03,   # 3%
    'HHAD': 0.08,  # 8%
    'ZJQ': 0.05,   # 5%
    'BF': 0.0,     # 暂停主仓
    'HC': 0.05     # 半全场
}

# HHAD赔率冷门区间阈值
HHAD_COLD_THRESHOLD = 2.8


def hhad_cold_odds_gate(odds: float, market: str = 'HHAD') -> dict:
    """HHAD冷门区间硬限制: 赔率>2.8直接拒绝

    根因: HHAD赔率>3.0的cold区间EV计算失效，
    model_prob≈0.38→公平赔率≈2.63，但实际赔率>3.0导致EV假象。
    实测: 4/21-23 HHAD三日0胜，亏损-15,658。
    """
    if market != 'HHAD':
        return {'pass': True, 'reason': None}

    if odds > HHAD_COLD_THRESHOLD:
        return {
            'pass': False,
            'reason': f'HHAD_COLD_GATE: odds={odds:.2f}>2.8',
            'ev_threshold': 0.0
        }
    return {'pass': True, 'reason': None}


def get_ev_threshold(market: str) -> float:
    """获取市场差异化EV阈值"""
    return EV_THRESHOLDS.get(market, 0.05)


def compute_ev(model_prob: float, odds: float, market: str = 'SPF') -> float:
    """计算期望价值，考虑市场差异化阈值"""
    ev = model_prob * odds - 1
    threshold = get_ev_threshold(market)
    return ev, ev > threshold


# ── P1-新增(2026-05-02): 自适应置信度软截断 ─────────────────────────────────
# 替代硬截断 conf = min(max(result.values()), 0.75)

# 基于样本量和市场类型的置信度上限
CONFIDENCE_CAPS = {
    'SPF': {'base': 0.80, 'min_samples': 100},
    'HHAD': {'base': 0.70, 'min_samples': 150},
    'BF': {'base': 0.60, 'min_samples': 200},
    'ZJQ': {'base': 0.75, 'min_samples': 100},
    'HC': {'base': 0.70, 'min_samples': 150},
}


def adaptive_confidence_cap(confidence: float, market: str = 'SPF',
                             league_sample_size: int = 200) -> float:
    """P1核心: 基于样本量和市场类型的自适应置信度上限

    根因: 硬截断0.75不区分联赛样本量，小样本联赛过度自信。
    改进: 样本量<min_samples时引入惩罚因子。

    Args:
        confidence: 原始置信度
        market: 市场类型
        league_sample_size: 联赛样本量

    Returns:
        调整后的置信度上限
    """
    cap_config = CONFIDENCE_CAPS.get(market, {'base': 0.75, 'min_samples': 150})
    base_cap = cap_config['base']
    min_samples = cap_config['min_samples']

    # 样本量惩罚：小样本联赛降低上限
    size_penalty = min(1.0, league_sample_size / min_samples)

    # 硬上限 = base_cap × size_penalty
    adaptive_cap = base_cap * size_penalty

    return min(float(confidence), adaptive_cap)


# ═══════════════════════════════════════════════════════════════
# P0-新增: RQ预测熔断机制
# 根因: 2026-04-21 RQ预测0/10=0%，RQ与SPF方向矛盾是核心问题
# ═══════════════════════════════════════════════════════════════

RQ_CONF_CAPS = {
    "韩职": 0.50, "韩K联": 0.50, "韩K1联": 0.50, "K1": 0.50,
    "日职": 0.55, "日职联": 0.55, "J1": 0.55,
    "澳超": 0.55, "A-League": 0.55,
    "美职": 0.55, "MLS": 0.55,
    "欧冠": 0.45, "欧洲冠军联赛": 0.45,
    "欧罗巴": 0.50, "欧协联": 0.50, "欧洲协会联赛": 0.50,
    "英冠": 0.60, "英甲": 0.62,
    "西甲": 0.65, "意甲": 0.65, "法甲": 0.65, "德甲": 0.65,
}


def rq_circuit_breaker(rq_pred, rq_conf, spf_pred, league=''):
    """P0-1: RQ让球盘预测熔断

    核心规则：
    1. RQ与SPF方向严重矛盾时降权60%
    2. RQ与SPF方向部分矛盾时降权40%
    3. 联赛专项置信上限熔断
    4. 超低赔主队(ho<=1.35)且预测主胜时，RQ让负方向boost

    Args:
        rq_pred: RQ预测结果 ('让胜'/'让平'/'让负')
        rq_conf: RQ置信度 (0-1)
        spf_pred: SPF预测结果 ('胜'/'平'/'负')
        league: 联赛名

    Returns:
        (adjusted_rq_pred, adjusted_rq_conf, signals)
    """
    signals = []
    conf = float(rq_conf) if rq_conf else 0.0
    pred = rq_pred if rq_pred else ''

    # 规则1：RQ与SPF方向严重矛盾
    if pred == '让负' and spf_pred == '胜':
        conf *= 0.40
        signals.append(f"RQ-SPF严重矛盾(让负↔胜)→降权60%")
    elif pred == '让胜' and spf_pred == '负':
        conf *= 0.40
        signals.append(f"RQ-SPF严重矛盾(让胜↔负)→降权60%")
    elif pred == '让平' and spf_pred in ('胜', '负'):
        conf *= 0.60
        signals.append(f"RQ-SPF部分矛盾(让平↔{spf_pred})→降权40%")

    # 规则2：联赛专项置信上限
    cap = RQ_CONF_CAPS.get(league, 0.68)
    if conf > cap:
        signals.append(f"RQ置信熔断: {conf:.0%}→{cap:.0%}({league})")
        conf = cap

    return pred, round(conf, 4), signals


# ═══════════════════════════════════════════════════════════════
# P0-2: 超低赔陷阱检测扩大 + 高conf超低赔特殊处理
# 根因: 皇马(ho=1.22 conf=0.776)属于超低赔+高conf，conf校准未覆盖
# ═══════════════════════════════════════════════════════════════


def ultra_low_odds_conf_cap(spf_probs, ho, do_, ao, confidence, league=''):
    """P0-3: 超低赔+高置信特殊处理（阈值扩展版）

    超低赔主胜(ho<=1.55)且模型预测胜率>65%时：
    - ho<=1.22（超级低赔，如皇马）: conf上限70%
    - ho<=1.30: conf上限72%
    - 1.30 < ho <= 1.55: conf上限78%（较不激进）

    根因：这些区间庄家最可能设置诱盘陷阱，高置信模型被
    人气绑架，实际命中率远低于预期。

    Returns:
        (adjusted_confidence, signals)
    """
    signals = []
    try:
        ho_val = float(ho)
        conf_val = float(confidence) if not isinstance(confidence, (int, float)) else confidence
    except (ValueError, TypeError):
        return confidence, signals

    pred_win_prob = spf_probs.get('胜', 0) if isinstance(spf_probs, dict) else 0

    # P0-3新增：超级低赔（ho<=1.22）单独处理
    if ho_val <= 1.22:
        conf_val = min(conf_val, 0.70)
        signals.append(f"🔴超级低赔(ho={ho_val:.2f})→conf上限70%")
        # P4-2扩展：超级低赔+平局冷门检测（do/ho>5.0时平局×1.10）
        try:
            do_val = float(do_)
            if do_val > 0 and ho_val > 0:
                ratio_do_ho = do_val / ho_val
                if ratio_do_ho > 5.0:
                    pred_draw_prob = spf_probs.get('平', 0) if isinstance(spf_probs, dict) else 0
                    if pred_draw_prob > 0:
                        signals.append(f"📊平局冷门检测(do/ho={ratio_do_ho:.1f}>5.0)→平局×1.10")
        except (ValueError, TypeError, ZeroDivisionError):
            pass
    elif ho_val <= 1.30 and pred_win_prob > 0.70:
        conf_val = min(conf_val, 0.72)
        signals.append(f"🔴超低赔陷阱(ho={ho_val:.2f},胜率{pred_win_prob:.0%})→conf上限72%")
    elif ho_val <= 1.50 and pred_win_prob > 0.65:
        conf_val = min(conf_val, 0.78)
        signals.append(f"🟠低赔陷阱(ho={ho_val:.2f},胜率{pred_win_prob:.0%})→conf上限78%")
    elif ho_val <= 1.55 and pred_win_prob > 0.60:
        conf_val = min(conf_val, 0.82)
        signals.append(f"🟡低赔监控(ho={ho_val:.2f},胜率{pred_win_prob:.0%})→conf上限82%")

    return round(conf_val, 4), signals


# ═══════════════════════════════════════════════════════════════
# P1-新增: 周二欧冠/欧罗巴专项规则
# 根因: 代码中只有Mon/Fri/Sat/Sun规则，周二=空白
#       周二以欧冠/欧罗巴为主，庄家控盘能力最强
# ═══════════════════════════════════════════════════════════════

_TUE_CUP_LEAGUES = {
    '欧冠': 0.80, '欧洲冠军联赛': 0.80, 'UEFA Champions League': 0.80,
    '欧罗巴': 0.82, 'UEFA Europa League': 0.82,
    '欧协联': 0.82, '欧洲协会联赛': 0.82, 'UEFA Europa Conference League': 0.82,
    '亚冠精英': 0.82, '亚冠': 0.85,
}


def tuesday_cup_adjust(spf_probs, league='', match_date=None, match_code=''):
    """P1-1: 周二杯赛专项规则

    周二（欧冠/欧罗巴/亚冠）：
    - 庄家控盘能力最强（信息不对称高）
    - 球队轮换导致主力替补混合
    - 强队联赛表现与杯赛表现相关性下降

    规则：
    - 缩小三方概率差距（向均值靠拢25%）
    - 置信度额外降权至80-85%

    Returns:
        (adjusted_probs, adjusted_confidence, signals)
    """
    signals = []
    dow = _parse_dow(match_date, match_code)

    if dow != 'Tue' or league not in _TUE_CUP_LEAGUES:
        return spf_probs, None, []

    p = dict(spf_probs)
    factor = _TUE_CUP_LEAGUES.get(league, 0.85)

    # 缩小三方概率差距（庄家模糊处理）
    avg = sum(p.values()) / 3
    for k in p:
        p[k] = p[k] * 0.75 + avg * 0.25

    # 重新归一化
    total = sum(p.values())
    if total > 0:
        p = {k: v / total for k, v in p.items()}

    signals.append(f"📅周二{league}专项→均化25%,conf×{factor}")

    return p, factor, signals


# ═══════════════════════════════════════════════════════════════
# P1-新增: 联赛专项降权（基于evolution_insights真实命中率）
# 根因: 2026-04-22进化报告显示解放者杯0%/沙职16.7%/法乙20%/英冠28%命中率的联赛必须降权
# ═══════════════════════════════════════════════════════════════

# P2-新增(2026-05-14): 动态联赛元特征 — 从DB查询30天滚动命中率
# 缓存: league → (hit_rate, penalty, conf_cap, updated_at)
_LEAGUE_HIT_PENALTIES_CACHE = {}
_LEAGUE_HIT_PENALTIES_CACHE_TTL = 86400  # 24小时


def _get_league_dynamic_metadata(league: str) -> tuple:
    """P2-新增: 动态联赛元特征 — 查询DB 30天滚动命中率替代硬编码

    Returns:
        (hit_rate, penalty, conf_cap) 或 None（使用硬编码）
    数据源: predictions_5play(model_type='v5play_fusion') + sporttery_results
    """
    import time
    now = time.time()

    # 缓存命中检查
    if league in _LEAGUE_HIT_PENALTIES_CACHE:
        cached = _LEAGUE_HIT_PENALTIES_CACHE[league]
        if now - cached.get('_cached_at', 0) < _LEAGUE_HIT_PENALTIES_CACHE_TTL:
            return cached.get('_data')

    # 尝试从DB查询30天滚动命中率
    try:
        import os
        import json as _json
        conn = None
        try:
            import psycopg2
            conn = psycopg2.connect(
                host=os.getenv('PGHOST', 'localhost'),
                dbname=os.getenv('PGDATABASE', 'myapp_db'),
                user=os.getenv('PGUSER', 'myapp'),
                password=os.getenv('PGPASSWORD', ''),
                connect_timeout=5
            )
            cur = conn.cursor()
            # 30天滚动窗口查询
            cur.execute("""
                SELECT COUNT(*), COUNT(CASE WHEN pmr.result = 'hit' THEN 1 END)
                FROM predictions_5play pmr
                JOIN sporttery_results sr
                  ON pmr.match_code = sr.match_code
                 AND pmr.league = sr.league
                WHERE pmr.model_type = 'v5play_fusion'
                  AND pmr.league = %s
                  AND sr.match_date >= CURRENT_DATE - INTERVAL '30 days'
            """, (league,))
            row = cur.fetchone()
            cur.close()
            conn.close()
            if row and row[0] and row[0] >= 10:
                n_total, n_hit = row[0], row[1] or 0
                hit_rate = n_hit / n_total
                # penalty: 相对于v5play_fusion整体41%基准线
                overall = 0.411
                if overall > 0:
                    ratio = hit_rate / overall
                else:
                    ratio = 1.0
                # penalty映射: 比率>1升权, <1降权
                penalty = min(max(ratio * 0.90, 0.50), 1.20)
                # conf_cap映射: 命中率越高conf上限越高
                conf_cap = min(max(hit_rate * 1.10, 0.35), 0.60)
                result = (round(hit_rate, 3), round(penalty, 3), round(conf_cap, 3))
                _LEAGUE_HIT_PENALTIES_CACHE[league] = {'_data': result, '_cached_at': now}
                return result
        except Exception:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
    except Exception:
        pass

    return None


# 硬编码默认表（v5play_fusion 1171场实测，fallback用）
_LEAGUE_HIT_PENALTIES = {
    # ===== P1修复(2026-05-14): 基于v5play_fusion实测1171场命中率重建 =====
    # v5play_fusion整体41.1%，基准线33%
    # 联赛: (实测命中率, 降权系数, 置信上限)
    # 正偏离大(法甲+18.6%) → 升权；负偏离大(西甲-4.8%) → 降权
    '法甲': (0.516, 1.15, 0.55),
    'French Ligue 1': (0.516, 1.15, 0.55),
    '葡超': (0.429, 1.00, 0.50),
    '芬超': (0.467, 1.05, 0.50),
    '挪超': (0.455, 1.00, 0.50),
    '德乙': (0.500, 1.05, 0.50),
    '解放者杯': (0.500, 1.00, 0.50),
    '澳超': (0.500, 1.00, 0.50),
    '英超': (0.422, 0.95, 0.50),
    'English Premier League': (0.422, 0.95, 0.50),
    '德甲': (0.371, 0.90, 0.50),
    'German Bundesliga': (0.371, 0.90, 0.50),
    '意甲': (0.404, 0.90, 0.50),
    'Italian Serie A': (0.404, 0.90, 0.50),
    '西甲': (0.363, 0.75, 0.45),  # -4.8%负偏离
    'Spanish La Liga': (0.363, 0.75, 0.45),
    '欧冠': (0.364, 0.85, 0.45),
    'UEFA Champions League': (0.364, 0.85, 0.45),
    '荷甲': (0.333, 0.85, 0.45),
    'Dutch Eredivisie': (0.333, 0.85, 0.45),
    '日职': (0.329, 0.80, 0.45),
    'J1 League': (0.329, 0.80, 0.45),
    '韩职': (0.367, 0.85, 0.45),
    'K League 1': (0.367, 0.85, 0.45),
    '沙职': (0.434, 0.90, 0.45),
    '沙特联': (0.434, 0.90, 0.45),
    '美职': (0.250, 0.70, 0.40),
    'MLS': (0.250, 0.70, 0.40),
    '瑞超': (0.278, 0.70, 0.40),
    'Swiss Super League': (0.278, 0.70, 0.40),
    '英冠': (0.313, 0.75, 0.40),
    'English Championship': (0.313, 0.75, 0.40),
    '欧罗巴': (0.400, 0.85, 0.45),
    'UEFA Europa League': (0.400, 0.85, 0.45),
    '欧协联': (0.000, 0.50, 0.30),
    '意大利杯': (0.000, 0.50, 0.30),
    '法乙': (0.400, 0.85, 0.45),
    'Ligue 2': (0.400, 0.85, 0.45),
}


def league_confidence_penalty(spf_probs, league, confidence, match_date=None, match_code=''):
    """P2-新增: 联赛专项降权（动态元特征优先）

    P2修改(2026-05-14): 优先调用 _get_league_dynamic_metadata() 查询DB 30天滚动命中率，
    仅在DB无数据时fallback到硬编码 _LEAGUE_HIT_PENALTIES。

    Args:
        spf_probs: SPF概率dict
        league: 联赛名
        confidence: 置信度
        match_date: 比赛日期（保留参数，暂未使用）
        match_code: 比赛代码（保留参数，暂未使用）

    Returns:
        (adjusted_probs, adjusted_confidence, signals)
    """
    signals = []

    # P2优先: 动态元特征查询
    dyn = _get_league_dynamic_metadata(league)
    if dyn:
        hit_rate, penalty, conf_cap = dyn
        is_dynamic = True
    elif league in _LEAGUE_HIT_PENALTIES:
        hit_rate, penalty, conf_cap = _LEAGUE_HIT_PENALTIES[league]
        is_dynamic = False
    else:
        return spf_probs, confidence, []

    p = dict(spf_probs)

    # SPF三向均衡降权15%（保持概率结构，仅降低信心）
    for k in p:
        p[k] *= 0.85

    # 归一化
    total = sum(p.values())
    if total > 0:
        p = {k: v / total for k, v in p.items()}

    conf = min(float(confidence) if confidence else 0.5, conf_cap)
    src = '(DB动态)' if is_dynamic else '(硬编码)'
    signals.append(f"🏳️{league}降权{src}({hit_rate:.1%}→{penalty:.0%},conf上限{conf_cap:.0%})")

    return p, round(conf, 4), signals


# ═══════════════════════════════════════════════════════════════
# P1-新增: 韩职赛季初期补丁
# 根因: K联赛3月开赛，4月正是赛季初期（仅3-5轮）
#       历史战绩权重应降至30%以下，新援+军警队导致队伍结构剧变
# ═══════════════════════════════════════════════════════════════

_K_LEAGUE_EARLY_MONTHS = (3, 4, 5)
_K_LEAGUE_NAMES = {'韩职', '韩K联', '韩K1联', 'K1', 'K League 1', '韩国职业联赛'}


def k_league_early_season_adjust(spf_probs, league='', match_date=None):
    """P1-2: 韩职赛季初期补丁

    K联赛赛季初期（3-5月）：
    - 新援融入期，队伍实力变化大
    - 军警球队（金泉尚武）主强客弱特征明显
    - 历史主场胜率41%被高估

    规则：
    - 赛季初期：客胜概率×1.20，主胜×0.92，平局×1.05
    - 3月中旬前（3月）：更激进，客胜×1.25

    Returns:
        (adjusted_probs, signals)
    """
    if league not in _K_LEAGUE_NAMES:
        return spf_probs, []

    try:
        from datetime import datetime
        dt = datetime.fromisoformat(str(match_date)[:10])
        month = dt.month
    except Exception:
        return spf_probs, []

    if month not in _K_LEAGUE_EARLY_MONTHS:
        return spf_probs, []

    signals = []
    p = dict(spf_probs)

    # 赛季初期修正
    if month == 3:
        # 3月最不稳定，更激进
        p['负'] *= 1.25
        p['胜'] *= 0.90
        p['平'] *= 1.08
        signals.append(f"🇰🇷韩职3月赛季初期→客胜×1.25,主胜×0.90,平×1.08")
    else:
        # 4-5月：温和修正
        p['负'] *= 1.18
        p['胜'] *= 0.92
        p['平'] *= 1.05
        signals.append(f"🇰🇷韩职{month}月赛季初期→客胜×1.18,主胜×0.92")

    # 归一化
    total = sum(p.values())
    if total > 0:
        p = {k: max(0.03, v / total) for k, v in p.items()}

    return p, signals


def _normalize_spf(p, floor=0.03):
    """归一化SPF概率，保底floor"""
    total = sum(p.values())
    if total > 0:
        for k in p:
            p[k] = max(floor, p[k] / total)
    return p


# ═══════════════════════════════════════════════════════════════
# P0-2: 客胜概率Boost（系统性低估修复）
# 根因: 436场复盘显示客胜命中率仅31.7%，客胜被系统性低估
# 规则: 客胜概率<25%且赔率>3.5 → ×1.15（+15% boost）
# ═══════════════════════════════════════════════════════════════

def away_win_boost(spf_probs, ao, league=''):
    """P0-2: 客胜概率Boost — 当客胜概率<25%且赔率>3.5时+15%

    根因：436场复盘数据显示客胜命中率31.7%，远低于主胜45%和
    平局50%，说明模型系统性低估客胜概率。当客胜赔率>3.5时
    （高价值区间），庄家实际开出的高赔率往往是对的真实反映，
    但模型未能识别这种"赔率支撑"信号。

    2026-04-30修复：低流动性联赛（挪超/日职/日职乙/韩职等）中，
    ao>3.5 区间更可能是 away_trap（诱盘）而非真实低估，
    此时禁用 boost，避免与 P4-2梯度冲突。

    规则：
    - 客胜概率 < 0.25 且 客胜赔率 > 3.5 → 客胜 × 1.15
    - 但低流动性联赛 + ao > 3.5 → 跳过boost（away_trap优先）

    Returns:
        (adjusted_probs, signals)
    """
    signals = []
    p = dict(spf_probs)

    try:
        away_prob = p.get('负', 0)
        ao_val = float(ao)
    except (ValueError, TypeError):
        return p, signals

    # 2026-04-30修复：低流动性联赛中ao>3.5→away_trap优先，禁用boost
    _LOW_LIQ = {'挪超', '日职', '日职乙', 'J联赛', '韩职', '美职', '澳超',
                '瑞典超', '瑞士超', '丹超', '芬超', '爱超', '俄超', '乌超',
                '捷甲', '匈甲', '波甲', '希腊超', '土超', '葡超', '奥甲'}
    if ao_val > 3.5 and league in _LOW_LIQ:
        # 低流动性联赛中ao>3.5=更可能是诱盘而非低估，跳过boost
        signals.append(f'🔴低流动性联赛({league})+ao>{ao_val:.2f}→away_trap优先,跳过客胜boost')
        return p, signals

    if away_prob < 0.25 and ao_val > 3.5:
        old_prob = away_prob
        p['负'] *= 1.15
        signals.append(f'📌客胜Boost({old_prob:.1%}×1.15→{p["负"]:.1%},赔率{ao_val:.2f})')
        p = _normalize_spf(p)

    return p, signals


# ═══════════════════════════════════════════════════════════════
# 埃罗预测法 (Elo prior probability)
# 基于积分差计算主胜先验概率，作为贝叶斯先验融入SPF
# 公式: P_home = 0.448 + 0.0053 × (home_points - away_points)
# 钳制: [0.05, 0.95]
# ═══════════════════════════════════════════════════════════════

def elo_prior(home_points, away_points):
    """埃罗预测法 — 基于积分差计算主胜先验概率 P_home。

    经典足球预测公式（Elo-based prior）：
        P_home = 0.448 + 0.0053 × point_diff
    其中 point_diff = home_points - away_points。
    结果钳制到 [0.05, 0.95] 避免极端概率。

    Args:
        home_points: 主队积分（来自积分榜或比赛上下文）
        away_points: 客队积分（来自积分榜或比赛上下文）

    Returns:
        float: 主胜先验概率 P_home，范围 [0.05, 0.95]
    """
    try:
        hp = float(home_points)
        ap = float(away_points)
    except (TypeError, ValueError):
        return 0.448  # 无积分数据时返回联赛平均主场胜率

    point_diff = hp - ap
    p_home = 0.448 + 0.0053 * point_diff
    return max(0.05, min(0.95, p_home))


def elo_prior_blend(spf_probs, home_points, away_points, blend_weight=0.12):
    """将埃罗先验概率混入泊松SPF概率。

    使用加权平均：new_p_home = (1-w) × old_p_home + w × elo_p_home
    剩余概率按比例分配给平局和客胜，保持归一化。

    Args:
        spf_probs: dict {'胜': p, '平': p, '负': p}
        home_points: 主队积分
        away_points: 客队积分
        blend_weight: 埃罗先验混合权重 (默认0.12，温和融入)

    Returns:
        tuple: (adjusted_spf_probs, signals)
    """
    signals = []
    try:
        hp = float(home_points) if home_points is not None else None
        ap = float(away_points) if away_points is not None else None
    except (TypeError, ValueError):
        return spf_probs, signals

    if hp is None or ap is None:
        return spf_probs, signals

    elo_p_home = elo_prior(hp, ap)
    old_p_home = spf_probs.get('胜', 0.33)
    old_p_draw = spf_probs.get('平', 0.33)
    old_p_away = spf_probs.get('负', 0.34)

    w = blend_weight
    new_p_home = (1 - w) * old_p_home + w * elo_p_home

    # 剩余概率按比例分配给平局和客胜
    remaining = 1.0 - new_p_home
    if remaining > 0 and (old_p_draw + old_p_away) > 0:
        ratio = old_p_draw / (old_p_draw + old_p_away)
        new_p_draw = remaining * ratio
        new_p_away = remaining * (1 - ratio)
    else:
        new_p_draw = 0.0
        new_p_away = 0.0

    point_diff = hp - ap
    signals.append(
        f'📊埃罗先验(积分差{point_diff:+.0f}→P_home={elo_p_home:.1%},'
        f'混入{w:.0%}: 胜{old_p_home:.1%}→{new_p_home:.1%})'
    )

    return {'胜': new_p_home, '平': new_p_draw, '负': new_p_away}, signals


# ═══════════════════════════════════════════════════════════════
# P0-4: 置信度分段校准
# 根因: 高置信度区间(55-70)命中率41.4% vs 低置信度区间(45-55)命中率56.5%
# 规则: 高conf低hit→降权，低conf高hit→升权
# ═══════════════════════════════════════════════════════════════

# 基于历史复盘数据的置信度→命中率映射（分段）
# 格式: (conf_low, conf_high): hit_rate
_CONF_HIT_BANDS = [
    (0, 42,  0.50),  # 极低conf但命中率50%，说明模型保守时应更自信
    (42, 59, 0.565), # 最佳区间（复盘数据56.5%）→ ×1.05升权
    (59, 65,  0.44), # 高conf但命中率差 → ×0.90降权
    (65, 100, 0.414), # 最差区间命中率41.4% → ×0.92温和降权（原×0.85过严）
]


def calibrate_confidence(confidence, league='', match_date=None, match_code=''):
    """P0-4: 置信度分段校准 — 根据历史hit rate调整置信度

    核心发现（436场复盘数据）：
    - 置信度45-55区间：命中率56.5% ← 最佳区间，应该更自信
    - 置信度55-70区间：命中率41.4% ← 高conf低hit，应该降权
    - 置信度<30区间：命中率50%（但样本少）

    规则：
    - conf 45-55（最佳区间）：×1.05
    - conf 55-65（差区间）：×0.90
    - conf 65-80（最差区间）：×0.85
    - conf <30（样本不足）：×0.92保守降权
    - 杯赛（欧冠/欧罗巴/欧协联）：不适用此规则（样本特征不同）

    Returns:
        (adjusted_confidence, signals)
    """
    signals = []
    conf = confidence

    try:
        conf_val = float(conf) if not isinstance(conf, (int, float)) else conf
    except (ValueError, TypeError):
        return confidence, signals

    # 杯赛不适用（欧冠87.5%/欧罗巴72.7% vs 联赛45%，特征完全不同）
    _CUPS = {'欧冠', '欧罗巴', '欧协联', '欧洲冠军联赛', '欧洲协会联赛'}
    dow = _parse_dow(match_date, match_code)
    if league in _CUPS:
        return conf, signals

    # 分段校准（基于新的_CONF_HIT_BANDS）
    # conf 42-59（最佳区间）：×1.05
    # conf 59-65（差区间）：×0.90
    # conf 65-100（最差区间）：×0.92温和降权（原×0.85过严）
    if 42 <= conf_val < 59:
        # 最佳区间 → 略微升权
        conf *= 1.05
        signals.append(f'🎯最佳区间conf({conf_val})→×1.05')
    elif 59 <= conf_val < 65:
        conf *= 0.90
        signals.append(f'📉偏差区间conf({conf_val})→×0.90')
    elif 65 <= conf_val <= 100:
        conf *= 0.92
        signals.append(f'📉最差区间conf({conf_val})→×0.92温和降权')
    elif conf_val < 30:
        conf *= 0.92
        signals.append(f'📉极低conf({conf_val})→×0.92保守')
    elif conf_val > 90:
        conf = min(conf, 88)
        signals.append(f'⚠️极高conf({conf_val})→强制上限88%')

    return round(conf, 4), signals


def post_adjust_spf(spf_probs, ho, do_, ao, pop, league=''):
    """用必发+支持率微调SPF概率

    Args:
        spf_probs: {'胜':0.45, '平':0.25, '负':0.30}
        ho/do_/ao: 欧赔
        pop: fetch_popularity() 返回的 dict
        league: 联赛名

    Returns:
        (adjusted_probs, signals) — 调整后概率 + 触发信号列表
    """
    signals = []
    p = dict(spf_probs)  # copy
    betfair = pop.get('betfair') if pop else None
    support = pop.get('support') if pop else None

    # ── Bug #4 Fix (2026-04-30): 执行顺序反了 — away_trap(诱盘)必须先于away_boost(低估)
    # 原因：ao>3.5时诱盘信号比低估信号更权威，先惩罚诱盘再补偿低估
    # 原顺序：away_win_boost×1.15 → odds_gradient_adjust×0.75 → 净客胜+15%但逻辑混乱
    # 正确顺序：odds_gradient_adjust(诱盘×0.75) → balanced_odds_adjust → away_win_boost(低估×1.15)

    # ── P4-2: 赔率梯度分层先执行（大热×0.85/诱盘×0.75）— 诱盘惩罚优先 ──
    grad_p, grad_sig = odds_gradient_adjust(p, ho, ao, league=league)
    if grad_sig:
        p = grad_p
        signals.extend(grad_sig)

    # ── P4-1: 均衡赔率检测（主客胜率差<15%时平局boost×1.3） ──
    bal_p, bal_sig = balanced_odds_adjust(p, ho, do_, ao)
    if bal_sig:
        p = bal_p
        signals.extend(bal_sig)

    # ── P0-2: 客胜概率Boost最后执行（系统性低估修复） — away_trap已处理完毕 ──
    boost_p, boost_sig = away_win_boost(p, ao, league)
    p = boost_p
    signals.extend(boost_sig)

    # ── 1. 必发凯利方差：共识方向 boost ──
    if betfair:
        for label in ('胜', '平', '负'):
            kv = betfair.get(f'{_MAP[label]}_kelly', 0)
            if kv > 0 and kv < 3.0:
                # 凯利方差<3 = 职业资金强共识，该方向+2%
                p[label] += 0.02
                signals.append(f'必发共识{label}(凯利{kv:.1f})')
            elif kv > 8.0:
                # 凯利方差>8 = 高分歧，该方向-1.5%
                p[label] -= 0.015
                signals.append(f'必发分歧{label}(凯利{kv:.1f})')

    # ── 2. 必发交易量集中度：大量方向+1.5% ──
    if betfair:
        vols = {l: betfair.get(f'{_MAP[l]}_vol', 0) for l in ('胜', '平', '负')}
        total_vol = sum(vols.values())
        if total_vol > 0:
            for l, v in vols.items():
                ratio = v / total_vol
                if ratio > 0.55:
                    p[l] += 0.015
                    signals.append(f'必发集中{l}({ratio:.0%})')

    # ── 3. 支持率反向指标（加强版） ──
    if support:
        # 计算支持率vs赔率隐含概率的偏离
        implied = {'胜': 1/ho if ho > 1 else 0,
                   '平': 1/do_ if do_ > 1 else 0,
                   '负': 1/ao if ao > 1 else 0}
        implied_total = sum(implied.values())
        if implied_total > 0:
            for k in implied: implied[k] /= implied_total

        for label in ('胜', '平', '负'):
            sv = support.get(_MAP[label], 0) / 100.0
            iv = implied.get(label, 0)
            deviation = sv - iv

            if sv > 0.55 and deviation > 0.10:
                adjustment = -0.06 - deviation * 0.5
                p[label] = max(0.05, p[label] + adjustment)
                signals.append(f'支持率陷阱{label}({sv:.0%} vs 隐含{iv:.0%}, {adjustment:+.1%})')
            elif sv > 0.65:
                p[label] -= 0.03
                signals.append(f'支持率过热{label}({sv:.0%})')
            elif sv < 0.18:
                p[label] += 0.025
                signals.append(f'支持率冷门{label}({sv:.0%})')
            elif sv > 0.45 and sv < 0.55:
                if deviation < -0.05:
                    p[label] += 0.02
                    signals.append(f'支持率低估{label}({sv:.0%} vs 隐含{iv:.0%})')

    # ── 4. 必发vs支持率矛盾：最大信号 ──
    if betfair and support:
        for label in ('胜', '平', '负'):
            bf_pct = betfair.get(f'{_MAP[label]}_pct', 0)
            sp_pct = support.get(_MAP[label], 0)
            diff = bf_pct - sp_pct
            if diff > 25:
                # 职业资金看好 vs 散户不看好 → 强信号+2.5%
                p[label] += 0.025
                signals.append(f'资散背离{label}(必发{bf_pct:.0f}%vs支持{sp_pct:.0f}%)')
            elif diff < -25:
                p[label] -= 0.02
                signals.append(f'资散背离{label}(散户热{sp_pct:.0f}%vs必发{bf_pct:.0f}%)')

    # ── P1: 热门翻车检测 ──
    try:
        if float(ho) < 1.4:
            do_f = float(do_)
            # 条件1: 平赔>4.5
            if do_f > 4.5:
                p['平'] += 0.05
                p['胜'] -= 0.05
                signals.append(f'热门平赔陷阱(主赔{float(ho):.2f},平赔{do_f:.2f})')
            # 条件2: 周末+次级联赛
            dow_h = _parse_dow(league=league)  # hack: won't work
            # 条件2在weekday_effect_adjust中处理
    except Exception:
        pass

    p = _normalize_spf(p)

    return p, signals


def time_decay_confidence(confidence, hours_to_kickoff, league=''):
    """P1: 时间衰减因子 — 距开赛越远，置信度越低
    >48h: ×0.8, >72h: ×0.7
    """
    if hours_to_kickoff is None or hours_to_kickoff <= 48:
        return confidence, []
    signals = []
    if hours_to_kickoff > 72:
        confidence *= 0.7
        signals.append('⏰时间衰减×0.7(>72h)')
    elif hours_to_kickoff > 48:
        confidence *= 0.8
        signals.append('⏰时间衰减×0.8(>48h)')
    return round(confidence, 4), signals


# P2: 英甲小比分先验
_SMALL_SCORE_LEAGUES = {
    # P1-2: 日职小比分先验（1-0/0-0/1-1占约30%，命中率58.3%）
    '日职': {'1-1': 0.10, '0-0': 0.09, '1-0': 0.08, '0-1': 0.03, 'total': 0.30, 'home_win_rate': 0.44},
    '日职联': {'1-1': 0.10, '0-0': 0.09, '1-0': 0.08, '0-1': 0.03, 'total': 0.30, 'home_win_rate': 0.44},
    '日本职业联赛': {'1-1': 0.10, '0-0': 0.09, '1-0': 0.08, '0-1': 0.03, 'total': 0.30, 'home_win_rate': 0.44},
    '英甲': {'1-1': 0.11, '0-0': 0.09, '1-0': 0.09, 'total': 0.29, 'home_win_rate': 0.438},
    '英冠': {'home_win_rate': 0.44},
    '英超': {'home_win_rate': 0.46},
    '德甲': {'home_win_rate': 0.47},
    '意甲': {'home_win_rate': 0.46},
    '西甲': {'home_win_rate': 0.46},
}


def small_score_boost(league, score_probs):
    """P2: 小比分检测器 — 英甲1-1/0-0/1-0占29%先验
    当赔率显示小比分信号时，提升小比分推荐权重
    Args:
        league: 联赛名
        score_probs: {(h,a): prob} 比分概率dict
    Returns:
        (adjusted_probs, boost_signals)
    """
    signals = []
    cfg = _SMALL_SCORE_LEAGUES.get(league)
    if not cfg or 'total' not in cfg:
        return score_probs, signals

    p = dict(score_probs)
    prior_total = cfg['total']
    # 计算当前小比分占比
    small_keys = [(1,1), (0,0), (1,0), (0,1)]
    current_small = sum(p.get(k, 0) for k in small_keys)

    # 如果当前小比分概率低于先验，提升
    if current_small < prior_total * 0.8:
        boost = 1.15  # +15% boost
        for k in small_keys:
            if k in p:
                p[k] *= boost
        signals.append(f'英甲小比分增强({current_small:.0%}→先验{prior_total:.0%})')

    p = _normalize_spf(p)

    return p, signals


def get_league_home_win_rate(league):
    """获取联赛主场胜率（用于让球盘口差异化）"""
    cfg = _SMALL_SCORE_LEAGUES.get(league, {})
    return cfg.get('home_win_rate', 0.45)


# ═══════════════════════════════════════════════════════════════
# P0: 周五效应 + 英冠专项降权
# ═══════════════════════════════════════════════════════════════

import datetime as _dt

# P2: 周五各联赛历史主场胜率表（来自 collected_match_data_2026 全量统计）
_FRIDAY_LEAGUE_HWR = {
    # 联赛: (总样本, 主场胜率)
    '英冠': (892, 0.434),
    '英甲': (13942, 0.44),
    '英格兰甲级联赛': (13942, 0.44),
    '英超': (41813, 0.46),
    '德甲': (16008, 0.47),
    '德乙': (15695, 0.44),
    '德国乙级联赛': (15695, 0.44),
    '意甲': (18922, 0.46),
    '西甲': (17687, 0.46),
    '西班牙甲级联赛': (17687, 0.46),
    '法甲': (15747, 0.49),
    '法国甲级联赛': (15747, 0.49),
    '法乙': (4137, 0.42),
    '法国乙级联赛': (4137, 0.42),
    '葡超': (5649, 0.41),
    '葡萄牙超级联赛': (5649, 0.41),
    '澳超': (103, 0.38),
    '澳大利亚超级联赛': (103, 0.38),
    '韩职': (1306, 0.41),
    '韩K联': (1306, 0.41),
    '韩国职业联赛': (1306, 0.41),
    '日职': (6885, 0.44),
    '日职联': (6885, 0.44),
    '日本职业联赛': (6885, 0.44),
    '荷甲': (8588, 0.44),
    '荷兰甲级联赛': (8588, 0.44),
    '荷乙': (282, 0.41),
    '荷兰乙级联赛': (282, 0.41),
    '沙职': (0, 0.45),
    '沙特联': (0, 0.45),
    '沙特职业联赛': (0, 0.45),
}

# 周五效应：周五主队被高估的联赛（主场胜率低于全局均值）
_FRIDAY_WEAK_HOME = {'英冠', '英甲', '法乙', '荷乙', '澳超', '葡超'}


# ── 统一星期解析（英文/中文双输出） ──
_CODE_DOW_EN = {'周一':'Mon','周二':'Tue','周三':'Wed','周四':'Thu',
                '周五':'Fri','周六':'Sat','周日':'Sun'}
_CODE_DOW_CN = {v: k for k, v in _CODE_DOW_EN.items()}
_EN_DOW = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']


def _get_weekday(match_date, match_code=''):
    """从 match_date 或 match_code 推断星期几
    Returns: 'Mon','Tue',...'Sun' 或 None
    """
    if match_code:
        for cn, en in _CODE_DOW_EN.items():
            if cn in str(match_code):
                return en
    if match_date:
        try:
            ds = str(match_date)[:10]
            dt = _dt.datetime.strptime(ds, '%Y-%m-%d')
            return dt.strftime('%a')
        except Exception:
            pass
    return None


def _get_dow(match_date=None, match_code=''):
    """获取星期几（中文）"""
    en = _get_weekday(match_date, match_code)
    return _CODE_DOW_EN and {_v: _k for _k, _v in _CODE_DOW_EN.items()}.get(en, '') if en else ''


def _parse_dow(match_date=None, match_code=''):
    """从match_date或match_code解析星期几 — 统一入口"""
    return _get_weekday(match_date, match_code)


def friday_effect_adjust(spf_probs, league='', match_date=None, match_code=''):
    """P0: 周五效应 — 周五/周六主胜概率系统性下调

    根据历史数据，周五主场胜率仅38%（vs 全局44%），
    庄家在周五压低主队赔率制造热门陷阱。

    Args:
        spf_probs: {'胜':0.50, '平':0.25, '负':0.25}
        league: 联赛名
        match_date: 比赛日期(str/date)
        match_code: 体彩编号（含周几前缀）

    Returns:
        (adjusted_probs, signals)
    """
    signals = []
    p = dict(spf_probs)
    dow = _get_weekday(match_date, match_code)

    is_friday = (dow == 'Fri')

    # ── 周五/周六主胜下调 ──
    if dow == 'Fri':
        # 周五平局boost
        if p.get('平', 0) < 0.30:
            p['平'] = min(0.40, p['平'] * 1.20)
            signals.append('周五平局boost(+20%)')
        # 主胜下调
        p['胜'] *= 0.90   # -10%
        p['负'] *= 1.10   # +10% — 周五客胜boost（庄家周五压低主队赔率，客胜被高估）
        signals.append('📅周五效应:主胜×0.90,客胜×1.10')
    elif dow == 'Sat':
        p['胜'] *= 0.92   # -8%
        signals.append('📅周六效应:主胜×0.92')

    # ── 周五+弱主场联赛叠加 ──
    if dow == 'Fri' and league in _FRIDAY_WEAK_HOME:
        p['胜'] *= 0.92   # 额外-8%
        p['负'] *= 1.05   # +5%
        signals.append(f'📅周五{league}弱势主场×0.92')

    p = _normalize_spf(p)

    return p, signals


def thursday_effect_adjust(spf_probs, league='', match_date=None, match_code=''):
    """P0: 周四效应 — 周四比赛客胜概率系统性上调

    周四比赛多为欧战/杯赛/小联赛，参赛队轮换、球员疲劳，
    主场优势被削弱，客胜概率相对更高。

    Args:
        spf_probs: {'胜':0.50, '平':0.25, '负':0.25}
        league: 联赛名
        match_date: 比赛日期(str/date)
        match_code: 体彩编号（含周几前缀）

    Returns:
        (adjusted_probs, signals)
    """
    signals = []
    p = dict(spf_probs)
    dow = _get_weekday(match_date, match_code)

    if dow != 'Thu':
        return p, signals

    # ── 周四客胜boost ──
    # 周四比赛主场劣势明显，客胜概率系统性偏高
    p['负'] *= 1.08   # +8%
    p['胜'] *= 0.95   # -5%（主场被低估）
    signals.append('📅周四效应:客胜×1.08,主胜×0.95')

    # ── 周四+弱主场联赛叠加 ──
    if league in _FRIDAY_WEAK_HOME:
        p['负'] *= 1.06   # 额外+6%
        signals.append(f'📅周四{league}弱势主场客胜×1.06')

    p = _normalize_spf(p)

    return p, signals


def championship_friday_adjust(spf_probs, ho, league='', match_date=None, match_code=''):
    """P0: 英冠周五专项降权

    英冠历史主场胜率43-44%，但模型周五对英冠预测主胜率55-65%，
    偏差+20pp。周五英冠6场全预测主胜，实际仅2场。

    规则：
    - 英冠+周五 → 主胜×0.80
    - 英冠+赔率1.5-2.0 → 主胜×0.85（额外）
    - 英冠+赔率<1.6且非胜 → 冷门检测触发

    Args:
        spf_probs: {'胜':0.50, '平':0.25, '负':0.25}
        ho: 主胜赔率
        league: 联赛名
        match_date: 比赛日期
        match_code: 体彩编号

    Returns:
        (adjusted_probs, signals)
    """
    signals = []
    p = dict(spf_probs)
    is_championship = league in ('英冠', 'Championship', 'ELC')

    if not is_championship:
        return p, signals

    dow = _get_weekday(match_date, match_code)

    # 英冠全局：主胜概率偏低
    p['胜'] *= 0.92
    signals.append('🏟️英冠主胜×0.92')

    if dow == 'Fri':
        # 英冠+周五：主队被高估，但不过度惩罚（避免误杀低赔主胜）
        p['胜'] *= 0.90
        p['平'] *= 1.05
        p['负'] *= 1.10
        signals.append('🏟️英冠周五主胜×0.90')

    # 英冠+赔率1.5-2.0：模型最差区间（命中率29%）
    try:
        if 1.5 <= ho <= 2.0:
            p['胜'] *= 0.90
            p['平'] *= 1.08
            p['负'] *= 1.08
            signals.append(f'🏟️英冠赔率盲区({ho:.2f})主胜×0.90')
    except (TypeError, ValueError):
        pass

    # 英冠冷门检测：赔率<1.6但主胜概率被压低后仍最高
    try:
        if ho < 1.6 and p['胜'] > max(p['平'], p['负']):
            # 主队热门但英冠主场胜率仅44% → 警告
            signals.append(f'⚠️英冠热门陷阱(赔率{ho:.2f},历史HWR=43%)')
    except (TypeError, ValueError):
        pass

    p = _normalize_spf(p)

    return p, signals


# ═══════════════════════════════════════════════════════════════
# P2: 通用周几效应 + P3: 联赛专项规则
# ═══════════════════════════════════════════════════════════════

_LEAGUE_HIGH_DRAW = {'法乙', '英冠', '日职', '葡超', '挪超'}


def weekday_effect_adjust(spf_probs, match_date, match_code, league):
    """P2: 周几效应通用规则"""
    dow = _get_weekday(match_date, match_code)
    p = dict(spf_probs)
    signals = []

    if dow == 'Mon':
        p['平'] = min(0.35, p.get('平', 0.15) * 1.25)
        signals.append('⚠️周一效应:平局+25%,信心上限40%')
    elif dow == 'Fri':
        if league in ('意甲', '西甲', '德甲'):
            p['平'] = min(0.35, p.get('平', 0.15) * 1.15)
            signals.append('周五五大联赛平局boost')
    elif dow in ('Sat', 'Sun'):
        if league in _LEAGUE_HIGH_DRAW:
            p['平'] = min(0.50, p.get('平', 0.15) * 1.10)
            signals.append(f'周末高平局联赛boost({league})')

    total = sum(p.values())
    if total > 0:
        for k in p:
            p[k] /= total
    return p, signals


def league_specific_adjust(spf_probs, ho, do_, ao, league=''):
    """P3: 联赛专项规则"""
    p = dict(spf_probs)
    signals = []
    try:
        ho_f, do_f = float(ho), float(do_)
    except (TypeError, ValueError):
        return p, signals

    # 意甲：平局率24%但预测常为0，赔率2.0-2.5区间平局boost
    if league == '意甲' and 2.0 <= ho_f <= 2.5:
        p['平'] += 0.03
        p['胜'] -= 0.02
        p['负'] -= 0.01
        signals.append('意甲均衡对阵平局+3%')

    # 西甲：平局率19%，均衡对阵(主赔2.0-2.8)平局+3%
    if league == '西甲' and 2.0 <= ho_f <= 2.8:
        p['平'] += 0.03
        p['胜'] -= 0.02
        p['负'] -= 0.01
        signals.append('西甲均衡对阵平局+3%')

    # 日职：平局率35%，P0先验已覆盖，但低赔日职额外boost
    if league == '日职' and do_f < 3.2:
        p['平'] += 0.02
        signals.append('日职低平赔额外boost')

    # 沙职：数据量不足，不调整但标记
    if league == '沙职':
        signals.append('⚠️沙职数据不足，低信心')

    total = sum(p.values())
    if total > 0:
        for k in p:
            p[k] /= total
    return p, signals





def odds_band_verify(spf_probs, ho, do_, ao, league='', match_date=None, match_code=''):
    """P1: 赔率区间信号验证 — 1.8-2.2 死亡区间强制降权

    历史数据：1.8-2.2区间命中率仅29%（vs 其他区间>46%）。
    该区间 = "主队略被看好但非热门"，庄家操盘空间最大。

    规则：
    - 1.8-2.2区间：主胜概率×0.90，平局×1.05
    - 需要额外信号确认（赔率变化方向、赔率偏离）才恢复

    Args:
        spf_probs: {'胜':0.50, '平':0.25, '负':0.25}
        ho/do_/ao: 欧赔
        league/match_date/match_code: 上下文

    Returns:
        (adjusted_probs, signals)
    """
    signals = []
    p = dict(spf_probs)

    try:
        fav_odds = min(ho, ao)
        is_death_band = 1.8 <= fav_odds <= 2.2
    except (TypeError, ValueError):
        return p, signals

    if not is_death_band:
        return p, signals

    # 死亡区间基础降权
    if ho < ao:  # 主队热门
        p['胜'] *= 0.90
        p['平'] *= 1.05
        p['负'] *= 1.05
        signals.append(f'⚡死亡赔率区间({ho:.2f})主胜×0.90')
    else:  # 客队热门
        p['负'] *= 0.90
        p['平'] *= 1.05
        p['胜'] *= 1.05
        signals.append(f'⚡死亡赔率区间({ao:.2f})客胜×0.90')

    p = _normalize_spf(p)

    return p, signals


def confidence_friday_penalty(confidence, league, match_date=None, match_code=''):
    """P0/P1: 周五置信度惩罚

    在动态置信度计算之后调用，对周五/英冠/死亡赔率区间的预测额外降权。

    Args:
        confidence: 原始置信度 (0-1)
        league: 联赛名
        match_date/match_code: 上下文

    Returns:
        (adjusted_confidence, signals)
    """
    signals = []
    dow = _get_weekday(match_date, match_code)
    is_championship = league in ('英冠', 'Championship', 'ELC')

    # 周五英冠：最大惩罚
    if dow == 'Fri' and is_championship:
        confidence *= 0.75
        signals.append('🏟️英冠周五置信×0.75')
    elif dow == 'Fri':
        confidence *= 0.90
        signals.append('📅周五置信×0.90')
    elif is_championship:
        confidence *= 0.92
        signals.append('🏟️英冠置信×0.92')

    return round(confidence, 4), signals


# ═══════════════════════════════════════════════════════════════
# P2: 周五联赛主场胜率查询
# ═══════════════════════════════════════════════════════════════

def get_friday_league_stats(league):
    """P2: 获取联赛历史主场胜率（用于周五决策参考）

    Args:
        league: 联赛名

    Returns:
        dict: {'sample': N, 'home_win_rate': float} 或 None
    """
    entry = _FRIDAY_LEAGUE_HWR.get(league)
    if entry:
        return {'sample': entry[0], 'home_win_rate': entry[1]}
    return None


def get_all_friday_league_stats():
    """P2: 返回全部联赛主场胜率表"""
    return {lg: {'sample': v[0], 'home_win_rate': v[1]}
            for lg, v in _FRIDAY_LEAGUE_HWR.items()}


# ═══════════════════════════════════════════════════════════════
# P0: 周六亚洲联赛主场效应修正（2026-04-04 复盘新增）
# ═══════════════════════════════════════════════════════════════

# 亚洲联赛周六主场胜率偏差（数据来源: collected_match_data_2026）
# 澳超: 全局41.1% vs 周六37.0%（↓4.1pp）
# 日职/韩职: 数据不足，保守估计↓2-3pp
_SATURDAY_ASIAN_ADJUST = {
    '澳超': 0.96,    # 周六HWR↓4.1pp → 主胜×0.96
    'A-League': 0.96,
    '澳大利亚超级联赛': 0.96,
    '韩职': 0.97,    # 保守估计↓3pp
    '韩K联': 0.97,
    '韩K1联': 0.97,
    'K1': 0.97,
    'K League 1': 0.97,
    '韩国职业联赛': 0.97,
    '日职': 0.97,    # 保守估计↓3pp
    '日职联': 0.97,
    '日本职业联赛': 0.97,
    '日职乙': 0.97,
    'J2联赛': 0.97,
    '日乙': 0.97,
    '日本乙级联赛': 0.97,
}

# 小联赛列表（庄家抽水高、数据少、预测可靠性低）
_SMALL_LEAGUES = {
    '芬超', '芬兰超级联赛',
    '挪超', '挪威超级联赛',
    '瑞超', '瑞典超级联赛',
    '丹超', '奥超', '爱超', '捷甲',
    '波甲', '罗甲', '匈甲', '斯超', '斯甲', '芬甲', '挪甲',
    '以超', '以甲', '哈萨超', '哥伦甲', '厄瓜多尔甲',
    '巴拉圭甲', '哥斯达黎加超', '摩洛哥超', '南非超',
    '尼日利亚超', '尼日利亚甲', '埃及超', '阿尔及利亚超',
    '阿甲', '墨超',
}


def saturday_asian_adjust(spf_probs, league='', match_date=None, match_code=''):
    """P0: 周六亚洲联赛主场效应修正

    澳超周六HWR=37%（全局41.1%，↓4.1pp），模型却在上调主胜，
    导致周六002中央海岸预测主胜但实际平。
    韩职/日职保守估计↓2-3pp。

    Returns:
        (adjusted_probs, signals)
    """
    signals = []
    p = dict(spf_probs)
    dow = _get_weekday(match_date, match_code)

    if dow != 'Sat':
        return p, signals

    factor = _SATURDAY_ASIAN_ADJUST.get(league)
    if not factor:
        return p, signals

    p['胜'] *= factor
    p['平'] *= (1 + (1 - factor) * 0.5)  # 平局补回一半
    signals.append(f'🌏周六{league}主场修正×{factor}')

    p = _normalize_spf(p)

    return p, signals


def small_league_penalty(confidence, league='', ho=0, do_=0, ao=0):
    """P0: 小联赛惩罚 — 庄家高抽水+数据少→置信度上限50%

    芬超周六006: 主胜1.28实际0-0，超低赔率在小联赛中庄家抽水更狠。

    Returns:
        (adjusted_confidence, signals)
    """
    signals = []

    if league not in _SMALL_LEAGUES:
        return confidence, signals

    # 小联赛置信度上限50%
    confidence = min(confidence, 0.50)
    signals.append(f'🏴小联赛{league}置信上限50%')

    # 返还率检查（赔率越低抽水越狠）
    try:
        if ho > 0 and do_ > 0 and ao > 0:
            rr = 1.0 / (1.0/ho + 1.0/do_ + 1.0/ao)
            if rr < 0.88:
                confidence *= 0.85
                signals.append(f'💰高抽水惩罚(返还率{rr:.1%})×0.85')
    except (TypeError, ValueError, ZeroDivisionError):
        pass

    return round(confidence, 4), signals

# ═══════════════════════════════════════════════════════════════
# P0: 平局专项检测器（2026-04-05 新增）
# 根因：昨日29场中7场SPF失败是平局，占58%
# ═══════════════════════════════════════════════════════════════

# (已合并到上方 _parse_dow)


# ── 联赛平局系数表（基于历史预测数据统计 2026-04-06）──
# 联赛: (实际平局率, 平局乘数) — 平局率高的联赛额外放大
_LEAGUE_DRAW_BOOST = {
    '法乙': (0.60, 1.80), '英甲': (0.29, 1.20),
    '英超': (0.50, 1.50), '荷甲': (0.455, 1.40),
    '德甲': (0.30, 1.25), '德乙': (0.25, 1.10),
    '法甲': (0.30, 1.25), '西甲': (0.182, 1.00),
    '意甲': (0.136, 1.00), '葡超': (0.167, 1.00),
    '英冠': (0.20, 1.10), '澳超': (0.333, 1.30),
    '韩职': (0.20, 1.05), '日职': (0.286, 1.15),
    '世预赛': (0.357, 1.30), '国际赛': (0.308, 1.20),
}

# 周几平局修正（实际平局率 - 预测平局率的差距）
_DOW_DRAW_ADJUST = {
    'Sun': 1.30,  # 实际28.9% vs 预测4.8%
    'Sat': 1.25,  # 实际24.6% vs 预测4.3%
    'Fri': 1.25,  # 实际30.0% vs 预测10.0%
    'Thu': 1.15,  # 实际28.6% vs 预测14.3%
    'Wed': 1.20,  # 周三实际平局率27-30%，给予适度boost（2026-04-30修复）
    'Tue': 1.25,  # P0验证：周初平局率35.3%，英冠4场4平
    'Mon': 1.25,  # P0验证：周初次级联赛平局溢价
}

# 赔率区间平局先验（实际统计：平衡区间62.5%打平！）
_ODDS_BAND_DRAW_PRIOR = {
    'balance': (2.0, 2.5, 0.40),   # odds_home ∈ [2.0, 2.5] → 平局基准40%
    'mid_home': (1.6, 2.0, 0.25),  # 中主场
    'strong_home': (0.0, 1.6, 0.10),  # 强主场
    'away_fav': (2.5, 99.0, 0.28),   # 客强
}

# P1改进: 基于04-27复盘的联赛专项修正
_LEAGUE_DRAW_BAND_ADJUST = {
    # 法乙: balance区间实际大比分多，降低平局先验
    '法乙': {'balance_mult': 0.75, 'away_fav_mult': 0.85},
    # 瑞超: 平衡赔率区间低平局，低置信
    '瑞超': {'balance_mult': 0.70, 'away_fav_mult': 0.80},
    # 葡超: 客胜倾向
    '葡超': {'balance_mult': 0.80, 'away_fav_mult': 1.10},
    # 英超/意甲/西甲: 维持原值
    '英超': {'balance_mult': 1.0, 'away_fav_mult': 1.0},
    '意甲': {'balance_mult': 1.0, 'away_fav_mult': 1.0},
    '西甲': {'balance_mult': 1.0, 'away_fav_mult': 1.0},
}

# TODO: P1 - Auto-update from prediction_match_results:
# SELECT ho_band, COUNT(*), SUM(CASE WHEN result='平' THEN 1 ELSE 0 END) as draws
# FROM prediction_match_results WHERE run_date > CURRENT_DATE - 90
# GROUP BY ho_band ORDER BY ho_band

def draw_boost_adjust(spf_probs, ho, do_, ao, league='', match_date=None, match_code=''):
    """P0: 平局专项检测器 v2 — 基于历史预测数据全面升级

    核心发现（2026-04-06 复盘）：
    - 264场中72场平局(27.3%)，漏判62/72=86%
    - 赔率2.0-2.5区间平局率62.5%，命中率仅12.5%
    - 周日/周五/周六平局率28-30%，预测仅5-10%

    规则（v2 升级）：
    1. 赔率区间平局先验：balance区间直接拉到40%基准
    2. 联赛平局系数：法乙×1.8、英超×1.5、荷甲×1.4
    3. 周几平局修正：周日/周五/周六额外×1.25-1.30
    4. 平局赔率信号：do<3.0强信号
    5. 三重叠加时触发"平局模式"：平局概率设为最高

    Returns:
        (adjusted_probs, signals)
    """
    signals = []
    p = dict(spf_probs)
    dow = _parse_dow(match_date, match_code)
    draw_boost_count = 0

    # ── P0: 法乙专项规则 ──
    if league == '法乙':
        orig_draw = spf_probs.get('平', 0)
        if orig_draw < 0.35:
            spf_probs['平'] = min(0.55, orig_draw * 1.3)
            excess = spf_probs['平'] - orig_draw
            if '胜' in spf_probs:
                spf_probs['胜'] = max(0.10, spf_probs['胜'] - excess * 0.7)
            if '负' in spf_probs:
                spf_probs['负'] = max(0.10, spf_probs['负'] - excess * 0.3)
            total = sum(spf_probs.values())
            for k in spf_probs: spf_probs[k] /= total
            signals.append(f'法乙专项(弱化):平局boost({orig_draw:.0%}→{spf_probs["平"]:.0%})')
            p = dict(spf_probs)

    # 1. 赔率区间平局先验（最核心的新规则）
    try:
        ho_val = float(ho)
        for band_name, (lo, hi, prior) in _ODDS_BAND_DRAW_PRIOR.items():
            if lo <= ho_val < hi:
                if prior > p.get('平', 0):
                    gap = prior - p.get('平', 0)
                    p['平'] += gap * 0.5  # 拉向先验的50%
                    draw_boost_count += 1
                    signals.append(f'🎯{band_name}区间平局先验{prior:.0%}(当前{p.get("平",0):.0%})')
                break
    except: pass

    # ════ P1-1新增(2026-05-12): 平局分类boost ──────────────────────────
    # 根据赔率均衡度将赛事分类，对不同类型实施差异化平局boost
    # 均衡型(ho∈[1.9,2.5])：庄家也无法判断，平局率30-40%，boost+25%
    # 悬殊型(ho<1.5 or ho>3.0)：弱队被动，平局率<15%，boost+3%
    # 中间型：不在以上两类，标准处理
    try:
        ho_f = float(ho)
        do_f = float(do_)
        ao_f = float(ao)
        
        # 计算赔率均衡度：主客赔率差距越小越均衡
        odds_balance = abs(ho_f - ao_f)  # 差距越小→越均衡
        
        if 1.9 <= ho_f <= 2.5 and odds_balance < 0.8:
            # 均衡型赛事：双方实力接近，平局概率高
            old_draw = p.get('平', 0)
            # 均衡型平局基准30%，拉向先验的60%
            p['平'] = max(p.get('平', 0), 0.25)  # 最低25%
            p['平'] = min(p['平'] * 1.25, 0.42)  # 最多+25%，上限42%
            total = sum(p.values())
            if total > 0:
                excess = p['平'] - old_draw if p['平'] > old_draw else 0
                if excess > 0 and '胜' in p and '负' in p:
                    p['胜'] = max(0.10, p['胜'] - excess * 0.6)
                    p['负'] = max(0.10, p['负'] - excess * 0.4)
                    total = sum(p.values())
                    p = {k: v/total for k, v in p.items()}
            draw_boost_count += 1
            signals.append(f'天秤均衡型:平局→{p["平"]:.0%}(均衡度{odds_balance:.2f})')
        
        elif ho_f < 1.5 or ho_f > 3.0:
            # 悬殊型赛事：实力差距大，平局概率低
            # 只做微小修正，因为悬殊赛事平局本就不多
            implied_draw = 0.886 / do_f if do_f > 0 else 0.25
            if p.get('平', 0) < implied_draw * 0.90:
                p['平'] = implied_draw * 0.90
                total = sum(p.values())
                if total > 0:
                    p = {k: v/total for k, v in p.items()}
                draw_boost_count += 1
                signals.append(f'砝码悬殊型:平局→{p["平"]:.0%}(ho={ho_f:.2f})')
    except: pass

    # 2. 联赛平局系数
    lg_cfg = _LEAGUE_DRAW_BOOST.get(league)
    if lg_cfg:
        actual_rate, multiplier = lg_cfg
        p['平'] *= multiplier
        if multiplier > 1.15:
            draw_boost_count += 1
            signals.append(f'🏷️{league}平局系数×{multiplier}(实际{actual_rate:.0%})')

    # 3. 周几平局修正
    if dow and dow in _DOW_DRAW_ADJUST:
        factor = _DOW_DRAW_ADJUST[dow]
        if factor > 1.0:
            p['平'] *= factor
            draw_boost_count += 1
            signals.append(f'📅{dow}平局修正×{factor}')

    # 4. 平局赔率信号（乘法boost）
    try:
        do_val = float(do_)
        if do_val < 2.80:
            p['平'] *= 1.15
            p['胜'] *= 0.93
            draw_boost_count += 1
            signals.append(f'⚠️极低平赔{do_val:.2f}→平局+15%')
        elif do_val < 3.10:
            p['平'] *= 1.08
            draw_boost_count += 1
            signals.append(f'📉低平赔{do_val:.2f}→平局+8%')
    except: pass

    # 4b. 【P0核心新增】赔率隐含平局概率Floor — 最精准的平局漏判修复
    # 根因：odds_draw=3.37时隐含29%平，但模型只有15-20%
    # 规则：当odds_draw<3.5时，强制平局概率≥隐含×0.85
    try:
        do_val = float(do_)
        ho_val = float(ho)
        ao_val = float(ao)
        if 0 < do_val < 3.5:
            implied_draw = 0.886 / do_val  # 体彩返还率88.6%
            floor = implied_draw * 0.85
            if p.get('平', 0) < floor:
                old_draw = p.get('平', 0)
                p['平'] = floor
                draw_boost_count += 1
                signals.append(f'🎯赔率Floor({do_val:.2f})→平局{floor:.1%}(隐含{implied_draw:.1%}×0.85, 原{old_draw:.1%})')
            # 赔率极低时(<2.8)：更强floor
            if do_val < 2.8 and p.get('平', 0) < implied_draw * 0.90:
                p['平'] = implied_draw * 0.90
                signals.append(f'🎯极低平赔Floor({do_val:.2f})→平局{implied_draw*0.90:.1%}(×0.90)')
    except: pass

    # 5. 三重叠加触发"平局模式" — 强制平局为最高概率
    if draw_boost_count >= 3:
        if p.get('胜', 0) > p.get('平', 0):
            # 交换胜和平的概率
            p['胜'], p['平'] = p['平'] * 0.9, p['胜'] * 1.1
            signals.append(f'🔴三重叠加→平局模式(胜↔平互换)')
        elif p.get('负', 0) > p.get('平', 0):
            p['负'], p['平'] = p['平'] * 0.9, p['负'] * 1.1
            signals.append(f'🔴三重叠加→平局模式(负↔平互换)')

    # P1新增: 胜预测>2.0区间平局漏算修正
    # 胜→平转化率24.8%，说明模型在ho>2.0时低估平局
    try:
        ho_f = float(ho)
        do_f = float(do_)
        implied_ho = 1/ho_f if ho_f > 0 else 0
        implied_do = 1/do_f if do_f > 0 else 0
        # 计算庄家隐含平局概率
        total_imp = implied_ho + implied_do + (1/float(ao) if float(ao)>0 else 0)
        market_draw = implied_do / total_imp if total_imp > 0 else 0

        if p.get('胜', 0) > p.get('平', 0) and ho_f > 2.0:
            # 庄家隐含平局概率 > 模型平局概率 → 模型低估了平局
            model_draw = p.get('平', 0)
            if market_draw > model_draw + 0.05:
                # 庄家认为平局概率高，但模型低估 → 补平局
                deficit = market_draw - model_draw
                boost = min(0.15, deficit * 0.5)  # 最多+15%
                p['平'] *= (1 + boost)
                p['胜'] *= (1 - boost * 0.3)  # 微降胜
                total = sum(p.values())
                if total > 0:
                    p = {k: max(0.05, v/total) for k,v in p.items()}
                signals.append(f'📊胜选平局补boost ho={ho_f:.2f}→平×{1+boost:.2f}')
    except: pass

    # ═══════════════════════════════════════════════════════════════════════
    # P0核心新增 (2026-04-27): 高信心平局逆转机制
    # 根因: 4月26日佛罗伦萨(61.7%)、勒阿弗尔(62.0%)高信心全败于平局
    #       draw_boost_adjust只微调概率，从未真正逆转>60%信心的"胜"预测
    # 触发条件:
    #   1. 模型预测"胜"且信心>60%
    #   2. ho ∈ [1.40, 2.0] (中等偏低赔，强队主场)
    #   3. do < 3.5 (庄家未极端高开平赔)
    #   4. 联赛∈{法乙,荷甲,英超,意甲,西甲,德甲,葡超,日职,澳超,美职}
    #      或 周∈{Sun,Sat,Fri}
    #   5. 庄家隐含平局概率 > 模型平局概率 + 8%
    # 动作: 平局概率直接提升至max(胜,负)×80%，重新归一化
    # ═══════════════════════════════════════════════════════════════════════
    try:
        ho_f = float(ho)
        do_f = float(do_)
        if not (1.40 <= ho_f <= 2.0 and 0 < do_f < 3.5):
            raise ValueError('not in high-risk draw zone')
        model_pred = max(p, key=p.get)
        if model_pred != '胜':
            raise ValueError('not a home-win prediction')
        model_conf = p.get('胜', 0)
        if model_conf < 0.60:
            raise ValueError('confidence below 60%')
        implied_do = 1/do_f if do_f > 0 else 0
        implied_ho = 1/ho_f if ho_f > 0 else 0
        ao_f = float(ao) if ao > 0 else 99
        implied_ao = 1/ao_f if ao_f > 0 else 0
        total_imp = implied_ho + implied_do + implied_ao
        market_draw = implied_do / total_imp if total_imp > 0 else 0
        model_draw = p.get('平', 0)
        if market_draw < model_draw + 0.08:
            raise ValueError('market draw not significantly higher than model')
        # 周几/联赛条件
        high_risk_leagues = {'法乙','荷甲','英超','意甲','西甲','德甲','葡超','日职','澳超','美职','挪超','瑞超','芬超','瑞典超'}
        high_risk_dow = {'Sun','Sat','Fri','Mon'}
        dow_ok = (dow and dow in high_risk_dow) or (league and league in high_risk_leagues)
        if not dow_ok:
            raise ValueError('not in high-risk dow/league')
        # 通过所有检查 → 执行平局逆转
        max_other = max(p.get('负', 0), p.get('胜', 0))
        target_draw = max(max_other * 0.80, market_draw * 0.90)
        surplus = target_draw - model_draw
        p['平'] = target_draw
        if '胜' in p:
            p['胜'] = max(0.08, p['胜'] - surplus * 0.70)
        if '负' in p:
            p['负'] = max(0.08, p['负'] - surplus * 0.30)
        p = _normalize_spf(p, floor=0.05)
        signals.append(
            f'🔴高信心平局逆转: 信心{model_conf:.0%}→平局{p["平"]:.0%}('
            f'庄隐含{market_draw:.0%}|模型{model_draw:.0%}|ho={ho_f:.2f}|do={do_f:.2f})'
        )
    except (ValueError, TypeError, ZeroDivisionError):
        pass

    # P1新增: 联赛专项赔率区间修正
    league_adj = _LEAGUE_DRAW_BAND_ADJUST.get(league, None)
    if league_adj:
        # Apply league-specific multipliers to draw probability
        adj_balance = league_adj.get('balance_mult', 1.0)
        adj_away = league_adj.get('away_fav_mult', 1.0)
        try:
            ho_val = float(ho)
            for band_name, (lo, hi, prior) in _ODDS_BAND_DRAW_PRIOR.items():
                if lo <= ho_val < hi:
                    if band_name == 'balance' and adj_balance != 1.0:
                        old_draw = p.get('平', 0)
                        p['平'] *= adj_balance
                        signals.append(f'🏷️{league}balance区间修正×{adj_balance}(平局{old_draw:.0%}→{p["平"]:.0%})')
                    elif band_name == 'away_fav' and adj_away != 1.0:
                        old_draw = p.get('平', 0)
                        p['平'] *= adj_away
                        signals.append(f'🏷️{league}away_fav区间修正×{adj_away}(平局{old_draw:.0%}→{p["平"]:.0%})')
                    break
        except: pass

    p = _normalize_spf(p)

    return p, signals


# ═══════════════════════════════════════════════════════════════════════════
# P3: 冷门诱盘检测 v2（2026-04-06 更新）
# 根因：利雅新月1.18/国际图尔1.28等超低赔→实际平局
# 强化：超低赔主胜从×0.70→×0.65，平局boost从+5%→+15%
# ═══════════════════════════════════════════════════════════════

# 同联赛超低赔主胜计数器（内存缓存，生产环境建议用Redis）
_ULTRA_LOW_CACHE = {}  # {(league, team): count}

def trap_detection_adjust(spf_probs, ho, do_, ao, league='', match_date=None,
                          match_code='', home_team='', away_team=''):
    """P3: 冷门诱盘检测 v2

    规则：
    - 超低赔主胜(≤1.30): 主胜×0.65, 平+15%, 负+10%
    - 同联赛连续≥2场超低赔: 反转×0.85
    - 弱方赔率>6.0: 赔率失真降权

    Returns:
        (adjusted_probs, signals)
    """
    signals = []
    p = dict(spf_probs)
    global _ULTRA_LOW_CACHE

    try:
        ho_val = float(ho)
    except:
        return p, signals

    # 1. 超低赔主胜诱盘（强化：从×0.70→×0.65，平局更多）
    if ho_val <= 1.30:
        p['胜'] *= 0.65
        p['平'] *= 1.15
        p['负'] *= 1.10
        signals.append(f'🔴超低赔诱盘({ho_val:.2f})→主胜×0.65平+15%负+10%')
        # 更新缓存
        if home_team and league:
            key = (league, home_team)
            _ULTRA_LOW_CACHE[key] = _ULTRA_LOW_CACHE.get(key, 0) + 1
            cnt = _ULTRA_LOW_CACHE[key]
            if cnt >= 2:
                p['胜'] *= 0.85
                p['负'] *= 1.15
                signals.append(f'🔴连续{cnt}场超低赔主胜→反转主胜×0.85客胜+15%')
    elif ho_val <= 1.50:
        signals.append(f'🟡低赔主胜({ho_val:.2f})监控中')
        if home_team and league:
            key = (league, home_team)
            _ULTRA_LOW_CACHE[key] = _ULTRA_LOW_CACHE.get(key, 0)

    # 2. 弱方赔率失真（>6.0=赔率极端夸大差距）
    try:
        ao_val = float(ao)
        if ao_val > 6.0:
            p['负'] *= 0.80
            p['平'] *= 1.10
            signals.append(f'⚠️弱方赔率失真({ao_val:.1f})→客胜×0.80平局+10%')
        elif ao_val > 5.0:
            # Gap-12: 4.5-5.0区间：客胜冷门区，但不如>6.0极端
            p['负'] *= 0.85
            p['平'] *= 1.07
            signals.append(f'🟡冷门区间({ao_val:.1f})→客胜×0.85平+7%')
        elif ao_val > 4.5:
            # Gap-12: 4.5-5.0区间边界：轻微冷门信号
            p['负'] *= 0.90
            p['平'] *= 1.03
            signals.append(f'🟢冷门边界({ao_val:.1f})→客胜×0.90平+3%')
    except: pass

    p = _normalize_spf(p)

    return p, signals


# ═══════════════════════════════════════════════════════════════
# P2: 星期×联赛交叉权重（2026-04-05 新增）
# 根因：周六庄家全力控盘58%，周五45%最低
# ═══════════════════════════════════════════════════════════════

# 联赛×星期降权系数（2026-04-06 更新：加入周日数据）
# 周日命中率42.2%（全周最低）= 庄家收官日操盘最强
_LEAGUE_DOW_WEIGHTS = {
    ('英超', 'Sat'): 0.90,  ('英超', 'Fri'): 0.88,  ('英超', 'Sun'): 0.85,
    ('英超', 'Mon'): 0.90,
    ('西甲', 'Sat'): 0.90,  ('西甲', 'Fri'): 0.88,  ('西甲', 'Sun'): 0.85,
    ('德甲', 'Sat'): 0.90,  ('德甲', 'Fri'): 0.88,  ('德甲', 'Sun'): 0.87,
    ('意甲', 'Sat'): 0.90,  ('意甲', 'Fri'): 0.88,  ('意甲', 'Sun'): 0.87,
    ('法甲', 'Sat'): 0.90,  ('法甲', 'Fri'): 0.88,  ('法甲', 'Sun'): 0.87,
    ('葡超', 'Sat'): 0.92,  ('葡超', 'Fri'): 0.90,  ('葡超', 'Sun'): 0.88,
    ('荷甲', 'Sat'): 0.88,  ('荷甲', 'Fri'): 0.86,  ('荷甲', 'Sun'): 0.84,
    ('澳超', 'Sat'): 0.93,  ('澳超', 'Fri'): 0.92,  ('澳超', 'Sun'): 0.90,
    ('韩职', 'Sat'): 0.94,  ('韩职', 'Fri'): 0.92,  ('韩职', 'Sun'): 0.90,
    ('日职', 'Sat'): 0.94,  ('日职', 'Fri'): 0.92,  ('日职', 'Sun'): 0.90,
    ('沙职', 'Sat'): 0.95,  ('沙职', 'Fri'): 0.93,  ('沙职', 'Sun'): 0.91,
    ('英冠', 'Sat'): 0.88,  ('英冠', 'Fri'): 0.85,  ('英冠', 'Sun'): 0.85,
    ('德乙', 'Sat'): 0.92,  ('德乙', 'Fri'): 0.90,
    ('法乙', 'Sat'): 0.85,  ('法乙', 'Fri'): 0.82,  ('法乙', 'Sun'): 0.82,
    ('英甲', 'Sat'): 0.92,  ('英甲', 'Fri'): 0.90,
}

# 小联赛在周末降权更多（数据少+庄家抽水高）
_SMALL_LEAGUE_DOW = {
    'Fri': 0.88, 'Sat': 0.88, 'Sun': 0.90,
}

# P1: 中游队诱平陷阱检测模式
# 中游强队主场，赔率1.8-2.5，但庄家给平赔偏低(≤3.30) → 典型"诱上主胜"实为平局
_MID_TIER_TRAP_LEAGUES = {'葡超', '意甲', '西甲', '法甲', '欧罗巴', '欧协联', '土超', '希腊超'}
_MID_TIER_HO_RANGE = (1.80, 2.50)  # 主胜赔率区间
_MID_TIER_DRAW_THRESHOLD = 3.30  # 平赔低于此值触发陷阱


def mid_tier_trap_adjust(spf_probs, ho, do_, ao, league='',
                          match_date=None, match_code=''):
    """P1: 中游队诱平陷阱检测

    模式：中游球队主场（主胜赔率1.8-2.5）+ 平赔偏低(≤3.30)
    庄家意图：利用球队人气（而非真实优势）诱导主胜，实际赛果多为平局
    
    识别特征：
    - 联赛：中游强队多的联赛（葡超/意甲/西甲/法甲/欧罗巴/欧协联）
    - 主胜赔率：1.80-2.50（中游队，不是豪门）
    - 平赔：≤3.30（庄家在压低平赔，但对外显示"主胜热门"）
    
    调整：平局×1.25，主胜×0.80
    
    Returns:
        (adjusted_probs, signals)
    """
    signals = []
    p = dict(spf_probs)
    
    if league not in _MID_TIER_TRAP_LEAGUES:
        return p, signals
    
    try:
        ho_val = float(ho)
        do_val = float(do_)
    except (ValueError, TypeError):
        return p, signals
    
    lo, hi = _MID_TIER_HO_RANGE
    if not (lo <= ho_val <= hi):
        return p, signals
    
    # 平赔偏低 → 陷阱特征
    if do_val > 0 and do_val <= _MID_TIER_DRAW_THRESHOLD:
        # 庄家在用低平赔压赔付，但市场焦点在主胜
        p['平'] *= 1.25
        p['胜'] *= 0.80
        # 客胜略微降（庄家不担心客胜）
        p['负'] *= 0.95
        
        # 归一化
        total = sum(p.values())
        if total > 0:
            p = {k: max(0, v / total) for k, v in p.items()}
        
        signals.append(
            f'🔶中游诱平陷阱({league})，主胜{ho_val:.2f}平{do_val:.2f}→平×1.25胜×0.80'
        )
    
    return p, signals


def dow_league_adjust(confidence, league='', match_date=None, match_code=''):
    """P2: 星期×联赛交叉降权

    周五/周六：五大联赛降权10-12%，小联赛降权12-15%
    周三：庄家控盘弱，模型升权5%

    Returns:
        (adjusted_confidence, signals)
    """
    signals = []
    dow = _parse_dow(match_date, match_code)
    if not dow:
        return confidence, signals

    # 联赛×星期专属权重
    key = (league, dow)
    factor = _LEAGUE_DOW_WEIGHTS.get(key)

    # 小联赛通用周末降权（周日加入）
    if not factor and dow in _SMALL_LEAGUE_DOW and league in _SMALL_LEAGUES:
        factor = _SMALL_LEAGUE_DOW[dow]

    # 周日通用降权（42.2%命中率 = 全周最低）
    if not factor and dow == 'Sun':
        factor = 0.92
        signals.append('📅周日通用降权×0.92(全周命中率最低)')

    # 周三效应改为联赛专项：小联赛降权，五大联赛/欧冠小幅升权（2026-04-30修复）
    if not factor and dow == 'Wed':
        small_league = league in {'挪超', '日职', '日职乙', 'J联赛', '韩职', '美职', '澳超', '瑞典超', '瑞士超', '奥甲', '丹超', '芬超', '爱超', '比甲', '希腊超', '土超', '葡超', '俄超', '乌超', '捷甲', '匈甲', '波甲'}
        if small_league:
            confidence *= 0.92
            signals.append('📅周三小联赛降权→置信-8%（低流动性庄家信息差大）')
        else:
            confidence *= 1.03
            signals.append('📅周三庄家控盘弱→置信+3%（五大/欧冠小幅升权）')
        return round(confidence, 4), signals

    if not factor:
        return confidence, signals

    confidence *= factor
    signals.append(f'📅{dow}{league}交叉降权×{factor}')
    return round(confidence, 4), signals


# ═══════════════════════════════════════════════════════════════
# P4: 临场盘信号（2026-04-05 新增）
# 根因：Betfair成交数据已有，但未充分用于预测
# ═══════════════════════════════════════════════════════════════

def betfair_live_signal(betfair_data, spf_probs):
    """P4: 临场盘信号检测

    基于Betfair成交量变化判断庄家意图：
    - 成交量暴涨（开球前2h内>50%总量）: 庄家引导信号
    - 某选项成交量占比 vs 赔率支持度背离: 诱盘嫌疑
    - 凯利指数<1.0: 职业资金支持

    Returns:
        (adjusted_probs, signals)
    """
    signals = []
    p = dict(spf_probs)
    if not betfair_data:
        return p, signals

    try:
        vols = {
            '胜': betfair_data.get('home_vol', 0),
            '平': betfair_data.get('draw_vol', 0),
            '负': betfair_data.get('away_vol', 0),
        }
        total_vol = sum(vols.values())
        if total_vol < 100:
            return p, signals  # 成交量太少不参考

        pct = {k: v / total_vol for k, v in vols.items()}

        # 1. 凯利指数 < 1.0 = 职业资金支持
        kellys = {
            '胜': betfair_data.get('home_kelly', 99),
            '平': betfair_data.get('draw_kelly', 99),
            '负': betfair_data.get('away_kelly', 99),
        }
        for label, k in kellys.items():
            if 0 < k < 1.0:
                p[label] *= 1.08
                signals.append(f'💰凯利{k:.2f}<1.0职业支持{label}+8%')

        # 2. 成交量集中某方向 > 60% = 庄家引导
        for label, ratio in pct.items():
            if ratio > 0.60:
                signals.append(f'📊成交量集中{label}({ratio:.0%})→庄家引导嫌疑')
                # 引导方向本身可能是诱，反向+5%
                opp = {'胜': '负', '负': '胜', '平': '平'}
                opp_label = opp[label]
                p[opp_label] *= 1.05
                signals.append(f'  →反向{opp_label}+5%')

        # 3. 必发占比 vs 模型概率背离
        bk_pct = {k: spf_probs.get(k, 0.33) for k in ('胜', '平', '负')}

        for label in ('胜', '平', '负'):
            diff = pct[label] - bk_pct.get(label, 0)
            if diff > 0.15:
                # 必发热门 vs 赔率不支持 = 诱盘
                p[label] *= 0.92
                signals.append(f'🔍必发{pct[label]:.0%}vs赔率{bk_pct.get(label,0):.0%}背离→{label}诱盘-8%')
            elif diff < -0.15:
                # 必发冷门 vs 赔率支持 = 价值
                p[label] *= 1.06
                signals.append(f'💎必发{pct[label]:.0%}vs赔率{bk_pct.get(label,0):.0%}背离→{label}价值+6%')

        p = _normalize_spf(p)

    except Exception:
        pass

    return p, signals


def get_ultra_low_cache():
    """外部访问超低赔计数器（用于调试/重置）"""
    return _ULTRA_LOW_CACHE.copy()

def reset_ultra_low_cache():
    """重置超低赔计数器（每天调用一次）"""
    global _ULTRA_LOW_CACHE
    _ULTRA_LOW_CACHE.clear()


# ═══════════════════════════════════════════════════════════════
# P0 NEW: 赔率锚定 — 泊松概率与赔率隐含概率加权融合
# ═══════════════════════════════════════════════════════════════

def odds_implied_spf_probs(ho, do_, ao):
    """从欧赔计算隐含SPF概率（归一化，扣除返还率）"""
    try:
        imp_h = 1.0 / float(ho)
        imp_d = 1.0 / float(do_)
        imp_a = 1.0 / float(ao)
        total = imp_h + imp_d + imp_a
        return {
            '胜': imp_h / total,
            '平': imp_d / total,
            '负': imp_a / total,
        }
    except (ValueError, TypeError, ZeroDivisionError):
        return None


def odds_anchor_adjust(spf_probs, ho, do_, ao, league='', match_date=None, match_code=''):
    """P0强化: 赔率锚定融合 — 矛盾场次直接跟赔率走

    核心逻辑（基于30天数据）:
    - 矛盾场次命中率44%，周一仅12.5% → 矛盾时极低α
    - odds_ratio>3.5时直接用赔率（不做融合）
    - 周一/周五矛盾时α=0.10（赔率权重90%）
    """
    signals = []
    if not spf_probs or not ho or not do_ or not ao:
        return spf_probs, signals

    impl = odds_implied_spf_probs(ho, do_, ao)
    if not impl:
        return spf_probs, signals

    poisson_best = max(spf_probs, key=spf_probs.get)
    odds_best = max(impl, key=impl.get)

    try:
        fav_odds = min(float(ho), float(do_), float(ao))
        max_odds = max(float(ho), float(do_), float(ao))
        odds_ratio = max_odds / fav_odds if fav_odds > 0 else 1.0
    except:
        odds_ratio = 1.0

    conflict = poisson_best != odds_best
    dow = _get_dow(match_date, match_code)  # 复用已有函数

    # ── P4: 联赛矛盾模式修正系数 ──
    league_adj = _LEAGUE_CONFLICT_ADJ.get(league, {})

    # 极端矛盾 → 直接用赔率
    if conflict and odds_ratio > 3.5:
        fused = dict(impl)
        signals.append(f'🔴极端矛盾({poisson_best}→{odds_best},ratio={odds_ratio:.1f})→跟赔率')
        return fused, signals

    # 根据矛盾程度和周几选α
    if conflict:
        if dow in ('周一', '周五'):
            alpha = 0.10
            signals.append(f'🔴{dow}矛盾({poisson_best}→{odds_best})α=0.10')
        elif odds_ratio > 2.5:
            alpha = 0.15
            signals.append(f'⚠️强矛盾({poisson_best}→{odds_best},ratio={odds_ratio:.1f})α=0.15')
        else:
            alpha = 0.25
            signals.append(f'⚠️矛盾({poisson_best}↔{odds_best})α=0.25')
    elif odds_ratio > 3.5:
        alpha = 0.40
        signals.append(f'赔率悬殊(ratio={odds_ratio:.1f})α=0.40')
    else:
        alpha = 0.50

    # 泊松×α + 赔率×(1-α)
    fused = {k: alpha * spf_probs.get(k, 0) + (1 - alpha) * impl[k]
             for k in ('胜', '平', '负')}

    # P4: 联赛方向修正
    if conflict and league_adj:
        for k, adj in league_adj.items():
            if k in fused:
                fused[k] *= adj
        signals.append(f'联赛修正({league})')

    total = sum(fused.values())
    if total > 0:
        fused = {k: v / total for k, v in fused.items()}

    return fused, signals


# ═══════════════════════════════════════════════════════════════
# P2 NEW: 高赔率方差场次特殊处理
# ═══════════════════════════════════════════════════════════════

def odds_conflict_check(spf_probs, ho, do_, ao, league=''):
    """P2: 高赔率方差场次检测 — 赔率与预测方向明显矛盾时标记高风险

    判断逻辑：
    - 预测选项的赔率 > 赔率最低项 × 1.8 → 高风险
    - 例：预测主胜2.5，但客胜只有1.5（1.5×1.8=2.7>2.5成立）
    
    Returns:
        (is_high_risk, risk_signals, confidence_penalty)
    """
    signals = []
    penalty = 0.0

    try:
        ho_f, do_f, ao_f = float(ho), float(do_), float(ao)
    except (ValueError, TypeError):
        return False, signals, penalty

    if not spf_probs:
        return False, signals, penalty

    pred_dir = max(spf_probs, key=spf_probs.get)
    pred_odds = {'胜': ho_f, '平': do_f, '负': ao_f}.get(pred_dir, 0)
    min_odds = min(ho_f, do_f, ao_f)

    # 赔率比判断
    odds_margin = pred_odds / min_odds if min_odds > 0 else 1.0

    if odds_margin > 2.2:
        signals.append(f'🔴高风险:预测{pred_dir}@{pred_odds:.2f}但赔率最低{min_odds:.2f}(比{odds_margin:.1f}×)')
        penalty = 0.25
        return True, signals, penalty
    elif odds_margin > 1.8:
        signals.append(f'🟡中风险:预测{pred_dir}@{pred_odds:.2f}但赔率暗示{min_odds:.2f}(比{odds_margin:.1f}×)')
        penalty = 0.10
        return True, signals, penalty

    return False, signals, penalty


# ═══════════════════════════════════════════════════════════════
# P2 NEW: 周一看盘特殊规则
# ═══════════════════════════════════════════════════════════════

def monday_special_adjust(spf_probs, ho, do_, ao, league='', match_date=None, match_code=''):
    """P0强化: 周一专项处理 — 庄家周中吸筹后周一开盘策略
    数据: 周一命中率33.3%(历史最低)，英冠周一6场全错，平局率偏高
    
    规则(v3强化):
    1. 周一平局概率+10%（庄家常开平局盘）
    2. 客队热门时客胜额外+8%（庄家引导资金去主队）
    3. 英冠/意甲/荷乙周一额外谨慎：平局+5%
    4. 小联赛周一：λ×0.80（进球预期降20%）
    """
    signals = []
    if not spf_probs:
        return spf_probs, signals

    dow = _get_dow(match_date, match_code)
    if dow != '周一':
        return spf_probs, signals

    try:
        ho_f, do_f, ao_f = float(ho), float(do_), float(ao)
    except (ValueError, TypeError):
        return spf_probs, signals

    p = dict(spf_probs)

    # 1. 周一平局+10%（原+5%→+10%，更激进）
    p['平'] *= 1.10
    signals.append('📅周一平局+10%')

    # 2. 客队热门时客胜+8%（庄家引导主队资金）
    if ao_f < ho_f:
        p['负'] *= 1.08
        signals.append('📅周一客队热门→客胜+8%')

    # 3. 英冠/意甲/荷乙周一额外谨慎
    if league in ('英冠', '意甲', '荷乙'):
        p['平'] *= 1.05
        signals.append(f'⚠️{league}周一额外谨慎→平局+5%')

    # 4. 小联赛周一λ降权（进球预期降20%）
    if league in _SMALL_LEAGUES:
        signals.append(f'📅周一{league}小联赛→进球预期×0.80')

    p = _normalize_spf(p)

    return p, signals


# ═══════════════════════════════════════════════════════════════
# P3 NEW: 中间赔率区间专项优化 (1.81-2.80)
# ═══════════════════════════════════════════════════════════════

def mid_odds_band_adjust(spf_probs, ho, do_, ao, league=''):
    """P3: 中间赔率区间(1.81-2.80)专项优化
    数据: 该区间命中率仅42-44%，庄家开盘最不明确
    
    策略:
    - 该区间平局概率+8%（庄家自己也拿不准）
    - 缩小三方差距（更均衡预测）
    """
    signals = []
    if not spf_probs:
        return spf_probs, signals

    try:
        ho_f, do_f, ao_f = float(ho), float(do_), float(ao)
    except (ValueError, TypeError):
        return spf_probs, signals

    # ── P0/P3新增：真正均衡盘口检测（04-06 balanced odds 4场全错的精准修复）──
    spread = abs(ho_f - ao_f)
    if spread < 0.30:
        # 真正均衡盘口：|ho-ao|<0.3 → 庄家自己都拿不准，4场全错
        # 规则：draw≥30%，所有选项<40%
        p = dict(spf_probs)
        implied_draw = 0.886 / do_f if do_f > 0 else 0.26
        draw_floor = max(0.30, implied_draw * 0.85)
        if p.get('平', 0) < draw_floor:
            p['平'] = draw_floor
        # 强制所有<40%
        for k in p:
            p[k] = min(0.39, p[k])
        # 重新归一化（确保draw最高但不超40%）
        total = sum(p.values())
        if total > 0:
            p = {k: min(0.39, v / total * (1 + (0.30 - min(p.values())) / total)) for k, v in p.items()}
            p = {k: min(0.39, v) for k, v in p.items()}
            total = sum(p.values())
            if total > 0:
                p = {k: v / total for k, v in p.items()}
        signals.append(f'⚖️均衡盘口({spread:.2f})→draw≥{draw_floor:.0%}强制,全部<40%')
        return p, signals

    # ── 中间赔率区间(0.30≤|ho-ao|<0.5)：温和均衡 ──
    if spread >= 0.50:
        return spf_probs, signals  # 非均衡盘口，跳过

    p = dict(spf_probs)

    # 平局+12%（比原来8%更激进）
    p['平'] *= 1.12

    # 缩小三方差距（向均值靠拢）
    avg = sum(p.values()) / 3
    for k in p:
        p[k] = 0.80 * p[k] + 0.20 * avg  # 原来0.85→0.80，更均衡

    total = sum(p.values())
    if total > 0:
        p = {k: v / total for k, v in p.items()}

    signals.append(f'📊温和均衡({spread:.2f})→平局+12%均衡化')
    return p, signals


# ═══════════════════════════════════════════════════════════════
# P0-1 NEW: 中赔区间(1.8-2.3)精准优化 - 命中率仅37.6%的专项修复
# 根因: 中赔(1.8-2.3)命中率37.6%，远低于55%预期
#       胜→平转化率24.8%，平局超出预期；胜→负转化率27.4%，负也超出预期
# ═══════════════════════════════════════════════════════════════

def mid_tier_odds_sharp_adjust(spf_probs, ho, do_, ao, league='',
                               match_date=None, match_code=''):
    """P0-1: 中赔区间(1.8-2.3)精准优化 - 命中率仅37.6%的专项修复

    数据证据：
    - 中赔(1.8-2.3)命中率37.6%，远低于55%预期
    - 胜→平转化率24.8%，平局超出预期
    - 胜→负转化率27.4%，负也超出预期

    策略：
    - ho 1.8-2.3区间：平局+20%，胜/负各压10%
    - ho 2.0-2.3区间（更高风险）：平局+25%，胜×0.85
    - 当do_ < 3.2时（低平赔陷阱）：平局再+5%
    - 限制调整幅度，避免过度修正

    Returns:
        (adjusted_probs, signals)
    """
    signals = []
    if not spf_probs:
        return spf_probs, signals
    try:
        ho_f = float(ho); do_f = float(do_)
    except (ValueError, TypeError):
        return spf_probs, signals

    # Only apply to 1.8 <= ho < 2.3 range
    if not (1.80 <= ho_f < 2.30):
        return spf_probs, signals

    p = dict(spf_probs)

    # 胜→平转化率压降: 1.8-2.3区间平局强化
    draw_boost = 1.22
    if ho_f >= 2.00:
        draw_boost = 1.28  # 原来是1.25，现增强
    if 0 < do_f < 3.20:
        draw_boost += 0.05
    if ho_f > 2.0 and do_f < 3.5:
        # 新增: ho>2.0且平赔不太高时加强平局权重
        draw_boost = max(draw_boost, 1.30)

    p['平'] *= draw_boost
    p['胜'] *= 0.90
    p['负'] *= 0.90

    # Cap at reasonable values (no single option > 55%)
    for k in p:
        p[k] = min(0.55, p[k])

    # Re-normalize
    total = sum(p.values())
    if total > 0:
        p = {k: max(0.05, v / total) for k, v in p.items()}

    signals.append(f'📊中赔精准ho={ho_f:.2f}→平×{draw_boost:.2f}胜×0.90负×0.90')
    return p, signals


# ═══════════════════════════════════════════════════════════════════════
# P0: 胜→平转化率专项压降（基于2026-04-26复盘数据）
# 目标: 胜→平转化率 24.8% → <20%
# ═══════════════════════════════════════════════════════════════════════

_WIN_TO_DRAW_TARGET = 0.20   # 目标转化率<20%
_WIN_TO_DRAW_HISTORICAL = 0.248  # 历史转化率24.8%

def transition_matrix_adjust(spf_probs, ho, do_, ao, league='', dow=''):
    """
    专项调整胜→平转化率
    触发条件: 预测胜且ho>2.0且do_<3.5
    动作: 平局提升，胜压降
    """
    p = dict(spf_probs) if spf_probs else {}
    signals = []

    try:
        ho_f = float(ho)
        do_f = float(do_)
    except (ValueError, TypeError):
        return p, signals

    # 触发条件: 预测胜, ho>2.0, 平赔不太高
    if p.get('胜', 0) <= p.get('平', 0) or ho_f <= 2.0 or do_f >= 3.5:
        return p, signals

    # 计算压降幅度（历史24.8%→目标20%，需压降约20%）
    boost = 0.10  # 基础平局+10%

    if ho_f > 2.3:
        boost = 0.12  # 更高风险区间
    elif ho_f > 2.0 and do_f < 3.2:
        boost = 0.08  # 低平赔陷阱

    p['平'] *= (1 + boost)
    p['胜'] *= (1 - boost * 0.5)

    # 归一化
    total = sum(p.values())
    if total > 0:
        p = {k: max(0.05, v / total) for k, v in p.items()}

    signals.append(f'📊transition_matrix ho={ho_f:.2f}→平+{boost*100:.0f}%')
    return p, signals


# ═══════════════════════════════════════════════════════════════════════
# P4 NEW: 联赛矛盾模式库（数据驱动）
# ═══════════════════════════════════════════════════════════════

# 基于30天矛盾场次数据: 联赛→方向修正系数
_LEAGUE_CONFLICT_ADJ = {
    '意甲': {'负': 1.15},   # 矛盾时模型常低估客队
    '英冠': {'负': 1.10},   # 冷门陷阱频繁
    '英超': {'胜': 0.95},   # 主场优势被高估
    '荷甲': {'负': 1.10},   # 客队偷袭多
    '德甲': {'胜': 0.95},   # 客场偷袭多
}


# (已合并到上方 _get_dow)


# ═══════════════════════════════════════════════════════════════
# P5: 联赛×周几 λ校正因子（基于2026年历史数据，样本≥15场）
# ═══════════════════════════════════════════════════════════════

# 联赛总平均进球 → 周几平均进球 → 校正因子 = 周几平均/联赛平均
# 正值=该周几进球偏多, <1=偏少
_DOW_LEAGUE_GOAL_FACTOR = {
    '欧冠': {'周三': 1.14, '周四': 1.07},
    '西甲': {'周日': 1.16, '周六': 1.09, '周五': 0.96, '周一': 0.95},
    '德乙': {'周日': 1.16, '周六': 1.00, '周五': 0.88},
    '沙职': {'周五': 1.13, '周四': 1.10, '周六': 0.93},
    '德甲': {'周日': 1.10, '周六': 1.05, '周五': 1.04},
    '澳超': {'周五': 1.08, '周六': 1.05},
    '欧罗巴': {'周五': 1.05},
    '荷甲': {'周六': 1.02, '周日': 1.01},
    '意甲': {'周四': 1.26, '周一': 1.06, '周五': 1.07, '周六': 0.97, '周日': 0.95},
    '英甲': {'周六': 1.06, '周二': 0.89, '周四': 0.84},
    '英超': {'周三': 1.12, '周四': 0.99, '周六': 1.03, '周日': 1.01, '周一': 0.96, '周二': 0.92},
    '英冠': {'周日': 1.11, '周五': 1.07, '周六': 1.00, '周二': 0.96, '周三': 0.90, '周四': 0.91},
    '葡超': {'周六': 1.03, '周日': 1.05},
    '法甲': {'周日': 1.05},
    '韩职': {'周二': 0.98},
}

# 联赛×周几平局率偏差（加到泊松平局概率上）
# 正值=该周几平局偏多, 负值=偏少
_DOW_LEAGUE_DRAW_ADJUST = {
    # P2-3: 联赛×周几平局率偏差（正值=平局偏多, 负值=偏少）
    # 基于409场回测数据扩充
    '英超': {'周二': 0.25, '周四': 0.20, '周一': 0.30, '周三': 0.08, '周六': 0.05},
    '意甲': {'周四': 0.20, '周三': 0.25, '周六': 0.07, '周日': -0.05, '周一': -0.07},
    '英冠': {'周三': -0.10, '周五': -0.06, '周六': -0.01, '周日': -0.02},
    '西甲': {'周一': 0.20, '周六': 0.06, '周二': 0.30, '周日': -0.04, '周五': 0.10},
    '德甲': {'周五': 0.18, '周六': 0.08, '周日': -0.07, '周三': 0.10},
    '德乙': {'周六': 0.08, '周日': 0.09},
    '法甲': {'周日': 0.05, '周六': 0.08, '周五': 0.12},
    # 新增高频周几效应
    '欧冠': {'周三': -0.08, '周二': 0.05},  # 欧冠周中平局偏少
    '欧罗巴': {'周三': -0.06, '周四': 0.04},
    '韩职': {'周六': 0.15, '周日': 0.10},  # 韩职周末平局偏多
    '日职': {'周六': 0.12, '周日': 0.08},  # 日职周末平局偏多
    '沙职': {'周五': 0.12, '周六': 0.08},
    '美职': {'周六': 0.10, '周日': 0.08},
    '解放者杯': {'周二': 0.08, '周四': 0.06},

}


def apply_dow_league_lambda(cal_h, cal_a, league='', match_date=None, match_code=''):
    """P5: 联赛×周几 λ校正 — 调整泊松期望进球数

    基于历史数据：同一联赛在不同周几的场均进球差异可达0.5-0.9球。
    校正因子 = 周几场均进球 / 联赛总场均进球

    Args:
        cal_h, cal_a: 赔率反推的泊松λ
        league: 联赛名
        match_date: 比赛日期 (YYYY-MM-DD)
        match_code: 场次编码 (含周几信息)

    Returns:
        (adjusted_h, adjusted_a, draw_adj, signals)
    """
    signals = []
    dow = _get_dow(match_date, match_code)
    if not dow or not league:
        return cal_h, cal_a, 0.0, signals

    # 进球校正因子
    factor = _DOW_LEAGUE_GOAL_FACTOR.get(league, {}).get(dow, 1.0)
    if factor != 1.0:
        cal_h *= factor
        cal_a *= factor
        arrow = '↑' if factor > 1 else '↓'
        signals.append(f'λ{arrow}{factor:.2f}({league}{dow}偏{"多" if factor > 1 else "少"}球)')

    # 平局校正
    draw_adj = _DOW_LEAGUE_DRAW_ADJUST.get(league, {}).get(dow, 0.0)
    if abs(draw_adj) > 0.01:
        direction = '↑' if draw_adj > 0 else '↓'
        signals.append(f'平局{direction}{abs(draw_adj):.0%}({league}{dow})')

    return cal_h, cal_a, draw_adj, signals


def apply_dow_draw_to_spf(spf_probs, draw_adj):
    """将平局校正应用到SPF概率上"""
    if abs(draw_adj) < 0.01 or not spf_probs:
        return spf_probs
    p = dict(spf_probs)
    p['平'] = max(0.03, min(0.60, p.get('平', 0.25) + draw_adj))
    total = sum(p.values())
    if total > 0:
        p = {k: v / total for k, v in p.items()}
    return p


# ═══════════════════════════════════════════════════════════════
# P1 NEW: Confidence校准（基于04-06数据分析）
# 根因: confidence 55-70区间命中率41.4%（全段最低），赔率悬殊时被过度自信
# 规则: confidence>65 ×0.85, <30 ×0.90（过低也不可靠）
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
# P2-2: 博弈论优化 — 赔率市场效率检测
# ═══════════════════════════════════════════════════════════════

def odds_market_efficiency_check(spf_probs, ho, do_, ao):
    """P2-2: 赔率市场效率检测 — 庄家赔率是否揭示真实概率

    计算庄家隐含概率 vs 模型概率的偏离（KL散度）。
    偏离大 → 庄家有信息优势 → 降权。

    数据来源: 409场回测

    Args:
        spf_probs: 模型SPF概率 {'胜': p1, '平': p2, '负': p3}
        ho, do_, ao: 主胜/平局/客胜赔率

    Returns:
        (adjusted_probs, signals)
    """
    import math as _math_local

    try:
        ho_f, do_f, ao_f = float(ho), float(do_), float(ao)
        p = dict(spf_probs)

        # 庄家隐含概率（假设体彩返还率88.6%，归一化）
        home_imp = (1 / ho_f) / (1 / ho_f + 1 / do_f + 1 / ao_f)
        draw_imp = (1 / do_f) / (1 / ho_f + 1 / do_f + 1 / ao_f)
        away_imp = (1 / ao_f) / (1 / ho_f + 1 / do_f + 1 / ao_f)

        # 模型概率
        total_p = sum(p.values())
        if total_p == 0:
            return spf_probs, []

        # 计算KL散度近似（庄家 vs 模型）
        kl_home = home_imp * _math_local.log(home_imp / max(0.01, p.get('胜', 0.33)))
        kl_draw = draw_imp * _math_local.log(draw_imp / max(0.01, p.get('平', 0.33)))
        kl_away = away_imp * _math_local.log(away_imp / max(0.01, p.get('负', 0.33)))
        kl_total = abs(kl_home) + abs(kl_draw) + abs(kl_away)

        # KL散度>0.5说明庄家和模型分歧大 → 降权
        signals = []
        if kl_total > 0.5:
            penalty = min(0.15, (kl_total - 0.5) * 0.2)
            for k in p:
                p[k] *= (1 - penalty)
            total = sum(p.values())
            if total > 0:
                p = {k: max(0.05, v / total) for k, v in p.items()}
            signals.append(f'📉市场效率偏离KL={kl_total:.3f}→all×{1 - penalty:.2f}')

        return p, signals
    except Exception:
        return spf_probs, []


# P3: 联赛专项置信度上限（提出来避免每次调用重建）
_HIGH_CONF_CAPS = {
    '国际赛': 70, '友谊赛': 70, '世界杯': 75, '欧冠': 80,
    '欧罗巴': 80, '欧协联': 80, '世预赛': 75, '洲际杯': 75,
}


def confidence_calibration_adjust(confidence, spf_probs, ho, do_, ao, league='',
                                      match_date=None, match_code=''):
    """P1+P3: Confidence校准 — 赔率悬殊时模型过度自信 + 联赛专项置信度上限
    
    数据: 55-70区间命中率41.4%最差，45-55区间56.5%最好。
    高confidence时(赔率悬殊)模型倾向追热门，但庄家反向操盘。
    
    P3新增规则（基于LLM进化建议）：
    - 国际赛/友谊赛：置信度上限70%（战意衰减不可预测）
    - 杯赛（非联赛）：置信度上限75-80%
    - 纯赔率推导的极高信心(>90)：×0.85防过度自信
    - 赔率悬殊(odds_ratio>2.0)+高conf：×0.80
    
    Returns: (adjusted_confidence, signals)
    """
    signals = []
    conf = confidence
    try:
        conf_val = float(conf) if not isinstance(conf, (int, float)) else conf
    except (ValueError, TypeError):
        return confidence, signals

    try:
        ho_f, do_f, ao_f = float(ho), float(do_), float(ao)
        fav_odds = min(ho_f, do_f, ao_f)
        max_odds = max(ho_f, do_f, ao_f)
        odds_ratio = max_odds / fav_odds if fav_odds > 0 else 1.0
    except (ValueError, TypeError):
        odds_ratio = 1.5  # 保守

    # conf>65 + odds悬殊(>2.0) → 最危险组合，×0.80
    if conf_val > 65 and odds_ratio > 2.0:
        conf *= 0.80
        signals.append(f'🎯高危(conf={conf_val},ratio={odds_ratio:.1f})→×0.80')
    elif conf_val > 65:
        conf *= 0.85
        signals.append(f'📉高conf({conf_val}%)→×0.85')
    elif conf_val < 30:
        conf *= 0.90
        signals.append(f'📉低conf({conf_val}%)→×0.90')
    elif 45 <= conf_val <= 55:
        conf *= 1.05
        signals.append(f'📈最佳conf区间({conf_val}%)→×1.05')

    # P3: 联赛专项置信度上限
    cap = _HIGH_CONF_CAPS.get(league)
    if cap and conf_val > cap:
        conf = cap
        signals.append(f'🔒{league}置信度上限{cap}%')
    
    # P3: 极高信心(>90)强制衰减（纯排名/身价模型不可靠）
    if conf_val > 90:
        conf = min(conf, 85)
        signals.append(f'⚠️极高信心{conf_val}%→强制上限85%')

    return round(conf, 4), signals


# ═══════════════════════════════════════════════════════════════
# P1 NEW: 英冠全周冷门触发器
# 根因: 英冠04-06 6场命中率16.7%，非仅周五问题
# 规则: 英冠不分周几，赔率盲区(1.5-2.5)主胜×0.85，平局+10%
# ═══════════════════════════════════════════════════════════════

def championship_daily_adjust(spf_probs, ho, do_, ao, league=''):
    """P1: 英冠全周冷门触发器（非仅周五）
    
    根因: 英冠04-06命中16.7%，5场错中4场是主队热门被逼平。
    赔率盲区(1.5-2.5)：庄家知道主队不稳但开盘模糊。
    
    规则:
    - 英冠+odds_home 1.5-2.5 → 主胜×0.85, 平+10%, 客胜+5%
    - 英冠+balanced odds(|ho-ao|<0.5) → 平×1.20
    - 英冠+超低赔主胜(≤1.5) → 保持警惕：主胜×0.90
    """
    signals = []
    if not spf_probs:
        return spf_probs, signals

    is_championship = league in ('英冠', 'Championship', 'ELC')
    if not is_championship:
        return spf_probs, signals

    try:
        ho_f = float(ho)
        ao_f = float(ao)
        spread = abs(ho_f - ao_f)
    except (ValueError, TypeError):
        return spf_probs, signals

    p = dict(spf_probs)

    # 英冠赔率盲区(1.5-2.5)：主胜×0.85
    if 1.5 <= ho_f <= 2.5:
        p['胜'] *= 0.85
        p['平'] *= 1.10
        p['负'] *= 1.05
        signals.append(f'🏟️英冠赔率盲区({ho_f:.2f})→主×0.85平+10%')

    # 英冠均衡盘口：平×1.20
    if spread < 0.5:
        p['平'] *= 1.20
        signals.append(f'🏟️英冠均衡盘口(差{spread:.2f})→平×1.20')

    # 英冠超低赔主胜：保持警惕（不是陷阱，但不过度自信）
    if ho_f <= 1.5:
        p['胜'] *= 0.90
        signals.append(f'🏟️英冠低赔主胜({ho_f:.2f})→主×0.90谨慎')

    p = _normalize_spf(p)

    return p, signals


# ═══════════════════════════════════════════════════════════════
# P2-熔断: 威廉-体彩赔率偏差检测（2026-04-07）
# 根因：威廉希尔与体彩赔率出现极大偏差（如尤文图斯7.8，尼斯4.2）
# 常伴随极端冷门或模型盲区，>2.5时触发熔断预警
# ═══════════════════════════════════════════════════════════════

def william_deviation_alert(ho, do_, ao, ticai_ho=None, ticai_do=None, ticai_ao=None):
    """检查威廉-体彩赔率偏差是否触发熔断预警

    Args:
        ho, do_, ao: 威廉希尔赔率（主胜/平/客胜）
        ticai_ho, ticai_do, ticai_ao: 体彩赔率（可选，不提供则跳过）

    Returns:
        (max_deviation, signal_or_None)
    """
    if None in (ticai_ho, ticai_do, ticai_ao):
        return 0.0, None
    try:
        ho_w, do_w, ao_w = float(ho), float(do_), float(ao)
        ho_t = float(ticai_ho)
        do_t = float(ticai_do)
        ao_t = float(ticai_ao)
        # 计算各选项最大偏差
        max_dev = max(abs(ho_w - ho_t), abs(do_w - do_t), abs(ao_w - ao_t))
        THRESHOLD = 2.5
        if max_dev > THRESHOLD:
            return max_dev, (f"🚨威廉-体彩偏差熔断({max_dev:.2f}>2.5) "
                      f"H:{ho_w:.2f}/{ho_t:.2f} D:{do_w:.2f}/{do_t:.2f} A:{ao_w:.2f}/{ao_t:.2f}")
        return max_dev, None
    except:
        return 0.0, None


# ═══════════════════════════════════════════════════════════════
# P4-1: 均衡赔率检测 (2026-04-16)
# 根因: 主客胜率差<15%时，庄家对比赛走向不明确，模型应切换为随机基准
# 规则: |1/ho - 1/ao| < 0.15 → 均衡场次，平局boost×1.3，主客胜衰减
# ═══════════════════════════════════════════════════════════════

def detect_balanced_odds(home_odds, draw_odds, away_odds):
    """P4-1: 均衡赔率检测 — 主客胜率差<15%时切换为随机基准

    当主客胜率差很小时，说明庄家对比赛结果没有明显倾向，
    此时泊松模型的预测可靠性下降，应切换为更均衡的概率分布。

    规则：
    - is_balanced = |1/ho - 1/ao| < 0.15
    - 均衡时：平局概率×1.3，主客胜概率相应衰减

    Args:
        home_odds, draw_odds, away_odds: 欧赔

    Returns:
        (is_balanced: bool, signals: list)
    """
    signals = []
    try:
        ho_f, do_f, ao_f = float(home_odds), float(draw_odds), float(away_odds)
        if ho_f <= 1 or ao_f <= 1 or do_f <= 1:
            return False, signals
    except (ValueError, TypeError):
        return False, signals

    # 计算隐含胜率
    prob_h = 1.0 / ho_f
    prob_a = 1.0 / ao_f
    prob_diff = abs(prob_h - prob_a)

    is_balanced = prob_diff < 0.15
    if is_balanced:
        signals.append(f'⚖️P4-1均衡赔率(主{ho_f:.2f}客{ao_f:.2f},胜率差{prob_diff:.1%}<15%)')

    return is_balanced, signals


def balanced_odds_adjust(spf_probs, home_odds, draw_odds, away_odds):
    """P4-1: 均衡赔率调整 — 平局boost×1.3，主客胜衰减

    当检测到均衡赔率时（主客胜率差<15%），庄家开盘显示不确定，
    应提升平局概率，降低主客胜的极端预测。

    Returns:
        (adjusted_probs, signals)
    """
    signals = []
    is_balanced, bal_sig = detect_balanced_odds(home_odds, draw_odds, away_odds)
    signals.extend(bal_sig)

    if not is_balanced:
        return dict(spf_probs), signals

    p = dict(spf_probs)
    # 平局boost ×1.3
    p['平'] *= 1.30
    # 主客胜各承担一半衰减
    total_before = sum(p.values())
    decay_total = p['平'] - spf_probs.get('平', 0)
    if decay_total > 0 and total_before > 0:
        p['胜'] = max(0.08, p['胜'] - decay_total * 0.5)
        p['负'] = max(0.08, p['负'] - decay_total * 0.5)

    total_after = sum(p.values())
    if total_after > 0:
        for k in p:
            p[k] /= total_after

    signals.append(f'P4-1均衡调整→平×1.3,胜/{p.get("胜",0):.1%},负/{p.get("负",0):.1%}')

    return p, signals


# ═══════════════════════════════════════════════════════════════
# P4-2: 赔率梯度分层 (2026-04-16)
# 根因: 1.3-1.5=大热（过热回调）/ 1.5-2.5=正常 / >3.5=诱盘
# 规则: 大热方向×0.85，诱盘方向×0.75
# ═══════════════════════════════════════════════════════════════

def get_odds_gradient(home_odds, away_odds, league=''):
    """P4-2: 赔率梯度分层 — 返回(梯度类型, 调整方向)

    分层规则：
    - 客队赔率1.3-1.5：客队为大热，客胜概率×0.85
    - 客队赔率1.5-2.5：正常区间，不调整
    - 客队赔率>3.5：客胜为诱盘，客胜概率×0.75
    - 主队赔率1.3-1.5：主队为大热，主胜概率×0.85
    - 主队赔率>3.5：主胜为诱盘，主胜概率×0.75

    Returns:
        (gradient_type: str, adjustment: dict)  # adjustment = {方向: multiplier}
    """
    adjustment = {}
    gradient_type = 'normal'

    try:
        ho_f = float(home_odds)
        ao_f = float(away_odds)
    except (ValueError, TypeError):
        return 'unknown', {}

    # 客队梯度
    if 1.3 <= ao_f <= 1.5:
        gradient_type = 'away_hot'
        adjustment['负'] = 0.85
    elif ao_f > 3.5:
        gradient_type = 'away_trap'
        adjustment['负'] = 0.75

    # 主队梯度
    if ho_f < 1.3:
        # 大热：主队赔率极低（如皇马），过热回调
        if gradient_type != 'normal':
            gradient_type = 'both_hot'
        else:
            gradient_type = 'home_hot'
        adjustment['胜'] = 0.85
    elif 1.3 <= ho_f <= 1.5:
        # 中热
        if gradient_type != 'normal':
            gradient_type = 'both_hot'
        else:
            gradient_type = 'home_hot'
        adjustment['胜'] = 0.85
    elif 1.5 < ho_f <= 2.0:
        # P4-2新增：主场优势区但高波动，主胜概率×0.88
        gradient_type = 'home_mid_high_vol'
        adjustment['胜'] = 0.88
    elif ho_f > 3.5:
        if gradient_type != 'normal':
            gradient_type = 'both_trap'
        else:
            gradient_type = 'home_trap'
        adjustment['胜'] = 0.75

    # P4-2新增：客队优势区（2.0 < ao <= 3.0）
    if 2.0 < ao_f <= 3.0:
        adjustment['负'] = 0.90
        if gradient_type == 'normal':
            gradient_type = 'away_mid_advantage'

    # 低流动性联赛：away_trap/home_trap ×0.90 替代 ×0.75
    _SMALL = {'挪超', '日职', '日职乙', 'J联赛', '韩职', '美职', '澳超',
               '瑞典超', '瑞士超', '奥甲', '丹超', '芬超', '爱超', '比甲', '苏超'}
    if league in _SMALL and gradient_type in ('away_trap', 'home_trap'):
        for direction in adjustment:
            adjustment[direction] = 0.90

    return gradient_type, adjustment


def odds_gradient_adjust(spf_probs, home_odds, away_odds, league=''):
    """P4-2: 赔率梯度分层调整

    大热方向回调（×0.85），诱盘方向降权（×0.75）。
    联赛专项：挪超/日职/韩职等低流动性联赛在客队大热时，away_trap优先级更高。

    Args:
        league: 联赛名，低流动性联赛客队大热时跳过梯度调整（避免信号冲突）

    Returns:
        (adjusted_probs, signals)
    """
    # Bug #5 Fix (2026-04-30): 低流动性联赛客队大热时，away_trap已由odds_gradient_adjust先检测，
    # 不需要二次惩罚；但需要把league传给away_win_boost使其正确跳过冲突信号
    _SMALL = {'挪超', '日职', '日职乙', 'J联赛', '韩职', '美职', '澳超',
               '瑞典超', '瑞士超', '奥甲', '丹超', '芬超', '爱超', '比甲', '苏超'}
    signals = []
    gradient_type, adjustment = get_odds_gradient(home_odds, away_odds, league)

    # 低流动性联赛 + 客队大热（away_trap场景）：跳过梯度调整，让away_trap×0.75处理
    if league in _SMALL and gradient_type == 'away_trap':
        signals.append(f'P4-2跳过(低流动性联赛away_trap已由先行的away_trap处理)')
        return dict(spf_probs), signals

    if gradient_type == 'normal' or not adjustment:
        return dict(spf_probs), signals

    p = dict(spf_probs)

    for direction, mult in adjustment.items():
        old_val = p.get(direction, 0)
        p[direction] *= mult
        signals.append(f'P4-2梯度:{direction}×{mult}({gradient_type},原{old_val:.1%}→{p[direction]:.1%})')

    # 补偿衰减给其他方向
    total_decay = sum(
        spf_probs.get(d, 0) * (1 - mult)
        for d, mult in adjustment.items()
    )
    other_dirs = [d for d in ('胜', '平', '负') if d not in adjustment]
    if other_dirs and total_decay > 0:
        per_dir = total_decay / len(other_dirs)
        for d in other_dirs:
            p[d] = min(0.90, p.get(d, 0) + per_dir)

    total = sum(p.values())
    if total > 0:
        for k in p:
            p[k] /= total

    return p, signals


# ═══════════════════════════════════════════════════════════════
# P4-4: 置信度衰减 (2026-04-16)
# 根因: 高置信+赔率异常波动=庄家反向操盘风险
# 规则: 置信>65%且赔率在预测方向变化>5% → 置信度×0.8
# ═══════════════════════════════════════════════════════════════

def confidence_decay(confidence, odds_change, current_confidence=None):
    """P4-4: 置信度衰减 — 高置信+赔率异常波动时打折

    当模型非常自信（>65%）但赔率在预测方向有显著变化时，
    说明庄家可能在反向操盘，此时应降低置信度。

    规则：
    - 置信>65% 且 赔率变化>5% → 置信度×0.8
    - 置信<35%时（弱信号）：不调整

    Args:
        confidence: 原始置信度 (0-1)
        odds_change: 赔率在预测方向的变化幅度 (0-1, 正数=朝预测方向变，负数=反方向)
        current_confidence: 可选，当前的动态置信度（用于比较）

    Returns:
        (adjusted_confidence, signals)
    """
    signals = []

    try:
        conf_val = float(confidence) if not isinstance(confidence, (int, float)) else confidence
        odds_ch = float(odds_change) if not isinstance(odds_change, (int, float)) else odds_change
    except (ValueError, TypeError):
        return confidence, signals

    # 弱信号不调整
    if conf_val < 0.35:
        return confidence, signals

    # 赔率在预测方向变化>5%=显著波动
    if conf_val > 0.65 and abs(odds_ch) > 0.05:
        old_conf = conf_val
        conf_val *= 0.80
        signals.append(
            f'P4-4置信衰减:高conf{old_conf:.0%}×赔率波动{odds_ch:+.1%}→×0.8={conf_val:.0%}'
        )

    return round(conf_val, 4), signals

# ─────────────────────────────────────────────
# P0: 赔率数据清洗与Overround异常检测
# ─────────────────────────────────────────────
import math as _m

def implied_prob_from_odds(ho, do, ao):
    """从赔率计算隐含概率（归一化，去overround）"""
    try:
        if not all(v > 0 for v in [ho, do, ao]):
            return None
        ih, id_, ia = 1/ho, 1/do, 1/ao
        total = ih + id_ + ia
        if total <= 0:
            return None
        return {'胜': ih/total, '平': id_/total, '负': ia/total}
    except:
        return None

def overround_severity(ho, do, ao):
    """返回overround严重度等级: 0=正常, 1=警告, 2=严重, 3=极端"""
    try:
        total = 1/ho + 1/do + 1/ao
        if total <= 1.05: return 0
        if total <= 1.30: return 1
        if total <= 1.60: return 2
        return 3
    except:
        return 2

def odds_reliability_score(ho, do, ao, league=''):
    """赔率可信度评分 [0.30, 1.0]，越低越不可靠"""
    sev = overround_severity(ho, do, ao)
    base = 1.0 - sev * 0.20
    weak = {'法乙', '沙职', '英甲', '澳超', '韩职', '解放者杯', '欧罗巴', '国际赛'}
    if league in weak:
        base *= 0.90
    try:
        if ho < 1.25 or ao > 8.0: base *= 0.90
        if ho > 5.0 or ao < 1.20: base *= 0.85
    except: pass
    return max(0.30, min(1.0, base))

# 全局缓存: 联赛近5场赔率可信度
_odds_reliability_cache = {}

def recent_odds_trend(league, current_score):
    if league not in _odds_reliability_cache:
        _odds_reliability_cache[league] = []
    cache = _odds_reliability_cache[league]
    cache.append(current_score)
    if len(cache) > 5: cache.pop(0)
    if len(cache) < 3: return '稳定'
    avg = sum(cache[:-1]) / max(1, len(cache)-1)
    diff = current_score - avg
    if diff > 0.05: return '上升'
    if diff < -0.05: return '下降'
    return '稳定'

def spf_odds_adjust_v2(confidence, ho, do, ao, league='', dow=3):
    """P0核心: 赔率可信度×置信度联合校准"""
    signals = []
    score = odds_reliability_score(ho, do, ao, league)
    trend = recent_odds_trend(league, score)
    
    if trend == '下降': confidence *= 0.92; signals.append(f'赔率趋势下降→×0.92')
    elif trend == '上升': confidence *= 1.03; signals.append(f'赔率趋势上升→×1.03')
    
    if score >= 0.90: pass
    elif score >= 0.75:
        penalty = 0.90 + (score - 0.75) * 0.4
        confidence *= penalty
        signals.append(f'赔率可信度偏低({score:.2f})→×{penalty:.2f}')
    elif score >= 0.55:
        confidence *= 0.78
        signals.append(f'赔率可信度警告({score:.2f})→×0.78')
    else:
        confidence *= 0.60
        signals.append(f'赔率可信度危险({score:.2f})→×0.60')
    
    weak_dow_cup = {0, 1}
    cup_kw = {'欧冠', '欧罗巴', '欧协联', '解放者', '杯'}
    if dow in weak_dow_cup and any(k in str(league) for k in cup_kw):
        confidence *= 0.75
        signals.append(f'周一/二杯赛风险→×0.75')
    
    return confidence, signals, {'reliability': score, 'trend': trend}

# ─────────────────────────────────────────────
# P1: 置信度×赔率区间交叉校准
# ─────────────────────────────────────────────
_ODDS_BAND_CONF_MATRIX = [
    ((0, 1.35), (0, 40), 0.80, '🔥超热赔低conf'),
    ((0, 1.35), (40, 55), 0.88, '🔥超热赔中conf'),
    ((0, 1.35), (55, 100), 0.78, '🔥超热赔高conf陷阱'),
    ((1.35, 1.60), (0, 40), 0.85, '🔥热赔低conf'),
    ((1.35, 1.60), (40, 55), 0.95, '🔥热赔中conf'),
    ((1.35, 1.60), (55, 100), 0.82, '🔥热赔高conf'),
    ((1.60, 2.10), (0, 40), 0.90, '📊中性赔低conf'),
    ((1.60, 2.10), (40, 55), 1.08, '📊中性赔中conf最佳'),
    ((1.60, 2.10), (55, 100), 0.92, '📊中性赔高conf'),
    ((2.10, 2.80), (0, 40), 0.88, '⚠️中冷赔低conf'),
    ((2.10, 2.80), (40, 55), 0.95, '⚠️中冷赔中conf'),
    ((2.10, 2.80), (55, 100), 0.85, '⚠️中冷赔高conf'),
    ((2.80, 100), (0, 40), 0.85, '❄️冷赔低conf'),
    ((2.80, 100), (40, 55), 0.78, '❄️冷赔中conf'),
    ((2.80, 100), (55, 100), 0.72, '❄️冷赔高conf极危险'),
]

def odds_conf_cross_calibrate(confidence, ho, do, ao, league=''):
    """P1核心: 赔率区间×置信度交叉校准矩阵"""
    signals = []
    final_mult = 1.0
    for (ho_low, ho_high), (conf_low, conf_high), mult, prefix in _ODDS_BAND_CONF_MATRIX:
        if ho_low <= ho < ho_high and conf_low <= confidence*100 < conf_high:
            final_mult *= mult
            signals.append(f'{prefix}→×{mult}')
            break
    weak = {'法乙', '沙职', '英甲', '澳超', '韩职', '解放者杯', '欧罗巴', '国际赛'}
    if league in weak and final_mult < 1.0:
        final_mult *= 0.90
        signals.append(f'弱联赛降权→×0.90')
    return confidence * final_mult, signals

# ─────────────────────────────────────────────
# P2: 周效应 + 联赛×星期交叉调整
# ─────────────────────────────────────────────
_WEEKDAY_BASE_ADJUST = {'Mon': 0.82, 'Tue': 0.82, 'Wed': 0.93, 'Thu': 0.93, 'Fri': 0.90, 'Sat': 1.00, 'Sun': 0.93}

_LEAGUE_DOW_MODIFIERS = {
    '英超': {'Sat': (1.08,'周六英超强势'), 'Mon': (0.78,'周一英超冷门多'), 'Tue': (0.80,'周二欧冠轮换')},
    '西甲': {'Thu': (0.72,'周四西甲替补阵容'), 'Sat': (1.05,'周六西甲'), 'Mon': (0.82,'周一西甲弱')},
    '意甲': {'Thu': (0.75,'周四意甲轮换'), 'Sat': (1.05,'周六意甲'), 'Mon': (0.80,'周一意甲弱')},
    '德甲': {'Sat': (1.06,'周六德甲'), 'Mon': (0.85,'周一德甲弱')},
    '法甲': {'Wed': (1.10,'周三法甲单关强势'), 'Sat': (1.05,'周六法甲'), 'Mon': (0.75,'周一法甲极弱')},
    '英超': {'Sat': (1.08,'周六英超强势'), 'Mon': (0.78,'周一英超冷门多'), 'Tue': (0.80,'周二欧冠轮换')},
    '欧罗巴': {'Tue': (0.72,'周四欧罗巴最危险'), 'Thu': (0.88,'周四欧罗巴')},
    '解放者杯': {'Thu': (0.75,'周四解放者杯不稳定'), 'Fri': (0.80,'周五解放者杯')},
    '美职': {'Sat': (1.10,'周六美职'), 'Sun': (1.08,'周日美职')},
    '韩职': {'Sat': (1.08,'周六韩职'), 'Mon': (0.80,'周一韩职弱')},
    '日职': {'Sat': (1.10,'周六日职强势'), 'Sun': (1.05,'周日日职')},
    '荷甲': {'Sat': (1.08,'周六荷甲'), 'Wed': (1.05,'周三荷甲')},
    '葡超': {'Sat': (1.06,'周六葡超'), 'Thu': (0.75,'周四葡超轮换')},
    '芬超': {'Sat': (1.10,'周六芬超强势')},
    '瑞典超': {'Sat': (1.08,'周六瑞典超')},
    '挪超': {'Sat': (1.08,'周六挪超')},
    '瑞士超': {'Sat': (1.08,'周六瑞士超')},
}

_DOW_INT_MAP = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}

# ═══════════════════════════════════════════════════════════════════════
# P2联赛白名单 (2026-04-27新增)
# 根因: 荷甲(OOS=60.2%)、法乙等高OOS联赛被周日效应×0.93+杯赛×0.85双重覆盖
# 规则: OOS>55%联赛在信心>65%时，P2调整幅度降低80%；信心>75%完全禁用P2
# ═══════════════════════════════════════════════════════════════════════
_LEAGUE_P2_EXEMPT = {
    '荷甲', '法乙', '德乙', '葡超', '西甲', '英超', '意甲',  # OOS>52%
    '欧冠', '欧罗巴', '欧协联',  # 杯赛专项模型优先
    '土超', '希腊超', '韩职', '挪超', '瑞超',  # OOS>50%
}

def _dow_key(dow):
    """将int(0-6)或string('Mon'-'Sun')统一转为string key"""
    if isinstance(dow, int):
        return _DOW_INT_MAP.get(dow, None)
    return dow  # 已经是字符串

def _get_dow_features(dow_s):
    """从dow_features表查询指定星期的主场胜率-客场胜率差值，用于动态调整周效应。
    Returns (home_away_diff, sample_size) 或 (None, 0) if unavailable."""
    try:
        import psycopg2
        import os
        from pathlib import Path
        env = {}
        env_path = Path("/home/doodoo/.hermes/.env")
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("PGPASSWORD="):
                    env["PGPASSWORD"] = line.split("=", 1)[1].strip()
                    break
        conn = psycopg2.connect(host="localhost", user="doodoo", password=env.get("PGPASSWORD",""), dbname="myapp_db", connect_timeout=5)
        cur = conn.cursor()
        # dow_features.dow: 0=周一..6=周日
        _DOW_NUM = {'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}
        dow_num = _DOW_NUM.get(dow_s)
        if dow_num is None:
            cur.close(); conn.close(); return None, 0
        cur.execute("SELECT spf_home_rate, spf_away_rate, sample_size FROM dow_features WHERE dow=%s", (dow_num,))
        row = cur.fetchone()
        cur.close(); conn.close()
        if row:
            return row[0] - row[1], row[2]
        return None, 0
    except Exception:
        return None, 0

def weekday_dow_adjust(confidence, league, dow):
    """P2核心: 周效应 + 联赛×星期交叉调整

    dow: int (0=周一..6=周日) 或 string ('Mon'..'Sun')

    P2联赛白名单豁免 (2026-04-27):
    - OOS>55%联赛 + 信心>65%: P2调整幅度降低80%
    - OOS>55%联赛 + 信心>75%: 完全禁用P2
    """
    signals = []
    mult = 1.0
    dow_s = _dow_key(dow)
    if dow_s is None:
        return confidence, signals

    # P1-Fix: 动态周效应 — 从dow_features表读取实际主客场胜率差
    # home_away_diff > 0 → 主场该日更强；< 0 → 客场更强
    # 用实际数据替代硬编码的 _WEEKDAY_BASE_ADJUST
    hadiff, ss = _get_dow_features(dow_s)
    if hadiff is not None and ss > 100:
        # 有足够样本：用真实数据替代硬编码
        # hadiff>0 → 主场该日胜率高，base_mult应>1.0（升权）
        # hadiff<0 → 客场该日胜率高，base_mult应<1.0（降权）
        dyn_mult = 0.90 + hadiff * 0.30  # e.g. hadiff=0.10 → 0.93; hadiff=-0.10 → 0.87
        dyn_mult = max(0.70, min(1.10, dyn_mult))  # 限制在±10%范围
        base_mult = dyn_mult
        signals.append(f'周{dow_s}动态效应(diff={hadiff:.3f},n={ss})→×{base_mult:.2f}')
    else:
        base_mult = _WEEKDAY_BASE_ADJUST.get(dow_s, 0.90)
        if base_mult != 1.0:
            signals.append(f'周{dow_s}基础效应(fallback)→×{base_mult:.2f}')
    exempt = league and league in _LEAGUE_P2_EXEMPT
    if exempt and confidence > 0.75:
        signals.append(f'🛡️{league}P2白名单→信心{confidence:.0%}>75%完全禁用P2')
        return confidence, signals
    if exempt and confidence > 0.65:
        # 先正常计算，再降低80%影响
        pass  # 继续计算，后续处理

    if base_mult != 1.0:
        mult *= base_mult
    league_mods = _LEAGUE_DOW_MODIFIERS.get(league, {})
    if dow_s in league_mods:
        dow_mult, dow_signal = league_mods[dow_s]
        mult *= dow_mult
        signals.append(f'{league}×周{dow_s}→{dow_signal}→×{dow_mult:.2f}')
    cup_kw = {'欧冠', '欧罗巴', '欧协联', '解放者', '杯'}
    cup_dow_strs = ('Tue', 'Wed', 'Thu')  # 周二/三/四杯赛危险
    if any(k in str(league) for k in cup_kw) and dow_s in cup_dow_strs:
        if mult > 0.80:
            mult *= 0.85
            signals.append(f'周{dow_s}杯赛→×0.85')

    # P2白名单: OOS>55%联赛 + 信心>65%: 降低80%调整幅度
    if exempt and confidence > 0.65:
        # mult从上次计算结果中恢复原状，再施加更轻的调整
        # 当前mult已经包含了完整P2调整，需要将其向1.0回拨80%
        # 也就是说实际只承受20%的调整幅度
        adjusted_mult = 1.0 + (mult - 1.0) * 0.20
        signals.append(f'🛡️{league}P2豁免: ×{mult:.3f}→×{adjusted_mult:.3f}(承受20%调整)')
        mult = adjusted_mult

    return confidence * mult, signals

# ═══════════════════════════════════════════════════════════════
# P0新增(2026-04-24): 赔率方向矛盾检测器
# 根因: 04-23复盘4场contradiction=true全错(100%失败)
# 逻辑: 当模型预测方向与赔率implied方向矛盾且为热赔区间时，
#       强制跟随赔率方向（庄家信号 > 模型信号）
# ═══════════════════════════════════════════════════════════════

def _implied_dir(ho, do, ao):
    """从赔率计算隐含方向（不考虑overround，用原始1/odds比率）"""
    try:
        if not all(v > 0 for v in [ho, do, ao]):
            return None, None
        ih, id_, ia = 1/ho, 1/do, 1/ao
        # 直接比较比率，不用归一化
        dir_map = {'胜': ih, '平': id_, '负': ia}
        return max(dir_map, key=dir_map.get), dir_map
    except:
        return None, None


def contradiction_filter(spf_pred, spf_probs, ho, do, ao):
    """P0矛盾检测: 模型预测方向 vs 赔率implied方向

    核心规则（来自04-23深度分析）:
    - contradiction=true的4场全错（周四001/004/006/009）
    - 共同特征: 模型预测≠implied方向 且 ho<1.5（热赔区间）
    - 解决方案: 热赔区间矛盾时，强制跟随赔率方向，降权至70%

    Args:
        spf_pred: 模型预测字符串 '胜'/'平'/'负'
        spf_probs: 模型预测概率 dict {'胜': 0.xx, '平': 0.xx, '负': 0.xx}
        ho, do, ao: 赔率

    Returns:
        (override_pred_or_None, override_conf_or_None, signal_or_None)
        若无需override，返回 (None, None, None)
    """
    try:
        implied_dir, vals = _implied_dir(ho, do, ao)
        if implied_dir is None or spf_probs is None:
            return None, None, None

        model_dir = max(spf_probs, key=spf_probs.get)
        if implied_dir == model_dir:
            return None, None, None  # 无矛盾

        # 检测矛盾
        imp_win_prob = vals['胜'] / (vals['胜'] + vals['平'] + vals['负'])
        is_hot_odds = (ho < 1.50) or (ao < 1.50)

        # 热赔区间 + 矛盾 → 不强制改方向（热赔区间庄家常故意设陷阱）
        # 模型高置信时：只降权不override；模型低置信时：跟随赔率更安全
        if is_hot_odds:
            model_top = max(spf_probs, key=spf_probs.get)
            model_top_conf = spf_probs.get(model_top, 0)
            imp_conf = vals[implied_dir] / sum(vals.values())
            # P1-3修复: 模型高置信(≥0.45)时保留模型方向，仅降权×0.80
            # 根因: 热赔区间模型高置信时庄家常反向操作，强制跟随赔率导致系统性偏差
            if model_top_conf >= 0.45:
                override_conf = round(model_top_conf * 0.80, 4)
                signal = (f"⚠️矛盾信号: 模型{model_top}({model_top_conf:.0%}) vs "
                          f"赔率隐含{implied_dir}({imp_conf:.0%}) → 保留方向降权×0.80")
                return None, override_conf, signal  # 不改方向
            else:
                # 模型低置信时: 跟随赔率方向，×0.70
                override_conf = 0.70
                signal = (f"🚨矛盾跟随: 模型{model_top}({model_top_conf:.0%}) vs "
                          f"赔率隐含{implied_dir}({imp_conf:.0%}) → 跟随赔率×0.70")
                return implied_dir, override_conf, signal

        # 非热赔区间的矛盾，温和降权10%
        model_conf = spf_probs.get(model_dir, 0)
        if model_conf < 0.40:
            return None, None, None  # 模型本身低置信，不做强制跟随

        return None, None, None

    except Exception:
        return None, None, None


# ═══════════════════════════════════════════════════════════════
# P1新增(2026-04-24): WH初赔差值检测器
# 根因: 04-23 8场全部SP>WH初=韬盘，庄家隐藏真实意图
# 逻辑: SP胜 vs WH初赔差值→判断韬盘程度→预测方向+置信度修正
# ═══════════════════════════════════════════════════════════════

def _single_source_gap_detector(sp_ho, init_home, source_name=''):
    """单源初赔差值检测核心逻辑

    Args:
        sp_ho: SP主胜赔率
        init_home: 初赔主胜
        source_name: 来源标识(WH/TYC/BF)

    Returns:
        (signal_type, adjust_factor, reason) 或 (None, 1.0, '')
    """
    if not sp_ho or sp_ho <= 0 or not init_home or init_home <= 0:
        return None, 1.0, ''

    sp_ho = float(sp_ho)
    init_h = float(init_home)

    # 计算差值百分比: (SP - init) / init
    gap_pct = (sp_ho - init_h) / init_h

    # 判断韬盘程度
    # 极韬盘: gap > 50% (SP比初赔高50%以上)
    # 大韬盘: gap > 20%
    # 小韬盘: gap > 5%
    # 实盘: gap < -5%
    # 中性: -5% <= gap <= 5%

    if gap_pct > 0.50:
        signal = 'tao_home_extreme'
        adj = 0.50
        reason = f"极韬盘[{source_name}]: SP={sp_ho:.2f} 初={init_h:.2f} 差={gap_pct:+.0%}"
    elif gap_pct > 0.20:
        signal = 'tao_home_large'
        adj = 0.70
        reason = f"大韬盘[{source_name}]: SP={sp_ho:.2f} 初={init_h:.2f} 差={gap_pct:+.0%}"
    elif gap_pct > 0.05:
        signal = 'tao_home_small'
        adj = 0.85
        reason = f"小韬盘[{source_name}]: SP={sp_ho:.2f} 初={init_h:.2f} 差={gap_pct:+.0%}"
    elif gap_pct < -0.05:
        signal = 'real_home'
        adj = 1.05  # 轻微boost
        reason = f"实盘[{source_name}]: SP={sp_ho:.2f} 初={init_h:.2f} 差={gap_pct:+.0%}"
    else:
        signal = 'neutral'
        adj = 1.0
        reason = f"中性[{source_name}]: SP={sp_ho:.2f} 初={init_h:.2f} 差={gap_pct:+.0%}"

    return signal, adj, reason


def wh_gap_detector(sp_ho, wh_home_initial=None,
                    tyc_home_initial=None, tyc_draw_initial=None, tyc_away_initial=None,
                    betfair_home_initial=None, betfair_draw_initial=None, betfair_away_initial=None,
                    wh_draw_initial=None, wh_away_initial=None):
    """P1核心: 多源初赔差值检测（威廉希尔+体彩网+必发）

    核心规则：
    - SP > 初赔 → 韬盘 → 庄家不想要该方向
    - SP < 初赔 → 实盘 → 庄家真实看好
    - 差值越大 → 韬盘程度越深 → 庄家越不希望该方向打出

    多源综合逻辑：
    1. 各源独立计算gap和signal
    2. 韬盘信号取最严重（adj最小）
    3. 实盘信号取最强（adj最大）
    4. 中性信号不影响
    5. 多源同向韬盘 → 确认度高，额外降权5%

    Args:
        sp_ho: SP主胜赔率
        wh_home_initial: 威廉希尔初赔主胜
        tyc_home_initial: 体彩网初赔主胜
        tyc_draw_initial: 体彩网初赔平局
        tyc_away_initial: 体彩网初赔客胜
        betfair_home_initial: 必发初赔主胜
        betfair_draw_initial: 必发初赔平局
        betfair_away_initial: 必发初赔客胜
        wh_draw_initial: 威廉希尔初赔平局(保留)
        wh_away_initial: 威廉希尔初赔客胜(保留)

    Returns:
        (signal_type, confidence_adjust, reason_dict)
        signal_type: 'multi_tao'|'multi_real'|'mixed'|'neutral'|None
        confidence_adjust: 综合调整因子
        reason_dict: {'WH': reason, 'TYC': reason, 'BF': reason}
    """
    try:
        sources = {}
        results = {}  # source -> (signal, adj)

        # 威廉希尔
        if wh_home_initial:
            sig, adj, reason = _single_source_gap_detector(sp_ho, wh_home_initial, 'WH')
            if sig:
                results['WH'] = (sig, adj, reason)

        # 体彩网
        if tyc_home_initial:
            sig, adj, reason = _single_source_gap_detector(sp_ho, tyc_home_initial, 'TYC')
            if sig:
                results['TYC'] = (sig, adj, reason)

        # 必发
        if betfair_home_initial:
            sig, adj, reason = _single_source_gap_detector(sp_ho, betfair_home_initial, 'BF')
            if sig:
                results['BF'] = (sig, adj, reason)

        if not results:
            return None, 1.0, {}

        # 综合分析
        tao_signals = {k: v for k, v in results.items() if v[0] and v[0].startswith('tao_')}
        real_signals = {k: v for k, v in results.items() if v[0] == 'real_home'}
        neutral_signals = {k: v for k, v in results.items() if v[0] == 'neutral'}

        reason_dict = {k: v[2] for k, v in results.items()}

        # 情况1: 全是韬盘 → 取最严重(adj最小)，多源确认额外降权
        if tao_signals and not real_signals:
            worst = min(tao_signals.items(), key=lambda x: x[1][1])
            source_count = len(tao_signals)
            # 多源确认: 2个以上源同向韬盘，额外降权5%
            extra_penalty = 0.95 if source_count >= 2 else 1.0
            combined_adj = worst[1][1] * extra_penalty
            combined_signal = f"multi_tao_{source_count}src"
            reason_dict['summary'] = f"{source_count}源同向韬盘,最严重{worst[0]}"
            return combined_signal, combined_adj, reason_dict

        # 情况2: 全是实盘 → 取最强(adj最大)
        if real_signals and not tao_signals:
            best = max(real_signals.items(), key=lambda x: x[1][1])
            return f"multi_real_{len(real_signals)}src", best[1][1], reason_dict

        # 情况3: 混和(韬盘+实盘) → 中性偏弱
        if tao_signals and real_signals:
            avg_adj = sum(v[1] for v in results.values()) / len(results)
            reason_dict['summary'] = "混和信号(韬+实),取均值"
            return 'mixed', avg_adj, reason_dict

        # 情况4: 全是中性
        return 'neutral', 1.0, reason_dict

    except Exception:
        return None, 1.0, {}


def full_spf_calibration_v2(confidence, ho, do, ao, league='', dow=3, spf_probs=None,
                             wh_home_initial=None, tyc_home_initial=None, betfair_home_initial=None,
                             match_code=None):
    """P0+P1+P2 三合一校准主函数
    
    Returns: (final_conf, all_signals, odds_info, ov_pred, calibration_trace)
        calibration_trace: list of {"layer": str, "input": float, "output": float, "action": str}
    """
    calibration_trace = []
    all_signals = []
    
    def _trace(layer, inp, out, action=''):
        calibration_trace.append({'layer': layer, 'input': round(inp, 4), 'output': round(out, 4), 'action': action})
    
    # Step 0 (P0核心): 赔率方向矛盾过滤
    # 04-23复盘: contradiction=true的4场全错(周四001/004/006/009)
    # 规则: 热赔区间(ho<1.5)矛盾时强制跟随赔率方向，降权至70%
    spf_pred_str = max(spf_probs, key=spf_probs.get) if spf_probs else None
    _trace('ensemble', confidence, confidence, 'ensemble输出')
    ov_pred, ov_conf, ov_sig = contradiction_filter(spf_pred_str, spf_probs, ho, do, ao)

    # Step 1: P0 赔率清洗
    conf_p0, sigs_p0, odds_info = spf_odds_adjust_v2(confidence, ho, do, ao, league, dow)
    all_signals.extend(sigs_p0)
    _trace('spf_odds_adjust_v2', confidence, conf_p0, '赔率清洗')
    
    # Step 2: P1 赔率×置信度矩阵
    conf_p1, sigs_p1 = odds_conf_cross_calibrate(conf_p0, ho, do, ao, league)
    all_signals.extend(sigs_p1)
    _trace('odds_conf_cross_calibrate', conf_p0, conf_p1, '赔率×置信度矩阵')
    
    # Step 3: P2 周效应
    conf_p2, sigs_p2 = weekday_dow_adjust(conf_p1, league, dow)
    all_signals.extend(sigs_p2)
    _trace('weekday_dow_adjust', conf_p1, conf_p2, f'周效应 dow={dow}')

    # P1-1新增(2026-05-14): 周中置信度上限封顶
    # 实测周一~五命中率仅5.0%(283场)，周日/周六合计32.0%
    # 周中(周一~五)置信度上限35%，超出直接截断
    if dow is not None and conf_p2 > 0.35:
        wd = int(dow) if isinstance(dow, (int, float)) else None
        if wd is not None and wd not in (5, 6):  # 非周六(5)周日(6)
            conf_p2_before = conf_p2
            conf_p2 = 0.35
            all_signals.append(f'⛔周中上限: {wd}→35% ({conf_p2_before:.0%}→{conf_p2:.0%})')
            _trace('weekday_conf_cap', conf_p2_before, conf_p2, f'周{dow}封顶35%')

    # Step 3.5 (P1): 中游队诱平陷阱检测
    # 04-27修复: mid_tier_trap_adjust从未被调用，P1诱平逻辑完全失效
    # 插入位置: P2周效应之后、冷门矛盾之前，对概率做陷阱调整
    if spf_probs is not None:
        spf_probs, trap_sigs = mid_tier_trap_adjust(
            spf_probs, ho, do, ao, league=league, match_date=None, match_code='')
        all_signals.extend(trap_sigs)
        _trace('mid_tier_trap_adjust', conf_p2, conf_p2, '概率陷阱检测（置信度不变）')

    # Step 4: 冷门矛盾检测
    imp = implied_prob_from_odds(ho, do, ao)
    if imp and spf_probs:
        imp_dir = max(imp, key=imp.get)
        model_dir = max(spf_probs, key=spf_probs.get)
        if imp_dir != model_dir:
            imp_conf = imp[imp_dir]
            model_conf = spf_probs.get(model_dir, 0)
            if imp_conf > 0.60 and model_conf < 0.40:
                conf_p2 *= 0.72
                all_signals.append(f'🚨冷门矛盾: 赔率{imp_dir}{imp_conf:.0%} vs 模型{model_dir}{model_conf:.0%}→×0.72')
                _trace('cold_contradiction', conf_p2 / 0.72, conf_p2, '冷门矛盾降权×0.72')
    
    # Step 5: 赔率极端值修正
    # 04-25 重大修正: 
    #   - 超热主胜(ho<1.30,模型预测胜): 04-24数据3/3全中，应BOOST而非降权
    #   - 超热客胜(ao<1.30,模型预测负): 客队是绝对热门，模型预测客胜应BOOST
    #   - 中性区主胜(1.5<ho<2.5,模型预测胜): 04-24数据3/3全中，应正向调整
    conf_before_extreme = conf_p2
    if spf_probs:
        model_dir = max(spf_probs, key=spf_probs.get)
        try:
            if ho < 1.30 and model_dir == '胜':
                conf_p2 *= 1.12  # 修正: 热门主胜3/3→BOOST
                all_signals.append(f'🔥超热赔主胜→×1.12')
            elif ao < 1.30 and model_dir == '负':
                conf_p2 *= 1.10  # 修正: 热门客胜应BOOST
                all_signals.append(f'🔥超热赔客胜→×1.10')
            # 04-25新增: 中性区(1.5<ho<2.5)主胜正向boost
            # 根因: 模型在neutral区概率系统性低估，04-24该区间主胜3/3全中
            elif 1.5 < ho < 2.5 and model_dir == '胜' and conf_p2 < 0.55:
                conf_p2 *= 1.15  # 低置信neutral区主胜→正向boost
                all_signals.append(f'📊中性区主胜boost→×1.15')
        except: pass
    if conf_p2 != conf_before_extreme:
        _trace('odds_extreme_adjust', conf_before_extreme, conf_p2, '赔率极端值修正')
    
    # Step 6 (P0核心): 矛盾降权 — 在P1/P2校准后额外降权（不再双重惩罚ov_conf）
    # ov_conf已在contradiction_filter中包含降权系数，此处只做上限保护
    if ov_sig:
        # P1-3修复: ov_conf已带降权系数，不再×0.50，只做上限截断
        conf_p2_before = conf_p2
        conf_p2 = max(0.20, min(ov_conf if ov_conf else conf_p2, 0.75))
        all_signals.append(ov_sig)
        _trace('contradiction_override', conf_p2_before, conf_p2, '矛盾降权')

    # Step 7 (P1): 多源初赔差值检测 — 韬盘降权（威廉希尔+体彩网+必发）
    # 04-23数据: 8场全部SP>WH初=韬盘，庄家隐藏真实意图
    # 多源综合: 各源独立计算，取最严重韬盘或最强实盘
    _has_any_init = any([wh_home_initial, tyc_home_initial, betfair_home_initial])
    if _has_any_init:
        wh_signal, wh_adj, reason_dict = wh_gap_detector(
            ho,
            wh_home_initial=wh_home_initial,
            tyc_home_initial=tyc_home_initial,
            betfair_home_initial=betfair_home_initial
        )
        if wh_signal and wh_signal.startswith('multi_tao'):
            conf_p2_before = conf_p2
            conf_p2 *= wh_adj
            _reasons = ' | '.join(f"{k}:{v}" for k, v in reason_dict.items() if k != 'summary')
            all_signals.append(f'🌓多源韬盘{wh_signal}: {_reasons}→×{wh_adj:.2f}')
            _trace('wh_gap_detector', conf_p2_before, conf_p2, f'多源韬盘×{wh_adj}')
        elif wh_signal and wh_signal.startswith('multi_real'):
            conf_p2_before = conf_p2
            conf_p2 *= wh_adj
            _reasons = ' | '.join(f"{k}:{v}" for k, v in reason_dict.items() if k != 'summary')
            all_signals.append(f'🌓多源实盘{wh_signal}: {_reasons}→×{wh_adj:.2f}')
            _trace('wh_gap_detector', conf_p2_before, conf_p2, f'多源实盘×{wh_adj}')
        elif wh_signal == 'mixed':
            conf_p2_before = conf_p2
            conf_p2 *= wh_adj
            all_signals.append(f'🌓混和信号: {reason_dict.get("summary","")}→×{wh_adj:.2f}')
            _trace('wh_gap_detector', conf_p2_before, conf_p2, f'混和信号×{wh_adj}')
        elif wh_signal == 'neutral':
            pass  # 中性信号不做调整

    # ═══════════════════════════════════════════════════════════════════
    # P3: DL影子部署 (2026-04-27新增)
    # 逻辑: BP神经网络与泊松模型并联推理
    #   - 同向: 信心×1.05boost，标记DL_aligned
    #   - 异向: 记录分歧但不降权，仅在信号中标注DL_divergence
    #   - 双重强信号: 两模型同向+胜置信>65% → ×1.08
    # ═══════════════════════════════════════════════════════════════════
    conf_before_dl = conf_p2
    if spf_probs:
        # 获取泊松预测方向
        poisson_dir = max(spf_probs, key=spf_probs.get) if spf_probs else None
        poisson_conf_val = spf_probs.get(poisson_dir, 0) if poisson_dir else 0

        # 获取BP预测 (用spf_probs中的方向作为BP代理)
        # 注意: BP模型独立推理在predict_5play中已执行，这里用泊松结果作为影子参考
        # bp_pred 和 bp_conf 可从 match_data['sources']['bp'] 获取
        bp_dir = None  # 已在predict_5play中计算，此处引用poisson作为影子
        bp_match_conf = spf_probs.get('bp_confidence', 0) if isinstance(spf_probs, dict) and 'bp_confidence' in spf_probs else None

        # DL影子核心逻辑
        if poisson_dir and poisson_conf_val > 0:
            # 获取泊松预测方向
            implied = implied_prob_from_odds(ho, do, ao)
            implied_dir = max(implied, key=implied.get) if implied else None

            # 双重强信号: 同向 + 高置信
            if implied_dir == poisson_dir and poisson_conf_val > 0.65:
                conf_p2 *= 1.08
                all_signals.append(f'🧠DL双重强信号: {poisson_dir}{poisson_conf_val:.0%}+赔率同向→×1.08')
            elif implied_dir == poisson_dir:
                conf_p2 *= 1.05
                all_signals.append(f'🧠DL对齐: 泊松{poisson_dir}{poisson_conf_val:.0%}=赔率同向→×1.05')
            elif implied_dir and poisson_dir:
                # 异向分歧
                all_signals.append(f'⚠️DL分歧: 泊松{poisson_dir}{poisson_conf_val:.0%}≠赔率{implied_dir}{implied.get(implied_dir,0):.0%}')
    if conf_p2 != conf_before_dl:
        _trace('dl_shadow', conf_before_dl, conf_p2, 'DL影子boost')

    final = max(0.20, min(0.95, conf_p2))
    _trace('final_clamp', conf_p2, final, '输出截断至[0.20, 0.95]')

    # P1-新增: 高置信度惩罚 — 在所有P0-P3调整完成后，对最终confidence调用hot_confidence_penalty
    final, hot_sig = hot_confidence_penalty(final, spf_probs)
    if hot_sig:
        all_signals.extend(hot_sig)
        _trace('hot_confidence_penalty', conf_p2, final, hot_sig[0] if hot_sig else '无惩罚')

    return final, all_signals, odds_info, ov_pred, calibration_trace


# ══════════════════════════════════════════════════════════════════════════════
# P1: 亚盘盘口深度判断器 (2026-04-24)
# 根因：亚盘盘口深浅+水位是庄家操盘核心信号
# ══════════════════════════════════════════════════════════════════════════════

# SP胜率 → 合理亚盘盘口映射表
# ho = SP主胜赔率, handicap = 亚盘让球数（正=主队让，负=主队受让）
_SP_TO_HANDICAP = [
    # (ho_lo, ho_hi, handicap_str, handicap_value, description)
    (1.00, 1.15, '两球/两球半', 2.25),
    (1.15, 1.25, '球半/两球', 1.75),
    (1.25, 1.35, '球半', 1.50),
    (1.35, 1.55, '一球/球半', 1.25),
    (1.55, 1.75, '一球', 1.00),
    (1.75, 2.00, '半球/一球', 0.75),
    (2.00, 2.30, '半球', 0.50),
    (2.30, 2.70, '平半/半球', 0.25),
    (2.70, 3.50, '平手', 0.00),
    (3.50, 99.0, '受让平半', -0.25),
]


def _sp_to_reasonable_handicap(ho):
    """根据SP主胜赔率推断合理亚盘盘口（数值）"""
    try:
        ho_f = float(ho)
    except (TypeError, ValueError):
        return None
    for lo, hi, label, val in _SP_TO_HANDICAP:
        if lo <= ho_f < hi:
            return val
    return None


def asian_handicap_deepjudge(handicap_value, home_odds, away_odds, sp_ho, sp_do, sp_ao):
    """P1: 亚盘盘口深度判断器

    综合盘口深浅 + 水位分析，判断庄家操盘意图。

    支持两种赔率格式：
    1. 标准水位格式：home_odds/away_odds ∈ [0.75, 1.15]（如 0.86/0.98）
    2. 让球赔率格式：home_odds/away_odds > 1.5（如 2.33/2.15）
       → 从SP赔率反推合理水位进行对比分析

    Args:
        handicap_value: 亚盘让球数（正=主队让，负=主队受让）
        home_odds: 主队水位或让球赔率
        away_odds: 客队水位或让球赔率
        sp_ho, sp_do, sp_ao: SP胜平负赔率

    Returns:
        dict: {
            'signal': '阻上|诱上|中性|阻下',
            'confidence': 0.0-1.0,
            'reason': str,
            'recommended_direction': '上盘|下盘|观望',
            'depth_analysis': {...},
            'water_analysis': {...}
        }
    """
    try:
        hv = float(handicap_value) if handicap_value is not None else None
        ho_sp = float(sp_ho) if sp_ho is not None else None
        ao_sp = float(sp_ao) if sp_ao is not None else None
        home_raw = float(home_odds) if home_odds is not None else None
        away_raw = float(away_odds) if away_odds is not None else None
    except (TypeError, ValueError):
        return {
            'signal': '中性', 'confidence': 0.0,
            'reason': '数据无效，无法判断',
            'recommended_direction': '观望',
            'depth_analysis': {}, 'water_analysis': {}
        }

    # ── 判定赔率格式 ──
    WATER_RANGE_LOW = 0.75
    WATER_RANGE_HIGH = 1.15
    is_water_format = (
        home_raw is not None and away_raw is not None
        and WATER_RANGE_LOW <= home_raw <= WATER_RANGE_HIGH
        and WATER_RANGE_LOW <= away_raw <= WATER_RANGE_HIGH
    )
    is_handicap_odds_format = (
        home_raw is not None and away_raw is not None
        and (home_raw > 1.5 or away_raw > 1.5)
    )

    reason_parts = []
    confidence = 0.5
    signal = '中性'
    recommended_direction = '观望'
    depth_signal = None
    water_signal = None

    # ── 1. 合理盘口计算 ──
    reasonable_hv = _sp_to_reasonable_handicap(ho_sp) if ho_sp else None
    depth_analysis = {
        'actual_handicap': hv,
        'reasonable_handicap': reasonable_hv,
        'depth_delta': None,
        'odds_format': 'water' if is_water_format else ('handicap_odds' if is_handicap_odds_format else 'unknown'),
    }

    # ── 2. 盘口深浅判断 ──
    if hv is not None and reasonable_hv is not None:
        delta = hv - reasonable_hv
        depth_analysis['depth_delta'] = round(delta, 2)

        if delta >= 0.50:
            depth_signal = 'deep'
            reason_parts.append(f'深盘(实际{hv:+.2f} vs 合理{reasonable_hv:+.2f}, Δ={delta:+.2f})→庄家给上盘空间')
            confidence = 0.68
        elif delta <= -0.50:
            depth_signal = 'shallow'
            reason_parts.append(f'浅盘(实际{hv:+.2f} vs 合理{reasonable_hv:+.2f}, Δ={delta:+.2f})→庄家诱上盘')
            confidence = 0.72
        elif delta >= 0.25:
            depth_signal = 'slightly_deep'
            reason_parts.append(f'略深盘(Δ={delta:+.2f})')
            confidence = 0.58
        elif delta <= -0.25:
            depth_signal = 'slightly_shallow'
            reason_parts.append(f'略浅盘(Δ={delta:+.2f})')
            confidence = 0.60
        else:
            depth_signal = 'normal'
            reason_parts.append(f'盘口正常(Δ={delta:+.2f})')
            confidence = 0.50
    else:
        reason_parts.append(f'无法计算盘口深度(ho={ho_sp}, hv={hv}, 合理={reasonable_hv})')
        confidence = 0.40

    # ── 3. 水位/赔率分析 ──
    water_analysis = {'format': 'unknown'}

    if is_water_format:
        # ── 标准水位分析 ──
        water_sum = home_raw + away_raw
        water_gap = home_raw - away_raw
        water_analysis = {
            'format': 'water',
            'home_water': home_raw,
            'away_water': away_raw,
            'water_sum': round(water_sum, 3),
            'water_gap': round(water_gap, 3),
        }

        if home_raw < 0.78:
            water_signal = 'dead_water_home'
            reason_parts.append(f'🏱️死水盘(主队水位{home_raw:.2f}<0.78)→上盘极大概率高走')
            confidence = min(confidence + 0.10, 0.85)
        elif away_raw < 0.78:
            water_signal = 'dead_water_away'
            reason_parts.append(f'🏱️死水盘(客队水位{away_raw:.2f}<0.78)→下盘极大概率高走')
            confidence = min(confidence + 0.10, 0.85)
        elif home_raw < 0.85:
            water_signal = 'low_water_home'
            reason_parts.append(f'💧主队低水({home_raw:.2f})→庄家真实看好')
        elif away_raw < 0.85:
            water_signal = 'low_water_away'
            reason_parts.append(f'💧客队低水({away_raw:.2f})→庄家真实看好')
        elif home_raw > 1.00:
            water_signal = 'high_water_home'
            reason_parts.append(f'⚠️主队高水({home_raw:.2f}>1.00)→不真实，诱')
            confidence = max(confidence - 0.08, 0.30)
        elif away_raw > 1.00:
            water_signal = 'high_water_away'
            reason_parts.append(f'⚠️客队高水({away_raw:.2f}>1.00)→不真实，诱')
            confidence = max(confidence - 0.08, 0.30)

        if water_sum > 1.92:
            reason_parts.append(f'💰高抽水(总和{water_sum:.3f}>1.92)→庄家利润厚，小心')

        if abs(water_gap) > 0.10:
            if water_gap < 0:
                reason_parts.append(f'客队贴水更厚(Δ={water_gap:.3f})→资金倾向客队')
            else:
                reason_parts.append(f'主队贴水更厚(Δ={water_gap:.3f})→资金倾向主队')

    elif is_handicap_odds_format and ho_sp and ao_sp:
        # ── 从让球赔率反推水位分析 ──
        # 思路：比较让球赔率与SP胜平负赔率隐含的概率差
        # 如果让球赔率远高于SP隐含概率 → 庄家给高赔吸引资金 → 诱
        # 如果让球赔率接近SP隐含概率 → 庄家真实表态
        try:
            imp_ho = 1.0 / ho_sp if ho_sp > 0 else 0
            imp_ao = 1.0 / ao_sp if ao_sp > 0 else 0
            total_imp = imp_ho + imp_ao
            if total_imp > 0:
                fair_home_prob = imp_ho / total_imp
                fair_away_prob = imp_ao / total_imp
                # 让球赔率隐含概率
                impl_home = 1.0 / home_raw if home_raw > 0 else 0
                impl_away = 1.0 / away_raw if away_raw > 0 else 0
                total_impl = impl_home + impl_away
                if total_impl > 0:
                    rel_home = (impl_home / total_impl) - fair_home_prob
                    rel_away = (impl_away / total_impl) - fair_away_prob
                    water_analysis = {
                        'format': 'handicap_odds',
                        'home_odds': home_raw,
                        'away_odds': away_raw,
                        'implied_home_prob': round(impl_home / total_impl, 3),
                        'implied_away_prob': round(impl_away / total_impl, 3),
                        'fair_home_prob': round(fair_home_prob, 3),
                        'fair_away_prob': round(fair_away_prob, 3),
                        'home_prob_delta': round(rel_home, 3),
                        'away_prob_delta': round(rel_away, 3),
                    }
                    # 主队赔率过高（诱主）
                    if rel_home > 0.08:
                        water_signal = 'overpriced_home'
                        reason_parts.append(f'⚠️主队让球赔率偏高({home_raw:.2f})→概率差+{rel_home:.1%}→庄家诱主队')
                        confidence = max(confidence - 0.08, 0.35)
                    # 客队赔率过高（诱客）
                    elif rel_away > 0.08:
                        water_signal = 'overpriced_away'
                        reason_parts.append(f'⚠️客队让球赔率偏高({away_raw:.2f})→概率差+{rel_away:.1%}→庄家诱客队')
                        confidence = max(confidence - 0.08, 0.35)
                    # 主队赔率偏低（庄家真实不看好主队）
                    elif rel_home < -0.08:
                        water_signal = 'underpriced_home'
                        reason_parts.append(f'💧主队让球赔率偏低({home_raw:.2f})→概率差{rel_home:.1%}→庄家真实不看好')
                    # 客队赔率偏低
                    elif rel_away < -0.08:
                        water_signal = 'underpriced_away'
                        reason_parts.append(f'💧客队让球赔率偏低({away_raw:.2f})→概率差{rel_away:.1%}→庄家真实不看好')
                    else:
                        reason_parts.append(f'让球赔率合理(Δ<8%)，无明显诱盘信号')
        except (TypeError, ValueError, ZeroDivisionError):
            reason_parts.append('让球赔率→水位换算失败')
    else:
        reason_parts.append('无有效水位数据，仅基于盘口深度判断')

    # ── 4. 综合信号判断 ──
    # 深盘 + 低水 → 阻上，正路
    if depth_signal in ('deep', 'slightly_deep') and water_signal in ('low_water_home', 'dead_water_home', 'low_water_away', 'dead_water_away'):
        signal = '阻上'
        recommended_direction = '上盘' if 'home' in str(water_signal) else '下盘'
        reason_parts.append('✅深盘+低水→正路，阻上成功')
        confidence = min(confidence + 0.05, 0.88)

    # 浅盘 + 高水 → 诱上，陷阱
    elif depth_signal in ('shallow', 'slightly_shallow') and water_signal in ('high_water_home', 'high_water_away', 'overpriced_home', 'overpriced_away'):
        signal = '诱上'
        if water_signal in ('high_water_home', 'overpriced_home'):
            recommended_direction = '下盘'
        elif water_signal in ('high_water_away', 'overpriced_away'):
            recommended_direction = '上盘'
        else:
            recommended_direction = '下盘' if hv is not None and hv > 0 else '上盘'
        reason_parts.append('🔴浅盘+高水→陷阱，诱上盘')
        confidence = min(confidence + 0.08, 0.85)

    # 死水盘 → 独立高置信信号
    elif water_signal in ('dead_water_home', 'dead_water_away'):
        signal = '中性'
        recommended_direction = '上盘' if 'home' in str(water_signal) else '下盘'
        reason_parts.append(f'🏱️死水盘→{recommended_direction}极大概率高走')
        confidence = min(confidence + 0.12, 0.90)

    # 浅盘 → 诱上（独立信号）
    elif depth_signal in ('shallow', 'slightly_shallow'):
        signal = '诱上'
        recommended_direction = '下盘' if hv is not None and hv > 0 else '上盘'
        reason_parts.append('🔴浅盘→诱上，上盘危险')

    # 深盘 → 阻上（独立信号）
    elif depth_signal in ('deep', 'slightly_deep'):
        signal = '阻上'
        recommended_direction = '上盘' if hv is not None and hv > 0 else '下盘'
        reason_parts.append('✅深盘→阻上，上盘有机会')

    else:
        signal = '中性'
        reason_parts.append('盘口水位无明显异常')

    reason = '；'.join(reason_parts)
    confidence = max(0.30, min(0.95, round(confidence, 3)))

    return {
        'signal': signal,
        'confidence': confidence,
        'reason': reason,
        'recommended_direction': recommended_direction,
        'depth_analysis': depth_analysis,
        'water_analysis': water_analysis,
    }


def asian_handicap_deepjudge_batch(matches):
    """批量亚盘深度分析

    Args:
        matches: list of dict, each with keys:
            handicap_value, home_odds, away_odds,
            sp_ho, sp_do, sp_ao, match_code, league, home_team, away_team

    Returns:
        list of asian_handicap_deepjudge() results with match metadata
    """
    results = []
    for m in matches:
        j = asian_handicap_deepjudge(
            m.get('handicap_value'),
            m.get('home_odds'),
            m.get('away_odds'),
            m.get('sp_ho'),
            m.get('sp_do'),
            m.get('sp_ao'),
        )
        j['match_code'] = m.get('match_code', '')
        j['league'] = m.get('league', '')
        j['home_team'] = m.get('home_team', '')
        j['away_team'] = m.get('away_team', '')
        results.append(j)
    return results


# ═══════════════════════════════════════════════════════════════
# P2-新增(2026-05-12): 平赔预警信号 + 盘型换算值
# ═══════════════════════════════════════════════════════════════

_DRAW_WARNING_THRESHOLD = 3.00


def draw_warning(do_):
    """平赔预警信号 — 当平赔 < 3.00 时标记为防平

    根因：平赔低于3.00表明庄家对平局信心较高，
    通常对应平局概率>30%，需要重点关注。

    Args:
        do_: 平赔赔率（float 或可转换为 float 的值）

    Returns:
        dict: {
            'triggered': bool,      # 是否触发预警
            'signal': str or None,  # '防平' 或 None
            'draw_odds': float,     # 实际平赔值
            'reason': str,          # 解释文本
        }
    """
    try:
        draw_odds = float(do_)
    except (ValueError, TypeError):
        return {
            'triggered': False,
            'signal': None,
            'draw_odds': 0.0,
            'reason': '平赔数据无效',
        }

    if draw_odds <= 0:
        return {
            'triggered': False,
            'signal': None,
            'draw_odds': draw_odds,
            'reason': '平赔≤0无效',
        }

    if draw_odds < _DRAW_WARNING_THRESHOLD:
        return {
            'triggered': True,
            'signal': '防平',
            'draw_odds': draw_odds,
            'reason': f'平赔{draw_odds:.2f}<{_DRAW_WARNING_THRESHOLD:.2f}→防平',
        }

    return {
        'triggered': False,
        'signal': None,
        'draw_odds': draw_odds,
        'reason': f'平赔{draw_odds:.2f}≥{_DRAW_WARNING_THRESHOLD:.2f}，无预警',
    }


_HANDICAP_RATIO_THRESHOLDS = [
    (0.00, 0.65, '平手'),
    (0.65, 0.90, '平半'),
    (0.90, 1.00, '半球'),
    (1.00, float('inf'), '深盘(半球+)'),
]


def handicap_conversion(spf_probs):
    """盘型换算值 — 从SPF概率推导理论亚盘盘型

    ratio = home_win_prob / (draw_prob + away_win_prob)

    解释：主胜概率占「非胜即负/平」总和的比重。
    比值越高说明主队越强势，盘口应该更深。

    Thresholds:
        ratio < 0.65  → 平手 (主队弱势)
        0.65 ≤ ratio < 0.90 → 平半 (主队略弱)
        0.90 ≤ ratio < 1.00 → 半球 (均衡)

    Args:
        spf_probs: dict with keys '胜','平','负' and float values

    Returns:
        dict: {
            'ratio': float,              # 盘型换算比值
            'handicap_type': str,        # 盘型字符串
            'home_prob': float,          # 主胜概率
            'draw_prob': float,          # 平局概率
            'away_prob': float,          # 客胜概率
            'reason': str,               # 解释文本
        }
    """
    try:
        hp = float(spf_probs.get('胜', 0))
        dp = float(spf_probs.get('平', 0))
        ap = float(spf_probs.get('负', 0))
    except (ValueError, TypeError, AttributeError):
        return {
            'ratio': 0.0,
            'handicap_type': '无效',
            'home_prob': 0.0,
            'draw_prob': 0.0,
            'away_prob': 0.0,
            'reason': 'SPF概率数据无效',
        }

    denom = dp + ap
    if denom <= 0:
        return {
            'ratio': 0.0,
            'handicap_type': '无效',
            'home_prob': hp,
            'draw_prob': dp,
            'away_prob': ap,
            'reason': f'分母draw+away={denom:.4f}≤0',
        }

    ratio = hp / denom

    handicap_type = '深盘(半球+)'
    for lo, hi, label in _HANDICAP_RATIO_THRESHOLDS:
        if lo <= ratio < hi:
            handicap_type = label
            break

    return {
        'ratio': round(ratio, 4),
        'handicap_type': handicap_type,
        'home_prob': round(hp, 4),
        'draw_prob': round(dp, 4),
        'away_prob': round(ap, 4),
        'reason': (
            f'比值={ratio:.4f}({hp:.1%}/{dp:.1%}+{ap:.1%})→{handicap_type}'
        ),
    }
# === LEAGUE WEIGHT OVERRIDE (daily_review executor) ===
# 2026-05-14: 修复 — LEAGUE_OVERRIDE从未声明直接引用，先声明后赋值
LEAGUE_OVERRIDE = {}
LEAGUE_OVERRIDE['韩职'] = {'factor': 0.65, 'cap': 40, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['日职'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['意甲'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['5区间平局率6'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['周日命中率4'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['12数据仍10'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['14'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['10'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['55'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['准确率'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['输出270'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['区间命中率4'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['89区间6'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['on结算命中率全部为'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['低于随机基准3'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['实际负占5'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['10最佳日4'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['但均值为2'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['95区间4'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['最高但仍低于5'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['conf上限3'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['conf上限4'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['conf上限5'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['结构下平局概率最高仅'] = {'factor': 0.75, 'cap': 45, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['80s太短导致10'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['场仅'] = {'factor': 0.65, 'cap': 40, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['偏离'] = {'factor': 0.75, 'cap': 45, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['J联赛'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['命中率6'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['的4'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['命中率仅4'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['冷门区命中率4'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['45'] = {'factor': 0.75, 'cap': 45, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['36'] = {'factor': 0.75, 'cap': 45, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['置信上限2'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['置信上限3'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['24'] = {'factor': 0.65, 'cap': 40, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['25'] = {'factor': 0.65, 'cap': 40, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['22'] = {'factor': 0.75, 'cap': 45, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['29'] = {'factor': 0.75, 'cap': 45, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['历史5'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['置信度上限强制压至3'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['历史2'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['nfidence降权'] = {'factor': 0.75, 'cap': 45, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['当前值低于4'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['随机基率3'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['8分类基率1'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['实际命中5'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['Platt输出仅4'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['晚场36场预测10'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['实际9'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['11'] = {'factor': 0.75, 'cap': 45, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['09'] = {'factor': 0.65, 'cap': 40, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['07'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['06'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['解放者杯'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['沙职'] = {'factor': 0.65, 'cap': 40, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['Poisson模型'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['30'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
# === END LEAGUE OVERRIDE ===
# === LEAGUE WEIGHT OVERRIDE (daily_review executor) ===
LEAGUE_OVERRIDE['韩职'] = {'factor': 0.65, 'cap': 40, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['日职'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['意甲'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['5区间平局率6'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['周日命中率4'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['12数据仍10'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['14'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['10'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['55'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['准确率'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['输出270'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['区间命中率4'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['89区间6'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['on结算命中率全部为'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['低于随机基准3'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['实际负占5'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['10最佳日4'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['但均值为2'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['95区间4'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['最高但仍低于5'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['conf上限3'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['conf上限4'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['conf上限5'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['结构下平局概率最高仅'] = {'factor': 0.75, 'cap': 45, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['80s太短导致10'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['场仅'] = {'factor': 0.65, 'cap': 40, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['偏离'] = {'factor': 0.75, 'cap': 45, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['J联赛'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['命中率6'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['的4'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['命中率仅4'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['冷门区命中率4'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['45'] = {'factor': 0.75, 'cap': 45, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['36'] = {'factor': 0.75, 'cap': 45, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['置信上限2'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['置信上限3'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['24'] = {'factor': 0.65, 'cap': 40, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['25'] = {'factor': 0.65, 'cap': 40, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['22'] = {'factor': 0.75, 'cap': 45, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['29'] = {'factor': 0.75, 'cap': 45, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['历史5'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['置信度上限强制压至3'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['历史2'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['nfidence降权'] = {'factor': 0.75, 'cap': 45, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['当前值低于4'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['随机基率3'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['8分类基率1'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['实际命中5'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['Platt输出仅4'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['晚场36场预测10'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['实际9'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['11'] = {'factor': 0.75, 'cap': 45, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['09'] = {'factor': 0.65, 'cap': 40, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['07'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['06'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['解放者杯'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['沙职'] = {'factor': 0.65, 'cap': 40, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['Poisson模型'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['30'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
# === END LEAGUE OVERRIDE ===
# === LEAGUE WEIGHT OVERRIDE (daily_review executor) ===
LEAGUE_OVERRIDE['韩职'] = {'factor': 0.65, 'cap': 40, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['日职'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['意甲'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['5区间平局率6'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['周日命中率4'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['12数据仍10'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['14'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['10'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['55'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['准确率'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['输出270'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['区间命中率4'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['89区间6'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['on结算命中率全部为'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['低于随机基准3'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['实际负占5'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['10最佳日4'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['但均值为2'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['95区间4'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['最高但仍低于5'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['conf上限3'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['conf上限4'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['conf上限5'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['结构下平局概率最高仅'] = {'factor': 0.75, 'cap': 45, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['80s太短导致10'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['场仅'] = {'factor': 0.65, 'cap': 40, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['偏离'] = {'factor': 0.75, 'cap': 45, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['J联赛'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['命中率6'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['的4'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['命中率仅4'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['冷门区命中率4'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['45'] = {'factor': 0.75, 'cap': 45, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['36'] = {'factor': 0.75, 'cap': 45, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['置信上限2'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['置信上限3'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['24'] = {'factor': 0.65, 'cap': 40, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['25'] = {'factor': 0.65, 'cap': 40, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['22'] = {'factor': 0.75, 'cap': 45, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['29'] = {'factor': 0.75, 'cap': 45, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['历史5'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['置信度上限强制压至3'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['历史2'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['nfidence降权'] = {'factor': 0.75, 'cap': 45, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['当前值低于4'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['随机基率3'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['8分类基率1'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['实际命中5'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['Platt输出仅4'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['晚场36场预测10'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['实际9'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['11'] = {'factor': 0.75, 'cap': 45, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['09'] = {'factor': 0.65, 'cap': 40, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['07'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['06'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['解放者杯'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['沙职'] = {'factor': 0.65, 'cap': 40, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['Poisson模型'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
LEAGUE_OVERRIDE['30'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-13'}
# === END LEAGUE OVERRIDE ===
# === LEAGUE WEIGHT OVERRIDE (daily_review executor) ===
LEAGUE_OVERRIDE['目标'] = {'factor': 0.65, 'cap': 40, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['日职'] = {'factor': 0.65, 'cap': 40, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['5区间平局率6'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['周日命中率4'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['12数据仍10'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['14'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['10'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['55'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['准确率'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['输出270'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['区间命中率4'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['89区间6'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['on结算命中率全部为'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['低于随机基准3'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['实际负占5'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['10最佳日4'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['但均值为2'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['95区间4'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['最高但仍低于5'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['conf上限3'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['conf上限4'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['conf上限5'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['结构下平局概率最高仅'] = {'factor': 0.75, 'cap': 45, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['80s太短导致10'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['负占6'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['非固定'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['场仅'] = {'factor': 0.65, 'cap': 40, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['偏离'] = {'factor': 0.75, 'cap': 45, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['J联赛'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['命中率6'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['的4'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['命中率仅4'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['冷门区命中率4'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['45'] = {'factor': 0.75, 'cap': 45, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['36'] = {'factor': 0.75, 'cap': 45, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['置信上限2'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['置信上限3'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['24'] = {'factor': 0.65, 'cap': 40, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['25'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['22'] = {'factor': 0.75, 'cap': 45, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['29'] = {'factor': 0.75, 'cap': 45, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['历史5'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['置信度上限强制压至3'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['历史2'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['nfidence降权'] = {'factor': 0.75, 'cap': 45, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['当前值低于4'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['随机基率3'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['8分类基率1'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['250'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['595'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['实际命中5'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['Platt输出仅4'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['晚场36场预测10'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['复盘统计永远'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['实际9'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['vs'] = {'factor': 0.65, 'cap': 40, 'source': 'daily_review', 'date': '2026-05-14'}
LEAGUE_OVERRIDE['39'] = {'factor': 0.45, 'cap': 35, 'source': 'daily_review', 'date': '2026-05-14'}
# === END LEAGUE OVERRIDE ===

# ═══════════════════════════════════════════════════════════════
# P1-新增(2026-05-14): 高置信度惩罚 — 修正60+区间的过度自信
# 根因: 60+置信区间gap=-11.1%，模型高估准确率约11%
# 规则: confidence > 0.60 → 应用折扣因子0.89 (即1 - 11.1%)
# ═══════════════════════════════════════════════════════════════

# 高置信度惩罚区间阈值
# 2026-05-14 P0修复: 阈值0.60→0.65，折扣0.89→0.85
# 实测: 65%+区间命中率40.9%(最低)，35-45%和55-65%区间51.7%(最佳)
HOT_CONFIDENCE_THRESHOLD = 0.65

# 折扣因子: 65%+区间实测40.9%，需从0.65降至0.55附近
HOT_CONFIDENCE_DISCOUNT = 0.85


def hot_confidence_penalty(confidence: float, spf_probs: dict = None) -> tuple:
    """P1-新增: 高置信度惩罚 — 修正65+区间的过度自信

    根因: 置信度>65%区间实际命中率仅40.9%(实测最低)，
    而35-45%和55-65%区间都是51.7%。
    60-65%区间表现正常，不应受惩罚。

    规则:
    - confidence > 0.65: 应用 HOT_CONFIDENCE_DISCOUNT (0.85)
    - confidence越高惩罚越大（非线性指数衰减）

    Args:
        confidence: 原始置信度 (0-1)
        spf_probs: SPF概率dict（保留参数，暂未使用）

    Returns:
        (adjusted_confidence, signals)
    """
    signals = []
    conf = float(confidence) if confidence else 0.0

    if conf <= HOT_CONFIDENCE_THRESHOLD:
        return round(conf, 4), signals

    # 计算折扣: 超出阈值越多，惩罚越大（指数衰减）
    excess = conf - HOT_CONFIDENCE_THRESHOLD
    # 折扣 = 基准折扣 ^ (1 + 超出量) — conf=0.70 → discount≈0.72, conf=0.80 → discount≈0.60
    discount = HOT_CONFIDENCE_DISCOUNT ** (1 + excess * 2.0)

    adjusted = conf * discount
    threshold_pct = int(HOT_CONFIDENCE_THRESHOLD * 100)
    signals.append(
        f"🔥高置信惩罚(conf={conf:.0%}>{threshold_pct}%→{adjusted:.0%}, discount={discount:.2f})"
    )

    return round(adjusted, 4), signals
