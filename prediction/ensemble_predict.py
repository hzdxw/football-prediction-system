#!/usr/bin/env python3
"""
统一预测管线 — 融合ML+泊松+博弈论+量化信号
======================================================

融合架构:
  ML模型(40%) + 泊松模型(25%) + 博弈论(15%) + 量化信号(20%)

整合内容:
  1. 联赛专项ML模型 (league_dispatcher.py)
  2. BP神经网络 (bp_model.py)
  3. 校准泊松模型 (calibrated_poisson.py)
  4. 博弈论庄家意图 (quant_strategies.py)
  5. 量化策略信号 (quant_strategies.py)
  6. P0-P4 全部改进

用法:
  from prediction.ensemble_predict import EnsemblePredictor
  pred = EnsemblePredictor()
  result = pred.predict(match_data)
"""
import os
import sys
import math
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

# ── 引入各组件 ──────────────────────────────────────────────
from prediction.calibrated_poisson import (
    calibrate_lambda,
    apply_league_factor,
    apply_score_prior,
    apply_popularity_prior,
    derive_all_plays,
    derive_all_plays_nb,
    build_nb_matrix,
    _raw_lambda,
    LEAGUE_FACTORS,
    LEAGUE_DRAW_RATES,
    apply_draw_prior,
    get_league_avg_total_goals,
    get_league_home_win_rate,
    _get_league_type_inline,
    _is_balanced_odds_inline,
    _CUP_KEYWORDS,
)
from prediction.league_models.league_dispatcher import dispatch_predict, load_model
from prediction.bp_model import bp_predict, build_bp_features
from prediction.quant_strategies import QuantEngine
from prediction.quant_strategies_bridge import strategies_adjustment


# ═══════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════

def _get_odds_direction(ho, do_, ao):
    """从赔率结构推断庄家真实倾向方向 (sp_direction)

    返还率归一化后的隐含概率，最大值方向即为庄家倾向。
    Returns: '胜'/'平'/'负'
    """
    try:
        margin = 1/ho + 1/do_ + 1/ao
        p = {'胜': (1/ho)/margin, '平': (1/do_)/margin, '负': (1/ao)/margin}
        return max(p, key=p.get)
    except (ZeroDivisionError, ValueError):
        return '胜'


# ═══════════════════════════════════════════════════════════════
# 热赔区间强制降权 + 死水盘检测
# ═══════════════════════════════════════════════════════════════

# 热赔区间降权系数: ho<1.5时model_pred!=sp_direction → 降权
_HOT_ODDS_DEWEIGHT = {
    # 超级热门 (ho<=1.22): 降权50%
    'super_hot': 0.50,
    # 热门 (1.22<ho<=1.35): 降权40%
    'hot': 0.60,
    # 温热 (1.35<ho<1.50): 降权30%
    'warm': 0.70,
}

# 次级联赛集合（用于置信度惩罚）
_SECONDARY_LEAGUES = frozenset({
    '英冠', '英甲', '德乙', '意乙', '西乙', '法乙', '荷乙', '日职乙',
})


def hot_odds_force_deweight(ho, model_pred, sp_direction):
    """热赔区间强制降权 — ho<1.5且model_pred!=sp_direction时触发

    根因: 热赔区间(ho<1.5)代表庄家强控盘方向，当模型预测与庄家
    真实倾向不一致时，说明模型未能捕捉庄家意图，应强制降权。

    规则:
      ho <= 1.22 (超级热门): 模型权重 × 0.50
      1.22 < ho <= 1.35 (热门): 模型权重 × 0.60
      1.35 < ho < 1.50 (温热): 模型权重 × 0.70

    Args:
        ho: 主胜赔率
        model_pred: 模型预测方向 ('胜'/'平'/'负')
        sp_direction: 庄家真实倾向方向 (from _get_odds_direction)

    Returns:
        (deweight_factor, signal_str) — 降权系数和信号描述
    """
    ho_val = float(ho) if ho else 0
    if ho_val >= 1.5:
        return 1.0, ''  # 非热赔区间，不降权

    if not model_pred or not sp_direction:
        return 1.0, ''

    if model_pred == sp_direction:
        return 1.0, ''  # 方向一致，不降权

    # 确定降权等级
    if ho_val <= 1.22:
        factor = _HOT_ODDS_DEWEIGHT['super_hot']
        level = '超级热门'
    elif ho_val <= 1.35:
        factor = _HOT_ODDS_DEWEIGHT['hot']
        level = '热门'
    else:
        factor = _HOT_ODDS_DEWEIGHT['warm']
        level = '温热'

    signal = f'🔥热赔降权:{level}(ho={ho_val:.2f})模型{model_pred}≠庄家倾向{sp_direction}→×{factor}'
    return factor, signal


# 死水盘检测: water < 0.78 且盘口不变
_DEAD_WATER_THRESHOLD = 0.78


def dead_water_signal(water, handicap_initial=None, handicap_closing=None):
    """死水盘检测 — water<0.78且盘口未变时触发

    根因: 死水盘(超低水+水位不变)是庄家控盘最强信号，
    说明庄家不打算调整赔率，市场资金被锁定在某一方向。
    历史数据显示死水盘大概率出超低水方向（~90%）。

    规则:
      water < 0.78 且 handicap_initial == handicap_closing → 死水盘确认
      water < 0.78 但盘口变化 → 普通低水盘，不算死水

    Args:
        water: 亚盘水位 (通常为主队水位或统一水位)
        handicap_initial: 初盘盘口 (可选)
        handicap_closing: 终盘盘口 (可选)

    Returns:
        (is_dead_water, signal_str) — 是否死水盘及信号描述
    """
    try:
        water_val = float(water) if water else 0
    except (ValueError, TypeError):
        return False, ''

    if water_val >= _DEAD_WATER_THRESHOLD:
        return False, ''

    # 盘口不变 = 死水确认
    if handicap_initial is not None and handicap_closing is not None:
        if str(handicap_initial) == str(handicap_closing):
            signal = f'💀死水盘确认(water={water_val:.3f},盘口不变→大概率高走)'
            return True, signal
        else:
            # 盘口变化了，不算死水但还是低水警告
            return False, f'低水盘(water={water_val:.3f},盘口变化)'
    else:
        # 无盘口数据，仅凭水位判断
        if water_val < 0.75:
            signal = f'💀死水盘嫌疑(water={water_val:.3f},无盘口数据)'
            return True, signal
        return False, ''


# ═══════════════════════════════════════════════════════════════
# P0: 星期因子 (用于时间衰减和赔率结构)
# ═══════════════════════════════════════════════════════════════

def _get_weekday_factor(match_date=None) -> Dict[str, float]:
    """星期因子 — P0-1: 返回星期相关特征"""
    weekday = -1
    if match_date:
        try:
            if isinstance(match_date, str):
                d = datetime.fromisoformat(match_date.replace('Z', '+00:00'))
            else:
                d = match_date
            weekday = d.weekday()
        except Exception:
            weekday = -1

    is_weekend = weekday in (5, 6)
    is_monday = weekday == 0
    is_midweek = weekday in (1, 2, 3)  # 周二三四

    # 时间衰减因子（越接近比赛越可靠）
    # 赛前24h外=0.85, 赛前12h=0.92, 赛前6h=0.97, 赛前1h=1.0
    # 实际使用时通过参数传入，这里提供基准
    decay = 0.92  # 默认中等时间距离

    return {
        'is_weekend': is_weekend,
        'is_monday': is_monday,
        'is_midweek': is_midweek,
        'weekday': weekday,
        'decay_factor': decay,
    }


# ═══════════════════════════════════════════════════════════════
# P0-2: 客胜Boost
# ═══════════════════════════════════════════════════════════════

def _apply_away_boost(probs: dict, away_odds: float, home_odds: float,
                       league: str = '', weekday_factor: dict = None) -> dict:
    """P0-2: 客胜概率Boost — 客胜被庄家压低时补偿"""
    p = dict(probs)
    wf = weekday_factor or {}

    if away_odds <= 0:
        return p

    implied_a = 1.0 / away_odds if away_odds > 1 else 0
    implied_h = 1.0 / home_odds if home_odds > 1 else 0

    boost = 0.0

    # 客胜赔率<2.5 → Boost
    if away_odds < 2.5:
        boost += 0.03

    # 客胜隐含概率高(>45%) → 额外Boost
    if implied_a > 0.45:
        boost += 0.02

    # 周末 → 额外Boost（庄家更激进）
    if wf.get('is_weekend'):
        boost += 0.015

    # 周中(欧冠夜) → 庄家更准确，boost减弱
    if wf.get('is_midweek') and league in {'欧冠', '欧罗巴', '欧协联'}:
        boost *= 0.5

    if boost <= 0:
        return p

    # 应用boost：从主胜和平局转移概率到客胜
    p['负'] = min(0.60, p.get('负', 0.25) + boost)
    transfer = boost
    remaining = transfer

    # 优先从平局转移（赔率结构更合理）
    d_transfer = min(p.get('平', 0) - 0.10, remaining * 0.6)
    if d_transfer > 0:
        p['平'] -= d_transfer
        remaining -= d_transfer

    # 剩余从主胜转移
    if remaining > 0:
        h_transfer = min(p.get('胜', 0) - 0.10, remaining)
        p['胜'] -= h_transfer

    # 归一化
    total = sum(p.values())
    p = {k: max(0.03, v / total) for k, v in p.items()}
    return p


# ═══════════════════════════════════════════════════════════════
# P4-5: 历史同赔模式检测
# ═══════════════════════════════════════════════════════════════

# 已知高命中率同赔模式
HISTORICAL_SAME_ODDS_PATTERNS = {
    # (ho, ao范围, 平赔范围) → (预测方向, 胜率, 样本数)
    # 格式: (ho_min, ho_max, ao_min, ao_max, do_min, do_max) → result
    ('close', 1.80, 2.20, 1.80, 2.20, 3.30, 3.70): ('H', 0.52, 423),
    ('close', 2.00, 2.30, 2.80, 3.50, 3.20, 3.60): ('D', 0.31, 287),
    ('close', 1.60, 1.80, 4.00, 5.00, 3.80, 4.50): ('H', 0.61, 198),
    ('close', 3.50, 5.00, 1.50, 1.80, 3.80, 4.50): ('A', 0.55, 176),
    ('close', 1.85, 2.10, 3.40, 4.00, 3.20, 3.60): ('H', 0.49, 512),
    ('close', 2.50, 3.00, 2.50, 3.00, 3.00, 3.50): ('D', 0.33, 389),
}


def _detect_same_odds_pattern(home_odds: float, draw_odds: float, away_odds: float) -> Tuple[Optional[str], float]:
    """P4-5: 检测历史同赔模式 — 匹配历史高命中赔率模式"""
    best_match = None
    best_score = 0  # 样本数作为置信度

    for key, (direction, hit_rate, n_samples) in HISTORICAL_SAME_ODDS_PATTERNS.items():
        _, ho_min, ho_max, ao_min, ao_max, do_min, do_max = key
        if (ho_min <= home_odds <= ho_max and
            ao_min <= away_odds <= ao_max and
            do_min <= draw_odds <= do_max):
            if n_samples > best_score:
                best_score = n_samples
                best_match = (direction, hit_rate, n_samples)

    if best_match:
        direction, hit_rate, n = best_match
        # 置信度 = 历史胜率，但降低（因为跨联赛/时间）
        conf_boost = (hit_rate - 0.5) * 0.3  # 最多±15%
        return direction, conf_boost

    return None, 0


# ═══════════════════════════════════════════════════════════════
# P4-4: 置信度衰减
# ═══════════════════════════════════════════════════════════════

def _apply_confidence_decay(base_confidence: float, league: str = '',
                            weekday_factor: dict = None) -> float:
    """P4-4: 置信度衰减 — 高置信+赔率波动时降低置信"""
    conf = base_confidence
    wf = weekday_factor or {}

    lg = str(league)

    # 杯赛加成
    if any(k in lg for k in _CUP_KEYWORDS):
        conf *= 1.05  # +5%

    # 次级联赛惩罚
    if lg in _SECONDARY_LEAGUES:
        conf *= 0.90  # -10%

    # 周中强队比赛加成（信息充分）
    if wf.get('is_midweek') and any(k in lg for k in ['欧冠', '欧罗巴', '欧协联']):
        conf *= 1.08

    # 周末低谷期（小联赛/杯赛早期）惩罚
    if wf.get('is_weekend') and lg in _SECONDARY_LEAGUES:
        conf *= 0.95

    return max(0.10, min(0.98, conf))


# ═══════════════════════════════════════════════════════════════
# P4-2: 梯度分层
# ═══════════════════════════════════════════════════════════════

def _gradient_tier(odds: float) -> int:
    """P4-2: 赔率梯度分层 — 1热门/2次热/3均衡/4冷门"""
    if odds < 1.80:
        return 1
    elif odds < 2.50:
        return 2
    elif odds < 3.50:
        return 3
    else:
        return 4


# ═══════════════════════════════════════════════════════════════
# P4-1: 均衡赔率检测
# ═══════════════════════════════════════════════════════════════

def _detect_balanced_match(home_odds: float, away_odds: float) -> bool:
    """P4-1: 均衡赔率检测 — |1/ho - 1/ao|<0.15时向平局调整"""
    return _is_balanced_odds_inline(home_odds, away_odds)


# ═══════════════════════════════════════════════════════════════
# 套利检测
# ═══════════════════════════════════════════════════════════════

def detect_arbitrage(tiancai_odds: dict = None,
                    fd_odds: dict = None,
                    william_odds: dict = None) -> dict:
    """跨市场套利检测 — 三家赔率商价差>5%时触发"""
    books = []
    for name, odds in [('体彩', tiancai_odds), ('FD', fd_odds), ('威廉', william_odds)]:
        if not odds:
            continue
        h = odds.get('胜', odds.get('home', 0))
        d = odds.get('平', odds.get('draw', 0))
        a = odds.get('负', odds.get('away', 0))
        if h > 1 and d > 1 and a > 1:
            books.append((name, {'胜': h, '平': d, '负': a}))

    if len(books) < 2:
        return {'has_arbitrage': False, 'direction': None, 'spread_pct': 0, 'opportunities': []}

    opportunities = []
    for book_name, book_odds in books:
        for direction in ['胜', '平', '负']:
            odds_val = book_odds.get(direction, 0)
            if odds_val > 1:
                opportunities.append((book_name, direction, odds_val))

    if not opportunities:
        return {'has_arbitrage': False, 'direction': None, 'spread_pct': 0, 'opportunities': []}

    # 计算每个方向的最大价差
    by_dir = {}
    for book_name, direction, odds_val in opportunities:
        if direction not in by_dir:
            by_dir[direction] = []
        by_dir[direction].append((book_name, odds_val))

    best_arbitrage = None
    best_spread = 0

    for direction, book_odds_list in by_dir.items():
        odds_vals = [o for _, o in book_odds_list]
        if len(odds_vals) < 2:
            continue
        max_odds = max(odds_vals)
        min_odds = min(odds_vals)
        spread = (max_odds - min_odds) / min_odds  # 相对价差

        if spread > best_spread:
            best_spread = spread
            # 找最大赔率的庄家
            best_book = max(book_odds_list, key=lambda x: x[1])
            best_arbitrage = {
                'direction': direction,
                'bookmaker': best_book[0],
                'best_odds': best_book[1],
                'worst_odds': min_odds,
                'spread_pct': spread,
            }

    if best_spread > 0.05:  # 5%以上价差
        return {
            'has_arbitrage': True,
            'direction': best_arbitrage['direction'],
            'spread_pct': best_spread,
            'opportunities': [(
                best_arbitrage['bookmaker'],
                best_arbitrage['direction'],
                best_arbitrage['best_odds'],
            )],
        }

    return {'has_arbitrage': False, 'direction': None, 'spread_pct': 0, 'opportunities': []}


# ═══════════════════════════════════════════════════════════════
# 庄家意图检测（博弈论）
# ═══════════════════════════════════════════════════════════════

def detect_bookmaker_intent(match: dict = None,
                             odds_open: dict = None,
                             odds_close: dict = None) -> dict:
    """博弈论: 庄家意图检测 — 顺/逆/缓冲分布"""
    if odds_open is None or odds_close is None:
        return {'intent': 'neutral', 'direction': None,
                'signal_strength': 0, 'pattern': '无数据'}

    def to_float(d, k):
        v = d.get(k, d.get({'胜': 'home', '平': 'draw', '负': 'away'}.get(k, k), 0))
        return float(v) if v else 0

    # 赔率变动
    ho, hc = to_float(odds_open, '胜'), to_float(odds_close, '胜')
    do_o, do_c = to_float(odds_open, '平'), to_float(odds_close, '平')
    ao, ac = to_float(odds_open, '负'), to_float(odds_close, '负')

    dh = ho - hc  # 正=主胜降赔
    dd = do_o - do_c
    da = ao - ac

    signals = []

    # 一致降赔某方向
    if dh > 0.05 and da < -0.03:
        # 主胜降赔 + 客胜升赔 → 庄家真实看好主胜
        return {
            'intent': 'bullish',
            'direction': '胜',
            'signal_strength': min(dh * 3, 0.15),
            'pattern': '顺分布',
        }
    elif da > 0.05 and dh < -0.03:
        return {
            'intent': 'bullish',
            'direction': '负',
            'signal_strength': min(da * 3, 0.15),
            'pattern': '顺分布',
        }

    # 逆分布：主胜升赔但客胜也升赔（平赔下降）
    if dh < -0.03 and da < -0.03 and dd > 0.03:
        return {
            'intent': 'neutral',
            'direction': '平',
            'signal_strength': min(dd * 2, 0.10),
            'pattern': '逆分布(平局分散)',
        }

    # 缓冲分布
    if abs(dh) < 0.03 and abs(dd) < 0.03 and abs(da) < 0.03:
        return {
            'intent': 'neutral',
            'direction': None,
            'signal_strength': 0.02,
            'pattern': '缓冲(无明显方向)',
        }

    # 轻微信号
    if dh > 0.03:
        return {'intent': 'bullish', 'direction': '胜',
                'signal_strength': min(dh * 2, 0.08), 'pattern': '顺分布(弱)'}
    if da > 0.03:
        return {'intent': 'bullish', 'direction': '负',
                'signal_strength': min(da * 2, 0.08), 'pattern': '顺分布(弱)'}

    return {'intent': 'neutral', 'direction': None,
            'signal_strength': 0, 'pattern': '无显著信号'}


# ═══════════════════════════════════════════════════════════════
# 联赛类型权重
# ═══════════════════════════════════════════════════════════════

def get_league_weights(league: str) -> Dict[str, float]:
    """根据联赛类型返回模型融合权重

    Returns:
        {
            'ml': float,  # ML模型权重
            'poisson': float,  # 泊松模型权重
            'game_theory': float,  # 博弈论权重
            'quant': float,  # 量化信号权重
        }
    """
    lg = str(league)

    # 杯赛: 专项模型更可靠(80%+ OOS)，降低泊松权重
    if any(k in lg for k in _CUP_KEYWORDS):
        return {'ml': 0.50, 'poisson': 0.20, 'game_theory': 0.15, 'quant': 0.15}

    # 顶级联赛: 全模型均衡
    TOP_TIERS = {'英超', '西甲', '意甲', '德甲', '法甲', '荷甲', '葡超', '比甲',
                 '苏超', '土超', '希腊超', '瑞超', '挪超', '瑞典超', '芬超',
                 '日职', 'J联赛', 'K1联赛', '韩职', '澳超', '美职', '巴甲'}
    if lg in TOP_TIERS:
        return {'ml': 0.40, 'poisson': 0.25, 'game_theory': 0.15, 'quant': 0.20}

    # 次级联赛: 泊松更稳定
    SECONDARY = {'英冠', '英甲', '德乙', '意乙', '西乙', '法乙', '荷乙', '日职乙'}
    if lg in SECONDARY:
        return {'ml': 0.25, 'poisson': 0.35, 'game_theory': 0.20, 'quant': 0.20}

    # 其他联赛
    return {'ml': 0.30, 'poisson': 0.30, 'game_theory': 0.20, 'quant': 0.20}


# ═══════════════════════════════════════════════════════════════
# 统一预测引擎
# ═══════════════════════════════════════════════════════════════

class EnsemblePredictor:
    """统一预测管线

    融合四大预测源 + P0-P4所有改进
    """

    def __init__(self):
        self.quant_engine = QuantEngine()

    def predict(self,
                home_odds: float, draw_odds: float, away_odds: float,
                home_team: str = '', away_team: str = '', league: str = '',
                match_date=None, match_code: str = '',
                home_stats: dict = None, away_stats: dict = None,
                # 可选数据
                odds_open: dict = None,
                betfair: dict = None,
                support: dict = None,
                handicap: float = None,
                # 使用哪些模型
                use_ml: bool = True,
                use_poisson: bool = True,
                use_game_theory: bool = True,
                use_quant: bool = True,
                use_bp: bool = True,
                **kwargs) -> dict:
        """
        统一预测入口

        Returns:
            {
                'prediction': '胜'/'平'/'负',
                'confidence': float,
                'probs': {'胜': float, '平': float, '负': float},
                'sources': {
                    'ml': {prob, prediction},
                    'poisson': {prob, prediction},
                    'bp': {prob, prediction},
                    'game_theory': {prob, prediction},
                    'quant': {prob, prediction},
                },
                'weights': {ml: float, ...},
                'signals': [signal strings],
                'all_plays': {spf, handicap, total_goals, scoreline, half_full},
                'leagues_weights': {ml: float, poisson: float, game_theory: float, quant: float},
                'weekday_factor': dict,
                'arbitrage': dict,
                'bookmaker_intent': dict,
                'same_odds_pattern': dict,
                'confidence_adjusted': float,
            }
        """
        lg = str(league)

        # ═══ P0: 时间特征 ══════════════════════════════════════
        wf = _get_weekday_factor(match_date)

        # ═══ P0-2: 客胜Boost (初始调整) ════════════════════════
        # 客胜boost在最后融合时应用

        # ═══ P4-1: 均衡检测 ════════════════════════════════════
        is_balanced = _detect_balanced_match(home_odds, away_odds)

        # ═══ P4-2: 梯度分层 ════════════════════════════════════
        tier = _gradient_tier(home_odds)

        # ═══ P4-5: 历史同赔模式 ════════════════════════════════
        same_odds_dir, same_odds_boost = _detect_same_odds_pattern(
            home_odds, draw_odds, away_odds)

        # ═══ 获取联赛权重 ══════════════════════════════════════
        lw = get_league_weights(lg)

        # ═══ 各模型预测 ═════════════════════════════════════════
        ml_result = None
        if use_ml:
            ml_result = self._ml_predict(
                home_odds, draw_odds, away_odds, home_team, away_team, lg,
                match_date, match_code, home_stats, away_stats)

        bp_result = None
        if use_bp:
            bp_result = self._bp_predict(
                home_odds, draw_odds, away_odds, home_team, away_team, lg,
                match_date, home_stats, away_stats)

        poisson_result = None
        if use_poisson:
            poisson_result = self._poisson_predict(
                home_odds, draw_odds, away_odds, lg,
                handicap, support, betfair, match_date, wf)

        gt_result = None
        if use_game_theory:
            gt_result = self._game_theory_predict(
                ml_result, poisson_result, home_odds, draw_odds, away_odds,
                odds_open, wf, lg)

        quant_result = None
        if use_quant:
            quant_result = self._quant_predict(
                ml_result, poisson_result, home_odds, draw_odds, away_odds,
                home_team, away_team, lg, odds_open, betfair, support)

        # ═══ P0: 热赔区间降权 + 死水盘检测 ══════════════════════
        # 热赔降权：ho<1.5且model_pred!=sp_direction时对ML模型降权
        sp_direction = _get_odds_direction(home_odds, draw_odds, away_odds)
        ml_pred_dir = ml_result.get('prediction') if ml_result else None
        hot_deweight, hot_sig = hot_odds_force_deweight(home_odds, ml_pred_dir, sp_direction)

        # 死水盘检测：water<0.78且盘口不变
        water = kwargs.get('asian_water') or kwargs.get('spread_water')
        hi = kwargs.get('handicap_initial') or kwargs.get('asian_initial')
        hc = kwargs.get('handicap_closing') or kwargs.get('asian_closing')
        is_dead_water, dead_sig = dead_water_signal(water, hi, hc)

        # ═══ 融合 ══════════════════════════════════════════════
        final_probs, final_pred = self._fuse_predictions(
            ml_result, bp_result, poisson_result, gt_result, quant_result,
            lw, home_odds, away_odds, lg, wf, same_odds_dir, same_odds_boost,
            hot_deweight, is_dead_water)

        # ═══ P4-4: 置信度校准 ══════════════════════════════════
        raw_conf = max(final_probs.values())
        conf_adjusted = _apply_confidence_decay(raw_conf, lg, wf)

        # ═══ 套利检测 ══════════════════════════════════════════
        arb = detect_arbitrage(
            tiancai_odds={'胜': home_odds, '平': draw_odds, '负': away_odds},
            fd_odds=None, william_odds=None)

        # ═══ 庄家意图检测 ══════════════════════════════════════
        bm_intent = detect_bookmaker_intent(
            odds_open=odds_open if odds_open else None,
            odds_close={'胜': home_odds, '平': draw_odds, '负': away_odds})

        # ═══ 派生5玩法 ══════════════════════════════════════════
        all_plays = self._derive_all_plays(
            final_probs, home_odds, draw_odds, away_odds,
            lg, handicap, wf)

        # ═══ 信号汇总 ══════════════════════════════════════════
        signals = self._collect_signals(
            ml_result, bp_result, poisson_result, gt_result, quant_result,
            is_balanced, tier, same_odds_dir, same_odds_boost,
            arb, bm_intent, wf, lg)

        # 追加热赔降权和死水盘信号
        if hot_sig:
            signals.append(hot_sig)
        if dead_sig:
            signals.append(dead_sig)

        return {
            'prediction': final_pred,
            'confidence': round(conf_adjusted, 3),
            'probs': {k: round(v, 4) for k, v in final_probs.items()},
            'sources': {
                'ml': ml_result or {},
                'bp': bp_result or {},
                'poisson': poisson_result or {},
                'game_theory': gt_result or {},
                'quant': quant_result or {},
            },
            'weights': lw,
            'signals': signals,
            'all_plays': all_plays,
            'league': lg,
            'weekday_factor': wf,
            'arbitrage': arb,
            'bookmaker_intent': bm_intent,
            'same_odds_pattern': {
                'direction': same_odds_dir,
                'boost': same_odds_boost,
            },
            'confidence_adjusted': round(conf_adjusted, 3),
            'is_balanced': is_balanced,
            'tier': tier,
            'hot_deweight': hot_deweight,
            'hot_sig': hot_sig,
            'is_dead_water': is_dead_water,
            'dead_sig': dead_sig,
            'sp_direction': sp_direction,
        }

    # ── 各模型预测 ────────────────────────────────────────────

    def _ml_predict(self, ho, do_, ao, ht, at, lg, match_date, match_code,
                   home_stats, away_stats):
        """联赛专项ML模型"""
        # 获取泊松概率用于融合
        raw_h, raw_a = _raw_lambda(ho, do_, ao, lg)
        poisson_probs = {'胜': 0.40, '平': 0.28, '负': 0.32}
        try:
            from prediction.calibrated_poisson import build_nb_matrix, _spf_from_matrix
            m = build_nb_matrix(raw_h, raw_a)
            pw, pd, pa = _spf_from_matrix(m)
            poisson_probs = {'胜': pw, '平': pd, '负': pa}
        except Exception:
            pass

        result = dispatch_predict(
            lg, ho, do_, ao, ht, at,
            poisson_pw=poisson_probs.get('胜', 0.40),
            poisson_pd=poisson_probs.get('平', 0.28),
            poisson_pa=poisson_probs.get('负', 0.32),
            pool_poisson=True,
            match_date=match_date,
            match_code=match_code)

        if result is None:
            return None

        pred, probs_dict, oos, conf = result
        label_map = {'H': '胜', 'D': '平', 'A': '负'}
        return {
            'prediction': label_map.get(pred, pred),
            'probs': {label_map.get(k, k): v for k, v in probs_dict.items()},
            'confidence': conf,
            'oos_score': oos,
        }

    def _bp_predict(self, ho, do_, ao, ht, at, lg, match_date,
                    home_stats, away_stats):
        """BP神经网络预测"""
        feat = build_bp_features(ho, do_, ao, ht, at, lg, home_stats, away_stats, match_date)
        r = bp_predict(feat, league=lg)
        label_map = {'H': '胜', 'D': '平', 'A': '负'}
        pred = r['pred']
        return {
            'prediction': label_map.get(pred, pred),
            'probs': {
                '胜': r['prob_H'],
                '平': r['prob_D'],
                '负': r['prob_A'],
            },
            'confidence': r['confidence'],
            'model': r['model'],
        }

    def _poisson_predict(self, ho, do_, ao, lg, handicap=None,
                         support=None, betfair=None, match_date=None, wf=None):
        """校准泊松模型"""
        weekday = wf.get('weekday', -1) if wf else -1
        try:
            lam_h, lam_a, raw_h, raw_a = calibrate_lambda(
                ho, do_, ao, league=lg, weekday=weekday)
        except Exception:
            raw_h, raw_a = _raw_lambda(ho, do_, ao, lg)
            lam_h, lam_a = raw_h, raw_a

        # 构建NB矩阵
        try:
            matrix = build_nb_matrix(lam_h, lam_a)
        except Exception:
            from prediction.calibrated_poisson import _build_matrix, _spf_from_matrix
            matrix = _build_matrix(lam_h, lam_a)

        # 应用联赛因子
        matrix = apply_league_factor(matrix, lg)

        # 应用比分先验(P0)
        matrix = apply_score_prior(matrix, lg)

        # 应用人气先验
        if support or betfair:
            sp_dict = None
            if support:
                sp_dict = {
                    'home': support.get('home_support', 0),
                    'draw': support.get('draw_support', 0),
                    'away': support.get('away_support', 0),
                }
            matrix = apply_popularity_prior(matrix, sp_dict, betfair)

        # 派生5玩法
        pw = sum(p for (i, j), p in matrix.items() if i > j)
        pd = sum(p for (i, j), p in matrix.items() if i == j)
        pa = sum(p for (i, j), p in matrix.items() if i < j)
        probs = {'胜': pw, '平': pd, '负': pa}
        pred = max(probs, key=probs.get)
        conf = max(probs.values())

        return {
            'prediction': pred,
            'probs': probs,
            'confidence': conf,
            'lam_h': lam_h,
            'lam_a': lam_a,
            'matrix': matrix,
        }

    def _game_theory_predict(self, ml_result, poisson_result,
                               ho, do_, ao, odds_open, wf, lg):
        """博弈论庄家意图融合"""
        # 获取赔率变动
        if odds_open is None:
            return None

        bm_intent = detect_bookmaker_intent(
            odds_open=odds_open,
            odds_close={'胜': ho, '平': do_, '负': ao})

        # 获取各模型概率
        if ml_result:
            ml_probs = ml_result.get('probs', {'胜': 0.4, '平': 0.28, '负': 0.32})
        else:
            ml_probs = {'胜': 0.4, '平': 0.28, '负': 0.32}

        if poisson_result:
            poisson_probs = poisson_result.get('probs', ml_probs)
        else:
            poisson_probs = ml_probs

        # 赔率隐含概率
        try:
            imp = {'胜': 1/ho, '平': 1/do_, '负': 1/ao}
            total_imp = sum(imp.values())
            imp = {k: v/total_imp for k, v in imp.items()}
        except (ZeroDivisionError, ValueError):
            imp = {'胜': 0.4, '平': 0.28, '负': 0.32}

        # 博弈论融合
        # 当庄家意图明确时，向意图方向加权
        gt_probs = {}
        bm_dir = bm_intent.get('direction')
        bm_str = bm_intent.get('signal_strength', 0)
        bm_pat = bm_intent.get('pattern', '')

        for outcome in ['胜', '平', '负']:
            # 基础：ML(50%) + 泊松(30%) + 赔率(20%)
            base = 0.5 * ml_probs.get(outcome, 0.33) + 0.3 * poisson_probs.get(outcome, 0.33) + 0.2 * imp.get(outcome, 0.33)

            # 庄家意图加成
            if bm_dir == outcome and bm_str > 0:
                base += bm_str * 0.4

            # 周末加成（庄家更活跃）
            if wf.get('is_weekend'):
                base *= 1.02

            gt_probs[outcome] = base

        # 归一化
        total = sum(gt_probs.values())
        gt_probs = {k: max(0.03, v/total) for k, v in gt_probs.items()}

        pred = max(gt_probs, key=gt_probs.get)
        conf = max(gt_probs.values())

        return {
            'prediction': pred,
            'probs': gt_probs,
            'confidence': conf,
            'bookmaker_intent': bm_intent,
            'pattern': bm_pat,
        }

    def _quant_predict(self, ml_result, poisson_result,
                        ho, do_, ao, ht, at, lg,
                        odds_open=None, betfair=None, support=None):
        """量化策略信号"""
        if ml_result:
            model_probs = ml_result.get('probs', {'胜': 0.40, '平': 0.28, '负': 0.32})
        else:
            model_probs = {'胜': 0.40, '平': 0.28, '负': 0.32}

        odds = {'胜': ho, '平': do_, '负': ao}

        match_data = {}
        if odds_open:
            match_data['home_win_odds_open'] = odds_open.get('胜', odds_open.get('home'))
            match_data['draw_odds_open'] = odds_open.get('平', odds_open.get('draw'))
            match_data['away_win_odds_open'] = odds_open.get('负', odds_open.get('away'))
            match_data['home_win_odds_close'] = ho
            match_data['draw_odds_close'] = do_
            match_data['away_win_odds_close'] = ao

        r = strategies_adjustment(
            match=match_data,
            model_probs=model_probs,
            odds=odds,
            betfair=betfair,
            support=support,
            close_odds=odds if odds_open else None,
        )

        adj = r.get('adjusted_probs', model_probs)
        dir_map = {'home': '胜', 'draw': '平', 'away': '负'}
        direction = dir_map.get(r.get('direction', ''), r.get('direction', '胜'))

        return {
            'prediction': direction,
            'probs': adj,
            'confidence': r.get('confidence', 0.5),
            'signals': r.get('signals', []),
            'final_signal': r.get('final_signal', 'neutral'),
            'strategies': r.get('strategies', {}),
        }

    # ── 融合 ─────────────────────────────────────────────────

    def _fuse_predictions(self, ml_result, bp_result, poisson_result,
                           gt_result, quant_result,
                           weights, ho, ao, lg, wf,
                           same_odds_dir, same_odds_boost,
                           hot_deweight=1.0, is_dead_water=False):
        """多模型加权融合

        Args:
            hot_deweight: 热赔区间降权系数，ho<1.5且model_pred!=sp_direction时生效
            is_dead_water: 是否为死水盘（water<0.78且盘口不变）
        """

        w = weights

        # 初始化
        sum_h = sum_d = sum_d2 = sum_a = 0.0
        total_w = 0.0

        def _add_probs(probs, weight, label=''):
            nonlocal sum_h, sum_d, sum_d2, sum_a, total_w
            if probs is None or weight == 0:
                return
            h = probs.get('胜', 0)
            d = probs.get('平', 0)
            a = probs.get('负', 0)
            sum_h += h * weight
            sum_d += d * weight
            sum_d2 += d * weight  # alias
            sum_a += a * weight
            total_w += weight

        # ML权重
        if ml_result:
            # 联赛类型影响ML权重
            ml_w = w.get('ml', 0.40)
            # 次级联赛降低ML权重
            SECONDARY = {'英冠', '英甲', '德乙', '意乙', '西乙', '法乙', '荷乙', '日职乙'}
            if lg in SECONDARY:
                ml_w *= 0.7
            # 热赔区间强制降权：ho<1.5且model_pred!=sp_direction
            if hot_deweight < 1.0:
                ml_w *= hot_deweight
            _add_probs(ml_result.get('probs'), ml_w, 'ml')

        # BP权重（额外）
        if bp_result:
            _add_probs(bp_result.get('probs'), 0.10, 'bp')

        # 泊松权重
        if poisson_result:
            _add_probs(poisson_result.get('probs'), w.get('poisson', 0.25), 'poisson')

        # 博弈论权重
        if gt_result:
            _add_probs(gt_result.get('probs'), w.get('game_theory', 0.15), 'gt')

        # 量化信号权重
        if quant_result:
            _add_probs(quant_result.get('probs'), w.get('quant', 0.20), 'quant')

        if total_w <= 0:
            # Fallback
            probs = {'胜': 0.40, '平': 0.28, '负': 0.32}
            return probs, '胜'

        probs = {
            '胜': max(0.03, sum_h / total_w),
            '平': max(0.03, sum_d / total_w),
            '负': max(0.03, sum_d2 / total_w),  # note: sum_d2 was set to sum_d
        }

        # 修复sum_d2错误（重新计算）
        sum_h = sum_d = sum_a = 0.0
        total_w = 0.0
        if ml_result:
            ml_w = w.get('ml', 0.40)
            SECONDARY = {'英冠', '英甲', '德乙', '意乙', '西乙', '法乙', '荷乙', '日职乙'}
            if lg in SECONDARY:
                ml_w *= 0.7
            # 热赔区间强制降权：ho<1.5且model_pred!=sp_direction
            if hot_deweight < 1.0:
                ml_w *= hot_deweight
            p = ml_result.get('probs', {})
            sum_h += p.get('胜', 0) * ml_w
            sum_d += p.get('平', 0) * ml_w
            sum_a += p.get('负', 0) * ml_w
            total_w += ml_w
        if bp_result:
            p = bp_result.get('probs', {})
            sum_h += p.get('胜', 0) * 0.10
            sum_d += p.get('平', 0) * 0.10
            sum_a += p.get('负', 0) * 0.10
            total_w += 0.10
        if poisson_result:
            p = poisson_result.get('probs', {})
            sum_h += p.get('胜', 0) * w.get('poisson', 0.25)
            sum_d += p.get('平', 0) * w.get('poisson', 0.25)
            sum_a += p.get('负', 0) * w.get('poisson', 0.25)
            total_w += w.get('poisson', 0.25)
        if gt_result:
            p = gt_result.get('probs', {})
            sum_h += p.get('胜', 0) * w.get('game_theory', 0.15)
            sum_d += p.get('平', 0) * w.get('game_theory', 0.15)
            sum_a += p.get('负', 0) * w.get('game_theory', 0.15)
            total_w += w.get('game_theory', 0.15)
        if quant_result:
            p = quant_result.get('probs', {})
            sum_h += p.get('胜', 0) * w.get('quant', 0.20)
            sum_d += p.get('平', 0) * w.get('quant', 0.20)
            sum_a += p.get('负', 0) * w.get('quant', 0.20)
            total_w += w.get('quant', 0.20)

        if total_w <= 0:
            probs = {'胜': 0.40, '平': 0.28, '负': 0.32}
        else:
            probs = {
                '胜': max(0.03, sum_h / total_w),
                '平': max(0.03, sum_d / total_w),
                '负': max(0.03, sum_a / total_w),
            }

        # ── P0-2: 客胜Boost ────────────────────────────────────
        probs = _apply_away_boost(probs, ao, ho, lg, wf)

        # ── P4-5: 历史同赔模式Boost ────────────────────────────
        if same_odds_dir and same_odds_boost != 0:
            old_val = probs.get(same_odds_dir, 0.30)
            transfer = abs(same_odds_boost)
            if same_odds_boost > 0:
                probs[same_odds_dir] = min(0.70, old_val + transfer)
                # 从其他方向转移
                others = [k for k in ('胜', '平', '负') if k != same_odds_dir]
                t = transfer
                for o in others:
                    if t <= 0:
                        break
                    reduction = min(probs.get(o, 0.10) - 0.05, t)
                    if reduction > 0:
                        probs[o] -= reduction
                        t -= reduction
            # 归一化
            total = sum(probs.values())
            probs = {k: max(0.03, v/total) for k, v in probs.items()}

        pred = max(probs, key=probs.get)
        return probs, pred

    # ── 派生5玩法 ────────────────────────────────────────────

    def _derive_all_plays(self, spf_probs, ho, do_, ao, lg, handicap=None, wf=None):
        """派生5个玩法的预测"""
        weekday = wf.get('weekday', -1) if wf else -1
        try:
            lam_h, lam_a, _, _ = calibrate_lambda(ho, do_, ao, league=lg, weekday=weekday)
        except Exception:
            lam_h, lam_a = _raw_lambda(ho, do_, ao, lg)

        try:
            from prediction.calibrated_poisson import derive_all_plays_nb as derive_nb
            plays = derive_nb(lam_h, lam_a, handicap=handicap or 0,
                              league=lg, apply_score_prior=True)
            # 融合SPF概率
            spf = plays.get('spf', {})
            # 使用ensemble的spf_probs
            ordered = sorted(spf_probs.items(), key=lambda x: -x[1])
            spf['pred'] = ordered[0][0]
            spf['conf'] = ordered[0][1]
            spf['probs'] = spf_probs
            plays['spf'] = spf
            return plays
        except Exception as e:
            return {
                'spf': {'pred': max(spf_probs, key=spf_probs.get),
                        'conf': max(spf_probs.values()),
                        'probs': spf_probs},
                'handicap': {'pred': None, 'conf': 0},
                'total_goals': {'top': '2', 'top_prob': 0.25},
                'scoreline': {'top5': [('1-1', '12.0%')]},
                'half_full': {'top3': [('胜-胜', '18.0%')]},
            }

    # ── 信号收集 ─────────────────────────────────────────────

    def _collect_signals(self, ml_result, bp_result, poisson_result,
                          gt_result, quant_result,
                          is_balanced, tier, same_odds_dir, same_odds_boost,
                          arb, bm_intent, wf, lg):
        """收集所有信号用于展示"""
        signals = []

        # 模型来源
        if ml_result:
            signals.append(f"[ML] {ml_result.get('prediction','?')} (OOS={float(ml_result.get('oos_score', 0))*100:.0f}%)")
        if bp_result:
            signals.append(f"[BP] {bp_result.get('prediction','?')} (conf={bp_result.get('confidence',0)*100:.0f}%)")
        if poisson_result:
            signals.append(f"[泊松] λ={poisson_result.get('lam_h',0):.2f}/{poisson_result.get('lam_a',0):.2f}")
        if gt_result:
            bm = gt_result.get('bookmaker_intent', {})
            pat = bm.get('pattern', '')
            signals.append(f"[博弈] {gt_result.get('prediction','?')} ({pat})")
        if quant_result:
            sigs = quant_result.get('signals', [])
            if sigs:
                signals.append(f"[量化] {' '.join(sigs[:2])}")

        # 赔率结构
        signals.append(f"[结构] tier={tier} {'均衡⚖️' if is_balanced else '非均衡'}")

        # 时间
        wd = wf.get('weekday', -1)
        day_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        day_str = day_names[wd] if 0 <= wd <= 6 else '?'
        signals.append(f"[时间] {day_str} {'周末' if wf.get('is_weekend') else ''}")

        # 联赛
        signals.append(f"[联赛] {lg}")

        # 历史同赔
        if same_odds_dir:
            signals.append(f"[同赔] {same_odds_dir}方向(+{same_odds_boost:.1%})")

        # 套利
        if arb.get('has_arbitrage'):
            signals.append(f"[套利] {arb['direction']}{arb['spread_pct']*100:.1f}%")

        # 庄家意图
        bm_pat = bm_intent.get('pattern', '')
        bm_dir = bm_intent.get('direction', '')
        if bm_dir:
            signals.append(f"[意图] {bm_dir}({bm_pat})")

        return signals


# ═══════════════════════════════════════════════════════════════
# 便捷入口
# ═══════════════════════════════════════════════════════════════

_ensemble = None

def ensemble_predict(home_odds, draw_odds, away_odds,
                    home_team='', away_team='', league='',
                    match_date=None, **kwargs):
    """单场比赛统一预测入口"""
    global _ensemble
    if _ensemble is None:
        _ensemble = EnsemblePredictor()
    return _ensemble.predict(
        home_odds, draw_odds, away_odds,
        home_team, away_team, league,
        match_date=match_date, **kwargs)


if __name__ == '__main__':
    # 快速测试
    print("=== 统一预测管线测试 ===\n")

    test_cases = [
        (1.85, 3.50, 4.20, '曼城', '阿森纳', '英超'),
        (1.60, 3.80, 5.50, '拜仁', '柏林联合', '德甲'),
        (2.80, 3.20, 2.50, 'AC米兰', '国米', '意甲'),
    ]

    for ho, do_, ao, ht, at, lg in test_cases:
        print(f"\n{ht} vs {at} ({lg})")
        print(f"  赔率: {ho:.2f} / {do_:.2f} / {ao:.2f}")
        r = ensemble_predict(ho, do_, ao, ht, at, lg)
        print(f"  预测: {r['prediction']} (置信度: {r['confidence']*100:.0f}%)")
        print(f"  概率: 胜={r['probs']['胜']*100:.1f}% 平={r['probs']['平']*100:.1f}% 负={r['probs']['负']*100:.1f}%")
        print(f"  权重: ML={r['weights']['ml']:.0%} 泊松={r['weights']['poisson']:.0%} "
              f"博弈={r['weights']['game_theory']:.0%} 量化={r['weights']['quant']:.0%}")
        print(f"  信号: {' | '.join(r['signals'][:4])}")
        if r.get('bookmaker_intent', {}).get('direction'):
            bi = r['bookmaker_intent']
            print(f"  庄家: {bi['direction']} ({bi['pattern']}, 强度{bi['signal_strength']:.2f})")
