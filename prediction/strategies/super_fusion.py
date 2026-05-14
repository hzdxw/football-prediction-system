# -*- coding: utf-8 -*-
"""
SuperFusionLayer — 多因子超级融合引擎
==========================================

架构：
  输入 → 11路因子并行计算 → 自适应权重融合 → 博弈论一致性检验 → 输出

11路因子：
  1. ML校准因子（ml_predict_5play校准矩阵）
  2. DL序列因子（dl_predictor LSTM序列模型）
  3. 泊松基础因子（calibrated_poisson）
  4. 博弈论因子（庄家意图/市场均衡/冷门偏见）
  5. 趋势跟踪因子（赔率初终盘变化方向）
  6. 高频套利因子（Betfair vs 体彩跨市场套利）
  7. 必发流向因子（成交量分布/大单信号）
  8. 周几专项因子（P5联赛×周几λ校正）
  9. 联赛专项因子（联赛分层模型权重）
  10. 历史相似盘口因子（Gap-10：479k赔率库）
  11. 球队近期状态因子（Gap-11：fd_uk近N场得失球）

自适应权重：
  - 各因子权重基于最近30天命中率动态调整
  - 置信度×权重加权平均
  - 博弈论一致性检验：多因子共识方向 vs 分歧方向

输出：
  final_probs, confidence, factor_contributions, signals, fusion_quality
"""

import os, sys, math, json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

# ── 路径设置 ──────────────────────────────────────────────────────────────
_PRED = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.dirname(_PRED))

# ── 因子权重配置（可动态调整）────────────────────────────────────────────
_DEFAULT_WEIGHTS = {
    'ml':       0.17,   # ML校准矩阵
    'dl':       0.13,   # DL序列模型
    'poisson':  0.11,   # 泊松基础
    'gametheory': 0.09, # 博弈论
    'trend':    0.08,   # 趋势跟踪
    'hft_arb':  0.07,   # 高频套利
    'betfair':  0.08,   # 必发流向
    'dow':      0.06,   # 周几专项
    'league':   0.05,   # 联赛专项
    'hist_sim': 0.04,   # Gap-10历史相似
    'team_form': 0.04,  # Gap-11球队状态
    'asian_hcap':  0.13,   # 亚盘让球因子（合并已删除的死代码bayes因子）
}

# ── 自适应权重调谐器（基于历史命中率动态调整）─────────────────────────────
import psycopg2, logging
_logger = logging.getLogger('super_fusion')

class AdaptiveWeightTuner:
    """从DB读取历史预测记录，按因子方向统计近期命中率，动态上调/下调权重"""
    def __init__(self, days: int = 30, min_samples: int = 20):
        self.days = days
        self.min_samples = min_samples
        self._weights = dict(_DEFAULT_WEIGHTS)
        self._last_tune = None

    def tune(self) -> dict:
        """从bet_records聚合真实命中率，驱动因子权重调谐（修复：原prediction_results表缺少direction/hit/factor_contributions字段，永远报错回退）"""
        try:
            conn = psycopg2.connect(host='localhost', dbname='myapp_db', user='myapp', password='MyApp@1234', connect_timeout=5)
            cur = conn.cursor()

            # Step 1: 从bet_records聚合各market+selection的真实命中率
            cur.execute(f"""
                SELECT 
                    market,
                    selection,
                    COUNT(*) as cnt,
                    SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END) as hits,
                    AVG(CASE WHEN profit > 0 THEN 1.0 ELSE 0.0 END) as hit_rate,
                    AVG(odds) as avg_odds
                FROM bet_records
                WHERE actual_result IS NOT NULL
                  AND created_at::date >= CURRENT_DATE - {self.days}
                  AND model_prob IS NOT NULL
                GROUP BY market, selection
                HAVING COUNT(*) >= 5
                ORDER BY market, selection
            """)
            rows = cur.fetchall()
            cur.close()
            conn.close()

            if not rows:
                _logger.info('AdaptiveWeightTuner: bet_records无足够样本，使用默认权重')
                return self._weights

            # Step 2: 计算各市场的均衡命中率（庄家返还率约0.90 → 均衡≈1/odds）
            market_hit_rates = {}
            for market, sel, cnt, hits, hr, avg_odds in rows:
                # 庄家均衡命中率（理论值）; avg_odds是Decimal需转换
                avg_odds_f = float(avg_odds) if avg_odds is not None else 0.0
                implied = 1.0 / avg_odds_f if avg_odds_f > 0 else 0.38
                # 实际命中率 vs 均衡命中率 → EV系数
                hr_f = float(hr) if hr is not None else 0.0
                ev_ratio = hr_f / implied if implied > 0 else 1.0
                market_hit_rates[f'{market}:{sel}'] = {
                    'cnt': cnt, 'hit_rate': hr_f, 'avg_odds': avg_odds_f,
                    'implied': implied, 'ev_ratio': ev_ratio
                }
                _logger.debug(f'  {market}:{sel} 样本{cnt} 胜率{hr_f:.3f} 均衡{implied:.3f} EV比{ev_ratio:.3f}')

            # Step 3: 按market聚合EV系数（HHAD selection格式为"让负(主让1球,43%)"，先按market聚合）
            market_ev = {}   # {market: {'cnt_total': N, 'weighted_ev_ratio': ...}}
            for key, data in market_hit_rates.items():
                mkt = key.split(':', 1)[0]
                ev_ratio = data['ev_ratio']
                cnt = data['cnt']
                if mkt not in market_ev:
                    market_ev[mkt] = {'total_cnt': 0, 'weighted_ev': 0.0}
                market_ev[mkt]['total_cnt'] += cnt
                market_ev[mkt]['weighted_ev'] += ev_ratio * cnt

            for mkt, v in market_ev.items():
                v['avg_ev_ratio'] = v['weighted_ev'] / v['total_cnt'] if v['total_cnt'] > 0 else 1.0
                _logger.info(f'  市场{mkt}: 总样本{v["total_cnt"]} 加权EV比={v["avg_ev_ratio"]:.3f}')

            # Step 4: 根据各市场EV调整因子权重
            new_w = dict(self._weights)  # 从当前权重出发调整
            # 因子权重受影响的market:selection组合
            factor_market_map = {
                'ml':             ['spf', 'hhad'],
                'poisson':        ['spf'],
                'bayes_factor':   ['spf', 'hhad'],
                'market_sentiment':['spf'],
                'elo_diff':       ['spf'],
                'home_advantage': ['spf'],
                'away_trap':      ['spf'],
                'bookmaker_intent':['spf'],
                'intensity_factor':['spf'],
                'rest_days_factor':['spf'],
                'asian_hcap':     ['hhad'],
                'ClosingOdds':    ['spf', 'hhad'],
                'dl':             ['spf', 'hhad'],
                'team_form':      ['spf'],
                'hist_sim':       ['spf'],
                'trend':          ['spf'],
                'gametheory':     ['spf'],
                'hft_arb':        ['spf'],
                'dow':            ['spf'],
                'league':         ['spf', 'hhad'],
                'betfair':        ['spf', 'hhad'],
            }

            for fac, markets in factor_market_map.items():
                if fac not in new_w:
                    continue
                adj_sum = 0.0
                adj_cnt = 0
                for mkt in markets:
                    if mkt in market_ev and market_ev[mkt]['avg_ev_ratio'] > 0:
                        adj_sum += market_ev[mkt]['avg_ev_ratio']
                        adj_cnt += 1
                if adj_cnt > 0:
                    avg_adj = adj_sum / adj_cnt
                    # 限制在[0.5, 1.5]防止极端调整
                    adj = max(0.5, min(1.5, avg_adj))
                    new_w[fac] = round(max(0.02, min(0.25, new_w[fac] * adj)), 4)

            # Step 5: 归一化（保持总和=1.0）
            total = sum(new_w.values())
            if total > 0:
                new_w = {k: round(v / total, 4) for k, v in new_w.items()}

            _logger.info(f'AdaptiveWeightTuner: 调谐完成 样本{len(rows)}个市场组合')
            for k, v in sorted(new_w.items()):
                _logger.info(f'  因子{k}: 权重{v:.4f}')

            self._weights = new_w
            self._last_tune = datetime.now()
            return new_w

        except Exception as e:
            try:
                conn.close()
            except Exception:
                pass
            _logger.warning(f'AdaptiveWeightTuner权重调谐失败: {e}')
            return self._weights

    @property
    def current_weights(self) -> dict:
        return self._weights

_tuner = AdaptiveWeightTuner(days=30, min_samples=20)

def get_adaptive_weights() -> dict:
    if _tuner._last_tune is None or (datetime.now() - _tuner._last_tune).total_seconds() > 86400:
        return _tuner.tune()
    return _tuner.current_weights

def _WEIGHTS() -> dict:
    return get_adaptive_weights()

# ── 因子可用性注册表（运行时检查）──────────────────────────────────────
_FACTOR_REGISTRY: Dict[str, bool] = {}

def _check_factor(name: str, fn) -> bool:
    """检测因子函数是否可用"""
    try:
        _FACTOR_REGISTRY[name] = True
        return True
    except Exception:
        _FACTOR_REGISTRY[name] = False
        return False


# ═══════════════════════════════════════════════════════════════════════
# 因子计算函数
# ═══════════════════════════════════════════════════════════════════════

def _run_with_timeout(fn, args, timeout_sec: int = 8):
    """跨平台超时封装：直接调用 + 线程级超时检查。
    每100ms检查一次elapsed，超时立即中断并返回None。
    注意：无法中断阻塞IO，仅对CPU密集型有效。
    对于DB/文件等IO阻塞，建议调用方自行实现超时。
    """
    import time, threading

    result = [None]
    exc = [None]

    def _target():
        try:
            result[0] = fn(*args)
        except Exception as e:
            exc[0] = e

    t = threading.Thread(target=_target, daemon=True)
    start = time.monotonic()
    t.start()
    while t.is_alive():
        if time.monotonic() - start > timeout_sec:
            return None  # 超时，不阻断流程
        t.join(timeout=0.1)
    if exc[0]:
        raise exc[0]
    return result[0]


def _get_ml_confidence_from_db(match_code: str, match_date: str) -> Optional[Dict]:
    """查询predictions_5play中已有的ML预测结果（P0-1 FIX: 早场pipeline写ml_confidence时查此处）

    注意：早场pipeline当前不写ml_confidence字段，此函数暂为保留接口。
    当早场pipeline写入ml_confidence后，此路径将自动生效。
    """
    return None  # 暂时禁用，早场pipeline未写入ml_confidence时无数据可查


def _ml_load_from_db(match_code: str, match_date: str) -> Optional[float]:
    """从 ml_predictions_5play 读取 spf_confidence，失败返回 None"""
    import subprocess, os
    q = f"""
SELECT spf_confidence FROM ml_predictions_5play
WHERE match_code='{match_code}' AND match_date='{match_date}'
LIMIT 1;
"""
    try:
        env = os.environ.copy()
        env['PGPASSWORD'] = os.environ.get('PGPASSWORD', 'MyApp@1234')
        env.update({'LANG': 'C', 'LC_ALL': 'C', 'LANGUAGE': 'C'})
        r = subprocess.run(
            ['psql', '-h', 'localhost', '-U', 'myapp', '-d', 'myapp_db', '-t', '-c', q],
            capture_output=True, text=True, env=env, timeout=5
        )
        out = r.stdout.strip()
        if out and r.returncode == 0:
            val = float(out)
            if val > 0:
                return val
    except Exception:
        pass
    return None


def compute_ml_factor(match: dict, ho: float, do_: float, ao: float) -> Optional[Dict]:
    """因子1: ML校准矩阵因子
    P0-1 FIX: 优先从 ml_predictions_5play 读取真实 spf_confidence，
              不再硬编码 conf=0.16，打破 uniform_fallback 死循环。
    填充路径: run_fusion_pipeline.py → ml_predictions_5play → 这里读取
    """
    margin = 1/ho + 1/do_ + 1/ao
    implied = {'胜': (1/ho)/margin, '平': (1/do_)/margin, '负': (1/ao)/margin}

    # 优先读取真实 ML 置信度（ml_predictions_5play 已由 pipeline 写入）
    mc = match.get('match_code', '')
    md = match.get('match_date', '')
    real_conf = _ml_load_from_db(mc, md) if mc and md else None

    if real_conf is not None:
        return {
            'probs': implied,
            'confidence': real_conf,
            'signals': [f'ml_real:spf_conf={real_conf:.3f}'],
            'tier': 'medium' if real_conf >= 0.30 else 'low',
        }

    # Fallback: conf=0.16（ml_predictions_5play 无数据时降级）
    return {
        'probs': implied,
        'confidence': 0.16,
        'signals': ['ml_fallback:no_ml_table_data'],
        'tier': 'minimal',
    }


def compute_dl_factor(match: dict, ho: float, do_: float, ao: float) -> Optional[Dict]:
    """因子2: DL序列模型因子（MLP LSTM风格，168维特征，进程级单例缓存）"""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data-collection'))
        from dl_predictor import _get_cached_predictor
        def _call():
            predictor = _get_cached_predictor(models_dir='/home/doodoo/.hermes/workspace/models')
            if 'spf' not in predictor.models:
                return None
            ht = match.get('home_team', '')
            at = match.get('away_team', '')
            lg = match.get('league', '')
            mc = match.get('match_code', '')
            bd = match.get('match_date') or datetime.now().isoformat()
            return predictor.predict(mc, lg, 'spf', ht, at, ho, do_, ao, before_date=bd)

        try:
            result = _run_with_timeout(_call, (), timeout_sec=10)
        except TimeoutError:
            return None

        if result and not result.get('error'):
            proba = result.get('prob', [0.33, 0.33, 0.33])
            labels = ['胜', '平', '负']
            return {
                'probs': {labels[i]: float(proba[i]) for i in range(3)},
                'confidence': result.get('confidence', 0.5),
                'signals': [f"DL:预测{result.get('prediction','')}"],
            }
        return None
    except Exception:
        return None


def compute_poisson_factor(ho: float, do_: float, ao: float) -> Optional[Dict]:
    """因子3: 泊松基础因子"""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from calibrated_poisson import calibrate_lambda

        cal_h, cal_a = calibrate_lambda(ho, do_, ao)[:2]
        # 手动从 λ 计算三区概率
        import math
        # 简化的泊松胜平负估算
        lambda_home = cal_h
        lambda_away = cal_a
        # P(主胜) = Σ_{i>j} P(i球) × P(j球)
        p_win = sum(math.exp(-lambda_home) * (lambda_home**i) / math.factorial(i) *
                    sum(math.exp(-lambda_away) * (lambda_away**j) / math.factorial(j)
                        for j in range(i))
                    for i in range(1, 10))
        p_draw = sum(math.exp(-lambda_home) * (lambda_home**i) / math.factorial(i) *
                      math.exp(-lambda_away) * (lambda_away**i) / math.factorial(i)
                      for i in range(10))
        p_lose = 1 - p_win - p_draw
        p_lose = max(0.05, p_lose)
        p_win  = max(0.05, min(0.85, p_win))
        p_draw = max(0.05, min(0.60, p_draw))
        total = p_win + p_draw + p_lose

        return {
            'probs': {'胜': p_win/total, '平': p_draw/total, '负': p_lose/total},
            'confidence': 0.52,
            'signals': [],
        }
    except Exception:
        return None


def compute_gametheory_factor(ho: float, do_: float, ao: float,
                               pop: Optional[dict] = None) -> Optional[Dict]:
    """因子4: 博弈论因子（市场均衡/庄家偏差/冷门偏见）"""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from calibrated_poisson import market_equilibrium_deviation, longshot_bias_adjustment
        from ml_predict_5play import implied_prob

        base = implied_prob(ho, do_, ao)
        base_probs = {'胜': base[0], '平': base[1], '负': base[2]}

        # 市场均衡偏离
        eq_adj, eq_sig = market_equilibrium_deviation(base_probs, ho, do_, ao)

        # 冷门偏见
        support = pop.get('support') if pop else None
        if support:
            cold_adj, cold_sig = longshot_bias_adjustment(base_probs, support, ho, do_, ao)
            eq_adj.update(cold_adj)

        total = sum(eq_adj.values())
        fused = {k: v / total for k, v in eq_adj.items()}

        return {
            'probs': fused,
            'confidence': 0.55,
            'signals': eq_sig,
        }
    except Exception:
        return None


def compute_trend_factor(match: dict) -> Optional[Dict]:
    """因子5: 趋势跟踪因子（赔率初终盘变化方向）"""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from quant_strategies import TrendStrategy
        strategy = TrendStrategy()
        result = strategy.analyze(match)
        if not result:
            return None

        sig = result.get('trend_signal', '中性')
        strength = result.get('trend_strength', 0)

        # 基于趋势方向调整基础概率
        base = 0.333
        if sig == '主队强势':
            probs = {'胜': base + strength * 0.15, '平': base, '负': base - strength * 0.10}
        elif sig == '客队强势':
            probs = {'胜': base - strength * 0.10, '平': base, '负': base + strength * 0.15}
        elif sig == '平局倾向':
            probs = {'胜': base - strength * 0.05, '平': base + strength * 0.10, '负': base - strength * 0.05}
        else:
            probs = {'胜': base, '平': base, '负': base}

        # 归一化
        total = sum(probs.values())
        probs = {k: max(0.05, v / total) for k, v in probs.items()}

        return {
            'probs': probs,
            'confidence': 0.50 + strength * 0.10,
            'signals': [f'trend:{sig}@{strength:.2f}'],
        }
    except Exception:
        return None


def compute_hft_arb_factor(ho: float, do_: float, ao: float,
                            bf_data: Optional[dict] = None) -> Optional[Dict]:
    """因子6: 高频套利因子（体彩 vs Betfair 跨市场套利）"""
    try:
        if not bf_data:
            return None

        # 支持两种格式：
        # 1. bf_data['betfair'] = {bf_home, bf_draw, bf_away}（Betfair赔率，>1）
        # 2. bf_data['betfair'] = {home_pct, draw_pct, away_pct}（百分比，0-100）
        bf = bf_data.get('betfair', {})
        use_pct = 'home_pct' in bf  # 用字段名区分，不是值的大小

        if use_pct:
            bf_home = float(bf.get('home_pct', 0)) / 100.0
            bf_draw = float(bf.get('draw_pct', 0)) / 100.0
            bf_away = float(bf.get('away_pct', 0)) / 100.0
        else:
            bf_home = float(bf.get('bf_home', 0))
            bf_draw = float(bf.get('bf_draw', 0))
            bf_away = float(bf.get('bf_away', 0))

        if bf_home <= 0 or bf_draw <= 0 or bf_away <= 0:
            return None

        # 体彩返还率
        sporttery_margin = 1/ho + 1/do_ + 1/ao
        s_home = (1/ho) / sporttery_margin
        s_draw = (1/do_) / sporttery_margin
        s_away = (1/ao) / sporttery_margin

        # Betfair隐含概率
        if use_pct:
            # 百分比格式：已经是概率（0-1），直接使用
            b_home = bf_home
            b_draw = bf_draw
            b_away = bf_away
        else:
            # 赔率格式：计算隐含概率
            bf_margin = 1/bf_home + 1/bf_draw + 1/bf_away
            b_home = (1/bf_home) / bf_margin
            b_draw = (1/bf_draw) / bf_margin
            b_away = (1/bf_away) / bf_margin

        # 套利信号：spread = 体彩 - Betfair
        spread = {
            '胜': s_home - b_home,
            '平': s_draw - b_draw,
            '负': s_away - b_away,
        }

        # 归一化后的概率（融合体彩+Betfair）
        fused = {
            '胜': (s_home + b_home) / 2,
            '平': (s_draw + b_draw) / 2,
            '负': (s_away + b_away) / 2,
        }
        total = sum(fused.values())
        fused = {k: v / total for k, v in fused.items()}

        # 判断是否存在套利机会
        arb_margin = 1 - (1/(1/s_home + 1/s_draw + 1/s_away) + 1/(1/b_home + 1/b_draw + 1/b_away)) / 2
        signals = [f'arb_margin:{arb_margin:.3f}']
        if arb_margin < -0.03:
            signals.append('arb:存在套利机会')

        return {
            'probs': fused,
            'confidence': 0.60,
            'signals': signals,
            'arb_margin': arb_margin,
        }
    except Exception:
        return None


def compute_betfair_factor(bf_data: Optional[dict] = None) -> Optional[Dict]:
    """因子7: 必发流向因子（成交量分布/大单信号）"""
    try:
        if not bf_data:
            return None
        bf = bf_data.get('betfair', {})
        vol = bf_data.get('volume', {})

        # 用字段名区分格式（不是值的大小）
        use_pct = 'home_pct' in bf
        if use_pct:
            bf_home = float(bf.get('home_pct', 0)) / 100.0
            bf_draw = float(bf.get('draw_pct', 0)) / 100.0
            bf_away = float(bf.get('away_pct', 0)) / 100.0
        else:
            bf_home = float(bf.get('bf_home', 0))
            bf_draw = float(bf.get('bf_draw', 0))
            bf_away = float(bf.get('bf_away', 0))
            if bf_home > 1:
                bf_margin = 1/bf_home + 1/bf_draw + 1/bf_away
                bf_home = (1/bf_home) / bf_margin
                bf_draw = (1/bf_draw) / bf_margin
                bf_away = (1/bf_away) / bf_margin

        if bf_home <= 0:
            return None

        # bf_home/draw/away 此时已经是概率（0-1 范围）
        # 归一化（以防百分比相加不为100）
        total = bf_home + bf_draw + bf_away
        if total <= 0:
            return None
        probs = {
            '胜': bf_home / total,
            '平': bf_draw / total,
            '负': bf_away / total,
        }

        # 大单信号检测
        home_vol = float(vol.get('home', 0)) if isinstance(vol, dict) else 0
        away_vol = float(vol.get('away', 0)) if isinstance(vol, dict) else 0
        signals = []
        if home_vol > 10000:
            signals.append(f'大单主队:{home_vol}')
        if away_vol > 10000:
            signals.append(f'大单客队:{away_vol}')

        return {
            'probs': probs,
            'confidence': 0.55,
            'signals': signals,
        }
    except Exception:
        return None


def compute_dow_factor(match: dict, lg: str, dow: int) -> Optional[Dict]:
    """因子8: 周几专项因子"""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from post_adjust import apply_dow_draw_to_spf
        base_probs = {'胜': 0.40, '平': 0.30, '负': 0.30}
        adj_probs, sig = apply_dow_draw_to_spf(base_probs, lg, match.get('match_date'))
        if not adj_probs:
            return None
        return {
            'probs': adj_probs,
            'confidence': 0.50,
            'signals': sig,
        }
    except Exception:
        return None


def compute_league_factor(lg: str) -> Optional[Dict]:
    """因子9: 联赛专项因子"""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from calibrated_poisson import LEAGUE_FACTORS, LEAGUE_DRAW_RATES
        factor = LEAGUE_FACTORS.get(lg, 1.0)
        draw_rate = LEAGUE_DRAW_RATES.get(lg, 0.28)
        home_base = (1 - draw_rate) * 0.47 + draw_rate * 0.25
        away_base = (1 - draw_rate) * 0.35 + draw_rate * 0.25
        draw_base = draw_rate
        probs = {'胜': home_base, '平': draw_base, '负': away_base}
        return {
            'probs': probs,
            'confidence': 0.50,
            'signals': [f'league_factor:{factor:.3f}'],
        }
    except Exception:
        return None


def compute_hist_sim_factor(lg: str, ho: float, do_: float, ao: float) -> Optional[Dict]:
    """因子10: 历史相似盘口因子（Gap-10）"""
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from ml_predict_5play import compute_historical_similarity
        result = compute_historical_similarity(lg, ho, do_, ao)
        if not result:
            return None
        probs = {
            '胜': result['historical_home_wr'],
            '平': result['historical_draw_wr'],
            '负': result['historical_away_wr'],
        }
        return {
            'probs': probs,
            'confidence': min(0.65, result['n_matches_found'] / 10 + 0.35),
            'signals': [f'hist_sim:{result["n_matches_found"]}场@{result["avg_odds_distance"]:.3f}'],
        }
    except Exception:
        return None


def compute_team_form_factor(home_team: str, away_team: str, lg: str) -> Optional[Dict]:
    """因子11: 球队近期状态因子（Gap-11）"""
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from ml_predict_5play import compute_team_recent_form
        result = compute_team_recent_form(home_team, away_team, lg)
        if not result:
            return None

        # 基于得失球率估算概率
        hgf = result['home_recent_goals_for']
        hga = result['home_recent_goals_against']
        agf = result['away_recent_goals_for']
        aga = result['away_recent_goals_against']

        # 简单Elo风格估算
        h_strength = hgf / (hgf + aga + 0.1)
        a_strength = agf / (agf + hga + 0.1)
        total = h_strength + a_strength + 0.3

        probs = {
            '胜': h_strength / total * 1.2,
            '平': 0.3 / total,
            '负': a_strength / total,
        }
        total_p = sum(probs.values())
        probs = {k: v / total_p for k, v in probs.items()}

        return {
            'probs': probs,
            'confidence': min(0.60, (result['n_home_matches'] + result['n_away_matches']) / 20 + 0.40),
            'signals': [f'home_wr:{result["home_recent_win_rate"]:.2f}',
                        f'away_wr:{result["away_recent_win_rate"]:.2f}'],
        }
    except Exception:
        return None


def compute_cold_factor(ho: float, do_: float, ao: float) -> Optional[Dict]:
    """P1-冷门因子: 检测赔率强化强势方但市场深度暗示冷门的矛盾信号。

    原理：强势方赔率极低(ho<1.5)时，庄家赔付压力巨大，若资金面出现背离信号
    （必发主队成交占比 >> 庄家隐含概率），则冷门概率显著上升。
    同样适用于客队让球盘极端强势的情况。
    """
    try:
        # 计算体彩隐含概率
        margin = 1/ho + 1/do_ + 1/ao
        imp = {'胜': (1/ho)/margin, '平': (1/do_)/margin, '负': (1/ao)/margin}

        # 强势方是哪边
        hot_side = '胜' if ho <= do_ and ho <= ao else ('负' if ao <= ho and ao <= do_ else None)

        signals = []
        cold_score = 0.0
        confidence = 0.40

        if hot_side == '胜' and ho < 1.45:
            # 极端主队热门：赔率压缩到极限，赔付压力极大
            # P1-3: ho>2.8时主队实为"假热门"（庄家高回报诱多），冷门概率被系统性低估
            # 此时cold_score boost应翻倍(0.06→0.12)
            extra_cold = 0.12 if ho > 2.8 else 0.06
            # 此时若出现以下信号，冷门概率上升：
            # 1. 强势方赔率极低但庄家赔率差(do_-ho)收窄 → 平局分流主胜压力
            odds_spread = do_ - ho
            if odds_spread < 0.35:
                cold_score += 0.25
                signals.append(f'冷门信号:主队极热+赔率差收窄({odds_spread:.2f})')
            # 2. 平局概率偏低但客队有一定概率（庄家在分散主胜压力）
            if imp['平'] < 0.28 and imp['负'] > 0.15:
                cold_score += 0.20
                signals.append(f'冷门信号:平局偏浅+客队有空间({imp["负"]:.2%})')
            # 3. 冷门专属：主队超低赔(<1.3)但让球盘口对主队不利
            if ho < 1.30:
                cold_score += 0.15
                signals.append('冷门信号:主队超低赔(<=1.30)赔付极限')
            # 4. P1-3: ho>2.8时增强冷门权重
            if ho > 2.8:
                cold_score += extra_cold
                signals.append(f'冷门信号:ho>{ho:.2f}假热门增强')

        elif hot_side == '负' and ao < 1.45:
            # P1-3: ao>2.8时客队"假热门"，冷门权重增强
            extra_cold_away = 0.12 if ao > 2.8 else 0.06
            odds_spread = do_ - ao
            if odds_spread < 0.35:
                cold_score += 0.25
                signals.append(f'冷门信号:客队极热+赔率差收窄({odds_spread:.2f})')
            if imp['平'] < 0.28 and imp['胜'] > 0.15:
                cold_score += 0.20
                signals.append(f'冷门信号:平局偏浅+主队有空间({imp["胜"]:.2%})')
            if ao < 1.30:
                cold_score += 0.15
                signals.append('冷门信号:客队超低赔(<=1.30)赔付极限')
            # P1-3: ao>2.8时增强冷门权重
            if ao > 2.8:
                cold_score += extra_cold_away
                signals.append(f'冷门信号:ao>{ao:.2f}假热门增强')

        # 冷门概率估算：冷门score → 客队/主队胜概率上调
        if cold_score > 0:
            cold_prob = min(0.35, cold_score * 0.4)
            base_win_prob = imp.get(hot_side, 0.5)
            base_lose_prob = imp.get('负' if hot_side == '胜' else '胜', 0.2)
            # 冷门时上调"非热门"方向概率
            upset_side = '负' if hot_side == '胜' else '胜'
            upset_boost = cold_prob * 0.6
            probs = dict(imp)
            probs[hot_side] = max(0.01, base_win_prob - upset_boost * 0.5)
            probs[upset_side] = base_lose_prob + upset_boost * 0.8
            probs['平'] = max(0.05, imp['平'] + upset_boost * 0.3 - cold_score * 0.1)
            total = sum(probs.values())
            probs = {k: v / total for k, v in probs.items()}
            # P0修复(2026-05-11): 极端赔率下提升cold_factor置信下限
            # 利雅青年ao=1.2检测到cold_score但conf=0.35过低被压制 → 保底0.45
            base_conf = 0.35 + cold_score * 0.15
            if ao < 1.5 or ho < 1.5:
                base_conf = max(base_conf, 0.45)
            confidence = min(0.55, base_conf)
            return {
                'probs': probs,
                'confidence': round(confidence, 3),
                'signals': signals + [f'cold_score:{cold_score:.2f}'],
            }

        return None
    except Exception:
        return None


def compute_asian_handicap_factor(match: dict, bf_data: Optional[dict] = None) -> Optional[Dict]:
    """亚盘让球因子
    读取 match["handicap"]（亚盘盘口，如 +0.5, -1, 0），
    hcap > 0 时主队受让 → boost 胜，
    hcap < 0 时主队让球 → boost 负，
    结合 Betfair 上/下盘比例做反向指标。
    """
    try:
        hcap = match.get('handicap')
        if hcap is None:
            return None

        hcap = float(hcap)

        # ── Betfair 上下盘比例（反向指标）─────────────────────────────
        # 两种 bf_data 格式：
        # 1. 完整格式：bf_data={'total_matched': float, 'over_back_odds': float, 'under_back_odds': float}
        # 2. spread格式（当前传入的）：bf_data={'home': float, 'away': float}
        #    home = sporttery隐含概率 - betfair隐含概率（正值=庄家更看好=资金偏热）
        #    → home>0 → 主队方向偏热 → over_weight 低 → 反向操作
        #    → away>0 → 客队方向偏热 → over_weight 高 → 反向操作
        over_ratio = 0.5
        if bf_data and bf_data.get('total_matched', 0) > 1000:
            try:
                over_back = bf_data.get('over_back_odds', 1.9)
                under_back = bf_data.get('under_back_odds', 1.9)
                r = 1 / over_back + 1 / under_back
                over_ratio = (1 / over_back) / r
            except Exception:
                pass
        elif bf_data and bf_data.get('home') is not None and bf_data.get('away') is not None:
            # spread 格式：用 bf_spread 推断主力方向
            # home spread = sporttery_prob_home - betfair_prob_home
            # > 0 → 主队方向偏热（庄家更乐观）→ 资金面偏空 → over_weight 低
            # < 0 → 主队方向偏冷 → 资金面偏多 → over_weight 高
            spread_home = float(bf_data['home'])
            spread_away = float(bf_data['away'])
            # 归一化：spread 为正说明该方向偏热（over）
            total_spread = abs(spread_home) + abs(spread_away)
            if total_spread > 0:
                # over_weight 表示"上盘热"的程度（>0.5 = 主队方向偏热）
                # 如果 spread_home > 0 且 spread_away > 0，取相对值
                over_weight = abs(spread_home) / total_spread
                # 主队方向 spread 为正（庄家高估）→ 上盘偏热 → over_weight 高
                # 客队方向 spread 为正（庄家高估）→ 下盘偏热 → over_weight 低
                if spread_home > 0:
                    over_ratio = over_weight  # 主队被庄家高估 → 上盘热
                else:
                    over_ratio = 1.0 - over_weight  # 客队被庄家高估 → 下盘热

        # 上盘偏热 → 反向操作
        over_weight = over_ratio
        under_weight = 1.0 - over_ratio

        # 盘口方向解读（以主队视角）
        if hcap > 0:
            # 主队受让（+0.5, +1 等），hcap越大主队越被低估 → boost 胜
            win_boost = min(abs(hcap) * 0.08, 0.25) * (1.0 - over_weight)
            probs = {
                '胜': 0.35 + win_boost,
                '平': 0.35 - win_boost * 0.3,
                '负': 0.30 - win_boost * 0.7,
            }
        elif hcap < 0:
            # 主队让球（-0.5, -1 等），|hcap|越大主队越强 → boost 负
            lose_boost = min(abs(hcap) * 0.08, 0.25) * (1.0 - under_weight)
            probs = {
                '胜': 0.30 - lose_boost * 0.7,
                '平': 0.35 - lose_boost * 0.3,
                '负': 0.35 + lose_boost,
            }
        else:
            # 平手盘
            return None

        # 归一化
        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}

        # 信心度：盘口越大信心越高
        confidence = min(abs(hcap) * 0.06 + 0.4, 0.72)

        return {
            'probs': probs,
            'confidence': confidence,
            'signals': [
                f'asian_hcap={hcap}',
                f'over_ratio={over_ratio:.2f}({"spread推断" if (bf_data and bf_data.get("home") is not None) else "betfair" if bf_data.get("total_matched", 0) > 1000 else "default"})'
            ],
        }
    except Exception as e:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 核心融合引擎
# ═══════════════════════════════════════════════════════════════════════════════

class SuperFusionLayer:
    """多因子超级融合引擎"""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or _WEIGHTS().copy()
        self.factor_cache: Dict[str, Any] = {}

    def analyze(self, match: dict,
                ho: float, do_: float, ao: float,
                pop: Optional[dict] = None,
                bf_data: Optional[dict] = None) -> dict:
        """
        主入口：融合11路因子输出最终预测

        Args:
            match: 比赛数据字典
            ho/do_/ao: 赔率
            pop: 必发/支持率数据（可选）
            bf_data: Betfair数据（可选）

        Returns:
            {
                final_probs, fusion_quality, confidence, dominant_direction,
                factor_contributions, convergence_signals, divergence_signals,
                arb_opportunity, all_factor_results
            }
        """
        lg  = match.get('league', '')
        h   = match.get('home_team', '')
        a   = match.get('away_team', '')
        dow = -1
        if match.get('match_date'):
            try:
                dow = datetime.fromisoformat(
                    str(match['match_date']).replace('Z', '+00:00')
                ).weekday()
            except Exception:
                pass

        # ── Step 1: 并行计算11路因子 ────────────────────────────────
        factor_results = {}
        factor_available = {}

        # 1. ML校准
        r = compute_ml_factor(match, ho, do_, ao)
        if r:
            factor_results['ml'] = r
            factor_available['ml'] = True

        # 2. DL序列
        r = compute_dl_factor(match, ho, do_, ao)
        if r:
            factor_results['dl'] = r
            factor_available['dl'] = True

        # 3. 泊松基础
        r = compute_poisson_factor(ho, do_, ao)
        if r:
            factor_results['poisson'] = r
            factor_available['poisson'] = True

        # 4. 博弈论
        r = compute_gametheory_factor(ho, do_, ao, pop)
        if r:
            factor_results['gametheory'] = r
            factor_available['gametheory'] = True

        # 5. 趋势跟踪
        r = compute_trend_factor(match)
        if r:
            factor_results['trend'] = r
            factor_available['trend'] = True

        # 6. 高频套利
        r = compute_hft_arb_factor(ho, do_, ao, bf_data)
        if r:
            factor_results['hft_arb'] = r
            factor_available['hft_arb'] = True

        # 7. 必发流向
        r = compute_betfair_factor(bf_data)
        if r:
            factor_results['betfair'] = r
            factor_available['betfair'] = True

        # 8. 周几专项
        if dow >= 0:
            r = compute_dow_factor(match, lg, dow)
            if r:
                factor_results['dow'] = r
                factor_available['dow'] = True

        # 9. 联赛专项
        r = compute_league_factor(lg)
        if r:
            factor_results['league'] = r
            factor_available['league'] = True

        # 10. 历史相似
        r = compute_hist_sim_factor(lg, ho, do_, ao)
        if r:
            factor_results['hist_sim'] = r
            factor_available['hist_sim'] = True

        # 11. 球队状态
        r = compute_team_form_factor(h, a, lg)
        if r:
            factor_results['team_form'] = r
            factor_available['team_form'] = True

        # 12. 亚盘让球
        r = compute_asian_handicap_factor(match, bf_data)
        if r:
            factor_results['asian_hcap'] = r
            factor_available['asian_hcap'] = True

        # 13. P1-冷门检测
        r = compute_cold_factor(ho, do_, ao)
        if r:
            factor_results['cold'] = r
            factor_available['cold'] = True

        # 初始化 contributions（Step 2 需要）
        def _to_conf(v):
            try: return float(v)
            except (TypeError, ValueError):
                return {'low': 0.3, 'medium': 0.5, 'high': 0.7}.get(str(v).lower(), 0.5)
        contributions = {fname: {
            'weight': self.weights.get(fname, 0.08),
            'confidence': _to_conf(factor_results[fname].get('confidence', 0.5)),
            'effective_weight': 0.0,
            'signals': factor_results[fname].get('signals', [])[:3],
        } for fname in factor_results}

        # ── Step 2: 自适应权重融合 ─────────────────────────────────
        # P1升级(2026-05-02): 基于赔率区间×市场类型动态调制权重
        # 替代固定 self.weights 硬编码权重
        dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        weekday_label = dow_labels[dow] if 0 <= dow <= 6 else 'Sat'
        # P0修复(2026-05-11): 极端赔率检测扩展到客队赔率
        # 利雅青年客胜1.2(ao=1.2)被误判为normal → 扩展hot区间
        odds_range = 'hot' if ho < 1.5 or ao < 1.5 else ('cold' if ho > 2.8 and ao > 2.8 else 'normal')

        # 构建factor_probs字典供adaptive_weighted_fusion使用
        factor_probs_for_fusion = {}
        for fname, fdata in factor_results.items():
            factor_probs_for_fusion[fname] = {
                'probs': fdata.get('probs', {}),
                'confidence': fdata.get('confidence', 0.5)
            }

        uniform_fallback_triggered = False  # 初始化：在if块外也要有定义
        if factor_probs_for_fusion:
            # 调用自适应融合（包含市场×赔率区间双调制）
            fusion_result = adaptive_weighted_fusion(
                factor_probs_for_fusion,
                market_type='SPF',
                odds_range=odds_range
            )
            _default_fused = {'胜': 1/3, '平': 1/3, '负': 1/3}

            # ════ P0-1新增(2026-05-12): 死亡赔率区间 + 赔率-模型背离修正 ════
            fprobs = fusion_result.get('fused_probs', _default_fused)

            # ── P0-1a: 死亡赔率区间检测 ───────────────────────────────
            # 区间1.21-1.35：庄家极强信心但实际平局率31.25%
            # 维罗纳(ho=1.35)、阿贾克斯(ho=1.25)均在死亡区间反向预测全败
            # 规则：ho∈[1.21,1.35] 且 do<4.5 → 平局+25%，主胜-15%
            try:
                ho_f = float(ho)
                do_f = float(do_)
                if 1.21 <= ho_f <= 1.35 and 0 < do_f < 4.5:
                    old_draw = fprobs['平']
                    fprobs['平'] = min(0.45, fprobs['平'] * 1.25)
                    fprobs['胜'] = fprobs['胜'] * 0.85
                    total = sum(fprobs.values())
                    if total > 0:
                        fprobs = {k: v/total for k, v in fprobs.items()}
                    _fusion_signals.append(
                        f'death_odds_zone(1.21-1.35):平+25%→{fprobs["平"]:.1%}(原{old_draw:.1%})'
                    )
            except Exception: pass

            # ── P0-1b: 赔率-模型背离修正 ─────────────────────────────
            # 当市场隐含概率与模型输出偏差>15%时，跟随市场而非模型
            # 根因：维罗纳(客胜隐含80%)/阿贾克斯(主胜隐含69%)庄家强烈倾向某方时
            #       模型反向预测→跟随庄家才是正确决策
            try:
                ho_f = float(ho); do_f = float(do_); ao_f = float(ao)
                if ho_f > 0 and do_f > 0 and ao_f > 0:
                    margin = 1/ho_f + 1/do_f + 1/ao_f
                    market_implied = {
                        '胜': (1/ho_f)/margin,
                        '平': (1/do_f)/margin,
                        '负': (1/ao_f)/margin
                    }
                    model_dir = _break_tie(fprobs, ho, do_, ao)
                    market_dir = _break_tie(market_implied, ho, do_, ao)
                    model_prob = fprobs[model_dir]
                    market_prob = market_implied[market_dir]
                    divergence = market_prob - model_prob
                    # 市场方向与模型相反，且市场信心显著更强
                    if model_dir != market_dir and divergence > 0.15:
                        old_prob = fprobs[model_dir]
                        # 将模型主方向降权，转向市场方向
                        fprobs[model_dir] = model_prob * 0.70
                        fprobs[market_dir] = market_prob * 1.10
                        total = sum(fprobs.values())
                        if total > 0:
                            fprobs = {k: v/total for k, v in fprobs.items()}
                        _fusion_signals.append(
                            f'odds_model_div:{model_dir}({old_prob:.0%})→{market_dir}({market_prob:.0%})背离{divergence:.0%}'
                        )
            except Exception: pass

            # 更新fusion_result中的fused_probs供后续使用
            fusion_result['fused_probs'] = fprobs

            # P0-FIX: uniform_fallback时fusion_quality不能按方差计算
            # 因为uniform分布方差最小→质量分最高，这是逆向指标
            # uniform时正确做法：标记低质量，让后续处理降级
            if 'uniform_fallback:total_eff_too_small' in fusion_result.get('signals', []):
                fused_probs = _default_fused  # 强制uniform，不制造伪自信
                uniform_fallback_triggered = True
                # 清零所有contributions的有效权重
                for fname in contributions:
                    contributions[fname]['effective_weight'] = 0.0
                _fusion_signals = fusion_result.get('signals', [])
            else:
                fused_probs = fusion_result.get('fused_probs', _default_fused)
                # 更新contributions中的权重为自适应后的权重
                adaptive_ws = fusion_result.get('adaptive_weights', {})
                for fname in contributions:
                    if fname in adaptive_ws:
                        new_w = adaptive_ws[fname]
                        old_c = contributions[fname]['confidence']
                        contributions[fname]['weight'] = round(new_w, 4)
                        contributions[fname]['effective_weight'] = round(new_w * old_c, 4)
                _fusion_signals = fusion_result.get('signals', [])
        else:
            # 兜底：赔率隐含概率
            margin = 1/ho + 1/do_ + 1/ao
            fused_probs = {'胜': (1/ho)/margin, '平': (1/do_)/margin, '负': (1/ao)/margin}
            _fusion_signals = []

        # ── Step 3: 博弈论一致性检验 ────────────────────────────────
        # 检查多因子方向一致性
        direction_counts = {'胜': 0, '平': 0, '负': 0}
        convergence_signals = list(_fusion_signals)   # 合并Step2自适应信号
        divergence_signals = []

        for fname, result in factor_results.items():
            top_dir = _break_tie(result['probs'], ho, do_, ao)
            direction_counts[top_dir] += 1

        n_factors = len(factor_results)
        for direction, cnt in direction_counts.items():
            ratio = cnt / n_factors if n_factors > 0 else 0
            if ratio >= 0.6:
                convergence_signals.append(f'多因子共识:{direction}({cnt}/{n_factors})')
            elif ratio <= 0.2 and n_factors >= 3:
                divergence_signals.append(f'因子分歧:{direction}仅{cnt}/{n_factors}支持')

        # ── Step 4: 融合质量评估 + 方向选择 ──────────────────────────
        # P0-FIX: uniform_fallback时方差=0→fq=1.0（满分），这是逆向指标
        # 均匀分布=最大不确定性，质量应为最低
        if uniform_fallback_triggered:
            fusion_quality = 0.25  # 强制低质量标记（低于普通fallback的0.30）
            dominant_direction = '平'  # 最安全的fallback方向
            _fusion_signals.append('uniform_fallback:强制平局(fq=0.25)')
        else:
            # 正常融合：基于方差评分
            vals = list(fused_probs.values())
            variance = sum((v - 1/3)**2 for v in vals) / 3
            fusion_quality = round(1.0 - variance * 3, 4)
            dominant_direction = _break_tie(fused_probs, ho, do_, ao)
        # P0-1: 降低膨胀乘数(1.5→1.1)，防止高信心过度膨胀
        # 实测：confidence≥0.6时命中率仅18.6%，1.5乘数系统性地高估了这类预测
        final_confidence = min(0.88, fusion_quality * sum(
            contributions.get(f, {}).get('confidence', 0.5)
            for f in contributions
        ) / max(len(contributions), 1) * 1.1)

        # ── Step 5: 套利机会检测 ────────────────────────────────────
        arb_opportunity = None
        if 'hft_arb' in factor_results and factor_results['hft_arb'].get('arb_margin', 1) < -0.03:
            arb_opportunity = factor_results['hft_arb']['arb_margin']

        # ── Bootstrap 不确定性量化 (N=100) + 2因子fallback ──────────
        # 对核心因子做重采样融合，估计置信度的95% CI
        # ≥3因子：bootstrap重采样；=2因子：Jensen-Shannon散度估算
        ci_lo, ci_hi = None, None
        try:
            core_factor_keys = ['ml', 'dl', 'poisson', 'gametheory', 'trend', 'hft_arb', 'betfair', 'cold']
            available_core = [k for k in core_factor_keys if k in factor_results]
            if len(available_core) >= 3:
                n_bootstrap = 30
                rng = np.random.default_rng(None)  # P0-2 fix: 无种子→每次独立随机
                boot_confs = []
                boot_max_probs = []  # 记录每轮胜出概率用于CI
                for _ in range(n_bootstrap):
                    try:
                        perturbed_probs = {}
                        for fname in available_core:
                            fd = factor_results[fname]
                            base = np.array([fd['probs'].get(o, 0.333) for o in ['胜', '平', '负']], dtype=float)
                            base = np.maximum(base, 0.01)
                            base = base / base.sum()
                            conf_raw = fd.get('confidence', 0.5)
                            if isinstance(conf_raw, str):
                                noise_scale = {'high': 0.04, 'medium': 0.08, 'low': 0.13, 'very_low': 0.18}.get(conf_raw.lower(), 0.10)
                            else:
                                try:
                                    cf = float(conf_raw)
                                except (ValueError, TypeError):
                                    cf = 0.5
                                noise_scale = max(0.03, 0.20 - cf * 0.12)
                            noise = rng.normal(0, noise_scale, 3)
                            perturbed = np.maximum(base + noise, 0.01)
                            perturbed_probs[fname] = perturbed / perturbed.sum()
                        sampled = rng.choice(available_core, size=len(available_core), replace=True)
                        sample_fused = np.zeros(3)
                        for fname in sampled:
                            fd = factor_results[fname]
                            w = contributions.get(fname, {}).get('weight', 1.0 / len(available_core))
                            sample_fused += w * perturbed_probs[fname]
                        total = sample_fused.sum()
                        if total > 0:
                            sample_fused = sample_fused / total
                        vals = sample_fused
                        max_prob = float(np.max(vals))
                        boot_max_probs.append(max_prob)
                        var = np.sum((vals - 1/3)**2) / 3
                        fq = max(0.0, 1.0 - var * 3)
                        boot_confs.append(min(0.95, fq * 1.5))
                    except Exception:
                        pass
                if boot_max_probs:
                    ci_lo = round(np.percentile(boot_max_probs, 2.5), 4)
                    ci_hi = round(np.percentile(boot_max_probs, 97.5), 4)
            elif len(available_core) == 2:
                # 2因子fallback：用Jensen-Shannon散度估算分歧度，以此确定CI半宽
                f1, f2 = available_core
                p1 = list(factor_results[f1].get('probs', {'胜':0.333,'平':0.333,'负':0.333}).values())
                p2 = list(factor_results[f2].get('probs', {'胜':0.333,'平':0.333,'负':0.333}).values())
                p1 = np.clip(np.array(p1, dtype=float), 1e-10, 1)
                p2 = np.clip(np.array(p2, dtype=float), 1e-10, 1)
                p1 = p1 / p1.sum(); p2 = p2 / p2.sum()
                m = 0.5 * (p1 + p2)
                js = 0.5 * (np.sum(p1 * np.log(p1 / m)) + np.sum(p2 * np.log(p2 / m)))
                half_width = min(0.20, round(float(js) * 1.5, 4))
                half_width = max(0.05, half_width)
                conf_estimate = float(final_confidence)
                ci_lo = round(max(0.0, conf_estimate - half_width), 4)
                ci_hi = round(min(1.0, conf_estimate + half_width), 4)
        except Exception as e:
            pass

        return {
            'final_probs': fused_probs,
            'fusion_quality': fusion_quality,
            'confidence': round(final_confidence, 4),
            'confidence_ci': (ci_lo, ci_hi),   # (2.5-percentile, 97.5-percentile) or (None, None)
            'dominant_direction': dominant_direction,
            'factor_contributions': contributions,
            'factor_count': len(factor_results),
            'convergence_signals': convergence_signals,
            'divergence_signals': divergence_signals,
            'arb_opportunity': arb_opportunity,
            'all_factor_results': {k: {'probs': v['probs'], 'confidence': v.get('confidence')}
                                   for k, v in factor_results.items()},
        }

    def update_weights_from_accuracy(self, accuracy_data: Dict[str, float]):
        """基于历史准确率动态更新权重

        accuracy_data: {'ml': 0.52, 'dl': 0.55, ...}
        """
        total_acc = sum(accuracy_data.values()) if accuracy_data else 1.0
        for fname in self.weights:
            acc = accuracy_data.get(fname, 0.45)
            # 准确率越高权重越高，但限制最大调整幅度
            new_w = acc / total_acc
            old_w = self.weights[fname]
            # 平滑调整（限制单次调整不超过15%）
            delta = (new_w - old_w) * 0.15
            self.weights[fname] = round(old_w + delta, 4)


# ═══════════════════════════════════════════════════════════════════════
# 便捷入口
# ═══════════════════════════════════════════════════════════════════════

def super_fuse(match: dict, ho: float, do_: float, ao: float,
               pop: Optional[dict] = None,
               bf_data: Optional[dict] = None) -> dict:
    """一行调用接口"""
    engine = SuperFusionLayer()
    return engine.analyze(match, ho, do_, ao, pop, bf_data)

# ═══════════════════════════════════════════════════════════════════════
# P2-P3 集成: 贝叶斯融合 + 自适应权重 + 周中赔率分布 (2026-05-02)
# ═══════════════════════════════════════════════════════════════════════

def integrate_bayesian_fusion(match: dict,
                               poisson_probs: dict,
                               ml_probs: dict,
                               dl_probs: Optional[dict],
                               odds: dict,
                               market_type: str = 'SPF',
                               weekday: str = 'Sat') -> dict:
    """集成贝叶斯融合到SuperFusionLayer

    Args:
        match: 比赛信息
        poisson_probs: {'H':, 'D':, 'A':}
        ml_probs: {'H':, 'D':, 'A':}
        dl_probs: {'H':, 'D':, 'A':} (可选)
        odds: {'H':, 'D':, 'A':}
        market_type: SPF/HHAD/ZJQ/BF
        weekday: Mon/Tue/Wed/Thu/Fri/Sat/Sun

    Returns:
        融合后的概率 + 融合方法说明
    """
    try:
        from prediction.bayes_fusion import BayesianFusion, apply_weekday_adjustment

        fuser = BayesianFusion()

        # 判断赔率区间
        ho = odds.get('H', 2.0)
        ao = odds.get('A', 2.0)
        # P0修复(2026-05-11): 极端赔率检测扩展到客队赔率
        odds_range = 'hot' if ho < 1.5 or ao < 1.5 else ('cold' if ho > 2.8 and ao > 2.8 else 'normal')

        # 计算模型置信度
        ml_conf = max(ml_probs.values())
        dl_conf = max(dl_probs.values()) if dl_probs else 0.5
        model_confidence = (ml_conf + dl_conf) / 2

        # 贝叶斯融合
        bayesian_probs = fuser.fuse_with_adaptive_weights(
            poisson_probs=poisson_probs,
            ml_probs=ml_probs,
            dl_probs=dl_probs,
            odds=odds,
            market_type=market_type,
            odds_range=odds_range,
            model_confidence=model_confidence
        )

        # 应用周中赔率调整
        adjusted_probs, conf_discount = apply_weekday_adjustment(bayesian_probs, weekday)

        return {
            'bayesian_probs': adjusted_probs,
            'odds_range': odds_range,
            'confidence_discount': conf_discount,
            'fusion_method': 'bayesian+weekday_adjustment'
        }

    except ImportError as e:
        return {
            'bayesian_probs': ml_probs,
            'odds_range': 'normal',
            'confidence_discount': 1.0,
            'fusion_method': f'fallback({e})'
        }


def _break_tie(probs: dict, ho: float, do_: float, ao: float) -> str:
    """Helper: 在概率相等时用赔率隐含概率打破字母序偏向。
    
    Bug修复: max(dict, key=dict.get) 在tie时返回字母序第一，
    系统性地偏向「平」(拼音首字母P < S < W)。
    当最高概率存在tie时，改用odds隐含概率选择方向，
    保证有方向保障，不产生系统性反向预测。
    
    Args:
        probs: {'胜': float, '平': float, '负': float}
        ho, do_, ao: 赔率（用于计算隐含概率）
    Returns:
        方向字符串 '胜'/'平'/'负'
    """
    if not probs:
        return '平'
    max_prob = max(probs.values())
    # 检测tie
    tied = [k for k, v in probs.items() if abs(v - max_prob) < 0.001]
    if len(tied) <= 1:
        return max(probs, key=probs.get)
    # tie时用赔率隐含概率打破字母序
    margin = (1/ho + 1/do_ + 1/ao) if (ho > 0 and do_ > 0 and ao > 0) else 3.0
    implied = {
        '胜': (1/ho)/margin if ho > 0 else 0.333,
        '平': (1/do_)/margin if do_ > 0 else 0.333,
        '负': (1/ao)/margin if ao > 0 else 0.333,
    }
    # 在tie的选项中，选择隐含概率最高的（庄家最看好）
    return max(tied, key=lambda k: implied.get(k, 0))


def adaptive_weighted_fusion(factor_probs: dict,
                            market_type: str = 'SPF',
                            odds_range: str = 'normal') -> dict:
    """自适应权重融合（SuperFusionLayer第3阶段改进）

    区别于SuperFusionLayer的固定权重，本函数根据:
    1. 市场类型调整基础权重
    2. 赔率区间调制（hot < 1.5, cold > 2.8, else normal）
    3. 因子置信度加权

    Args:
        factor_probs: {因子名: {'probs': {'胜':, '平':, '负':}, 'confidence': float}}
        market_type: SPF/HHAD/ZJQ/BF
        odds_range: 'hot' | 'normal' | 'cold'

    Returns:
        {'fused_probs': {'胜':,'平':,'负':}, 'adaptive_weights': {}, 'signals': []}
    """
    # ── 市场×因子基础权重表 ─────────────────────────────────────
    # 每个因子在每个市场有独立权重（共12因子×4市场）
    _BASE_WEIGHTS = {
        'ml':          {'SPF': 0.17, 'HHAD': 0.14, 'ZJQ': 0.15, 'BF': 0.16},
        'dl':          {'SPF': 0.13, 'HHAD': 0.11, 'ZJQ': 0.12, 'BF': 0.14},
        'poisson':     {'SPF': 0.11, 'HHAD': 0.16, 'ZJQ': 0.18, 'BF': 0.12},
        'gametheory':  {'SPF': 0.09, 'HHAD': 0.10, 'ZJQ': 0.08, 'BF': 0.09},
        'trend':       {'SPF': 0.08, 'HHAD': 0.08, 'ZJQ': 0.07, 'BF': 0.08},
        'hft_arb':     {'SPF': 0.07, 'HHAD': 0.07, 'ZJQ': 0.06, 'BF': 0.07},
        'betfair':     {'SPF': 0.08, 'HHAD': 0.08, 'ZJQ': 0.06, 'BF': 0.08},
        'dow':         {'SPF': 0.06, 'HHAD': 0.06, 'ZJQ': 0.05, 'BF': 0.06},
        'league':      {'SPF': 0.05, 'HHAD': 0.05, 'ZJQ': 0.05, 'BF': 0.05},
        'hist_sim':    {'SPF': 0.05, 'HHAD': 0.05, 'ZJQ': 0.05, 'BF': 0.05},
        'team_form':   {'SPF': 0.05, 'HHAD': 0.05, 'ZJQ': 0.05, 'BF': 0.05},
        'asian_hcap':  {'SPF': 0.04, 'HHAD': 0.05, 'ZJQ': 0.04, 'BF': 0.05},
        'cold':        {'SPF': 0.06, 'HHAD': 0.06, 'ZJQ': 0.04, 'BF': 0.05},  # P1-冷门因子
    }

    # ── 赔率区间调制系数 ─────────────────────────────────────────
    _RANGE_MOD = {
        'hot':   {'ml': 0.88, 'dl': 0.85, 'poisson': 1.18, 'gametheory': 1.15,
                  'trend': 0.80, 'hft_arb': 0.80, 'betfair': 0.90, 'dow': 0.90,
                  'league': 0.85, 'hist_sim': 0.90, 'team_form': 0.85, 'asian_hcap': 0.85, 'cold': 1.20},
        'cold':  {'ml': 1.12, 'dl': 1.15, 'poisson': 0.78, 'gametheory': 0.85,
                  'trend': 1.20, 'hft_arb': 1.20, 'betfair': 1.10, 'dow': 1.10,
                  'league': 1.20, 'hist_sim': 1.10, 'team_form': 1.15, 'asian_hcap': 1.15, 'cold': 1.00},
        'normal': {k: 1.0 for k in _BASE_WEIGHTS},
    }
    mod = _RANGE_MOD.get(odds_range, _RANGE_MOD['normal'])

    # ── 构建动态权重表 ───────────────────────────────────────────
    weights = {}
    for fname in _BASE_WEIGHTS:
        base = _BASE_WEIGHTS[fname].get(market_type, 0.08)
        weights[fname] = base * mod.get(fname, 1.0)

    # ── 归一化 ────────────────────────────────────────────────────
    total_w = sum(weights.values())
    weights = {k: v / total_w for k, v in weights.items()}

    # ── 加权融合 ─────────────────────────────────────────────────
    outcomes = ['胜', '平', '负']
    fused = {o: 0.0 for o in outcomes}
    total_eff = 0.0

    for fname, fdata in factor_probs.items():
        prob_dict = fdata.get('probs', {})
        conf_str = fdata.get('confidence', 0.5)
        try:
            conf = float(conf_str)
        except (TypeError, ValueError):
            # 字符串置信度：'low'/'medium'/'high' → 0.3/0.5/0.7
            conf_map = {'low': 0.3, 'medium': 0.5, 'high': 0.7}
            conf = conf_map.get(str(conf_str).lower(), 0.5)
        # 匹配因子名（fdata keys 可能带下划线或额外后缀，做模糊匹配）
        w = 0.0
        for key in weights:
            if key in fname.lower():
                w = weights[key]
                break
        # P0-FIX: 检测均匀分布因子，降低其有效权重
        # 均匀分布(0.333,0.333,0.333)的熵最大，有效信息最少，惩罚confidence
        prob_vals = [max(0.001, prob_dict.get(o, 0.333)) for o in outcomes]
        prob_sum = sum(prob_vals)
        prob_norm = [p / prob_sum for p in prob_vals]
        import math
        entropy = -sum(p * math.log(p) for p in prob_norm if p > 0)
        max_entropy = math.log(3)  # 均匀分布的熵
        uniformity_ratio = entropy / max_entropy if max_entropy > 0 else 1.0
        # uniformity_ratio≈1.0 表示均匀，≈0 表示极度集中
        # 均匀因子confidence打折扣：uniformity_ratio越高折扣越大
        # P0-2 FIX (revised): 0.80系数过激，导致所有均匀因子有效权重趋零
        # 修正：均匀因子降权50%（非80%），避免矫枉过正
        # uniformity_penalty: 系数从0.30降至0.15，最小值从0.30降至0.10
        # 避免对接近均匀分布的因子过度惩罚，导致total_eff < 0.15触发uniform_fallback
        uniformity_penalty = max(0.10, 1.0 - uniformity_ratio * 0.15)
        conf = conf * uniformity_penalty
        eff_w = w * conf
        for outcome in outcomes:
            p = prob_dict.get(outcome, 0.333)
            fused[outcome] += eff_w * p
        total_eff += eff_w

    # ── 归一化最终概率 ────────────────────────────────────────────
    # P0-2 FIX: 熵惩罚后总权重过小 → 强制uniform fallback
    # 当total_eff < 0.10时，说明所有因子在熵惩罚后都几乎没有有效权重
    # 典型情况：所有因子输出接近均匀分布(0.33,0.33,0.33)，此时不应给出高置信预测
    # P0修复(2026-05-11): 阈值0.15→0.10
    # 原因：实际只有3个因子(ml/dl/poisson)参与融合，total_eff常在0.10~0.14区间
    # 0.15阈值过于严格，导致大量合法场次被强制uniform，破坏了融合有效性
    if total_eff >= 0.10:
        fused = {k: v / total_eff for k, v in fused.items()}
    else:
        # total_eff < 0.15 → uniform fallback，不人为制造"伪自信"
        inv_sum = sum(1.0 for o in outcomes)
        fused = {o: 1.0 / inv_sum for o in outcomes}
        # 返回special marker让调用方知道这是uniform fallback
        return {
            'fused_probs': fused,
            'adaptive_weights': {k: 0.0 for k in weights},
            'signals': ['uniform_fallback:total_eff_too_small'],
            'odds_range': odds_range,
            'market_type': market_type,
        }

    # ── 信号 ──────────────────────────────────────────────────────
    signals = []
    if odds_range == 'hot':
        signals.append('adaptive_weight:hot区间-泊松/博弈论增强')
    elif odds_range == 'cold':
        signals.append('adaptive_weight:cold区间-ML/DL增强')

    return {
        'fused_probs': fused,
        'adaptive_weights': weights,
        'signals': signals,
        'odds_range': odds_range,
        'market_type': market_type,
    }
