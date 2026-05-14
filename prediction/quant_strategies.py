#!/usr/bin/env python3
"""
量化策略引擎 v1.0
四大量化策略：趋势策略 / 高频策略 / 套利策略 / 多因子

数据源:
  - titan007_*_matches: 初盘→终盘赔率变化
  - sporttery_support: 支持率 + 变化
  - betfair_trading_v2: 必发交易量/大单/Kelly
  - collected_match_data_2026: 体彩赔率

用法:
  from quant_strategies import QuantEngine
  engine = QuantEngine()
  signals = engine.analyze(match_data)
"""
from datetime import datetime

# ════ P1-2新增(2026-05-12): 联赛权重映射 ════
LEAGUE_WEIGHT_MAP = {
    # 五大联赛 + 重要杯赛 - 高可信度
    '英超': 1.0, '意甲': 1.0, '西甲': 1.0, '德甲': 1.0, '法甲': 1.0,
    '欧冠': 1.0, '欧洲冠军联赛': 1.0,
    # 次级联赛 - 中等可信度
    '荷甲': 0.85, '葡超': 0.85, '比甲': 0.85, '苏超': 0.80,
    '欧罗巴': 0.90, '欧协联': 0.75,
    # 小联赛 - 低可信度
    '中超': 0.70, 'J联赛': 0.75, 'K联赛': 0.75, '美职联': 0.70,
    # 默认
}


# === 数据库连接 ===
def _db_query(sql):
    import subprocess
    E = dict(PGPASSWORD='MyApp@1234')
    r = subprocess.run(
        ['psql', '-h', 'localhost', '-U', 'myapp', '-d', 'myapp_db', '-t', '-A', '-F', '|', '-c', sql],
        capture_output=True, text=True, env=E
    )
    return r.stdout.strip()

# ============================================================
# 策略一: 趋势策略 (Trend Following)
# 核心: 赔率变动方向 → 跟随或反转信号
# ============================================================

class TrendStrategy:
    """赔率趋势跟踪策略
    
    原理:
    - 庄家赔率调整是最强信号之一
    - 初盘→终盘的变化方向揭示庄家意图
    - 降盘降水 = 真正看好; 升盘升水 = 诱导
    
    信号类型:
    1. 一致降赔 (All Drop): 三个方向赔率同降某一方 → 强信号
    2. 降水不降盘: 减赔暗示，信号强度中等
    3. 反向升赔 (Reverse): 与基本面背离的赔率调整 → 陷阱
    """
    
    # 赔率变动阈值
    DROP_THRESHOLD = 0.05      # 降赔5分以上视为显著
    STRONG_DROP = 0.10         # 强降赔10分以上
    ASIAN_SHIFT = 0.25         # 亚盘升/降盘阈值
    
    def analyze(self, match: dict) -> dict:
        """分析单场比赛的赔率趋势
        
        Args:
            match: 包含 open/close 赔率数据的字典
        Returns:
            dict: trend_signal, trend_strength, trend_direction, details
        """
        signals = []
        
        # 1. 欧赔趋势 (home_win_odds_open → home_win_odds_close)
        euro = self._euro_trend(match)
        if euro:
            signals.append(euro)
        
        # 2. 亚盘趋势 (handicap_opening → handicap_close)
        asian = self._asian_trend(match)
        if asian:
            signals.append(asian)
        
        # 3. 大小球趋势 (ou_opening → ou_close)
        ou = self._ou_trend(match)
        if ou:
            signals.append(ou)
        
        # 4. 威廉希尔专项 (wh_open → wh_close)
        william = self._william_trend(match)
        if william:
            signals.append(william)
        
        # 综合评分
        return self._aggregate(signals)
    
    def _euro_trend(self, m):
        """欧赔1x2趋势分析"""
        ho, hc = m.get('home_win_odds_open'), m.get('home_win_odds_close')
        do_, dc = m.get('draw_odds_open'), m.get('draw_odds_close')
        ao, ac = m.get('away_win_odds_open'), m.get('away_win_odds_close')
        
        if not all([ho, hc, do_, dc, ao, ac]):
            return None
        
        try:
            ho, hc = float(ho), float(hc)
            do_, dc = float(do_), float(dc)
            ao, ac = float(ao), float(ac)
        except (ValueError, TypeError):
            return None
        
        dh = ho - hc  # 正=主胜降赔
        dd = do_ - dc
        da = ao - ac
        
        # 一致降赔信号
        if dh > self.DROP_THRESHOLD and da < -self.DROP_THRESHOLD:
            return {'type': 'euro_consistent_home', 'strength': min(dh * 5, 0.15),
                    'direction': 'home', 'detail': f'主赔-{dh:.2f}客赔+{abs(da):.2f}'}
        if da > self.DROP_THRESHOLD and dh < -self.DROP_THRESHOLD:
            return {'type': 'euro_consistent_away', 'strength': min(da * 5, 0.15),
                    'direction': 'away', 'detail': f'客赔-{da:.2f}主赔+{abs(dh):.2f}'}
        
        # 强降赔信号
        if dh > self.STRONG_DROP:
            return {'type': 'euro_strong_home', 'strength': min(dh * 3, 0.10),
                    'direction': 'home', 'detail': f'主赔强降{dh:.2f}'}
        if da > self.STRONG_DROP:
            return {'type': 'euro_strong_away', 'strength': min(da * 3, 0.10),
                    'direction': 'away', 'detail': f'客赔强降{da:.2f}'}
        
        return None
    
    def _asian_trend(self, m):
        """亚盘趋势分析"""
        ho = m.get('handicap_opening_numeric')
        hc = m.get('handicap_close_numeric')
        if ho is None or hc is None:
            return None
        
        try:
            ho, hc = float(ho), float(hc)
        except Exception:
            return None
        
        shift = ho - hc  # 正=升盘(利好主队)
        
        # 水位变化
        hw_open = m.get('handicap_opening_home_water')
        hw_close = m.get('handicap_close_home_water')
        water_change = 0
        if hw_open and hw_close:
            try:
                water_change = float(hw_close) - float(hw_open)
            except Exception:
                pass
        
        if abs(shift) >= self.ASIAN_SHIFT:
            direction = 'home' if shift > 0 else 'away'
            return {
                'type': f'asian_shift_{direction}',
                'strength': min(abs(shift) * 0.15, 0.12),
                'direction': direction,
                'detail': f'亚盘{("升盘" if shift > 0 else "降盘")}{abs(shift):.2f}水变{water_change:+.2f}'
            }
        
        # 降水信号
        if abs(shift) < 0.1 and water_change < -0.05:
            return {
                'type': 'asian_water_drop',
                'strength': min(abs(water_change) * 2, 0.06),
                'direction': 'home' if ho > 0 else 'away',
                'detail': f'亚盘水位下降{abs(water_change):.2f}'
            }
        
        return None
    
    def _ou_trend(self, m):
        """大小球趋势分析"""
        oo = m.get('ou_opening_numeric')
        oc = m.get('ou_close_numeric')
        if oo is None or oc is None:
            return None
        
        try:
            oo, oc = float(oo), float(oc)
        except Exception:
            return None
        
        diff = oc - oo  # 正=升盘(大球升)
        
        if abs(diff) >= 0.25:
            direction = 'over' if diff > 0 else 'under'
            return {
                'type': f'ou_shift_{direction}',
                'strength': 0.04,
                'direction': direction,
                'detail': f'大小球{"升" if diff > 0 else "降"}盘{abs(diff):.2f}'
            }
        
        return None
    
    def _william_trend(self, m):
        """威廉希尔专项: 初赔→终赔 (数据暂不完整，跳过)"""
        return None
    
    def _aggregate(self, signals):
        """聚合所有趋势信号"""
        if not signals:
            return {'signal': 'none', 'strength': 0, 'direction': 'neutral',
                    'details': '无显著趋势信号'}
        
        dir_score = {'home': 0, 'away': 0, 'over': 0, 'under': 0}
        for s in signals:
            d = s.get('direction', 'neutral')
            if d in dir_score:
                dir_score[d] += s['strength']
        
        best_dir = max(dir_score, key=dir_score.get)
        best_score = dir_score[best_dir]
        
        return {
            'signal': 'trend_follow' if best_score > 0.03 else 'weak',
            'strength': round(best_score, 4),
            'direction': best_dir,
            'details': ' | '.join(s['detail'] for s in signals)
        }


# ============================================================
# 策略二: 高频策略 (High-Frequency / Sharp Money Detection)
# 核心: 检测"聪明钱"流入 + 支持率异常变化
# ============================================================

class HighFreqStrategy:
    """高频/聪明钱检测策略
    
    原理:
    - 必发大单成交方向揭示机构意图
    - 支持率突然变化（散户涌入 vs 机构反向）
    - Kelly值异常 → 庄家定价偏差
    
    信号:
    1. 大单背离: 大单方向 vs 散户方向相反 → 跟大单
    2. Kelly偏差: kelly > 0 → 有正EV
    3. 支持率突变: Δsupport > 10% → 情绪信号
    """
    
    def analyze(self, match: dict, betfair: dict = None, support: dict = None) -> dict:
        """分析高频/聪明钱信号"""
        signals = []
        
        # 1. 必发聪明钱
        if betfair:
            bf = self._betfair_sharp(betfair)
            if bf:
                signals.append(bf)
        
        # 2. 支持率信号
        if support:
            sp = self._support_signal(support)
            if sp:
                signals.append(sp)
        
        return self._aggregate(signals)
    
    def _betfair_sharp(self, bf):
        """必发聪明钱检测"""
        try:
            home_pct = float(bf.get('home_pct', 0) or 0)
            away_pct = float(bf.get('away_pct', 0) or 0)
            home_big = float(bf.get('home_big_pct', 0) or 0)
            away_big = float(bf.get('away_big_pct', 0) or 0)
            home_kelly = float(bf.get('home_kelly', 0) or 0)
            away_kelly = float(bf.get('away_kelly', 0) or 0)
            total_vol = float(bf.get('total_volume', 0) or 0)
        except (ValueError, TypeError):
            return None
        
        details = []
        strength = 0
        direction = 'neutral'
        
        # 大单背离: 散户主队>60% 但大单客队>60% → 跟大单
        if home_pct > 55 and away_big > home_big + 15:
            direction = 'away'
            strength += 0.08
            details.append(f'大单背离:散户主{home_pct:.0f}%大单客{away_big:.0f}%')
        elif away_pct > 55 and home_big > away_big + 15:
            direction = 'home'
            strength += 0.08
            details.append(f'大单背离:散户客{away_pct:.0f}%大单主{home_big:.0f}%')
        
        # Kelly正值信号
        if home_kelly > 0.05:
            strength += min(home_kelly * 0.5, 0.06)
            direction = 'home' if direction == 'neutral' else direction
            details.append(f'主Kelly={home_kelly:.2f}')
        if away_kelly > 0.05:
            strength += min(away_kelly * 0.5, 0.06)
            direction = 'away' if direction == 'neutral' else direction
            details.append(f'客Kelly={away_kelly:.2f}')
        
        # 大成交量阈值
        if total_vol > 500000:  # 50万以上成交
            strength *= 1.3
            details.append(f'高成交{total_vol/10000:.0f}万')
        
        if not details:
            return None
        
        return {
            'type': 'sharp_money',
            'strength': round(min(strength, 0.15), 4),
            'direction': direction,
            'detail': ' | '.join(details)
        }
    
    def _support_signal(self, sp):
        """支持率变化信号"""
        try:
            hs = float(sp.get('home_support', 0) or 0)
            as_ = float(sp.get('away_support', 0) or 0)
            ds = float(sp.get('draw_support', 0) or 0)
            hp = float(sp.get('support_prev_home', 0) or 0)
            ap = float(sp.get('support_prev_away', 0) or 0)
        except (ValueError, TypeError):
            return None
        
        details = []
        strength = 0
        direction = 'neutral'
        
        # 支持率突变（>10%变化）
        d_home = hs - hp if hp > 0 else 0
        d_away = as_ - ap if ap > 0 else 0
        
        if abs(d_home) > 10:
            strength += 0.04
            details.append(f'主支持率变化{d_home:+.1f}%')
        if abs(d_away) > 10:
            strength += 0.04
            details.append(f'客支持率变化{d_away:+.1f}%')
        
        # 极端支持率 (>70%): 散户陷阱信号
        if hs > 70:
            strength += 0.03
            direction = 'away'  # 散户极端 → 反向
            details.append(f'主支持过热{hs:.0f}%→反指标')
        elif as_ > 70:
            strength += 0.03
            direction = 'home'
            details.append(f'客支持过热{as_:.0f}%→反指标')
        
        # 客队支持>45%反向指标（已验证：客胜率仅20%）
        if as_ >= 45:
            strength += 0.05
            direction = 'home'
            details.append(f'客支持{as_:.0f}%≥45→反向指标(历史客胜率20%)')
        
        if not details:
            return None
        
        return {
            'type': 'support_signal',
            'strength': round(strength, 4),
            'direction': direction,
            'detail': ' | '.join(details)
        }
    
    def _aggregate(self, signals):
        if not signals:
            return {'signal': 'none', 'strength': 0, 'direction': 'neutral', 'details': '无高频信号'}
        
        dir_score = {'home': 0, 'away': 0}
        for s in signals:
            d = s.get('direction', 'neutral')
            if d in dir_score:
                dir_score[d] += s['strength']
        
        best_dir = max(dir_score, key=dir_score.get)
        best_score = dir_score[best_dir]
        
        return {
            'signal': 'sharp_money' if best_score > 0.03 else 'weak',
            'strength': round(best_score, 4),
            'direction': best_dir,
            'details': ' | '.join(s['detail'] for s in signals)
        }


# ============================================================
# 策略三: 套利策略 (Statistical Arbitrage)
# 核心: 跨市场定价偏差 + Expected Value计算
# ============================================================

class ArbitrageStrategy:
    """统计套利策略
    
    原理:
    - 模型概率 vs 赔率隐含概率 → EV计算
    - 跨市场(欧赔/亚盘/必发)定价偏差
    - Closing Line Value (CLV): 跟终盘赔率的偏离
    
    信号:
    1. 正EV: 模型概率 > 赔率隐含概率 × (1+margin)
    2. CLV优势: 开盘赔率 > 终盘赔率 → 抓早
    3. 跨市场套利: 欧赔vs亚盘隐含不一致
    """
    
    MIN_EV = 0.05  # 最低5%正EV才触发
    
    def analyze(self, model_probs: dict, odds: dict, close_odds: dict = None) -> dict:
        """
        Args:
            model_probs: {home: 0.45, draw: 0.27, away: 0.28}
            odds: {home: 2.10, draw: 3.30, away: 3.50}
            close_odds: 终盘赔率（如有）
        """
        signals = []
        
        # 1. EV计算
        ev = self._calc_ev(model_probs, odds)
        if ev:
            signals.append(ev)
        
        # 2. CLV分析
        if close_odds:
            clv = self._clv_analysis(odds, close_odds)
            if clv:
                signals.append(clv)
        
        # 3. 返还率异常
        margin = self._margin_check(odds)
        if margin:
            signals.append(margin)
        
        return self._aggregate(signals)
    
    def _calc_ev(self, probs, odds):
        """计算三个方向的Expected Value"""
        results = []
        for outcome in ['home', 'draw', 'away']:
            p = probs.get(outcome, 0)
            o = odds.get(outcome, 0)
            if p <= 0 or o <= 0:
                continue
            implied = 1.0 / o
            ev_val = p * o - 1  # EV = P × Odds - 1
            
            if ev_val > self.MIN_EV:
                results.append({
                    'outcome': outcome,
                    'ev': ev_val,
                    'model_prob': p,
                    'implied_prob': implied,
                    'edge': p - implied,
                })
        
        if not results:
            return None
        
        best = max(results, key=lambda x: x['ev'])
        return {
            'type': 'positive_ev',
            'strength': round(min(best['ev'] * 2, 0.15), 4),
            'direction': best['outcome'],
            'detail': f"{best['outcome']}EV={best['ev']:.1%}(模型{best['model_prob']:.0%}vs隐含{best['implied_prob']:.0%})",
            'ev_data': results,
        }
    
    def _clv_analysis(self, open_odds, close_odds):
        """Closing Line Value分析"""
        best = None
        for outcome in ['home', 'draw', 'away']:
            oo = open_odds.get(outcome, 0)
            co = close_odds.get(outcome, 0)
            if oo <= 0 or co <= 0:
                continue
            diff = oo - co  # 正=开盘赔率高于终盘 → 有CLV优势
            if diff > 0.05:
                if best is None or diff > best[1]:
                    best = (outcome, diff)
        
        if not best:
            return None
        
        return {
            'type': 'clv_advantage',
            'strength': round(min(best[1] * 0.5, 0.08), 4),
            'direction': best[0],
            'detail': f'{best[0]}开盘{open_odds[best[0]]:.2f}>终盘{close_odds[best[0]]:.2f}'
        }
    
    def _margin_check(self, odds):
        """返还率/保证金异常检测"""
        h, d, a = odds.get('home', 0), odds.get('draw', 0), odds.get('away', 0)
        if h <= 0 or d <= 0 or a <= 0:
            return None
        
        overround = 1/h + 1/d + 1/a  # 应≈1.05-1.10
        margin = overround - 1
        
        if margin < 0.02:  # 异常低返还率 → 可能是错误定价
            return {
                'type': 'low_margin',
                'strength': 0.03,
                'direction': 'neutral',
                'detail': f'返还率异常高{(1-margin)*100:.1f}%→可能定价偏差'
            }
        
        return None
    
    def _aggregate(self, signals):
        if not signals:
            return {'signal': 'none', 'strength': 0, 'direction': 'neutral',
                    'details': '无套利信号', 'ev_data': None}
        
        dir_score = {'home': 0, 'draw': 0, 'away': 0}
        ev_data = None
        for s in signals:
            d = s.get('direction', 'neutral')
            if d in dir_score:
                dir_score[d] += s['strength']
            if 'ev_data' in s:
                ev_data = s['ev_data']
        
        best_dir = max(dir_score, key=dir_score.get)
        best_score = dir_score[best_dir]
        
        return {
            'signal': 'arbitrage' if best_score > 0.03 else 'weak',
            'strength': round(best_score, 4),
            'direction': best_dir,
            'details': ' | '.join(s['detail'] for s in signals),
            'ev_data': ev_data,
        }


# ============================================================
# 策略四: 多因子模型 (Multi-Factor Model)
# 核心: 因子库 + 因子评分 + IC测试
# ============================================================

class MultiFactorModel:
    """多因子模型
    
    因子体系:
    1. 价值因子 (Value): EV, Kelly, 赔率偏差
    2. 动量因子 (Momentum): 赔率变动方向, 连续降赔
    3. 情绪因子 (Sentiment): 支持率, 必发成交量
    4. 波动因子 (Volatility): 赔率波动幅度
    5. 流动性因子 (Liquidity): 必发成交量, 大单占比
    
    每个因子独立评分 → 加权合成 → 最终Alpha信号
    """
    
    # 因子权重（基于学术研究和回测调优）
    FACTOR_WEIGHTS = {
        'value': 0.30,
        'momentum': 0.25,
        'sentiment': 0.20,
        'volatility': 0.10,
        'liquidity': 0.15,
    }
    
    def score(self, match: dict, model_probs: dict, odds: dict,
              betfair: dict = None, support: dict = None) -> dict:
        """计算多因子评分"""
        factors = {}
        
        # 1. 价值因子
        factors['value'] = self._value_factor(model_probs, odds)
        
        # 2. 动量因子
        factors['momentum'] = self._momentum_factor(match)
        
        # 3. 情绪因子
        factors['sentiment'] = self._sentiment_factor(support, betfair)
        
        # 4. 波动因子
        factors['volatility'] = self._volatility_factor(match)
        
        # 5. 流动性因子
        factors['liquidity'] = self._liquidity_factor(betfair)
        
        # 加权合成
        composite = self._composite_score(factors)
        
        return {
            'factors': factors,
            'composite': composite,
            'alpha_signal': composite['direction'] if composite['score'] > 0.05 else 'neutral',
            'alpha_strength': composite['score'],
        }
    
    def _value_factor(self, probs, odds):
        """价值因子: 模型vs赔率偏差"""
        h_ev = probs.get('home', 0) * odds.get('home', 0) - 1 if odds.get('home', 0) > 0 else 0
        d_ev = probs.get('draw', 0) * odds.get('draw', 0) - 1 if odds.get('draw', 0) > 0 else 0
        a_ev = probs.get('away', 0) * odds.get('away', 0) - 1 if odds.get('away', 0) > 0 else 0
        
        evs = {'home': h_ev, 'draw': d_ev, 'away': a_ev}
        best_dir = max(evs, key=evs.get)
        best_ev = evs[best_dir]
        
        return {
            'score': round(max(0, best_ev) * 3, 4),  # 放大3倍
            'direction': best_dir if best_ev > 0 else 'neutral',
            'detail': f'最优{best_dir}EV={best_ev:.1%}',
        }
    
    def _momentum_factor(self, match):
        """动量因子: 赔率变动方向一致性"""
        # 复用趋势策略的逻辑，但返回因子评分
        try:
            ho = float(match.get('home_win_odds_open', 0) or 0)
            hc = float(match.get('home_win_odds_close', 0) or 0)
            ao = float(match.get('away_win_odds_open', 0) or 0)
            ac = float(match.get('away_win_odds_close', 0) or 0)
        except Exception:
            return {'score': 0, 'direction': 'neutral', 'detail': '无动量数据'}
        
        dh = ho - hc if ho > 0 and hc > 0 else 0
        da = ao - ac if ao > 0 and ac > 0 else 0
        
        if dh > 0.05 and da < -0.03:
            return {'score': round(min(dh * 2, 0.15), 4), 'direction': 'home',
                    'detail': f'主降{dh:.2f}客升{abs(da):.2f}'}
        elif da > 0.05 and dh < -0.03:
            return {'score': round(min(da * 2, 0.15), 4), 'direction': 'away',
                    'detail': f'客降{da:.2f}主升{abs(dh):.2f}'}
        
        return {'score': 0, 'direction': 'neutral', 'detail': '动量信号弱'}
    
    def _sentiment_factor(self, support, betfair):
        """情绪因子: 散户vs机构博弈"""
        score = 0
        direction = 'neutral'
        details = []
        
        if support:
            try:
                hs = float(support.get('home_support', 0) or 0)
                as_ = float(support.get('away_support', 0) or 0)
                
                # 客支持≥45%反向指标
                if as_ >= 45:
                    score += 0.08
                    direction = 'home'
                    details.append(f'客支持{as_:.0f}%≥45')
                elif hs >= 70:
                    score += 0.05
                    direction = 'away'
                    details.append(f'主支持过热{hs:.0f}%')
            except Exception:
                pass
        
        if betfair:
            try:
                home_big = float(betfair.get('home_big_pct', 0) or 0)
                away_big = float(betfair.get('away_big_pct', 0) or 0)
                if home_big > 65:
                    score += 0.06
                    if direction == 'neutral':
                        direction = 'home'
                    details.append(f'大单主{home_big:.0f}%')
                elif away_big > 65:
                    score += 0.06
                    if direction == 'neutral':
                        direction = 'away'
                    details.append(f'大单客{away_big:.0f}%')
            except Exception:
                pass
        
        return {'score': round(score, 4), 'direction': direction,
                'detail': ' | '.join(details) or '情绪中性'}
    
    def _volatility_factor(self, match):
        """波动因子: 赔率波动幅度"""
        try:
            ho = float(match.get('home_win_odds_open', 0) or 0)
            hc = float(match.get('home_win_odds_close', 0) or 0)
        except Exception:
            return {'score': 0, 'direction': 'neutral', 'detail': '无波动数据'}
        
        if ho <= 0 or hc <= 0:
            return {'score': 0, 'direction': 'neutral', 'detail': '无波动数据'}
        
        vol = abs(ho - hc) / ho  # 波动率
        
        if vol > 0.15:  # 高波动 → 信号不稳定
            return {'score': -0.03, 'direction': 'neutral',
                    'detail': f'高波动{vol:.1%}→降信心'}
        elif vol > 0.08:
            return {'score': 0.02, 'direction': 'neutral',
                    'detail': f'中等波动{vol:.1%}'}
        
        return {'score': 0.03, 'direction': 'neutral', 'detail': f'低波动{vol:.1%}→信号可靠'}
    
    def _liquidity_factor(self, betfair):
        """流动性因子"""
        if not betfair:
            return {'score': 0, 'direction': 'neutral', 'detail': '无必发数据'}
        
        try:
            vol = float(betfair.get('total_volume', 0) or 0)
        except Exception:
            return {'score': 0, 'direction': 'neutral', 'detail': '无成交数据'}
        
        if vol > 1000000:  # 百万以上
            return {'score': 0.06, 'direction': 'neutral', 'detail': f'高流动性{vol/10000:.0f}万'}
        elif vol > 100000:
            return {'score': 0.03, 'direction': 'neutral', 'detail': f'中等流动性{vol/10000:.0f}万'}
        
        return {'score': -0.02, 'direction': 'neutral', 'detail': f'低流动性{vol/10000:.0f}万'}
    
    def _composite_score(self, factors):
        """加权合成因子评分"""
        dir_score = {'home': 0, 'draw': 0, 'away': 0}
        total_score = 0
        
        for fname, weight in self.FACTOR_WEIGHTS.items():
            f = factors.get(fname, {})
            s = f.get('score', 0)
            d = f.get('direction', 'neutral')
            
            total_score += abs(s) * weight
            if d in dir_score:
                dir_score[d] += s * weight
        
        best_dir = max(dir_score, key=dir_score.get)
        best_score = dir_score[best_dir]
        
        return {
            'score': round(max(best_score, 0), 4),
            'direction': best_dir,
            'factor_details': {k: v.get('detail', '') for k, v in factors.items()},
        }


# ============================================================
# 诱盘检测 (适用于无必发/无FD历史数据的联赛)
# ============================================================

def detect_trap(league, home_team, away_team,
                 odds_home, odds_draw, odds_away,
                 match_time=None):
    """检测诱盘陷阱 — 适用于无必发/无FD数据的联赛

    Returns:
        dict: {
            'is_trap': bool,
            'trap_type': 'overvalued_home'/'overvalued_away'/'overvalued_draw'/None,
            'trap_confidence': float,   # 0-1
            'signal': str,              # 描述文本
            'avoid_direction': str/None # 建议避开的方向
        }
    """
    import math

    try:
        ho = float(odds_home); do_ = float(odds_draw); ao = float(odds_away)
    except (TypeError, ValueError):
        return {'is_trap': False, 'trap_type': None,
                'trap_confidence': 0.0, 'signal': '赔率无效', 'avoid_direction': None}

    # 阈值
    if ho > 3.5 and odds_home < 1.5:
        # 庄家给客队超低赔却抬高主队 — 诱导资金去主队
        conf = min(1.0, (ho - 3.5) * 0.3 + (1.5 - odds_home) * 2.0)
        return {
            'is_trap': True,
            'trap_type': 'overvalued_home',
            'trap_confidence': max(0.5, min(0.95, conf)),
            'signal': f'主队诱盘: ho={ho}但主队贴水<1.50',
            'avoid_direction': '胜'
        }

    if ho < 2.0 and odds_away > 4.5:
        # 主队超低赔但客队赔率偏高 — 庄家不想要客队资金
        conf = min(1.0, (4.5 - odds_away) * 0.3 + (2.0 - ho) * 1.0)
        return {
            'is_trap': True,
            'trap_type': 'overvalued_away',
            'trap_confidence': max(0.5, min(0.90, conf)),
            'signal': f'客队诱盘: ho={ho}主队低赔但客队>{4.5}',
            'avoid_direction': '负'
        }

    if odds_draw > 4.0 and do_ < 3.0:
        # 平局赔率开高但实际庄家压低 — 诱平
        conf = min(1.0, (odds_draw - 4.0) * 0.3 + (3.0 - do_) * 0.5)
        return {
            'is_trap': True,
            'trap_type': 'overvalued_draw',
            'trap_confidence': max(0.5, min(0.90, conf)),
            'signal': f'诱平陷阱: 平赔开{odds_draw:.2f}但实际{do_:.2f}',
            'avoid_direction': '平'
        }

    return {'is_trap': False, 'trap_type': None,
            'trap_confidence': 0.0, 'signal': '无诱盘信号', 'avoid_direction': None}


def check_and_apply_trap(market, league, home, away, odds_dict):
    """对无数据联赛应用诱盘检测，返回修正标记或None

    Args:
        market: 'spf'/'rqspf'/'bf'/'zjq'/'bqc'
        league, home, away: 联赛和队名
        odds_dict: {'home': x, 'draw': y, 'away': z} 或 {'h': x, 'd': y, 'a': z}

    Returns:
        str or None: 诱盘信号文本，无诱盘时返回None
    """
    # 统一字段名
    ho = odds_dict.get('home', odds_dict.get('h', 0))
    do_ = odds_dict.get('draw', odds_dict.get('d', 0))
    ao = odds_dict.get('away', odds_dict.get('a', 0))
    if not ho or not do_ or not ao:
        return None

    result = detect_trap(league, home, away, ho, do_, ao)
    if result['is_trap']:
        return (f"[诱盘:{result['trap_type']}置信{result['trap_confidence']:.0%}]"
                f"{result['signal']}→建议避开{result['avoid_direction']}")
    return None


# ============================================================
# 统一引擎: 聚合四大策略
# ============================================================

class QuantEngine:
    """量化策略引擎 - 聚合四大策略输出"""
    
    # 策略权重
    STRATEGY_WEIGHTS = {
        'trend': 0.25,
        'highfreq': 0.30,
        'arbitrage': 0.25,
        'multifactor': 0.20,
    }
    
    def __init__(self):
        self.trend = TrendStrategy()
        self.highfreq = HighFreqStrategy()
        self.arb = ArbitrageStrategy()
        self.mf = MultiFactorModel()
    
    def analyze(self, match: dict, model_probs: dict, odds: dict,
                betfair: dict = None, support: dict = None,
                close_odds: dict = None) -> dict:
        """
        全面分析单场比赛
        
        Returns:
            {
                'final_signal': 'strong_home' / 'home' / 'neutral' / 'away' / 'strong_away',
                'confidence': 0-1,
                'direction': 'home'/'draw'/'away',
                'strategies': {趋势/高频/套利/多因子各自的信号},
                'recommendation': str,
                'ev': float or None,
            }
        """
        results = {}
        
        # 1. 趋势策略
        results['trend'] = self.trend.analyze(match)
        
        # 2. 高频策略
        results['highfreq'] = self.highfreq.analyze(match, betfair, support)
        
        # 3. 套利策略
        results['arbitrage'] = self.arb.analyze(model_probs, odds, close_odds)
        
        # 4. 多因子模型
        mf_result = self.mf.score(match, model_probs, odds, betfair, support)
        results['multifactor'] = {
            'signal': mf_result['alpha_signal'],
            'strength': mf_result['alpha_strength'],
            'direction': mf_result['composite']['direction'],
            'details': mf_result['composite'].get('factor_details', {}),
        }
        
        # 聚合
        return self._final_decision(results, model_probs)
    
    def _final_decision(self, results, model_probs):
        """最终决策聚合"""
        dir_score = {'home': 0, 'draw': 0, 'away': 0}
        
        for sname, weight in self.STRATEGY_WEIGHTS.items():
            r = results.get(sname, {})
            s = r.get('strength', 0)
            d = r.get('direction', 'neutral')
            if d in dir_score:
                dir_score[d] += s * weight
        
        best_dir = max(dir_score, key=dir_score.get)
        best_score = dir_score[best_dir]
        
        # 信号强度分级
        if best_score > 0.08:
            signal = f'strong_{best_dir}'
        elif best_score > 0.03:
            signal = best_dir
        else:
            signal = 'neutral'
        
        # EV数据
        ev_data = results.get('arbitrage', {}).get('ev_data')
        best_ev = max((e['ev'] for e in ev_data), default=0) if ev_data else 0
        
        # 综合置信度
        confidence = min(0.5 + best_score, 0.95)
        
        # 推荐语
        rec_parts = []
        for sn, r in results.items():
            if r.get('strength', 0) > 0.02:
                rec_parts.append(f"[{sn}]{r.get('details', r.get('detail', ''))}")
        
        return {
            'final_signal': signal,
            'confidence': round(confidence, 3),
            'direction': best_dir,
            'strength': round(best_score, 4),
            'strategies': results,
            'recommendation': '\n'.join(rec_parts),
            'ev': round(best_ev, 3) if best_ev > 0 else None,
            'model_probs': model_probs,
        }


# ============================================================
# 历史回测工具
# ============================================================

def backtest_trend_strategy(league_table='titan007_premier_league_matches', limit=500):
    """回测趋势策略在历史数据上的表现
    
    原理:
    - 初赔→终赔变动方向 vs 实际赛果
    - 如果一致降赔主队方向 → 主胜命中率
    - 如果一致降赔客队方向 → 客胜命中率
    """
    sql = f"""
    SELECT home_team, away_team,
           home_win_odds_open, draw_odds_open, away_win_odds_open,
           home_win_odds_close, draw_odds_close, away_win_odds_close,
           home_score, away_score,
           handicap_opening_numeric, handicap_close_numeric
    FROM {league_table}
    WHERE home_win_odds_open IS NOT NULL AND home_win_odds_close IS NOT NULL
      AND home_score IS NOT NULL AND away_score IS NOT NULL
    ORDER BY match_date DESC NULLS LAST
    LIMIT {limit}
    """
    raw = _db_query(sql)
    
    if not raw:
        return {'error': 'no data'}
    
    total = 0
    trend_correct = 0
    reverse_correct = 0
    trend_total = 0
    reverse_total = 0
    
    stats = {
        'home_drop_correct': 0, 'home_drop_total': 0,
        'away_drop_correct': 0, 'away_drop_total': 0,
        'no_signal': 0, 'total': 0,
    }
    
    for line in raw.split('\n'):
        c = line.split('|')
        if len(c) < 10:
            continue
        
        try:
            ho, hc = float(c[2]), float(c[5])
            do_, dc = float(c[3]), float(c[6])
            ao, ac = float(c[4]), float(c[7])
            hs, as_ = int(c[8]), int(c[9])
        except (ValueError, IndexError):
            continue
        
        stats['total'] += 1
        result = 'home' if hs > as_ else ('draw' if hs == as_ else 'away')
        
        dh = ho - hc
        da = ao - ac
        
        # 主降客升
        if dh > 0.05 and da < -0.03:
            stats['home_drop_total'] += 1
            if result == 'home':
                stats['home_drop_correct'] += 1
        # 客降主升
        elif da > 0.05 and dh < -0.03:
            stats['away_drop_total'] += 1
            if result == 'away':
                stats['away_drop_correct'] += 1
        else:
            stats['no_signal'] += 1
    
    # 结果
    result = {
        'total_matches': stats['total'],
        'home_drop': {
            'total': stats['home_drop_total'],
            'correct': stats['home_drop_correct'],
            'hit_rate': round(stats['home_drop_correct'] / max(stats['home_drop_total'], 1), 3),
        },
        'away_drop': {
            'total': stats['away_drop_total'],
            'correct': stats['away_drop_correct'],
            'hit_rate': round(stats['away_drop_correct'] / max(stats['away_drop_total'], 1), 3),
        },
        'no_signal': stats['no_signal'],
        'signal_coverage': round((stats['home_drop_total'] + stats['away_drop_total']) / max(stats['total'], 1), 3),
    }
    
    return result


# ============================================================
# 策略五: 庄家操盘周期策略 (Day-of-Week Bookmaker Patterns)
# 核心: 周一到周日的庄家操盘手法差异
# ============================================================

class DayOfWeekStrategy:
    """庄家操盘周期策略

    原理:
    - 周一到周四: 机构有更多时间收集信息，赔率更精准
    - 周五: 联赛开盘期，赔率结构偏保守
    - 周六/日: 周末高峰，庄家利用情绪波动，赔率偏离基本面
    - 周中杯赛: 强队赔率偏紧，冷门概率更高

    规律:
    1. 周五比赛: 主队胜率被庄家略微高估 → 反向指标
    2. 周六/日: 散户涌入方向容易被庄家反向利用
    3. 周一: 庄家休息日结束，开盘偏紧，高赔率方向更有价值
    4. 周中欧冠: 赔率最精准，跟庄信号更可靠
    5. 赛季末(最后5轮): 战意因素主导，数据驱动信号减弱
    """

    def analyze(self, match: dict, weekday: int = -1) -> dict:
        """分析星期因子信号

        Args:
            match: 比赛数据
            weekday: 0=周一, 1=周二, ..., 6=周日

        Returns:
            {'signal': str, 'strength': float, 'direction': str, 'detail': str}
        """
        if weekday < 0:
            match_date = match.get('match_date') or match.get('date')
            if match_date:
                try:
                    if isinstance(match_date, str):
                        d = datetime.fromisoformat(match_date.replace('Z', '+00:00'))
                    else:
                        d = match_date
                    weekday = d.weekday()
                except Exception:
                    weekday = -1

        if weekday < 0:
            return {'signal': 'none', 'strength': 0, 'direction': 'neutral',
                    'detail': '无法识别星期'}

        league = str(match.get('league', ''))
        ho = match.get('home_win_odds_open') or match.get('home_odds')
        ao = match.get('away_win_odds_open') or match.get('away_odds')

        try:
            ho = float(ho) if ho else 2.0
            ao = float(ao) if ao else 3.5
        except (ValueError, TypeError):
            ho, ao = 2.0, 3.5

        is_cup = any(k in league for k in ['欧冠', '欧罗巴', '欧协联', '欧洲冠军'])

        if weekday == 4:  # 周五
            return {'signal': 'friday_pattern', 'strength': 0.04,
                    'direction': 'away', 'detail': '周五开盘保守→客队方向更有价值'}

        if weekday == 0:  # 周一
            return {'signal': 'monday_pattern', 'strength': 0.05,
                    'direction': 'home', 'detail': '周一庄家重新调整→主队方向'}

        if weekday in (1, 2, 3) and is_cup:  # 周中杯赛
            return {'signal': 'midweek_cup', 'strength': 0.06,
                    'direction': 'away', 'detail': '周中杯赛冷门多→客队方向'}

        if weekday in (5, 6):  # 周六/日
            if ho < 2.0:
                return {'signal': 'weekend_hot_home', 'strength': 0.05,
                        'direction': 'away', 'detail': '周末热门主队→庄家反向操作风险'}
            if ao < 2.5:
                return {'signal': 'weekend_hot_away', 'strength': 0.04,
                        'direction': 'home', 'detail': '周末热门客队→庄家反向操作风险'}
            return {'signal': 'weekend_active', 'strength': 0.03,
                    'direction': 'neutral', 'detail': '周末比赛庄家活跃→高波动'}

        return {'signal': 'none', 'strength': 0, 'direction': 'neutral',
                'detail': '普通比赛日'}


# ============================================================
# 高频信号触发条件
# ============================================================

HIGH_FREQ_THRESHOLDS = {
    'big_money_gap': 15,          # 大单方向 vs 散户方向差 > 15%
    'support_surge': 10,          # 支持率变化 > 10%
    'support_extreme': 70,        # 支持率极端值 > 70%
    'away_support_warning': 45,   # 客支持率 > 45% 反向指标
    'kelly_positive': 0.05,       # Kelly > 5% 正EV
    'volume_threshold': 500000,    # 成交额阈值(50万)
}


def detect_high_freq_trigger(betfair: dict = None, support: dict = None) -> dict:
    """检测高频信号触发条件"""
    triggers = []

    if betfair:
        try:
            home_big = float(betfair.get('home_big_pct', 0) or 0)
            away_big = float(betfair.get('away_big_pct', 0) or 0)
            home_pct = float(betfair.get('home_pct', 0) or 0)
            away_pct = float(betfair.get('away_pct', 0) or 0)
            vol = float(betfair.get('total_volume', 0) or 0)
            home_kelly = float(betfair.get('home_kelly', 0) or 0)
            away_kelly = float(betfair.get('away_kelly', 0) or 0)

            if home_pct > 55 and away_big > home_big + HIGH_FREQ_THRESHOLDS['big_money_gap']:
                triggers.append(('big_money_divergence_away', 0.08, 'away'))
            if away_pct > 55 and home_big > away_big + HIGH_FREQ_THRESHOLDS['big_money_gap']:
                triggers.append(('big_money_divergence_home', 0.08, 'home'))
            if home_kelly > HIGH_FREQ_THRESHOLDS['kelly_positive']:
                triggers.append(('kelly_home', min(home_kelly * 0.5, 0.06), 'home'))
            if away_kelly > HIGH_FREQ_THRESHOLDS['kelly_positive']:
                triggers.append(('kelly_away', min(away_kelly * 0.5, 0.06), 'away'))
            if vol > HIGH_FREQ_THRESHOLDS['volume_threshold'] and triggers:
                for i, (name, strength, direction) in enumerate(triggers):
                    triggers[i] = (name, strength * 1.3, direction)
        except (ValueError, TypeError):
            pass

    if support:
        try:
            hs = float(support.get('home_support', 0) or 0)
            as_ = float(support.get('away_support', 0) or 0)
            hp = float(support.get('support_prev_home', 0) or 0)
            ap = float(support.get('support_prev_away', 0) or 0)

            d_home = hs - hp if hp > 0 else 0
            d_away = as_ - ap if ap > 0 else 0
            if abs(d_home) > HIGH_FREQ_THRESHOLDS['support_surge']:
                direction = 'away' if d_home > 0 else 'home'
                triggers.append((f'support_surge_{direction}', 0.04, direction))
            if abs(d_away) > HIGH_FREQ_THRESHOLDS['support_surge']:
                direction = 'home' if d_away > 0 else 'away'
                triggers.append((f'support_surge_{direction}', 0.04, direction))
            if hs > HIGH_FREQ_THRESHOLDS['support_extreme']:
                triggers.append(('extreme_home_support', 0.05, 'away'))
            if as_ > HIGH_FREQ_THRESHOLDS['support_extreme']:
                triggers.append(('extreme_away_support', 0.05, 'home'))
            if as_ >= HIGH_FREQ_THRESHOLDS['away_support_warning']:
                triggers.append(('away_support_warning', 0.05, 'home'))
        except (ValueError, TypeError):
            pass

    if not triggers:
        return {'triggered': False, 'signal_type': 'none', 'strength': 0,
                'direction': 'neutral', 'details': '无高频触发信号'}

    best_trigger = max(triggers, key=lambda x: x[1])
    name, strength, direction = best_trigger

    return {
        'triggered': True,
        'signal_type': name,
        'strength': round(min(strength, 0.15), 4),
        'direction': direction,
        'details': f'{name}(强度{strength:.2f},方向{direction})',
        'all_triggers': [(n, round(s, 4), d) for n, s, d in triggers],
    }


def detect_bookmaker_intent(match: dict = None,
                             odds_open: dict = None,
                             odds_close: dict = None) -> dict:
    """博弈论: 庄家意图检测

    基于赔率变动方向判断庄家真实意图：
    - 顺分布: 初盘→终盘赔率向市场共识方向调整
    - 逆分布: 终盘赔率与初盘相反 → 庄家造热某方向
    - 缓冲分布: 赔率来回拉扯 → 庄家不确定
    """
    if odds_open is None or odds_close is None:
        return {'intent': 'neutral', 'direction': None,
                'signal_strength': 0, 'pattern': '无数据'}

    def _get(d, k, default=0.0):
        v = d.get(k, default)
        return float(v) if v else default

    ho = _get(odds_open, 'home', 0)
    do_o = _get(odds_open, 'draw', 0)
    ao = _get(odds_open, 'away', 0)
    hc = _get(odds_close, 'home', 0)
    dc = _get(odds_close, 'draw', 0)
    ac = _get(odds_close, 'away', 0)

    if ho <= 1 or hc <= 1:
        return {'intent': 'neutral', 'direction': None,
                'signal_strength': 0, 'pattern': '赔率无效'}

    dh = ho - hc
    dd = do_o - dc
    da = ao - ac

    if dh > 0.05 and da < -0.03:
        return {'intent': 'bullish', 'direction': 'home',
                'signal_strength': min(dh * 3, 0.15), 'pattern': '顺分布(主降)'}
    if da > 0.05 and dh < -0.03:
        return {'intent': 'bullish', 'direction': 'away',
                'signal_strength': min(da * 3, 0.15), 'pattern': '顺分布(客降)'}
    if dh < -0.03 and da < -0.03 and dd > 0.03:
        return {'intent': 'neutral', 'direction': 'draw',
                'signal_strength': min(dd * 2, 0.10), 'pattern': '逆分布(平局分散)'}
    if abs(dh) < 0.03 and abs(dd) < 0.03 and abs(da) < 0.03:
        return {'intent': 'neutral', 'direction': None,
                'signal_strength': 0.02, 'pattern': '缓冲(无明显方向)'}
    if dh > 0.03:
        return {'intent': 'bullish', 'direction': 'home',
                'signal_strength': min(dh * 2, 0.08), 'pattern': '顺分布(弱)'}
    if da > 0.03:
        return {'intent': 'bullish', 'direction': 'away',
                'signal_strength': min(da * 2, 0.08), 'pattern': '顺分布(弱)'}

    return {'intent': 'neutral', 'direction': None,
            'signal_strength': 0, 'pattern': '无显著信号'}


def detect_arbitrage(tiancai_odds: dict = None,
                    fd_odds: dict = None,
                    william_odds: dict = None) -> dict:
    """跨市场套利检测

    当三家赔率商对同一方向给出的赔率差>5%时，
    可能存在套利机会或庄家定价分歧。
    """
    books = []
    for name, odds in [('体彩', tiancai_odds), ('FD', fd_odds), ('威廉', william_odds)]:
        if not odds:
            continue
        h = odds.get('home', odds.get('胜', 0))
        d = odds.get('draw', odds.get('平', 0))
        a = odds.get('away', odds.get('负', 0))
        if h > 1 and d > 1 and a > 1:
            books.append((name, {'home': h, 'draw': d, 'away': a}))

    if len(books) < 2:
        return {'has_arbitrage': False, 'direction': None,
                'spread_pct': 0, 'opportunities': []}

    best = {}
    for book_name, book_odds in books:
        for direction in ('home', 'draw', 'away'):
            odds_val = book_odds.get(direction, 0)
            if odds_val > 1:
                if direction not in best or odds_val > best[direction][1]:
                    best[direction] = (book_name, odds_val)

    arbitrage = None
    max_spread = 0
    for direction, (book_name, best_odds) in best.items():
        min_odds = min(book_odds.get(direction, 0)
                      for _, book_odds in books
                      if book_odds.get(direction, 0) > 1)
        if min_odds <= 1:
            continue
        spread = (best_odds - min_odds) / min_odds
        if spread > max_spread:
            max_spread = spread
            arbitrage = {'direction': direction, 'bookmaker': book_name,
                        'best_odds': best_odds, 'worst_odds': min_odds,
                        'spread_pct': spread}

    if arbitrage and arbitrage['spread_pct'] > 0.05:
        return {'has_arbitrage': True, 'direction': arbitrage['direction'],
                'spread_pct': arbitrage['spread_pct'],
                'opportunities': [(arbitrage['bookmaker'], arbitrage['direction'],
                                   arbitrage['best_odds'])]}

    return {'has_arbitrage': False, 'direction': None,
            'spread_pct': 0, 'opportunities': []}


# ============================================================
# 附录: 赔率数学工具箱 (从 shenmeng-football-betting-analyzer 提取)
# 来源: skillhub shenmeng-football-betting-analyzer v1.3.1
# ============================================================

class KellyCalculator:
    """凯利指数计算器 - 价值投注识别与最优投注比例"""
    
    @staticmethod
    def calculate_true_probability(market_odds: float, margin: float = 0.05) -> float:
        """根据市场赔率和抽水率计算真实概率
        
        Args:
            market_odds: 市场赔率 (如 1.95)
            margin: 庄家抽水率 (默认5%)
        Returns:
            真实胜率估计
        """
        implied_prob = 1 / market_odds
        return implied_prob / (1 + margin)
    
    @staticmethod
    def calculate_kelly_criterion(true_probability: float, odds: float,
                                  fraction: float = 0.25) -> float:
        """计算凯利公式投注比例 (分数凯利保守策略)
        
        Args:
            true_probability: 真实胜率估计 (0-1)
            odds: 赔率 (小数赔率，如 2.0)
            fraction: 凯利分数 (默认0.25，保守)
        Returns:
            建议投注资金比例 (0-1)
        """
        if odds <= 1:
            return 0.0
        kelly = (true_probability * odds - 1) / (odds - 1)
        return max(0.0, kelly * fraction)
    
    @staticmethod
    def analyze_value(model_probability: float, market_odds: float,
                      min_edge: float = 0.05) -> dict:
        """分析价值投注
        
        Returns:
            dict: {
                is_value_bet: bool,
                edge: float,          # 边际
                kelly_fraction: float,
                recommendation: str
            }
        """
        implied_prob = 1 / market_odds
        edge = model_probability - implied_prob
        is_value = edge > min_edge
        kelly = KellyCalculator.calculate_kelly_criterion(
            model_probability, market_odds
        )
        
        if edge < -0.1:
            recommendation = "❌ 避免投注 - 赔率无价值"
        elif edge < 0:
            recommendation = "⚠️ 赔率略低 - 观望"
        elif edge < min_edge:
            recommendation = "➖ 价值不明显 - 可选"
        elif kelly < 0.02:
            recommendation = "✅ 有价值但比例小 - 小额娱乐"
        elif kelly < 0.05:
            recommendation = "✅ 有价值 - 适量投注"
        else:
            recommendation = "🔥 高价值 - 重点考虑"
        
        return {
            "is_value_bet": is_value,
            "edge": round(edge, 4),
            "kelly_fraction": round(kelly, 4),
            "recommendation": recommendation
        }


class OddsMovementAnalyzer:
    """赔率变化趋势分析器 - 跟踪赔率历史变动"""
    
    def __init__(self):
        self.movement_history: list = []
    
    def add_snapshot(self, timestamp: str, home_odds: float,
                     draw_odds: float, away_odds: float) -> None:
        """添加赔率快照"""
        self.movement_history.append({
            "time": timestamp,
            "home": home_odds,
            "draw": draw_odds,
            "away": away_odds
        })
    
    def analyze_trend(self) -> dict:
        """分析赔率趋势
        
        Returns:
            dict: {
                home_trend: str (🔼升/🔽降/➡️稳),
                draw_trend: str,
                away_trend: str,
                home_change_pct: float,
                draw_change_pct: float,
                away_change_pct: float,
                bookmaker_bias: str
            }
        """
        if len(self.movement_history) < 2:
            return {"error": "数据不足，需要至少2个快照"}
        
        first = self.movement_history[0]
        last = self.movement_history[-1]
        
        home_change = ((last["home"] - first["home"]) / first["home"]) * 100
        draw_change = ((last["draw"] - first["draw"]) / first["draw"]) * 100
        away_change = ((last["away"] - first["away"]) / first["away"]) * 100
        
        def _trend(val):
            if val > 5:   return "🔼 升"
            elif val < -5: return "🔽 降"
            else:          return "➡️ 稳"
        
        analysis = {
            "home_trend": _trend(home_change),
            "draw_trend": _trend(draw_change),
            "away_trend": _trend(away_change),
            "home_change_pct": round(home_change, 2),
            "draw_change_pct": round(draw_change, 2),
            "away_change_pct": round(away_change, 2),
        }
        
        # 判断机构倾向
        if home_change < -10 and away_change > 5:
            analysis["bookmaker_bias"] = "🏠 机构看好主队"
        elif away_change < -10 and home_change > 5:
            analysis["bookmaker_bias"] = "✈️ 机构看好客队"
        elif abs(home_change) < 5 and abs(away_change) < 5:
            analysis["bookmaker_bias"] = "⚖️ 机构态度平稳"
        else:
            analysis["bookmaker_bias"] = "🔄 机构调整中"
        
        return analysis


def calculate_stake(bankroll: float, confidence: float, odds: float,
                    kelly_fraction: float = 0.25,
                    max_stake_pct: float = 0.05,
                    league: str = "默认") -> dict:
    """计算建议投注金额 (资金管理)
    
    Args:
        bankroll: 总资金
        confidence: 置信度 (0-1)
        odds: 赔率
        kelly_fraction: 凯利分数 (默认0.25保守)
        max_stake_pct: 单次最大投注比例 (默认5%)
    
    Returns:
        dict: {fixed, kelly, confidence_weighted, recommended}
    """
    # 固定比例法 (2%)
    fixed_pct = 0.02
    fixed_stake = bankroll * fixed_pct
    
    # 凯利公式
    p = confidence
    q = 1 - p
    b = odds - 1
    kelly_pct = (p * b - q) / b if b > 0 else 0
    kelly_pct = max(0, kelly_pct * kelly_fraction)
    kelly_stake = bankroll * min(kelly_pct, max_stake_pct)
    
    # ════ P1-2新增(2026-05-12): 赔率档位差异化波动率调整 ════
    # 低赔率(1.5以下)：波动小，可适当加仓
    # 高赔率(3.0以上)：波动大，必须减仓
    if odds < 1.5:
        vol_adj = 0.6   # 低赔率低波动，安全边际高
    elif odds < 2.0:
        vol_adj = 0.8   # 轻微模糊区
    elif odds < 3.0:
        vol_adj = 1.0   # 正常波动
    else:
        vol_adj = 1.5   # 高赔率波动极大，减仓

    # 联赛差异调整
    league_adj = LEAGUE_WEIGHT_MAP.get(league, 0.75)
    
    # 最终仓位 = Kelly × 波动调整 × 联赛调整
    final_kelly_pct = kelly_pct * vol_adj * league_adj
    final_kelly_stake = bankroll * min(final_kelly_pct, max_stake_pct)
    
    # 置信度加权 (最高3%)
    confidence_stake = bankroll * (confidence * 0.03)
    
    recommended = min(final_kelly_stake, confidence_stake, bankroll * max_stake_pct)
    
    return {
        "fixed": round(fixed_stake, 2),
        "kelly": round(kelly_stake, 2),
        "final_kelly": round(final_kelly_stake, 2),
        "confidence_weighted": round(confidence_stake, 2),
        "recommended": round(recommended, 2),
        "vol_adj": vol_adj,
        "league_adj": league_adj,
    }


# === 独立运行 ===
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--backtest', action='store_true', help='回测趋势策略')
    parser.add_argument('--league', default='titan007_premier_league_matches')
    parser.add_argument('--limit', type=int, default=1000)
    args = parser.parse_args()

    if args.backtest:
        print(f"📊 回测趋势策略: {args.league}")
        r = backtest_trend_strategy(args.league, args.limit)
        print(json.dumps(r, indent=2, ensure_ascii=False))
    else:
        print("用法: python3 quant_strategies.py --backtest [--league TABLE] [--limit N]")
