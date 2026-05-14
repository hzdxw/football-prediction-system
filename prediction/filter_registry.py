# -*- coding: utf-8 -*-
"""
Filter Registry — 预测管线验证层
==================================

作用：EV过滤 + 冷赔率过滤，从 value_bet.py / ml_predict_5play.py 的内嵌逻辑中
      拆出来做成独立可注册/可组合的 Filter 模块。

使用方式：
    from prediction.filter_registry import get_filter, apply_filters, FILTER_REGISTRY

    # 获取已注册的 filter
    ev_filter = get_filter('ev_filter')
    cold_filter = get_filter('cold_filter')

    # 链式应用
    result = apply_filters(match_data, filters=['ev_filter', 'cold_filter'], policy='all')

    # 直接调用
    ev_result = ev_filter.check(match_data, spf_probs, odds, league)
    cold_result = cold_filter.check(match_data, odds)

注册表：
    'ev_filter'     — EV期望值过滤（动态阈值：联赛×赔率区间×市场类型）
    'cold_filter'   — 冷赔率区间过滤（HHAD >2.8 硬限制）
"""

from __future__ import annotations
import os, math
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FilterResult:
    """单个 filter 的检查结果"""
    name: str
    passed: bool
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    confidence_adjustment: float = 0.0   # 置信度调整（负数=降权）
    ev_threshold: float = 0.0
    ev_actual: float = 0.0

    def summary(self) -> str:
        status = "✅ PASS" if self.passed else "❌ REJECT"
        return f"[{self.name}] {status}: {self.reason}"


@dataclass
class EnsembleResult:
    """FilterEnsemble 的聚合结果"""
    passed: bool
    policy: str            # 'all' = 全部通过才通过, 'any' = 任一通过即通过
    results: List[FilterResult] = field(default_factory=list)
    final_confidence_adjustment: float = 0.0

    def reject_reasons(self) -> List[str]:
        return [r.reason for r in self.results if not r.passed]

    def summary(self) -> str:
        lines = [f"Ensemble({self.policy}): {'PASS' if self.passed else 'REJECT'}"]
        for r in self.results:
            lines.append(f"  {r.summary()}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# 基础 Filter 类
# ═══════════════════════════════════════════════════════════════════════════

class PredictionFilter:
    """预测过滤器基类"""

    name: str = "base"

    def check(self, match: dict, **kwargs) -> FilterResult:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name}>"


# ═══════════════════════════════════════════════════════════════════════════
# EV Filter — 期望值过滤
# ═══════════════════════════════════════════════════════════════════════════

class EVFilter(PredictionFilter):
    """
    期望值（EV）过滤

    阈值规则（联赛 × 赔率区间 × 市场类型）：
    - 单关：基准 5%，联赛波动系数 × 赔率区间系数
    - 2串1：最低 8%，组合风险叠加

    赔率区间系数：
    - do < 3.5 → 1.0
    - do >= 3.5 → 1.4（高赔平局需要更高EV才值得投）
    """

    name = "ev_filter"

    # 联赛 EV 乘数（法乙/英冠/挪超/苏超/澳超/韩职波动大 → 阈值更高）
    _LEAGUE_MULT: Dict[str, float] = {
        '法乙': 0.12, '瑞超': 0.10, '葡超': 0.10, '英冠': 0.12,
        '挪超': 0.10, '苏超': 0.10, '澳超': 0.12, '韩职': 0.10,
        '韩职联': 0.10, '日职': 0.08, '日职联': 0.08, 'J1': 0.08,
        '英超': 0.05, '意甲': 0.05, '西甲': 0.05, '德甲': 0.05, '法甲': 0.05,
        '欧冠': 0.08, '欧联': 0.08, '欧协联': 0.08,
    }

    # 赔率区间 × 市场类型的 EV 基准（5% × 联赛乘数 × 赔率区间系数）
    _BASE_EV: float = 0.05
    _PARLAY_MIN_EV: float = 0.08   # 2串1最低8%

    def check(self, match: dict,
              spf_probs: Optional[dict] = None,
              odds: Optional[dict] = None,
              league: Optional[str] = None,
              market: str = 'single',
              confidence: float = 0.5) -> FilterResult:
        """
        检查比赛是否满足 EV 阈值。

        Args:
            match: 比赛数据
            spf_probs: {'胜': float, '平': float, '负': float}
            odds: {'home': float, 'draw': float, 'away': float} 或 (ho, do_, ao)
            league: 联赛名
            market: 'single'=单关, 'parlay'=2串1
            confidence: 当前置信度（用于计算调整）

        Returns:
            FilterResult(passed=True/False, ev_actual, ev_threshold, ...)
        """
        if spf_probs is None:
            spf_probs = match.get('spf_probs', {})
        if odds is None:
            ho = float(match.get('home_odds', 0) or 0)
            do_ = float(match.get('draw_odds', 0) or 0)
            ao = float(match.get('away_odds', 0) or 0)
            odds = {'home': ho, 'draw': do_, 'away': ao}
        elif isinstance(odds, (list, tuple)):
            ho, do_, ao = odds
            odds = {'home': ho, 'draw': do_, 'away': ao}

        lg = league or match.get('league', '')

        # ── 计算隐含概率 ──
        ho = odds.get('home', 0)
        do_ = odds.get('draw', 0)
        ao = odds.get('away', 0)
        if ho <= 0 or do_ <= 0 or ao <= 0:
            return FilterResult(
                name=self.name, passed=False,
                reason="odds数据不完整",
                ev_threshold=0, ev_actual=0
            )

        margin = 1/ho + 1/do_ + 1/ao
        impl = {
            '胜': 1/ho / margin,
            '平': 1/do_ / margin,
            '负': 1/ao / margin,
        }

        # ── 动态阈值计算 ──
        # lg_mult 是联赛调整系数：五大联赛5%，弱联赛8-12%
        lg_mult = self._LEAGUE_MULT.get(lg, 0.08)
        # 平局赔率区间系数（高赔平局需要更高EV才值得）
        do_odds_mult = 1.4 if do_ >= 3.5 else 1.0
        # 基准 5% × 联赛系数（0.05-0.12）× 赔率区间系数
        threshold = self._BASE_EV * lg_mult * do_odds_mult
        # 确保不低于基准 EV 的联赛效应值（lg_mult=1.0 for 五大联赛）
        # 英超 lg_mult=0.05 → 5%×0.05=0.25% ❌  应为 5%×1.0=5%
        # 修复：lg_mult 语义是"联赛EV阈值倍率"，五大联赛=1.0，弱联赛=1.2-1.5
        # 重新映射（五大联赛基准=1.0，其他联赛+20-50%）
        _LEAGUE_EV_MAP = {
            '法乙': 1.2, '瑞超': 1.2, '葡超': 1.2, '英冠': 1.2,
            '挪超': 1.2, '苏超': 1.2, '澳超': 1.2, '韩职': 1.2,
            '韩职联': 1.2, '日职': 1.2, '日职联': 1.2, 'J1': 1.2,
            '英超': 1.0, '意甲': 1.0, '西甲': 1.0, '德甲': 1.0, '法甲': 1.0,
            '欧冠': 1.0, '欧联': 1.0, '欧协联': 1.0,
        }
        lg_rate = _LEAGUE_EV_MAP.get(lg, 1.2)
        threshold = self._BASE_EV * lg_rate * do_odds_mult
        if market == 'parlay':
            threshold = max(threshold, self._PARLAY_MIN_EV)

        # ── 找最优 EV ──
        best_ev = -999.0
        best_label = None
        best_adj_p = 0.0

        for label, label_odds in [('胜', ho), ('平', do_), ('负', ao)]:
            ip = impl.get(label, 0)
            poisson_p = spf_probs.get(label, 0)
            alpha = max(0, (poisson_p - ip) * 0.5)
            adj_p = max(0.05, min(0.90, ip + alpha))
            ev = adj_p * label_odds - 1
            if ev > best_ev:
                best_ev = ev
                best_label = label
                best_adj_p = adj_p

        # ── 置信度协同阈值（P1诊断 2026-05-07）────────────────────
        # 分析结论：conf>0.70 + recommend_strong → 49.3% vs 基准41.1%（+8.2%）
        # conf>0.70 本身无区分度（41.2%），真正信号来自 recommend_strong
        # 策略：conf>0.70 时降低EV阈值使更多好信号通过；conf≤0.50 时收紧阈值
        conf_thresh = 0.70
        low_conf_thresh = 0.50
        if confidence >= conf_thresh:
            threshold = threshold * 0.70   # 高置信：EV阈值降低30%（更容易通过）
        elif confidence <= low_conf_thresh:
            threshold = threshold * 1.50   # 低置信：EV阈值提高50%（更严格过滤）

        passed = best_ev >= threshold

        # ── 置信度调整 ──
        conf_adj = 0.0
        if not passed:
            # EV 不满足 → 降置信度
            gap = threshold - best_ev
            conf_adj = -min(gap * 2, 0.10)   # 最多降10%

        return FilterResult(
            name=self.name,
            passed=passed,
            reason=f"{best_label} EV={best_ev*100:.1f}% " +
                   (f"< 阈值{threshold*100:.1f}%" if not passed else "≥阈值"),
            ev_threshold=round(threshold, 4),
            ev_actual=round(best_ev, 4),
            confidence_adjustment=conf_adj,
            details={
                'best_label': best_label,
                'impl_prob': round(impl.get(best_label, 0) * 100, 1),
                'poisson_prob': round(spf_probs.get(best_label, 0) * 100, 1),
                'adjusted_prob': round(best_adj_p * 100, 1),
                'odds': label_odds,
                'market': market,
                'league': lg,
                'lg_mult': lg_mult,
                'do_odds_mult': do_odds_mult,
            }
        )


# ═══════════════════════════════════════════════════════════════════════════
# Cold Odds Filter — 冷赔率区间过滤
# ═══════════════════════════════════════════════════════════════════════════

class ColdOddsFilter(PredictionFilter):
    """
    冷赔率区间过滤（HHAD/让球专项）

    规则：
    - HHAD 赔率 > 2.8 → 命中率 < 20%，假信号严重 → 直接 REJECT
    - 超强热门（主胜 ≤ 1.20）且客胜 ≥ 3.75 → 客队极度低估，陷阱概率高 → REJECT
    - 赔率悬殊 > 5:1 且无亚盘数据确认 → REJECT
    """

    name = "cold_filter"

    # HHAD 冷门赔率阈值
    _HHAD_COLD_THRESHOLD: float = 2.8

    # 超强热门客胜陷阱阈值
    _SUPER_HOT_AWAY_MIN: float = 3.75   # 主胜 ≤ 1.20 时，客胜 > 此值 → 陷阱
    _SUPER_HOT_HOME_MAX: float = 1.20

    # 赔率悬殊硬限制
    _MAX_ODDS_RATIO: float = 5.0

    def check(self, match: dict,
              odds: Optional[dict] = None,
              handicap: Optional[float] = None,
              asian_data: Optional[dict] = None,
              **kwargs) -> FilterResult:
        """
        检查比赛是否处于冷赔率区间。

        Args:
            match: 比赛数据
            odds: {'home': float, 'draw': float, 'away': float} 或 (ho, do_, ao)
            handicap: 让球数（如 1.0, -1.0）
            asian_data: 亚盘数据（如 {'spread': float, 'home_odds': float, ...}）

        Returns:
            FilterResult
        """
        if odds is None:
            ho = float(match.get('home_odds', 0) or 0)
            do_ = float(match.get('draw_odds', 0) or 0)
            ao = float(match.get('away_odds', 0) or 0)
            odds = {'home': ho, 'draw': do_, 'away': ao}
        elif isinstance(odds, (list, tuple)):
            ho, do_, ao = odds
            odds = {'home': ho, 'draw': do_, 'away': ao}

        ho = odds.get('home', 0)
        do_ = odds.get('draw', 0)
        ao = odds.get('away', 0)
        lg = match.get('league', '')

        if ho <= 0 or ao <= 0:
            return FilterResult(name=self.name, passed=False,
                               reason="odds数据不完整")

        reasons: List[str] = []
        conf_adj = 0.0

        # ── 检查1: HHAD 冷门赔率 ──
        # 用平局赔率作为 HHAD 代理指标（主队让球后平局赔率通常 3.0-3.5）
        # 如果 do_ > _HHAD_COLD_THRESHOLD * 1.2 → 危险区
        if do_ > self._HHAD_COLD_THRESHOLD * 1.2:
            reasons.append(f"HHAD冷门区(do_={do_:.2f}>{self._HHAD_COLD_THRESHOLD*1.2:.1f})")
            conf_adj -= 0.08

        # ── 检查2: 超强热门陷阱 ──
        if ho <= self._SUPER_HOT_HOME_MAX and ao >= self._SUPER_HOT_AWAY_MIN:
            reasons.append(
                f"超强热门陷阱(home={ho:.2f}≤{self._SUPER_HOT_HOME_MAX}, "
                f"away={ao:.2f}≥{self._SUPER_HOT_AWAY_MIN})"
            )
            conf_adj -= 0.10

        # ── 检查3: 极端赔率悬殊且无亚盘确认 ──
        fav = min(ho, ao)
        underdog = max(ho, ao)
        odds_ratio = underdog / fav if fav > 0 else 999

        has_asian = asian_data is not None or match.get('handicap') or match.get('asian_handicap')
        if odds_ratio > self._MAX_ODDS_RATIO and not has_asian:
            reasons.append(f"极端赔率悬殊{odds_ratio:.1f}:1无亚盘确认")
            conf_adj -= 0.08

        # ── 检查4: 让球专项（HAD）─
        # 从 match 中取 handicap 或从 handicap 参数
        hc = handicap
        if hc is None:
            hc = float(match.get('handicap', 0) or 0)
        if hc != 0:
            # HHAD 让球赔率 > 2.8 → 冷门区
            hc_home = float(match.get('hc_win') or match.get('handicap_home_odds', 0) or 0)
            hc_away = float(match.get('hc_lose') or match.get('handicap_away_odds', 0) or 0)
            if hc_home > self._HHAD_COLD_THRESHOLD:
                reasons.append(f"HHAD主队让球赔率{hc_home:.2f}>2.8")
                conf_adj -= 0.08
            if hc_away > self._HHAD_COLD_THRESHOLD:
                reasons.append(f"HHAD客队受让赔率{hc_away:.2f}>2.8")
                conf_adj -= 0.08

        passed = len(reasons) == 0
        reason_str = "; ".join(reasons) if reasons else "区间正常"

        return FilterResult(
            name=self.name,
            passed=passed,
            reason=reason_str,
            confidence_adjustment=round(conf_adj, 4),
            details={
                'ho': ho, 'do_': do_, 'ao': ao,
                'odds_ratio': round(odds_ratio, 2),
                'has_asian': has_asian,
                'handicap': hc,
                'league': lg,
                'reject_count': len(reasons),
            }
        )


# ═══════════════════════════════════════════════════════════════════════════
# Filter Ensemble — 链式组合
# ═══════════════════════════════════════════════════════════════════════════

class FilterEnsemble:
    """
    过滤器组合器

    用法：
        ens = FilterEnsemble(filters=[ev_filter, cold_filter], policy='all')
        result = ens.apply(match, spf_probs, odds, league)

        if result.passed:
            # 通过所有过滤
        else:
            for reason in result.reject_reasons():
                print(f"  → {reason}")
    """

    def __init__(self,
                 filters: Optional[List[PredictionFilter]] = None,
                 policy: str = 'all'):
        """
        Args:
            filters: Filter 实例列表
            policy: 'all' = 全部通过才通过, 'any' = 任一通过即通过
        """
        self.filters = filters or []
        self.policy = policy  # 'all' | 'any'

    def add(self, f: PredictionFilter) -> 'FilterEnsemble':
        self.filters.append(f)
        return self

    def check(self, match: dict, **kwargs) -> EnsembleResult:
        results: List[FilterResult] = []
        for f in self.filters:
            try:
                r = f.check(match, **kwargs)
            except Exception as e:
                r = FilterResult(
                    name=getattr(f, 'name', repr(f)),
                    passed=False,
                    reason=f"检查异常: {e}"
                )
            results.append(r)

        if self.policy == 'all':
            passed = all(r.passed for r in results)
        else:  # 'any'
            passed = any(r.passed for r in results)

        total_conf_adj = sum(r.confidence_adjustment for r in results)

        return EnsembleResult(
            passed=passed,
            policy=self.policy,
            results=results,
            final_confidence_adjustment=round(total_conf_adj, 4)
        )


# ═══════════════════════════════════════════════════════════════════════════
# 全局注册表
# ═══════════════════════════════════════════════════════════════════════════

# 预实例化 singleton
_ev_filter_instance = EVFilter()
_cold_filter_instance = ColdOddsFilter()

FILTER_REGISTRY: Dict[str, PredictionFilter] = {
    'ev_filter':   _ev_filter_instance,
    'cold_filter': _cold_filter_instance,
}


def get_filter(name: str) -> Optional[PredictionFilter]:
    """按名字获取已注册的 filter"""
    return FILTER_REGISTRY.get(name)


def apply_filters(match: dict,
                  filters: Optional[List[str]] = None,
                  policy: str = 'all',
                  **kwargs) -> EnsembleResult:
    """
    按名字列表应用已注册的 filter。

    Args:
        match: 比赛数据
        filters: filter 名字列表，如 ['ev_filter', 'cold_filter']
        policy: 'all' = 全部通过, 'any' = 任一通过
        **kwargs: 传给各 filter.check() 的额外参数

    Returns:
        EnsembleResult
    """
    if filters is None:
        filters = list(FILTER_REGISTRY.keys())

    filter_objs = []
    for fname in filters:
        f = get_filter(fname)
        if f is not None:
            filter_objs.append(f)

    ens = FilterEnsemble(filters=filter_objs, policy=policy)
    return ens.check(match, **kwargs)


def get_default_ensemble(policy: str = 'all') -> FilterEnsemble:
    """获取默认的全量 filter 组合（ev_filter + cold_filter）"""
    return FilterEnsemble(
        filters=[_ev_filter_instance, _cold_filter_instance],
        policy=policy
    )
