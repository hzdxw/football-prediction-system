# -*- coding: utf-8 -*-
"""
贝叶斯概率融合框架 (P2-2026-05-02)
====================================

融合多个模型的不确定性，输出后验概率。
修复: 固定权重0.4/0.6无法自适应联赛/市场的问题。

数学原理:
  后验 ∝ 似然 × 先验
  P(O|D_models) ∝ P(D_models|O) × P(O)

三层融合:
  1. 赔率隐含概率作为市场似然 (market implied probability)
  2. 泊松概率作为先验 (poisson prior)
  3. ML/DL概率作为观测 (model observations)

与SuperFusionLayer的区别:
  - SuperFusionLayer: 线性加权 + 博弈论一致性
  - BayesianFusion: 概率乘法 + 不确定性感知
"""

import numpy as np
from typing import Dict, Optional


class BayesianFusion:
    """贝叶斯概率融合器"""

    # 默认先验权重（可动态调整）
    DEFAULT_PRIOR_WEIGHTS = {
        'poisson': 0.30,
        'ml': 0.40,
        'dl': 0.20,
        'market': 0.10
    }

    def __init__(self, prior_weights: Optional[Dict[str, float]] = None):
        self.prior_weights = prior_weights or dict(self.DEFAULT_PRIOR_WEIGHTS)

    def odds_to_implied_prob(self, odds: float, margin: float = 0.05) -> float:
        """赔率转隐含概率（去除庄家抽水）

        Args:
            odds: 赔率
            margin: 庄家抽水率（默认5%）

        Returns:
            隐含概率 (0-1)
        """
        if odds <= 0:
            return 0.333
        implied = 1.0 / odds
        # 去除margin
        return implied * (1 - margin)

    def calibrate_odds(self, ho: float, do_: float, ao: float) -> Dict[str, float]:
        """从赔率计算校准后的隐含概率

        Args:
            ho: 主胜赔率
            do_: 平局赔率
            ao: 客胜赔率

        Returns:
            {'H': float, 'D': float, 'A': float}
        """
        # 原始隐含概率
        raw = {
            'H': 1.0 / ho if ho > 0 else 0.333,
            'D': 1.0 / do_ if do_ > 0 else 0.333,
            'A': 1.0 / ao if ao > 0 else 0.333,
        }
        total = sum(raw.values())
        # 归一化
        return {k: v / total for k, v in raw.items()}

    def fuse(self,
             poisson_probs: Dict[str, float],
             ml_probs: Dict[str, float],
             dl_probs: Optional[Dict[str, float]] = None,
             odds: Optional[Dict[str, float]] = None,
             league_hist: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """贝叶斯融合：融合多个模型输出

        Args:
            poisson_probs: 泊松模型概率 {'H': 0.45, 'D': 0.28, 'A': 0.27}
            ml_probs: ML模型概率
            dl_probs: DL模型概率（可选）
            odds: 赔率 {'H': 2.0, 'D': 3.4, 'A': 3.8}
            league_hist: 联赛历史概率（可选）

        Returns:
            融合后的后验概率
        """
        # Step 1: 确定各成分
        has_dl = dl_probs is not None
        has_odds = odds is not None

        # Step 2: 计算先验（基于赔率隐含概率）
        if has_odds:
            market_probs = {
                'H': self.odds_to_implied_prob(odds.get('H', 2.0)),
                'D': self.odds_to_implied_prob(odds.get('D', 3.0)),
                'A': self.odds_to_implied_prob(odds.get('A', 3.0))
            }
        else:
            market_probs = None

        # Step 3: 先验融合（泊松 + 联赛历史）
        if league_hist:
            # 联赛历史作为先验
            prior_H = 0.45 * poisson_probs.get('H', 0.33) + 0.55 * league_hist.get('H', 0.33)
            prior_D = 0.45 * poisson_probs.get('D', 0.33) + 0.55 * league_hist.get('D', 0.33)
            prior_A = 0.45 * poisson_probs.get('A', 0.33) + 0.55 * league_hist.get('A', 0.33)
        else:
            prior_H = poisson_probs.get('H', 0.33)
            prior_D = poisson_probs.get('D', 0.33)
            prior_A = poisson_probs.get('A', 0.33)

        prior = {'H': prior_H, 'D': prior_D, 'A': prior_A}

        # Step 4: 似然融合（ML + DL + 赔率）
        if has_dl:
            # 完整融合：ML(40%) + DL(20%) + 泊松(30%) + 赔率(10%)
            ml_weight = self.prior_weights.get('ml', 0.40)
            dl_weight = self.prior_weights.get('dl', 0.20)
            poisson_weight = self.prior_weights.get('poisson', 0.30)
            market_weight = self.prior_weights.get('market', 0.10)
        else:
            # 无DL：ML(50%) + 泊松(40%) + 赔率(10%)
            ml_weight = 0.50
            dl_weight = 0.0
            poisson_weight = 0.40
            market_weight = 0.10

        # Step 5: 几何平均融合（比算术平均更保真）
        posterior = {}
        for outcome in ['H', 'D', 'A']:
            p_ml = ml_probs.get(outcome, 0.333)
            p_poisson = poisson_probs.get(outcome, 0.333)

            if has_dl:
                p_dl = dl_probs.get(outcome, 0.333)
            else:
                p_dl = 1.0

            if has_odds:
                p_market = market_probs.get(outcome, 0.333)
            else:
                p_market = 1.0

            # 几何平均
            product = (
                (p_ml ** ml_weight) *
                (p_dl ** dl_weight) *
                (p_poisson ** poisson_weight) *
                (p_market ** market_weight)
            )
            posterior[outcome] = product

        # Step 6: 归一化
        total = sum(posterior.values())
        if total > 0:
            posterior = {k: v / total for k, v in posterior.items()}
        else:
            # 兜底：均匀分布
            posterior = {'H': 0.333, 'D': 0.333, 'A': 0.333}

        return posterior

    def fuse_with_adaptive_weights(self,
                                   poisson_probs: Dict[str, float],
                                   ml_probs: Dict[str, float],
                                   dl_probs: Optional[Dict[str, float]],
                                   odds: Dict[str, float],
                                   market_type: str = 'SPF',
                                   odds_range: str = 'normal',
                                   model_confidence: float = 0.5) -> Dict[str, float]:
        """自适应权重融合：根据市场类型、赔率区间、置信度动态调整权重

        Args:
            poisson_probs: 泊松概率
            ml_probs: ML概率
            dl_probs: DL概率
            odds: 赔率
            market_type: 市场类型 (SPF/HHAD/ZJQ/BF)
            odds_range: 赔率区间 (hot/normal/cold)
            model_confidence: 模型置信度 (0-1)

        Returns:
            融合后概率
        """
        # 基础权重
        weights = {
            'ml': 0.40,
            'dl': 0.20,
            'poisson': 0.30,
            'market': 0.10
        }

        # 市场类型调整
        if market_type == 'HHAD':
            # 让球盘：提高泊松权重，降低ML权重
            weights = {'ml': 0.30, 'dl': 0.15, 'poisson': 0.45, 'market': 0.10}
        elif market_type == 'BF':
            # 比分市场：高不确定性，全部降低
            weights = {'ml': 0.35, 'dl': 0.25, 'poisson': 0.25, 'market': 0.15}
        elif market_type == 'ZJQ':
            # 总进球：泊松为主
            weights = {'ml': 0.30, 'dl': 0.20, 'poisson': 0.40, 'market': 0.10}

        # 赔率区间调整
        if odds_range == 'hot':
            # 热赔率区间：提高泊松/市场权重
            weights['poisson'] *= 1.2
            weights['market'] *= 1.2
            weights['ml'] *= 0.8
            weights['dl'] *= 0.8
        elif odds_range == 'cold':
            # 冷赔率区间：降低泊松权重（冷门区间泊松失真）
            weights['poisson'] *= 0.7
            weights['ml'] *= 1.15
            weights['dl'] *= 1.15

        # 置信度调制
        conf_modulator = 0.5 + 0.5 * model_confidence  # 0.5~1.0
        for k in weights:
            weights[k] *= conf_modulator

        # 归一化权重
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

        # 应用权重
        self.prior_weights = weights
        return self.fuse(poisson_probs, ml_probs, dl_probs, odds)


def bayesian_fuse(match_probs: dict, odds: dict,
                  market_type: str = 'SPF') -> dict:
    """便捷入口：贝叶斯融合一行调用

    Args:
        match_probs: {
            'poisson': {'H':, 'D':, 'A':},
            'ml': {'H':, 'D':, 'A':},
            'dl': {'H':, 'D':, 'A':} (可选)
          }
        odds: {'H':, 'D':, 'A':}
        market_type: SPF/HHAD/ZJQ/BF

    Returns:
        融合后的概率
    """
    fuser = BayesianFusion()

    # 判断赔率区间
    ho = odds.get('H', 2.0)
    odds_range = 'hot' if ho < 1.5 else ('cold' if ho > 2.8 else 'normal')

    # 计算模型置信度
    ml_conf = max(match_probs['ml'].values())
    dl_conf = max(match_probs.get('dl', {}).values()) if match_probs.get('dl') else 0.5
    model_confidence = (ml_conf + dl_conf) / 2

    return fuser.fuse_with_adaptive_weights(
        poisson_probs=match_probs['poisson'],
        ml_probs=match_probs['ml'],
        dl_probs=match_probs.get('dl'),
        odds=odds,
        market_type=market_type,
        odds_range=odds_range,
        model_confidence=model_confidence
    )


def apply_weekday_adjustment(probs: dict, weekday: str = 'Sat') -> tuple:
    """周中赔率分布调整

    基于历史数据，周中比赛（周一~周四）与周末在冷门分布上有显著差异：
    - 周中：冷门概率+8%（强队客场/一周双赛）
    - 周日：平局概率+5%（策略性防守）

    Args:
        probs: {'H':, 'D':, 'A':} 概率
        weekday: 'Mon'|'Tue'|'Wed'|'Thu'|'Fri'|'Sat'|'Sun'

    Returns:
        (调整后概率, 置信度折扣系数)
    """
    probs = dict(probs)
    total = sum(probs.values())
    probs = {k: v / total for k, v in probs.items()}

    conf_discount = 1.0
    if weekday in ('Mon', 'Tue', 'Wed', 'Thu'):
        # 周中冷门增强：降低主胜概率，平/客胜微增
        discount = 0.05
        probs['H'] = max(0.01, probs.get('H', 0.33) - discount)
        extra = discount * 0.6
        probs['A'] = probs.get('A', 0.33) + extra
        probs['D'] = probs.get('D', 0.33) + discount - extra
        conf_discount = 0.95
    elif weekday == 'Sun':
        # 周日平局增强
        discount = 0.04
        probs['D'] = probs.get('D', 0.33) + discount
        reduce_each = discount / 2
        probs['H'] = max(0.01, probs.get('H', 0.33) - reduce_each)
        probs['A'] = max(0.01, probs.get('A', 0.33) - reduce_each)
        conf_discount = 0.97
    else:  # Fri, Sat
        conf_discount = 1.0

    # 重新归一化
    total = sum(probs.values())
    probs = {k: v / total for k, v in probs.items()}

    return probs, conf_discount
