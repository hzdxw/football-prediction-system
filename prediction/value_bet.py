"""价值投注 — 赔率隐含为基准 + alpha + 平局专项检测

核心: 以赔率隐含概率为基准，泊松超额预测作为alpha。
超强热门(赔率<1.20)不推荐平局（泊松高λ虚增平局概率）。
平局信号>=2时用历史先验boost平局概率（但超强热门除外）。
"""
_KELLY_MAP = {'胜': 'home_kelly', '平': 'draw_kelly', '负': 'away_kelly'}
_SUP_MAP = {'胜': 'home', '平': 'draw', '负': 'away'}


def _draw_signals(ho, do, ao, lg, pop):
    score = 0; sigs = []
    # 改进4: 热方低赔(<1.6)时平局信号降权——庄控盘可能
    fav_odds = min(ho, ao)
    hot_draw_penalty = fav_odds < 1.6
    if 3.0 <= do < 3.5: score += 1; sigs.append('中平赔(29.4%)')
    if lg in ('国际赛', '友谊赛'): score += 1; sigs.append('国际赛平局+6%')
    if do <= min(ho, ao): score += 1; sigs.append('凸型分布')
    if pop and pop.get('betfair'):
        kv = pop['betfair'].get('draw_kelly')
        if kv is not None and float(kv) < 3: score += 1; sigs.append('必发平局共识')
    if pop and pop.get('support'):
        sv = pop['support'].get('draw', 0)
        if sv > 30: score += 1; sigs.append(f'支持率平{sv:.0f}%')
    if ho > 1.8 and ao > 2.0: score += 0.5; sigs.append('均势场')
    fav_odds = min(ho, ao)
    if 1.3 <= fav_odds < 2.0 and do > 3.5: score += 1; sigs.append('适度热方+高平赔')
    # 热方低赔时平局总分减半
    if hot_draw_penalty:
        score = score * 0.5
        sigs.append('⚠热方低赔<1.6平局降权')
    return score, sigs


def compute_value_bet(spf_probs, ho, do_, ao, lg, pop=None, is_friendly=False, market='single'):
    """is_friendly=True时不推荐价值投注（友谊赛不可预测）

    market: 'single'=单关, 'parlay'=2串1
    - 单关: 标准EV阈值，2元/注
    - 2串1: 更高EV阈值（≥8%才投），单位注数更高（组合风险）

    虚拟投注标准: 2元/注 (固定值)
    输出 units 字段表示投注注数
    """
    if is_friendly:
        return None
    margin = 1 / ho + 1 / do_ + 1 / ao
    impl = {'胜': 1 / ho / margin, '平': 1 / do_ / margin, '负': 1 / ao / margin}
    draw_score, draw_sigs = _draw_signals(ho, do_, ao, lg, pop)
    is_super_hot = min(ho, ao) < 1.20
    best = None
    for label, odds in [('胜', ho), ('平', do_), ('负', ao)]:
        ip = impl[label]
        poisson_p = spf_probs.get(label, 0)
        alpha = max(0, (poisson_p - ip) * 0.5)
        if label == '平' and is_super_hot:
            alpha = min(alpha, 0.01)
        if pop and pop.get('betfair'):
            kv = pop['betfair'].get(_KELLY_MAP[label])
            if kv is not None and float(kv) < 3: alpha += 0.025
        if pop and pop.get('support'):
            sv = pop['support'].get(_SUP_MAP[label], 33)
            if sv < 20: alpha += 0.02
            elif sv > 65: alpha -= 0.02
        if label == '平' and draw_score >= 2 and not is_super_hot:
            if do_ < 3.0: dp = 0.20
            elif do_ < 3.5: dp = 0.294
            elif do_ < 4.0: dp = 0.231
            else: dp = 0.077
            bw = min(0.6, draw_score * 0.15)
            ip = ip * (1 - bw) + dp * bw
            alpha += draw_score * 0.02
        elif label == '平' and lg in ('国际赛', '友谊赛') and do_ < 4.5 and not is_super_hot:
            alpha += 0.03
        adj_p = max(0.05, min(0.90, ip + alpha))
        ev = adj_p * odds - 1
        # P1: Dynamic EV threshold by league volatility and odds range
        _league_ev_multiplier = {
            '法乙': 0.12, '瑞超': 0.10, '葡超': 0.10, '英冠': 0.12,
            '挪超': 0.10, '苏超': 0.10, '澳超': 0.12, '韩职': 0.10,
            '日职': 0.08, '日职联': 0.08, 'J1': 0.08,
            '英超': 0.05, '意甲': 0.05, '西甲': 0.05, '德甲': 0.05, '法甲': 0.05,
            '欧冠': 0.08, '欧联': 0.08,
        }
        _league_mult = _league_ev_multiplier.get(lg, 0.08)
        # Odds-range adjustment: high-odds bets (do>3.5) need higher threshold
        _odds_mult = 1.0 if do_ < 3.5 else 1.4
        _dynamic_ev_threshold = 0.05 * _league_mult * _odds_mult

        # Gap-8: 2串1需要更高EV阈值（组合串关风险叠加）
        if market == 'parlay':
            _dynamic_ev_threshold = max(_dynamic_ev_threshold, 0.08)  # 2串1最低8% EV

        if ev < _dynamic_ev_threshold:
            continue  # Not a value bet under dynamic threshold
        if ev > _dynamic_ev_threshold and (best is None or ev > best['ev']):
            best = {'label': label, 'odds': odds,
                    'prob': round(adj_p * 100, 1),
                    'impl_prob': round(impl[label] * 100, 1),
                    'alpha': round(alpha * 100, 1),
                    'ev': round(ev * 100, 1),
                    # Gap-8: units按Kelly推算，2元/注
                    # Kelly_fraction × bankroll / (odds - 1) / 2
                    'units': 1,  # 2元/注基准，1注=2元
                    'market': market}
    if best and draw_score >= 2 and draw_sigs:
        best['draw_signals'] = draw_sigs
        best['draw_score'] = draw_score
    return best
