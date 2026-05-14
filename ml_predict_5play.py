"""
ml_predict_5play.py — ML v5 单一校准矩阵派生全部5玩法

架构: 赔率→校准λ→单一矩阵→派生5玩法
        强信号(SPF/让球/总进球)约束弱信号(比分)
"""
import os
import sys
import warnings
import importlib.util

import numpy as np

# LGBM训练时总会自动生成Column_N名字，预测时无可避免触发此警告，直接抑制
warnings.filterwarnings('ignore', message='X does not have valid feature names')

def _get_william_init(match):
    """从match dict提取威廉希尔初赔，返回(h,d,a)或None"""
    try:
        wh = float(match.get('william_home_open', 0) or 0)
        wd = float(match.get('william_draw_open', 0) or 0)
        wa = float(match.get('william_away_open', 0) or 0)
        if wh > 0 and wd > 0 and wa > 0:
            return (wh, wd, wa)
    except (ValueError, TypeError):
        pass
    return None

_spec = importlib.util.spec_from_file_location(
    '_cp', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prediction', 'calibrated_poisson.py'))
_cp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cp)
sys.modules['_cp'] = _cp

def _build_ev_gap_signal(value_bet):
    """P2-B: 价值赌注评分信号 — 从compute_value_bet结果提取EV gap摘要

    Returns:
        dict: {has_value, ev_pct, pred_prob, impl_prob, label, odds}
              或 None（无value）
    """
    if not value_bet:
        return None
    return {
        'has_value': True,
        'ev_pct': value_bet.get('ev', 0),
        'pred_prob': value_bet.get('prob', 0),
        'impl_prob': value_bet.get('impl_prob', 0),
        'label': value_bet.get('label', ''),
        'odds': value_bet.get('odds', 0),
    }


calibrate_lambda = _cp.calibrate_lambda
apply_league_factor = _cp.apply_league_factor
apply_score_prior = _cp.apply_score_prior
derive_all_plays = _cp.derive_all_plays
apply_popularity_prior = _cp.apply_popularity_prior

# Agent Harness — Filter Registry (P0接入 2026-05-06)
try:
    from prediction.filter_registry import get_filter, apply_filters, FILTER_REGISTRY
    _FILTER_REGISTRY_AVAILABLE = True
except ImportError:
    _FILTER_REGISTRY_AVAILABLE = False
    _fr_missing_warned = False

# Team Tier System — lazy-load module + data cache (P2接入 2026-05-12)
_TIER_MODULE = None
_TIER_CACHE = None

# Odds Trajectory — lazy-load (变盘信号 P2接入 2026-05-12)
_ODDS_TRAJ_MODULE = None
# Bookmaker Compare — lazy-load (三家对比信号 P2接入 2026-05-12)
_BOOKMAKER_COMPARE_MODULE = None

import json
_LOG_SAMPLES_FD = None
def _get_log_fd():
    global _LOG_SAMPLES_FD
    if _LOG_SAMPLES_FD is None:
        from datetime import datetime
        _ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        _path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'log_samples_{_ts}.jsonl')
        os.makedirs(os.path.dirname(_path), exist_ok=True)
        _LOG_SAMPLES_FD = open(_path, 'a', encoding='utf-8')
        print(f"📝 log_samples → {_path}")
    return _LOG_SAMPLES_FD

# ── 联赛大球/小球风格因子 ──
LEAGUE_GOAL_STYLE = {
    '德甲': 1.12, '中超': 1.10, '荷甲': 1.08, '英超': 1.05,
    '日职': 1.05, '挪超': 1.08, '瑞超': 1.05,
    '意甲': 0.92, '法甲': 0.90, '西甲': 0.95, '葡超': 0.95, '英冠': 0.93,
    '欧冠': 1.02, '欧联': 1.02, '世预赛': 1.00, '国际赛': 0.98,
}

# P1: 弱联赛列表（威廉希尔偏差时额外降权）
WEAK_LEAGUES = {'日职', '日职联', 'J1', '韩K联', '韩K1联', '韩职', 'K1',
                '泰超', '马来超', '越南联', '印尼甲', '中超', '中超旧名'}

# P1: 比分/总进球降权 - 命中率仅8.4%/9.7%
# 仅在高置信SPF+高置信联赛时推荐
HIGH_CONF_LEAGUES_BF = {'英甲', '欧冠', '德乙', '瑞超', '挪超', '欧联杯',
                         '欧协联', '韩K联', '荷乙', '西甲', '意甲'}

def apply_league_goal_style(cal_h, cal_a, league):
    """调整 λ 的联赛风格因子"""
    factor = LEAGUE_GOAL_STYLE.get(league, 1.0)
    return cal_h * factor, cal_a * factor

def implied_prob(h, d, a):
    r = 1/h + 1/d + 1/a
    return 1/h/r, 1/d/r, 1/a/r

_ODDS_HC = [(1.0,1.2,-2.25),(1.2,1.35,-1.75),(1.35,1.5,-1.25),(1.5,1.7,-0.75),
    (1.7,2.0,-0.25),(2.0,2.3,0.0),(2.3,2.8,0.25),(2.8,3.5,0.5),(3.5,5.0,0.75),(5.0,99,1.0)]

def expected_handicap(ho, ao):
    fav = min(ho, ao)
    for lo, hi, hc in _ODDS_HC:
        if lo <= fav < hi: return hc if fav == ho else -hc
    return 0.0

def water_level_classify(w):
    """7-tier水位分类: 超低<0.75 | 低0.75-0.85 | 中低0.86-0.90 | 中0.91-0.95 | 中高0.96-1.00 | 高1.00-1.08 | 超高>1.08"""
    if w < 0.75: return '超低水'
    if w <= 0.85: return '低水'
    if w <= 0.90: return '中低水'
    if w <= 0.95: return '中水'
    if w <= 1.00: return '中高水'
    if w <= 1.08: return '高水'
    return '超高水'

def compute_return_rate(h, d, a):
    """体彩隐含返还率 ≈ 1 / (1/h + 1/d + 1/a) — 宝典核心指标
    
    ⚠️ 注意：这是「返还率」(implied return rate, 理论≈88.6%)，
    不是「返奖率」(prize payout rate, 固定70%)。
    返还率用于判断庄家抽水程度和隐含概率计算。
    """
    try: return 1.0 / (1.0/h + 1.0/d + 1.0/a)
    except Exception: return 0.886  # 体彩理论返还率默认值

def compute_draw_deviation(d):
    """平赔偏离度 = draw_odds - 3.30（中庸平赔位置）"""
    return d - 3.30

def compute_odds_tail(odds):
    """赔率尾数大小：≥5为重尾(1)，<5为轻尾(0)"""
    try: return 1 if int(round(odds * 100)) % 10 >= 5 else 0
    except Exception: return 0

# ── Gap-10: 历史相似盘口特征 ──────────────────────────────────────────────
_LEAGUE_MAP = {
    'England: Premier League': 'E0', 'English Premier League': 'E0',
    'England: Championship': 'E1', 'English Championship': 'E1',
    'Germany: Bundesliga': 'D1', 'German Bundesliga': 'D1',
    'Germany: Bundesliga 2': 'D2', 'German 2. Bundesliga': 'D2',
    'Spain: La Liga': 'SP1', 'Spain: Primera Division': 'SP1', 'Spanish La Liga': 'SP1',
    'Spain: Segunda Division': 'SP2', 'Spanish Segunda': 'SP2',
    'Italy: Serie A': 'I1', 'Italian Serie A': 'I1',
    'Italy: Serie B': 'I2', 'Italian Serie B': 'I2',
    'France: Ligue 1': 'F1', 'French Ligue 1': 'F1',
    'France: Ligue 2': 'F2', 'French Ligue 2': 'F2',
    'Netherlands: Eredivisie': 'N1', 'Dutch Eredivisie': 'N1',
    'Netherlands: Eerste Divisie': 'N2', 'Dutch Eerste Divisie': 'N2',
    'Belgium: Jupiler League': 'B1', 'Belgian Jupiler League': 'B1',
    'Portugal: Primeira Liga': 'P1', 'Portuguese Primeira Liga': 'P1',
    'Turkey: Super Lig': 'T1', 'Turkish Super Lig': 'T1',
    'Greece: Super League': 'G1', 'Greek Super League': 'G1',
    'Scotland: Premier League': 'SC0', 'Scottish Premier League': 'SC0',
    'Scotland: Division One': 'SC1', 'Scottish Division One': 'SC1',
    'Austria: Bundesliga': 'A1', 'Austrian Bundesliga': 'A1',
    'Switzerland: Super League': 'SW1', 'Swiss Super League': 'SW1',
    'Czech Republic: Gambrinus Liga': 'CZE1',
    'Romania: Liga I': 'RUT1',
    'Poland: Ekstraklasa': 'PL1', 'Polish Ekstraklasa': 'PL1',
    'Denmark: Superliga': 'DK1', 'Danish Superliga': 'DK1',
    'Sweden: Allsvenskan': 'SWED1', 'Swedish Allsvenskan': 'SWED1',
    'Norway: Eliteserien': 'NOR1', 'Norwegian Eliteserien': 'NOR1',
    'Finland: Veikkausliiga': 'FIN1',
    'Australia: A-League': 'AUS1', 'A-League': 'AUS1',
    'Japan: J1 League': 'JAP1', 'Japanese J1 League': 'JAP1',
    'Japan: J2 League': 'JAP2', 'Japanese J2 League': 'JAP2',
    'South Korea: K League 1': 'KOR1', 'Korean K League 1': 'KOR1',
    'South Korea: K League 2': 'KOR2',
    'China: Super League': 'CN1', 'Chinese Super League': 'CN1',
    'Brazil: Serie A': 'BR1', 'Brazilian Serie A': 'BR1',
    'Argentina: Primera Division': 'ARG1',
    'USA: MLS': 'USA1', 'American MLS': 'USA1',
}

_HIST_CACHE = {}  # {(league_short, ho_bucket, do_bucket, ao_bucket): result}
_FD_CACHE = {}    # {(league_short, team, n): result}

def compute_historical_similarity(league_short, home_odds, draw_odds, away_odds, lookback_days=365, top_n=5):
    """Gap-10: 从479k历史赔率库中找相似盘口，统计历史跑出分布。

    league_short: 联赛短码如 'E0', 'D1', 'SP1'
    home_odds / draw_odds / away_odds: 当前赔率
    lookback_days: 只看近N天内的历史（避免联赛风格漂移）
    top_n: 取最近的N场相似盘口

    返回: {
        historical_home_wr, historical_draw_wr, historical_away_wr,
        historical_goal_avg, n_matches_found, avg_odds_distance
    }
    不足3场返回联赛全局均值。
    """
    import math, gzip, csv
    from collections import defaultdict

    cache_key = (league_short, round(home_odds, 2), round(draw_odds, 2), round(away_odds, 2))
    if cache_key in _HIST_CACHE:
        return _HIST_CACHE[cache_key]

    league_full = None
    for full, short in _LEAGUE_MAP.items():
        if short == league_short:
            league_full = full
            break

    matches = []
    try:
        with gzip.open(os.path.expanduser('~/.hermes/workspace/data-collection/kaggle_beatthebookie/closing_odds.csv.gz'), 'rt', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_league = row.get('league', '').strip()
                if league_full and row_league != league_full:
                    if row_league not in _LEAGUE_MAP:
                        continue
                    if _LEAGUE_MAP[row_league] != league_short:
                        continue
                elif not league_full:
                    continue

                try:
                    r_ho = float(row['avg_odds_home_win'])
                    r_do = float(row['avg_odds_draw'])
                    r_ao = float(row['avg_odds_away_win'])
                    if r_ho <= 0 or r_do <= 0 or r_ao <= 0:
                        continue
                except (ValueError, KeyError):
                    continue

                dist = math.sqrt((r_ho - home_odds)**2 + (r_do - draw_odds)**2 + (r_ao - away_odds)**2)
                if dist < 1.5:  # 赔率偏差超过1.5的过滤掉
                    try:
                        hs = int(row.get('home_score') or 0)
                        aw = int(row.get('away_score') or 0)
                        matches.append((dist, hs, aw))
                    except ValueError:
                        pass

    except Exception:
        _HIST_CACHE[cache_key] = None
        return None

    if len(matches) < 3:
        # 样本不足，返回联赛级别兜底统计
        fallback = _get_league_fallback_stats(league_short, league_full)
        if fallback:
            _HIST_CACHE[cache_key] = fallback
            return fallback
        _HIST_CACHE[cache_key] = None
        return None

    matches.sort(key=lambda x: x[0])
    top = matches[:top_n]
    n = len(top)
    home_wins = sum(1 for _, hs, aw in top if hs > aw)
    draws = sum(1 for _, hs, aw in top if hs == aw)
    away_wins = n - home_wins - draws
    total_goals = sum(hs + aw for _, hs, aw in top)
    avg_dist = sum(d for d, _, _ in top) / n

    result = {
        'historical_home_wr': round(home_wins / n, 4),
        'historical_draw_wr': round(draws / n, 4),
        'historical_away_wr': round(away_wins / n, 4),
        'historical_goal_avg': round(total_goals / n, 3),
        'n_matches_found': n,
        'avg_odds_distance': round(avg_dist, 4),
    }
    _HIST_CACHE[cache_key] = result
    return result

def _get_league_fallback_stats(league_short, league_full):
    """从closing_odds计算某联赛的全局统计（兜底用）"""
    import gzip, csv
    from collections import defaultdict
    league_rows = defaultdict(list)
    try:
        with gzip.open(os.path.expanduser('~/.hermes/workspace/data-collection/kaggle_beatthebookie/closing_odds.csv.gz'), 'rt', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rl = row.get('league', '').strip()
                if league_full and rl != league_full:
                    if rl in _LEAGUE_MAP and _LEAGUE_MAP[rl] == league_short:
                        pass
                    else:
                        continue
                elif not league_full:
                    continue
                try:
                    r_ho = float(row['avg_odds_home_win'])
                    r_do = float(row['avg_odds_draw'])
                    r_ao = float(row['avg_odds_away_win'])
                    hs = int(row.get('home_score') or 0)
                    aw = int(row.get('away_score') or 0)
                    if r_ho > 0 and r_do > 0 and r_ao > 0:
                        league_rows['all'].append((r_ho, r_do, r_ao, hs, aw))
                except Exception:
                    continue
    except Exception:
        return None

    rows = league_rows.get('all', [])
    if len(rows) < 10:
        return None
    n = len(rows)
    home_wins = sum(1 for _, _, _, hs, aw in rows if hs > aw)
    draws = sum(1 for _, _, _, hs, aw in rows if hs == aw)
    total_goals = sum(hs + aw for _, _, _, hs, aw in rows)
    return {
        'historical_home_wr': round(home_wins / n, 4),
        'historical_draw_wr': round(draws / n, 4),
        'historical_away_wr': round((n - home_wins - draws) / n, 4),
        'historical_goal_avg': round(total_goals / n, 3),
        'n_matches_found': n,
        'avg_odds_distance': 0.0,
    }

# ── Gap-11: 球队近期状态特征（来自fd_uk历史数据） ──────────────────────────
_FD_LEAGUE_FILES = {
    'E0': 'E0', 'E1': 'E1', 'E2': 'E2', 'E3': 'E3',
    'D1': 'D1', 'D2': 'D2',
    'SP1': 'SP1', 'SP2': 'SP2',
    'I1': 'I1', 'I2': 'I2',
    'F1': 'F1', 'F2': 'F2',
    'N1': 'N1',
    'B1': 'B1',
    'P1': 'P1',
    'T1': 'T1',
    'SC0': 'SC0', 'SC1': 'SC1',
    'A1': 'A1',
    'SW1': 'SW1',
    'DK1': 'DK1',
    'SWED1': 'SWED1',
    'NOR1': 'NOR1',
    'AUS1': 'AUS1',
    'JAP1': 'JAP1', 'JAP2': 'JAP2',
    'KOR1': 'KOR1',
    'CN1': 'CN1',
}

_TEAM_FORM_CACHE = {}

def _load_fd_uk_files(league_short):
    """懒加载fd_uk所有赛季CSV到内存（一次性读入，后继复用）"""
    cache_key = ('fd_cache', league_short)
    if cache_key in _FD_CACHE:
        return _FD_CACHE[cache_key]

    import pandas as pd
    league_code = _FD_LEAGUE_FILES.get(league_short)
    if not league_code:
        _FD_CACHE[cache_key] = pd.DataFrame()
        return _FD_CACHE[cache_key]

    base = os.path.expanduser('~/.hermes/workspace/data-collection/football_data_uk/')
    all_dfs = []
    try:
        for f in os.listdir(base):
            if f.startswith(league_code + '_') and f.endswith('.csv'):
                try:
                    df = pd.read_csv(os.path.join(base, f), low_memory=False)
                    all_dfs.append(df)
                except Exception:
                    continue
    except Exception:
        _FD_CACHE[cache_key] = pd.DataFrame()
        return _FD_CACHE[cache_key]

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        _FD_CACHE[cache_key] = combined
        return combined
    _FD_CACHE[cache_key] = pd.DataFrame()
    return _FD_CACHE[cache_key]

def _team_name_fuzzy_match(team_name, fd_teams):
    """在fd_uk球队列表中模糊匹配主客队名"""
    if not team_name or not fd_teams.size:
        return None
    # 精确匹配
    for t in fd_teams:
        if str(t).strip() == str(team_name).strip():
            return t
    # 包含匹配
    for t in fd_teams:
        if str(team_name).strip() in str(t).strip() or str(t).strip() in str(team_name).strip():
            return t
    return None

def compute_team_recent_form(home_team, away_team, league_short, n_recent=10):
    """Gap-11: 从fd_uk历史数据计算两队近期N场的得失球率。

    返回: {
        home_recent_goals_for, home_recent_goals_against,
        away_recent_goals_for, away_recent_goals_against,
        home_recent_win_rate, away_recent_win_rate,
        n_home_matches, n_away_matches
    }
    """
    import pandas as pd  # local import to avoid top-level pandas deps
    cache_key = (home_team, away_team, league_short, n_recent)
    if cache_key in _TEAM_FORM_CACHE:
        return _TEAM_FORM_CACHE[cache_key]

    df = _load_fd_uk_files(league_short)
    if df.empty:
        _TEAM_FORM_CACHE[cache_key] = None
        return None

    fd_teams = df['HomeTeam'].unique()
    h_match = _team_name_fuzzy_match(home_team, fd_teams)
    a_match = _team_name_fuzzy_match(away_team, fd_teams)

    def _get_recent_stats(team, is_home):
        if team is None:
            return None
        if is_home:
            mask = df['HomeTeam'] == team
            sub = df[mask].sort_values('Date', ascending=False).head(n_recent * 2)
        else:
            mask = df['AwayTeam'] == team
            sub = df[mask].sort_values('Date', ascending=False).head(n_recent * 2)

        if len(sub) < 3:
            return None

        goals_for_list = []
        goals_against_list = []
        results_list = []
        for _, row in sub.iterrows():
            if is_home:
                gf = int(row['FTHG']) if pd.notna(row['FTHG']) else 0
                ga = int(row['FTAG']) if pd.notna(row['FTAG']) else 0
                results_list.append(str(row.get('FTR', 'D')).strip().upper())
            else:
                gf = int(row['FTAG']) if pd.notna(row['FTAG']) else 0
                ga = int(row['FTHG']) if pd.notna(row['FTHG']) else 0
                results_list.append(str(row.get('FTR', 'D')).strip().upper())
            goals_for_list.append(gf)
            goals_against_list.append(ga)

        if len(goals_for_list) < 3:
            return None

        recent_gf = goals_for_list[:n_recent]
        recent_ga = goals_against_list[:n_recent]
        recent_res = results_list[:n_recent]
        n = len(recent_gf)
        wins = sum(1 for r in recent_res if (is_home and r == 'H') or (not is_home and r == 'A'))
        return {
            'gf': round(sum(recent_gf) / n, 3),
            'ga': round(sum(recent_ga) / n, 3),
            'win_rate': round(wins / n, 4),
            'n': n,
        }

    h_stats = _get_recent_stats(h_match, True) if h_match else None
    a_stats = _get_recent_stats(a_match, False) if a_match else None

    if h_stats is None and a_stats is None:
        _TEAM_FORM_CACHE[cache_key] = None
        return None

    def _default():
        return {'gf': 1.35, 'ga': 1.15, 'win_rate': 0.45, 'n': 0}

    result = {
        'home_recent_goals_for':     h_stats['gf'] if h_stats else _default()['gf'],
        'home_recent_goals_against': h_stats['ga'] if h_stats else _default()['ga'],
        'away_recent_goals_for':     a_stats['gf'] if a_stats else _default()['gf'],
        'away_recent_goals_against': a_stats['ga'] if a_stats else _default()['ga'],
        'home_recent_win_rate':      h_stats['win_rate'] if h_stats else _default()['win_rate'],
        'away_recent_win_rate':      a_stats['win_rate'] if a_stats else _default()['win_rate'],
        'n_home_matches':            h_stats['n'] if h_stats else 0,
        'n_away_matches':           a_stats['n'] if a_stats else 0,
    }
    _TEAM_FORM_CACHE[cache_key] = result
    return result


def compute_betfair_spread(ho, do_, ao, pop):
    """Gap-13: 必发交易偏度 = 体彩隐含概率 - 必发隐含概率

    体彩隐含概率: 从ho/do_/ao计算（庄家视角，偏中高）
    必发隐含概率: 从betfair的bf_home/bf_draw/bf_away计算（市场真实资金）
    spread > 0: 体彩比必发更看好该方向（庄家控盘信号）
    spread < 0: 必发比体彩更看好该方向（市场资金信号）

    pop格式: {'betfair': {'bf_home': x, 'bf_draw': x, 'bf_away': x}}
    """
    try:
        # 体彩隐含概率
        sporttery_margin = 1/ho + 1/do_ + 1/ao
        s_home = (1/ho) / sporttery_margin
        s_draw = (1/do_) / sporttery_margin
        s_away = (1/ao) / sporttery_margin

        bf = pop.get('betfair', {})
        # 兼容两种字段名：bf_home/bf_draw/bf_away（Betfair赔率>1）或 home_pct/draw_pct/away_pct（百分比 0-100）
        # 区分方式：字段名含 pct=百分比；字段名含 bf_/odds=赔率
        bf_home_raw = bf.get('bf_home') or bf.get('home_pct') or 0
        bf_draw_raw = bf.get('bf_draw') or bf.get('draw_pct') or 0
        bf_away_raw = bf.get('bf_away') or bf.get('away_pct') or 0

        # 判断是否为Betfair赔率格式：字段名含 bf_/odds/price 表示赔率（>1）
        # 字段名含 pct/percent 表示百分比（0-100）
        has_pct_fields = ('home_pct' in bf or 'bf_home' not in bf)
        if has_pct_fields:
            # 百分比格式：home_pct/draw_pct/away_pct（0-100），转为小数概率
            bf_home = float(bf_home_raw) / 100.0
            bf_draw = float(bf_draw_raw) / 100.0
            bf_away = float(bf_away_raw) / 100.0
        else:
            # 赔率格式：bf_home/bf_draw/bf_away（>1），计算隐含概率
            bf_home = float(bf_home_raw)
            bf_draw = float(bf_draw_raw)
            bf_away = float(bf_away_raw)

        if bf_home <= 0 or bf_draw <= 0 or bf_away <= 0:
            return None

        # 必发隐含概率（如果是赔率格式则计算，否则直接用概率值）
        if bf.get('bf_home'):
            bf_margin = 1/bf_home + 1/bf_draw + 1/bf_away
            b_home = (1/bf_home) / bf_margin
            b_draw = (1/bf_draw) / bf_margin
            b_away = (1/bf_away) / bf_margin
        else:
            # 已经是概率（百分比格式）
            b_home = bf_home
            b_draw = bf_draw
            b_away = bf_away

        # spread = 体彩 - 必发（庄家视角 - 市场视角）
        spread_home = s_home - b_home
        spread_draw = s_draw - b_draw
        spread_away = s_away - b_away

        # 综合spread信号：spread>0说明庄家比市场更看多该方向
        signal_strength = max(abs(spread_home), abs(spread_draw), abs(spread_away))
        dominant = 'home' if abs(spread_home) == max(abs(spread_home), abs(spread_draw), abs(spread_away)) \
                   else ('draw' if abs(spread_draw) == max(abs(spread_home), abs(spread_draw), abs(spread_away)) else 'away')

        return {
            'spread_home': round(spread_home, 4),
            'spread_draw': round(spread_draw, 4),
            'spread_away': round(spread_away, 4),
            'signal': f'bf_spread:{dominant}:{signal_strength:.3f}',
            'strength': round(signal_strength, 4),
        }
    except Exception:
        return None


def compute_three_company_signal(ho, do_, ao, wh=None, wd=None, wa=None, lh=None, ld=None, la=None, league=None):
    """三家比较信号：体彩 vs 威廉希尔 vs 立博 vs Interwetten(估算)

    DB字段: william_home_open/draw_open/away_open, ladbrokes_h/d/a
    理论骨架(94体系): 基于主客实力比计算合理赔率，与实际赔率比较
    Interwetten: 基于威廉骨架估算(误差±0.03五大联赛/±0.08小联赛, data_available=False)
    返回: dict with deviation scores, signal string, confidence_weight
    """
    # 动态导入（避免循环依赖）
    try:
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data-collection'))
        from crawl_interwetten import estimate_interwetten_from_william, get_interwetten_confidence_weight
        _HAS_INTERWETTEN = True
    except ImportError:
        _HAS_INTERWETTEN = False

    def implied(h, d, a):
        try:
            r = 1/h + 1/d + 1/a
            return 1/h/r, 1/d/r, 1/a/r
        except Exception:
            return None, None, None

    result = {'deviations': {}, 'signal': '一致', 'has_data': False,
              'interwetten': None, 'confidence_weight': 1.0}
    sp = implied(ho, do_, ao)  # 体彩隐含概率
    if sp[0] is None: return result

    devs = {}
    # 威廉希尔 vs 体彩
    if all(v is not None and v > 0 for v in [wh, wd, wa]):
        result['has_data'] = True
        wp = implied(wh, wd, wa)
        if wp[0] is not None:
            d_h = wp[0] - sp[0]
            d_d = wp[1] - sp[1]
            d_a = wp[2] - sp[2]
            devs['威廉'] = {'dev_h': round(d_h, 3), 'dev_d': round(d_d, 3), 'dev_a': round(d_a, 3)}
    # 立博 vs 体彩
    if all(v is not None and v > 0 for v in [lh, ld, la]):
        result['has_data'] = True
        lp = implied(lh, ld, la)
        if lp[0] is not None:
            d_h = lp[0] - sp[0]
            d_d = lp[1] - sp[1]
            d_a = lp[2] - sp[2]
            devs['立博'] = {'dev_h': round(d_h, 3), 'dev_d': round(d_d, 3), 'dev_a': round(d_a, 3)}

    result['deviations'] = devs
    if not devs:
        return result

    # 综合信号
    total_dev = sum(abs(d['dev_d']) for d in devs.values())
    if total_dev > 0.08:
        result['signal'] = '威廉立博压平' if all(abs(d.get('dev_d', 0)) > 0.03 for d in devs.values()) else '存在偏差'
    elif total_dev < 0.03:
        result['signal'] = '高度一致'
    else:
        result['signal'] = '轻微偏差'

    # ── Interwetten估算信号（基于威廉骨架，非真实数据）────────────
    # 当体彩平赔显著低于威廉平赔时 → 冷门预警（降权应用 confidence_weight）
    if _HAS_INTERWETTEN and wh and wd and wa and do_:
        try:
            est = estimate_interwetten_from_william(wh, wd, wa, league)
            if est:
                result['interwetten'] = est
                interwetten_d_est = est['d']
                est_confidence = est.get('confidence', 'low')
                # 估算数据降权系数
                weight = get_interwetten_confidence_weight(league)
                result['confidence_weight'] = weight

                if float(do_) < interwetten_d_est - est['error_estimate']:
                    if result.get('signal') not in ['威廉立博压平']:
                        result['signal'] = 'Interwetten压平→冷门预警'
                        result['cold_signal'] = True
                        result['signal_note'] = f'(估算降权×{weight}, confidence={est_confidence})'
        except Exception:
            pass
    else:
        result['confidence_weight'] = 0.5  # 无Interwetten时降权

    return result

def compute_euro_asian_gap(ho, ao, actual_hc):
    """欧亚一致性：欧赔隐含盘口 vs 实际亚盘的差异"""
    if actual_hc is None: return None
    implied = expected_handicap(ho, ao)
    return actual_hc - implied

def get_asian_change_signal(match_code):
    """从asian_handicap_odds/macau_asian_handicap获取亚盘变化模式信号
    
    返回: dict or None
      pattern: '升盘降水'/'降盘升水'/'升盘升水'/'降盘降水'/'无变化'
      direction: +1=利好上盘, -1=利空上盘, 0=中性
      (基于宝典理论：升盘降水→阻上→正路，降盘升水→诱→下盘)
    """
    try:
        import psycopg2
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.openclaw', '.env'))
        conn = psycopg2.connect(
            host=os.getenv('PGHOST','localhost'), dbname=os.getenv('PGDATABASE','myapp_db'),
            user=os.getenv('PGUSER','doodoo'), password=os.getenv('PGPASSWORD',''),
            connect_timeout=5)
        cur = conn.cursor()
        
        # 优先用macau数据（数值型更可靠）
        cur.execute("""SELECT initial_handicap_value, current_handicap_value,
                              initial_home_odds, current_home_odds
                       FROM macau_asian_handicap WHERE match_id LIKE %s
                       LIMIT 1""", (f'%{match_code}%',))
        row = cur.fetchone()
        if row and row[0] is not None and row[1] is not None:
            init_hc, cur_hc, init_w, cur_w = [float(x) if x else None for x in row]
            # Note: macau handicap uses negative for home favorite
            hc_change = cur_hc - init_hc if init_hc is not None and cur_hc is not None else 0
            w_rise = cur_w > init_w if init_w is not None and cur_w is not None else None
            cur.close(); conn.close()
        else:
            # Fallback to asian_handicap_odds (Chinese text)
            cur.execute("""SELECT initial_handicap, current_handicap,
                                  initial_home_water, current_home_water
                           FROM asian_handicap_odds WHERE match_id = %s
                           LIMIT 1""", (match_code,))
            row = cur.fetchone()
            cur.close(); conn.close()
            if not row: return None
            
            init_hc_str, cur_hc_str, init_w, cur_w = row
            from prediction.even_match_model import _parse_handicap_text
            # Parse Chinese handicap
            _M = {'平手':0,'平半':.25,'半球':.5,'半一':.75,'一球':1.,
                   '一球半':1.5,'球半':1.5,'半二':1.75,'二球':2.,'二球半':2.5}
            def _parse(raw):
                if raw is None: return None
                raw = raw.strip().replace(' ','')
                neg = '受让' in raw
                for k,v in _M.items():
                    if k in raw: return -v if neg else v
                try: return -float(raw) if neg else float(raw)
                except Exception: return None
            
            init_hc = _parse(init_hc_str)
            cur_hc = _parse(cur_hc_str)
            if init_hc is None or cur_hc is None: return None
            hc_change = cur_hc - init_hc
            try: w_rise = float(cur_w) > float(init_w) if cur_w and init_w else None
            except Exception: w_rise = None
        
        # Classify pattern (宝典理论)
        if abs(hc_change) < 0.01:
            pattern = '无变化'
            direction = 0
        elif hc_change < 0:  # 降盘 (盘口值变小 = 上盘难度降低)
            if w_rise is True: pattern = '降盘升水'; direction = -1  # 诱上→下盘
            elif w_rise is False: pattern = '降盘降水'; direction = 0  # 中性
            else: pattern = '降盘(水位未知)'; direction = -1
        else:  # 升盘
            if w_rise is False: pattern = '升盘降水'; direction = 1  # 阻上→正路
            elif w_rise is True: pattern = '升盘升水'; direction = 0  # 中性
            else: pattern = '升盘(水位未知)'; direction = 1
        
        return {'pattern': pattern, 'direction': direction,
                'hc_change': round(hc_change, 3)}
    except Exception:
        return None

def _build_saturday_features(ho, do_, ao, dow):
    """构建周六模型特征"""
    margin = 1/ho + 1/do_ + 1/ao
    p_h, p_d, p_a = 1/ho/margin, 1/do_/margin, 1/ao/margin
    is_sat = 1.0 if dow == 5 else 0.0
    return [
        p_h, p_d, p_a, margin, ho/ao, min(ho, ao), abs(ho-ao),
        do_/min(ho, ao), ao/ho, is_sat,
        1.0 if 1.8 <= ho <= 2.2 else 0.0,
        1.0 if do_ <= 3.0 else 0.0,
        1.0 if ho < 1.5 else 0.0,
        1.0 if ao < 1.5 else 0.0,
        1.0 if abs(ho-ao) < 0.3 else 0.0,
        p_h * is_sat, p_d * is_sat,
        1.0 if ho < ao else 0.0,
        1.0 if max(ho, do_, ao)/min(ho, do_, ao) < 1.5 else 0.0,
    ]

def predict_5play(match, _skip_super_fusion=False):
    h, a, mc, lg = (match.get('home_team', ''), match.get('away_team', ''),
                     match.get('match_code', ''), match.get('league', ''))
    ho = float(match.get('home_odds', match.get('standard_win', 2)))
    do_ = float(match.get('draw_odds', match.get('standard_draw', 3)))
    ao = float(match.get('away_odds', match.get('standard_lose', 3)))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prediction'))
    from ensemble_predict import ensemble_predict
    ens = ensemble_predict(ho, do_, ao, home_team=h, away_team=a, league=lg) or {}
    ens_pred = ens.get('prediction', '胜')
    ens_conf = ens.get('confidence', 0)
    sp_pt = (match.get('spread_point') or match.get('asian_handicap') 
             or match.get('asian_spread_line') or match.get('handicap_numeric') 
             or match.get('handicap'))
    sp_h = (match.get('spread_home') or match.get('asian_spread_home')
            or match.get('handicap_home_odds') or match.get('hc_win'))
    sp_a = (match.get('spread_away') or match.get('asian_spread_away')
            or match.get('handicap_away_odds') or match.get('hc_lose'))
    ou_line = match.get('total_line') or match.get('overunder_line') or match.get('over_line')
    ou_over = match.get('total_over') or match.get('over_odds')
    ou_under = match.get('total_under') or match.get('under_odds')

    # ── 优先用历史5场得失球，赔率反推仅作为兜底 ──
    # 队名映射：sporttery_spf简称 → collected_match_data_2026全称
    _name_map = {
        '麦克阿瑟': '麦克阿瑟FC', '纽喷气机': '纽卡斯尔喷气机',
        '布里斯班': '布里斯班狮吼', '悉尼FC': '悉尼FC',
        '维冈': '维冈竞技', '莱顿东方': '莱顿东方',
        '墨胜利': '墨尔本胜利', '墨尔本城': '墨尔本城',
        '珀斯光荣': '珀斯光荣', '阿德莱德': '阿德莱德联',
        '西悉尼': '西悉尼流浪者', '狮吼': '布里斯班狮吼',
        '中央海岸': '中央海岸水手',
    }
    h_lookup = _name_map.get(h, h)
    a_lookup = _name_map.get(a, a)
    hist_lambda = None
    try:
        from prediction.historical_lambda import get_match_lambda
        # 澳超CV极高(>0.8)，需更多样本才能稳定估计λ
        n_a_league = 15 if lg in ('澳超', 'A-League') else 5
        hist_lambda = get_match_lambda(h_lookup, a_lookup, n_recent=n_a_league, league=lg)
    except Exception:
        pass

    # ── P0-2: 时序权重计算（提至calibrate_lambda调用前）──
    if not hist_lambda:  # 只在无历史λ时计算，因为有历史λ就不用calibrate_lambda
        try:
            team_form_for_lambda = compute_team_recent_form(h, a, lg)
        except Exception:
            team_form_for_lambda = None
    else:
        team_form_for_lambda = None

    if hist_lambda:
        # 历史数据存在时，100%使用历史λ（不融合赔率）
        cal_h, cal_a, raw_h, raw_a = hist_lambda[0], hist_lambda[1], hist_lambda[0], hist_lambda[1]
    else:
        # 无历史数据时，才用赔率反推λ作为兜底
        cal_h, cal_a, raw_h, raw_a = calibrate_lambda(
            ho, do_, ao,
            spread_point=sp_pt, spread_home=sp_h, spread_away=sp_a,
            total_line=ou_line, total_over=ou_over, total_under=ou_under,
            recent_form=team_form_for_lambda,  # P0-2: 时序权重注入
        )

    # ── 联赛大球/小球风格因子 ──
    cal_h, cal_a = apply_league_goal_style(cal_h, cal_a, lg)

    # ── P5: 联赛×周几λ校正 ──
    _dow_draw_adj = 0.0
    _dl_signals = []
    try:
        from prediction.post_adjust import apply_dow_league_lambda
        cal_h, cal_a, _dow_draw_adj, _dl_signals = apply_dow_league_lambda(
            cal_h, cal_a, lg, match.get('match_date'), mc)
    except Exception:
        pass

    # ── DL独立计算（多源融合第一步，2026-05-05新增）──
    # 根因：DL从未在predict_5play主线被调用，只在ensemble_predict/SuperFusion中间接使用
    # 修复：DL作为独立计算源，与NB/v2并列，通过贝叶斯融合保证数学自洽
    _dl_spf = None
    _dl_tg = None
    _dl_bf_probs = None
    if not _skip_super_fusion:
        try:
            from data_collection.dl_predictor import _get_cached_predictor
            _dl = _get_cached_predictor(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))
            if _dl and hasattr(_dl, 'models') and _dl.models:
                # SPF的DL预测
                _dl_spf = _dl.predict(match_code=mc, league=lg, market='spf',
                                      home_team=h, away_team=a,
                                      home_odds=ho, draw_odds=do_, away_odds=ao)
                # 总进球的DL预测（zjq市场）
                # DL输出: ['0-1球','2球','3球','4+球'] → 映射为tg_dist格式
                _dl_tg_raw = _dl.predict(match_code=mc, league=lg, market='zjq',
                                         home_team=h, away_team=a,
                                         home_odds=ho, draw_odds=do_, away_odds=ao)
                if _dl_tg_raw and _dl_tg_raw.get('prob'):
                    _dl_tg_probs = _dl_tg_raw['prob']  # [p01, p2, p3, p4+]
                    _dl_tg = {
                        '0': _dl_tg_probs[0],   # P(tg=0 or 1)
                        '1': 0.0,
                        '2': _dl_tg_probs[1],   # P(tg=2)
                        '3': _dl_tg_probs[2],   # P(tg=3)
                        '4+': _dl_tg_probs[3],  # P(tg>=4)
                    }
                    _dl_tg['dist'] = [
                        ('0', _dl_tg_probs[0]),
                        ('2', _dl_tg_probs[1]),
                        ('3', _dl_tg_probs[2]),
                        ('4+', _dl_tg_probs[3]),
                    ]
        except Exception:
            pass

    # ── 英冠热门陷阱检测 ──
    _championship_cold_signal = None
    if lg == '英冠':
        try:
            from prediction.championship_cold_trap import detect_championship_cold_trap
            _cold = detect_championship_cold_trap(ho, do_, ao, '英冠')
            if _cold and _cold['is_cold_trap']:
                _championship_cold_signal = _cold
                # 降低主胜概率权重
                _adj = _cold['adjustment']
                _adj_prob = _adj['adjusted_home_prob']
                # 重新调整λ
                if _adj_prob < 0.50:
                    cal_h *= (_adj_prob / 0.50)  # 按比例降低
        except Exception:
            pass

    # ── 主客场λ调整（联赛级别）──
    home_adj_factor = None
    try:
        _ha_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'home_advantage.json')
        if os.path.exists(_ha_path):
            import json as _json
            with open(_ha_path) as _f:
                _ha_data = _json.load(_f)
            _ha_league = _ha_data.get(lg)
            if _ha_league and _ha_league.get('games', 0) >= 50:
                home_adj_factor = _ha_league['factor']
                adj = home_adj_factor ** 0.5  # sqrt to moderate effect
                cal_h = cal_h * adj
                cal_a = cal_a / adj
    except Exception:
        pass

    # ── 球队近期状态调整λ ──
    form_adj_info = []
    home_form = match.get('home_form')
    away_form = match.get('away_form')
    baseline = 1.35
    if home_form and home_form.get('matches', 0) >= 2:
        hf = 0.7 + 0.3 * home_form['avg_goals'] / baseline
        hf = max(0.6, min(1.4, hf))
        cal_h *= hf
        form_adj_info.append(f'主{home_form["avg_goals"]:.1f}球/场→λ×{hf:.2f}')
    if away_form and away_form.get('matches', 0) >= 2:
        af = 0.7 + 0.3 * away_form['avg_goals'] / baseline
        af = max(0.6, min(1.4, af))
        cal_a *= af
        form_adj_info.append(f'客{away_form["avg_goals"]:.1f}球/场→λ×{af:.2f}')

    # ── xG（预期进球）调整λ — understat数据 ──
    xg_adj_info = []
    try:
        from prediction.match_loader import get_xg_data
        # 确定赛季：优先用match_date推算
        _md = match.get('match_date', '')
        _season = '2024' if _md >= '2024-08-01' else ('2025' if _md >= '2025-08-01' else '2024')
        hxg = get_xg_data(h, lg, season=_season)
        axg = get_xg_data(a, lg, season=_season)
        xg_baseline = 1.35  # 五大联赛平均xG
        if hxg:
            h_xg_factor = 0.6 + 0.4 * hxg['avg_xg_for'] / xg_baseline
            h_xg_factor = max(0.7, min(1.3, h_xg_factor))
            cal_h *= h_xg_factor
            xg_adj_info.append(f'主xG={hxg["avg_xg_for"]:.2f}→λ×{h_xg_factor:.2f}')
        if axg:
            a_xg_factor = 0.6 + 0.4 * axg['avg_xg_for'] / xg_baseline
            a_xg_factor = max(0.7, min(1.3, a_xg_factor))
            cal_a *= a_xg_factor
            xg_adj_info.append(f'客xG={axg["avg_xg_for"]:.2f}→λ×{a_xg_factor:.2f}')
    except Exception:
        pass

    # ── Kalman Filter 动态实力调整λ ──
    try:
        from prediction.kalman_team_strength import kalman_adjust_lambda
        cal_h, cal_a = kalman_adjust_lambda(h, a, cal_h, cal_a)
    except Exception:
        pass

    # ── λ波动性CV调整（澳超CV>0.8需强惩罚，向联赛均值收缩）──
    cv_h = cv_a = None
    try:
        from prediction.historical_lambda import get_lambda_cv
        cv_h = get_lambda_cv(h_lookup, is_home=True, league=lg, n_recent=10)
        cv_a = get_lambda_cv(a_lookup, is_home=False, league=lg, n_recent=10)
        # 联赛λ均值（澳超主场1.66/客队参考1.40，全局1.35）
        lg_avg_h = 1.66 if lg in ('澳超','A-League') else 1.35
        lg_avg_a = 1.40 if lg in ('澳超','A-League') else 1.30
        # CV分段贝叶斯收缩：CV越高→收缩越强
        # 澳超CV>0.8属于极端，需30-35%收缩
        if cv_h:
            if cv_h > 0.8: cal_h = cal_h * 0.65 + lg_avg_h * 0.35  # 极端收缩
            elif cv_h > 0.7: cal_h = cal_h * 0.70 + lg_avg_h * 0.30  # 强收缩
            elif cv_h > 0.5: cal_h = cal_h * 0.80 + lg_avg_h * 0.20
            elif cv_h > 0.4: cal_h *= 0.95
        if cv_a:
            if cv_a > 0.8: cal_a = cal_a * 0.65 + lg_avg_a * 0.35
            elif cv_a > 0.7: cal_a = cal_a * 0.70 + lg_avg_a * 0.30
            elif cv_a > 0.5: cal_a = cal_a * 0.80 + lg_avg_a * 0.20
            elif cv_a > 0.4: cal_a *= 0.95
    except Exception:
        pass

    # ── 半场比分修正λ（如有半场数据）──
    ht_signals = []
    try:
        hs = match.get('half_score') or match.get('half_time_score')
        if hs and ':' in str(hs):
            parts = str(hs).split(':')
            hh, ha = int(parts[0]), int(parts[1])
            from prediction.historical_lambda import adjust_lambda_by_halftime
            cal_h, cal_a, ht_signals = adjust_lambda_by_halftime(cal_h, cal_a, hh, ha)
    except Exception:
        pass

    matrix = _cp._build_matrix(cal_h, cal_a)
    matrix = apply_league_factor(matrix, lg)
    matrix = apply_score_prior(matrix, lg)  # P0: 联赛历史比分先验
    # ── 支持率/必发先验修正 ──
    support = match.get('support')
    betfair = match.get('betfair')
    if support or betfair:
        matrix = apply_popularity_prior(matrix, support=support, betfair=betfair)
    hc_str = match.get('handicap', '')
    plays = derive_all_plays(matrix, handicap=hc_str, spf_result=ens_pred)

    # ── 体彩比分赔率融合 ──
    try:
        from prediction.scoreline_odds_fusion import fuse_scoreline_with_odds
        plays = fuse_scoreline_with_odds(plays, match.get('match_date'), match.get('match_code'))
    except Exception:
        pass

    # ── P1: 负二项比分模型（替代泊松比分） ──
    try:
        from prediction.calibrated_poisson import derive_all_plays_nb
        r_h = 1.8 if lg in ('德甲', '荷甲', '挪超') else 2.0  # 高进球联赛r更低=方差更大
        r_a = 1.8 if lg in ('德甲', '荷甲', '挪超') else 2.0
        nb_plays = derive_all_plays_nb(cal_h, cal_a, handicap=float(hc_str) if hc_str else 0, league=lg, r_h=r_h, r_a=r_a, use_score_prior=True)
        # 合并NB比分到现有scoreline（不整体替换，只更新top5）
        if 'scoreline' in nb_plays and 'top5' in nb_plays['scoreline']:
            _nb_sf = nb_plays['scoreline']
            if 'top5' in _nb_sf:
                plays['scoreline']['top5'] = _nb_sf['top5']
        plays['total_goals'] = nb_plays.get('total_goals', plays['total_goals'])
        plays['half_full'] = nb_plays.get('half_full', plays['half_full'])
    except Exception:
        pass

    # ── 比分最终确认（v2用市场赔率算λ，与SPF预测解耦）──
    # v2的泊松矩阵用市场赔率推导λ，不用post-processed概率
    # 这样比分和胜负预测才不会脱节
    spf_probs_final = plays.get('spf', {}).get('probs', {})
    _is_friendly_match = lg and any(kw in lg for kw in ['友谊', '热身', 'Friendly'])
    if spf_probs_final and not _is_friendly_match:
        try:
            from prediction.score_predictor_v2 import predict_scoreline_v2
            ou_line = float(match.get('over_line') or 2.75)
            ou_probs = {'total': ou_line, 'over': 0.52, 'under': 0.48}
            _tg = plays.get('total_goals', {}).get('expected')
            # 用市场赔率（ho/do/ao）推导spf_probs给v2用，而不是post-processed概率
            # 这样泊松矩阵的λ和胜平负预测完全对齐
            _ho = float(match.get('home_win', match.get('ho', 0)))
            _do = float(match.get('draw', match.get('do', 0)))
            _ao = float(match.get('away_win', match.get('ao', 0)))
            if _ho > 0 and _do > 0 and _ao > 0:
                _imp_total = 1/_ho + 1/_do + 1/_ao
                _mkt_probs = {
                    '胜': min(1.0, (1/_ho / _imp_total)),
                    '平': min(1.0, (1/_do / _imp_total)),
                    '负': min(1.0, (1/_ao / _imp_total)),
                }
            else:
                _mkt_probs = spf_probs_final  # fallback到post-processed
            _bf_scores, _bf_sigs, _ = predict_scoreline_v2(
                match_code=mc, league=lg,
                spf_probs=_mkt_probs, ou_probs=ou_probs,
                total_goals_ensemble=_tg)
            if _bf_scores:
                v2_top5 = sorted(_bf_scores.items(), key=lambda x: -x[1])[:5]
                plays['scoreline']['top5'] = [
                    (s, f'{p*100:.1f}%') for s, p in v2_top5]
                plays['scoreline']['top5_raw'] = plays['scoreline']['top5']
                top1_score = v2_top5[0]
                # ── 比分与SPF方向一致性修正 ──
                # 用ens_pred（函数开头的ensemble方向）判断，比post-processed spf更稳定
                # 平被P5抬高后spf_probs_final的max可能变成平，导致误判
                _spf_dir = ens_pred  # '胜'/'平'/'负'
                _t1 = top1_score[0]
                _h1, _a1 = int(_t1.split('-')[0]), int(_t1.split('-')[1])
                _conflict = (
                    (_spf_dir == '胜' and _h1 <= _a1) or
                    (_spf_dir == '负' and _h1 >= _a1) or
                    (_spf_dir == '平' and _h1 != _a1)
                )
                if _conflict and len(v2_top5) >= 3:
                    for _cand, _ in v2_top5[:3]:
                        _ch, _ca = int(_cand.split('-')[0]), int(_cand.split('-')[1])
                        _ok = (
                            (_spf_dir == '胜' and _ch > _ca) or
                            (_spf_dir == '负' and _ch < _ca) or
                            (_spf_dir == '平' and _ch == _ca)
                        )
                        if _ok:
                            top1_score = (_cand, _bf_scores[_cand])
                            break
                # ── end修正 ──
                plays['scoreline']['top'] = top1_score[0]
                plays['scoreline']['top_prob'] = top1_score[1]
                plays['scoreline']['top2'] = (v2_top5[1][0], f'{v2_top5[1][1]*100:.1f}%') if len(v2_top5) > 1 else ('', '0%')
                plays['scoreline']['top3'] = (v2_top5[2][0], f'{v2_top5[2][1]*100:.1f}%') if len(v2_top5) > 2 else ('', '0%')
                try:
                    from prediction.score_predictor_v2 import get_top_band
                    top_band_name, top_band_prob = get_top_band(_bf_scores)
                    # 比分历史命中率<10%，强制降权
                    top_band_prob = min(top_band_prob * 0.5, 0.35)
                    plays['scoreline']['band'] = (top_band_name, f'{top_band_prob*100:.1f}%')
                except Exception:
                    plays['scoreline']['band'] = ('unknown', '0%')

                # ══ 多源贝叶斯融合（2026-05-05重构）══
                # 根因：原v2派生tg/bqc本质是"一套派生另一套"，不是多源独立计算
                # 修复原则：NB/v2/DL三源独立计算 → 贝叶斯条件概率验证 → 数学自洽输出
                try:
                    from math import exp as _exp, factorial as _fact
                    from prediction.calibrated_poisson import _BQC

                    # ── 第一步：保存NB矩阵派生的tg/bqc作为独立源 ──
                    # nb_plays在line 900-912已计算，保存用于贝叶斯融合
                    _nb_tg = dict(plays.get('total_goals', {}).get('dist', []))
                    _nb_bqc = dict(plays.get('half_full', {}).get('top3', []))

                    # ── 第二步：v2独立计算bf/tg/bqc（已有代码复用）──
                    # v2的bf_scores已存在于_bf_scores，tg/bqc用v2的λ重新计算
                    tg_dist_v2 = {}
                    for score_str, score_prob in _bf_scores.items():
                        if score_str == '5+':
                            continue
                        try:
                            parts = score_str.split('-')
                            tg = int(parts[0]) + int(parts[1])
                            tg_dist_v2[tg] = tg_dist_v2.get(tg, 0) + score_prob
                        except (ValueError, IndexError):
                            continue
                    _v2_tg = dict(sorted(tg_dist_v2.items(), key=lambda x: -x[1])[:5])

                    # 用v2的λ参数做半全场
                    _expected_total = float(match.get('over_line') or 2.75)
                    _p_win = _mkt_probs.get('胜', 0)
                    _p_lose = _mkt_probs.get('负', 0)
                    _home_frac = _p_win / (_p_win + _p_lose + 1e-9)
                    _away_frac = _p_lose / (_p_win + _p_lose + 1e-9)
                    _v2_lam_h = _expected_total * _home_frac
                    _v2_lam_a = _expected_total * _away_frac
                    _lh1, _la1 = _v2_lam_h * 0.45, _v2_lam_a * 0.45
                    _lh2, _la2 = _v2_lam_h * 0.55, _v2_lam_a * 0.55

                    def _pp(k, lam):
                        if lam <= 0: return 0.0
                        return _exp(-lam) * lam ** k / _fact(max(1, k))

                    _bqc_dist = {}
                    for _hi in range(6):
                        for _hj in range(6):
                            _ph = _pp(_hi, _lh1) * _pp(_hj, _la1)
                            _hr = 1 if _hi > _hj else (0 if _hi == _hj else -1)
                            for _si in range(7 - _hi):
                                for _sj in range(7 - _hj):
                                    _ps = _pp(_si, _lh2) * _pp(_sj, _la2)
                                    _fr = 1 if _hi + _si > _hj + _sj else (0 if _hi + _si == _hj + _sj else -1)
                                    _bqc_dist[(_hr, _fr)] = _bqc_dist.get((_hr, _fr), 0) + _ph * _ps
                    _v2_bqc = dict(sorted(_bqc_dist.items(), key=lambda x: -x[1]))

                    # ── 第三步：DL独立计算（从DL的spf/hhad联合推断bf）──
                    # 如果有DL的tg预测，用它来验证/加权tg分布
                    _dl_tg_dist = None
                    if _dl_tg and _dl_tg.get('dist'):
                        _dl_tg_dist = dict(_dl_tg['dist'])

                    # ── 第四步：贝叶斯多源融合 ──
                    # 原则：tg和bf必须来自同一独立计算源，bqc与bf同源
                    # 融合策略：
                    #   1. BF方向：由v2和NB的共识决定（权重0.7 v2 + 0.3 NB）
                    #   2. TG分布：若DL有独立预测 → DL加权平均；否则 → v2/NB融合
                    #   3. BQC分布：跟随BF的λ参数体系

                    # 计算BF共识权重
                    _v2_bf_sum = sum(_bf_scores.values())
                    _v2_bf_norm = {k: v/_v2_bf_sum for k, v in _bf_scores.items() if k != '5+'}

                    # NB的bf_scores需要从nb_plays提取
                    _nb_bf_scores = {}
                    try:
                        _nb_top5 = nb_plays.get('scoreline', {}).get('top5', [])
                        for _s, _p in _nb_top5:
                            if isinstance(_p, str):
                                _p = float(_p.replace('%', '')) / 100.0
                            _nb_bf_scores[_s] = _p
                    except Exception:
                        _nb_bf_scores = {}

                    _nb_bf_sum = sum(_nb_bf_scores.values()) if _nb_bf_scores else 1.0
                    _nb_bf_norm = {k: v/_nb_bf_sum for k, v in _nb_bf_scores.items()}

                    # BF融合：v2(0.7) + NB(0.3)
                    _fused_bf = {}
                    _all_scores = set(_v2_bf_norm.keys()) | set(_nb_bf_norm.keys())
                    for _s in _all_scores:
                        if _s == '5+':
                            continue
                        _v = _v2_bf_norm.get(_s, 0.0) * 0.7
                        _n = _nb_bf_norm.get(_s, 0.0) * 0.3
                        _fused_bf[_s] = _v + _n

                    # 归一化BF
                    _bf_total = sum(_fused_bf.values())
                    if _bf_total > 0:
                        _fused_bf = {k: v/_bf_total for k, v in _fused_bf.items()}

                    # TG融合：如果有DL，用DL的tg分布做先验修正v2的tg
                    if _dl_tg_dist:
                        # DL的tg: {'0': p01, '2': p2, '3': p3, '4+': p4+}
                        # v2的tg: {0: p0, 1: p1, 2: p2, 3: p3, 4: p4, ...}
                        _fused_tg = {}
                        _dl_tg_total = sum(_dl_tg_dist.values())
                        for _tg_k, _v2_p in _v2_tg.items():
                            _dl_p = 0.0
                            if _tg_k == 0:
                                _dl_p = _dl_tg_dist.get('0', 0.0)
                            elif _tg_k == 2:
                                _dl_p = _dl_tg_dist.get('2', 0.0)
                            elif _tg_k == 3:
                                _dl_p = _dl_tg_dist.get('3', 0.0)
                            elif _tg_k >= 4:
                                _dl_p = _dl_tg_dist.get('4+', 0.0)
                            # 贝叶斯修正：P(tg|DL) ∝ P(DL|tg) × P(tg|v2)
                            # 简化为：融合权重0.6 v2 + 0.4 DL
                            _fused_tg[_tg_k] = _v2_p * 0.6 + (_dl_p / max(_dl_tg_total, 0.01)) * 0.4
                    else:
                        # 无DL时：v2(0.7) + NB(0.3)
                        _fused_tg = {}
                        for _tg_k, _v2_p in _v2_tg.items():
                            _nb_p = _nb_tg.get(_tg_k, 0.0) if _nb_tg else 0.0
                            _fused_tg[_tg_k] = _v2_p * 0.7 + _nb_p * 0.3

                    # 归一化TG
                    _tg_total = sum(_fused_tg.values())
                    if _tg_total > 0:
                        _fused_tg = {k: v/_tg_total for k, v in _fused_tg.items()}

                    # BQC跟随BF的λ体系（v2的λ）
                    _fused_bqc = _v2_bqc

                    # ── 第五步：数学自洽验证 ──
                    # 验证1：top1 bf的tg必须在tg_dist的top1
                    _top1_bf = max(_fused_bf, key=_fused_bf.get)
                    if _top1_bf != '5+':
                        try:
                            _bh, _ba = int(_top1_bf.split('-')[0]), int(_top1_bf.split('-')[1])
                            _bf_tg = _bh + _ba
                            _tg_top1 = max(_fused_tg, key=_fused_tg.get)
                            # 如果bf的tg不是tg_top1，发出警告但保留bf（因为bf权重更高）
                        except (ValueError, IndexError):
                            pass

                    # 验证2：SPF方向与bf方向必须一致
                    _spf_dir = ens_pred
                    _top1_parts = _top1_bf.split('-')
                    _th, _ta = int(_top1_parts[0]), int(_top1_parts[1])
                    _bf_aligned = (
                        (_spf_dir == '胜' and _th > _ta) or
                        (_spf_dir == '平' and _th == _ta) or
                        (_spf_dir == '负' and _th < _ta)
                    )
                    # 如果不一致，用spf方向修正top1
                    if not _bf_aligned:
                        _candidate_scores = sorted(_fused_bf.items(), key=lambda x: -x[1])
                        for _cand, _ in _candidate_scores[:3]:
                            _ch, _ca = int(_cand.split('-')[0]), int(_cand.split('-')[1])
                            _ok = (
                                (_spf_dir == '胜' and _ch > _ca) or
                                (_spf_dir == '平' and _ch == _ca) or
                                (_spf_dir == '负' and _ch < _ca)
                            )
                            if _ok:
                                _top1_bf = _cand
                                break

                    # ── 第六步：更新plays ──
                    # BF
                    _final_bf_sorted = sorted(_fused_bf.items(), key=lambda x: -x[1])[:5]
                    plays['scoreline']['top5'] = [
                        (s, f'{p*100:.1f}%') for s, p in _final_bf_sorted]
                    plays['scoreline']['top5_raw'] = plays['scoreline']['top5']
                    plays['scoreline']['top'] = _top1_bf
                    plays['scoreline']['top_prob'] = _fused_bf.get(_top1_bf, 0)
                    if len(_final_bf_sorted) > 1:
                        plays['scoreline']['top2'] = (_final_bf_sorted[1][0], f'{_final_bf_sorted[1][1]*100:.1f}%')
                    if len(_final_bf_sorted) > 2:
                        plays['scoreline']['top3'] = (_final_bf_sorted[2][0], f'{_final_bf_sorted[2][1]*100:.1f}%')
                    try:
                        from prediction.score_predictor_v2 import get_top_band
                        top_band_name, top_band_prob = get_top_band(_fused_bf)
                        # 比分历史命中率<10%，强制降权
                        top_band_prob = min(top_band_prob * 0.5, 0.35)
                        plays['scoreline']['band'] = (top_band_name, f'{top_band_prob*100:.1f}%')
                    except Exception:
                        plays['scoreline']['band'] = ('unknown', '0%')

                    # TG
                    _tg_sorted = sorted(_fused_tg.items(), key=lambda x: -x[1])
                    plays['total_goals'] = {
                        'dist': _tg_sorted[:5],
                        'top': _tg_sorted[0][0],
                        'top_prob': _tg_sorted[0][1],
                    }

                    # BQC
                    if _fused_bqc:
                        _bqc3 = [
                            (_BQC.get(k, str(k)), f'{v*100:.1f}%')
                            for k, v in sorted(_fused_bqc.items(), key=lambda x: -x[1])[:3]
                        ]
                        plays['half_full'] = {'top3': _bqc3}

                    # 记录融合元数据
                    plays['_bayesian_meta'] = {
                        'v2_bf_weight': 0.7, 'nb_bf_weight': 0.3,
                        'dl_tg_weight': 0.4 if _dl_tg_dist else 0.0,
                        'sources': ['v2', 'nb'] + (['dl'] if _dl_tg_dist else []),
                    }

                except Exception as _e:
                    # 贝叶斯融合失败时，保留v2原始计算作为fallback
                    pass
                # ══ 多源贝叶斯融合结束 ══
        except Exception:
            pass

    spf_probs = plays.get('spf', {}).get('probs', {})

    # ── 威廉希尔变赔方向因子 ──
    odds_shift = None
    try:
        from prediction.odds_shift import calc_odds_shift_factor
        odds_shift = calc_odds_shift_factor(mc)
        if odds_shift and spf_probs:
            # 变赔方向修正：降水方向 = 庄家真实意图
            shift_factor = 1 + odds_shift.get('home_shift', 0) * 0.1
            spf_probs['胜'] *= shift_factor
            # 归一化
            total = sum(spf_probs.values())
            for k in spf_probs: spf_probs[k] /= total
            plays['spf']['probs'] = spf_probs
    except Exception:
        pass

    # ── 赔率轨迹分析（初盘→终盘 升水/降水/变盘四分类+CLV）──
    _traj_signals = []
    _traj_adj = 0.0
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'notes', 'prediction', 'core'))
        from odds_trajectory import trajectory_adjust as _traj_adjust
        _traj = _traj_adjust(mc, spf_probs)
        if _traj and _traj.get('signals'):
            _traj_signals = _traj['signals']
            _traj_adj = _traj.get('confidence_adjust', 0.0)
        plays['spf']['probs'] = spf_probs
    except Exception:
        pass

    if '平' in spf_probs and spf_probs['平'] < 0.15:
        spf_probs['平'] = 0.15
        total_spf = spf_probs.get('胜', 0.4) + spf_probs['平'] + spf_probs.get('负', 0.4)
        for k in ['胜', '平', '负']:
            spf_probs[k] /= total_spf

    # ── 埃罗预测法先验融合（P0新增）──
    elo_data = match.get('elo')
    try:
        if elo_data and elo_data.get('home_points') is not None and elo_data.get('away_points') is not None:
            from prediction.post_adjust import elo_prior_blend
            spf_probs, _elo_signals = elo_prior_blend(
                spf_probs,
                elo_data.get('home_points'),
                elo_data.get('away_points')
            )
            if _elo_signals:
                _adj_signals.extend(_elo_signals)
            plays['spf']['probs'] = spf_probs
            if spf_probs:
                plays['spf']['pred'] = max(spf_probs, key=spf_probs.get)
    except Exception:
        pass
    # ── P5: 联赛×周几平局校正 ──
    if abs(_dow_draw_adj) > 0.01:
        try:
            from prediction.post_adjust import apply_dow_draw_to_spf
            spf_probs = apply_dow_draw_to_spf(spf_probs, _dow_draw_adj)
        except Exception:
            pass
    # ── P0: 赔率锚定融合（新增，在市场信号之前）──
    try:
        from prediction.post_adjust import odds_anchor_adjust
        spf_probs, _oa_sig = odds_anchor_adjust(spf_probs, ho, do_, ao, lg, match.get('match_date'), mc)
        if _oa_sig:
            _adj_signals.extend(['🔗赔率锚定'] + _oa_sig)
        plays['spf']['probs'] = spf_probs
        if spf_probs:
            plays['spf']['pred'] = max(spf_probs, key=spf_probs.get)
    except Exception as e:
        pass

    # ── P2: 周一看盘特殊规则（新增）──
    try:
        from prediction.post_adjust import monday_special_adjust
        spf_probs, _ms_sig = monday_special_adjust(spf_probs, ho, do_, ao, lg, match.get('match_date'), mc)
        if _ms_sig:
            _adj_signals.extend(_ms_sig)
            plays['spf']['probs'] = spf_probs
            if spf_probs:
                plays['spf']['pred'] = max(spf_probs, key=spf_probs.get)
    except Exception:
        pass

    # ── P3: 中间赔率区间优化（新增）──
    try:
        from prediction.post_adjust import mid_odds_band_adjust
        spf_probs, _mb_sig = mid_odds_band_adjust(spf_probs, ho, do_, ao, lg)
        if _mb_sig:
            _adj_signals.extend(_mb_sig)
            plays['spf']['probs'] = spf_probs
            if spf_probs:
                plays['spf']['pred'] = max(spf_probs, key=spf_probs.get)
    except Exception:
        pass

    # ── P0-1: 中赔1.8-2.3区间专项强化（新增）──
    try:
        from prediction.post_adjust import mid_tier_odds_sharp_adjust
        spf_probs, _mts_sig = mid_tier_odds_sharp_adjust(spf_probs, ho, do_, ao, lg, match.get('match_date'), mc)
        if _mts_sig:
            _adj_signals.extend(_mts_sig)
            plays['spf']['probs'] = spf_probs
            if spf_probs:
                plays['spf']['pred'] = max(spf_probs, key=spf_probs.get)
    except Exception:
        pass

    # P0新增: transition_matrix_adjust（胜→平转化率压降）
    try:
        from prediction.post_adjust import transition_matrix_adjust
        spf_probs, sigs_tm = transition_matrix_adjust(spf_probs, ho, do_, ao, lg)
        if sigs_tm:
            _adj_signals.extend(sigs_tm)
            plays['spf']['probs'] = spf_probs
            if spf_probs:
                plays['spf']['pred'] = max(spf_probs, key=spf_probs.get)
    except Exception:
        pass

    # ── P1: 英冠全周冷门触发器（新增，不分周几）──
    try:
        from prediction.post_adjust import championship_daily_adjust
        spf_probs, _cd_sig = championship_daily_adjust(spf_probs, ho, do_, ao, lg)
        if _cd_sig:
            _adj_signals.extend(_cd_sig)
            plays['spf']['probs'] = spf_probs
            if spf_probs:
                plays['spf']['pred'] = max(spf_probs, key=spf_probs.get)
    except Exception:
        pass

    # ── P2: 高赔率方差场次检测（新增）──
    try:
        from prediction.post_adjust import odds_conflict_check
        is_hr, _hr_sig, _hr_pen = odds_conflict_check(spf_probs, ho, do_, ao, lg)
        if is_hr and _hr_sig:
            _adj_signals.extend(_hr_sig)
    except Exception:
        pass

    # ── 市场信号后处理（必发+支持率调整SPF）——保留P0-P5信号──
    pop = None
    _mkt_signals = []
    try:
        from prediction.popularity import fetch_popularity
        from datetime import date as _date
        md = match.get('match_date') or _date.today().isoformat()
        pop = fetch_popularity(md, match_code=mc, home_team=h, away_team=a)
        from prediction.post_adjust import post_adjust_spf
        spf_probs, _mkt_signals = post_adjust_spf(spf_probs, ho, do_, ao, pop, lg)
        plays['spf']['probs'] = spf_probs
        # 重新确定SPF预测方向（取最高概率）
        if spf_probs:
            plays['spf']['pred'] = max(spf_probs, key=spf_probs.get)
    except Exception:
        pass
    # ── P0/P1: 周五效应 + 英冠专项 + 赔率区间降权 ──
    _adj_signals = []
    # 赔率轨迹信号（初盘→终盘 升水/降水/变盘四分类+CLV）
    if _traj_signals:
        _adj_signals.extend(['🔀赔率轨迹'] + _traj_signals)
    # P5: 联赛×周几λ校正信号（在line 718-723已计算）
    if _dl_signals:
        _adj_signals.extend(['📅λ校正'] + _dl_signals)
    try:
        from prediction.post_adjust import (
            friday_effect_adjust, championship_friday_adjust,
            odds_band_verify, confidence_friday_penalty,
            saturday_asian_adjust, small_league_penalty,
            draw_boost_adjust, trap_detection_adjust, mid_tier_trap_adjust, betfair_live_signal,
            weekday_effect_adjust, league_specific_adjust,
            thursday_effect_adjust
        )
        # P0: 联赛平局先验贝叶斯修正
        from prediction.calibrated_poisson import apply_draw_prior, market_equilibrium_deviation, longshot_bias_adjustment
        spf_probs = apply_draw_prior(spf_probs, lg)
        _adj_signals.append(f'P0平局先验({lg})')
        # P0: 周五效应（通用）
        spf_probs, _fsig = friday_effect_adjust(
            spf_probs, lg, match.get('match_date'), mc)
        _adj_signals.extend(_fsig)
        # P0: 英冠专项
        spf_probs, _csig = championship_friday_adjust(
            spf_probs, ho, lg, match.get('match_date'), mc)
        _adj_signals.extend(_csig)
        # P1: 赔率区间验证
        spf_probs, _bsig = odds_band_verify(
            spf_probs, ho, do_, ao, lg, match.get('match_date'), mc)
        _adj_signals.extend(_bsig)
        # P0: 周六亚洲联赛主场修正
        spf_probs, _asig = saturday_asian_adjust(
            spf_probs, lg, match.get('match_date'), mc)
        _adj_signals.extend(_asig)
        # P0: 平局专项检测（周四放在最后，因为平局boost影响所有方向）
        spf_probs, _dbsig = draw_boost_adjust(
            spf_probs, ho, do_, ao, lg, match.get('match_date'), mc)
        _adj_signals.extend(_dbsig)
        # P3: 冷门诱盘检测
        spf_probs, _trsig = trap_detection_adjust(
            spf_probs, ho, do_, ao, lg, match.get('match_date'), mc, h, a)
        _adj_signals.extend(_trsig)
        # P1: 中游队诱平陷阱检测（庄家诱导主胜实为平局）
        spf_probs, _mtsig = mid_tier_trap_adjust(
            spf_probs, ho, do_, ao, lg, match.get('match_date'), mc)
        _adj_signals.extend(_mtsig)
        # P2: 通用周几效应
        spf_probs, _wdsig = weekday_effect_adjust(
            spf_probs, match.get('match_date'), mc, lg)
        _adj_signals.extend(_wdsig)
        # P0: 周四效应
        spf_probs, _thsig = thursday_effect_adjust(
            spf_probs, lg, match.get('match_date'), mc)
        _adj_signals.extend(_thsig)
        # P3: 联赛专项规则
        spf_probs, _lssig = league_specific_adjust(
            spf_probs, ho, do_, ao, lg)
        _adj_signals.extend(_lssig)
        # P3: 博弈论-市场均衡偏离检测
        spf_probs, _meq_sig = market_equilibrium_deviation(spf_probs, ho, do_, ao)
        _adj_signals.extend(_meq_sig)
        # P3: 博弈论-冷门偏见修正
        if pop and pop.get('support'):
            spf_probs, _ls_sig = longshot_bias_adjustment(
                spf_probs, pop.get('support'), ho, do_, ao)
            _adj_signals.extend(_ls_sig)
        # P4: 临场盘信号（Betfair数据）
        if pop and pop.get('betfair'):
            spf_probs, _bfsig = betfair_live_signal(pop.get('betfair'), spf_probs)
            _adj_signals.extend(_bfsig)
        # 更新概率和预测方向
        plays['spf']['probs'] = spf_probs
        if spf_probs:
            plays['spf']['pred'] = max(spf_probs, key=spf_probs.get)
        _adj_signals.extend(_mkt_signals)
    except Exception:
        pass
    # ── P2: 档位系统信号（开盘方向 + 分布类型）──
    global _TIER_MODULE, _TIER_CACHE
    try:
        if _TIER_MODULE is None:
            import importlib.util
            _tspec = importlib.util.spec_from_file_location(
                'team_tier_db',
                os.path.expanduser('~/.hermes/workspace/notes/prediction/core/team_tier_db.py'))
            if _tspec and _tspec.loader:
                _tmod = importlib.util.module_from_spec(_tspec)
                _tspec.loader.exec_module(_tmod)
                _TIER_MODULE = _tmod
    except (ImportError, FileNotFoundError, OSError):
        pass

    if _TIER_MODULE:
        try:
            if _TIER_CACHE is None:
                _TIER_CACHE, _elos, _boundaries = _TIER_MODULE.load_team_elos()
            _tgap = _TIER_MODULE.get_tier_gap(h, a, _TIER_CACHE)
            _open = _TIER_MODULE.classify_opening(ho, _tgap)
            _dist = _TIER_MODULE.get_distribution_type(_tgap, 0)
            _adj_signals.append(f'📊档位差:{_tgap:+d} 开盘:{_open} 分布:{_dist}')
        except Exception:
            pass
    # ── P2: 变盘信号（赔率轨迹分析）──
    global _ODDS_TRAJ_MODULE
    try:
        if _ODDS_TRAJ_MODULE is None:
            import importlib.util
            _otspec = importlib.util.spec_from_file_location(
                'odds_trajectory',
                os.path.expanduser('~/.hermes/workspace/notes/prediction/core/odds_trajectory.py'))
            if _otspec and _otspec.loader:
                _otmod = importlib.util.module_from_spec(_otspec)
                _otspec.loader.exec_module(_otmod)
                _ODDS_TRAJ_MODULE = _otmod
    except (ImportError, FileNotFoundError, OSError):
        pass

    if _ODDS_TRAJ_MODULE:
        try:
            _var_sig = _ODDS_TRAJ_MODULE.get_variable_signal(mc, md)
            _adj_signals.append(f'📈变盘:{_var_sig}')
        except Exception:
            pass

    # ── P0新增: 威廉骨架特征 ───────────────────────────────
    try:
        from prediction.william_odds_base import compute_william_base, get_william_base_from_elo

        # 尝试加载ELO数据
        elo_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'prediction', 'team_elo.json')
        elo_map = {}
        if os.path.exists(elo_file):
            import json as _json
            with open(elo_file) as _f:
                elo_map = _json.load(_f)

        _wh = float(row.get('william_home_open', 0) or 0)
        _wd = float(row.get('william_draw_open', 0) or 0)
        _wa = float(row.get('william_away_open', 0) or 0)

        if _wh > 0 and _wd > 0 and _wa > 0:
            wb = get_william_base_from_elo(h, a, lg, elo_map)
            if wb:
                dev_h = _wh / wb['h'] if wb['h'] > 0 else 1.0
                dev_d = _wd / wb['d'] if wb['d'] > 0 else 1.0
                dev_a = _wa / wb['a'] if wb['a'] > 0 else 1.0
                if dev_h > 1.05 or dev_h < 0.95:
                    baodian_signals.append(f'威廉骨架偏差dev_h={dev_h:.2f}')
                if dev_d > 1.05:
                    baodian_signals.append(f'威廉骨架平赔偏高dev_d={dev_d:.2f}')
                result['william_base'] = wb
                result['william_deviation'] = {'h': round(dev_h,3), 'd': round(dev_d,3), 'a': round(dev_a,3)}
    except Exception as _e:
        pass  # 静默失败，不阻塞主流程

    # ── P2: 三家对比信号（体彩/威廉/立博）──
    global _BOOKMAKER_COMPARE_MODULE
    try:
        if _BOOKMAKER_COMPARE_MODULE is None:
            import importlib.util
            _bcspec = importlib.util.spec_from_file_location(
                'bookmaker_compare',
                os.path.expanduser('~/.hermes/workspace/notes/prediction/core/bookmaker_compare.py'))
            if _bcspec and _bcspec.loader:
                _bcmod = importlib.util.module_from_spec(_bcspec)
                _bcspec.loader.exec_module(_bcmod)
                _BOOKMAKER_COMPARE_MODULE = _bcmod
    except (ImportError, FileNotFoundError, OSError):
        pass

    if _BOOKMAKER_COMPARE_MODULE:
        try:
            _cons_sig = _BOOKMAKER_COMPARE_MODULE.get_consensus_signal(mc, md)
            _adj_signals.append(f'🏦三家:{_cons_sig}')
        except Exception:
            pass
    # ── 赔率-基本面融合 ──
    fusion_signals = []
    fusion_penalty = 0
    try:
        from prediction.team_loader import load_both
        from prediction.fusion_layer import fuse_predictions
        hd, ad = load_both(h, a, lg)
        fusion = fuse_predictions(match, ho, do_, ao, hd, ad)
        fusion_signals = fusion.get('signals', [])
        fusion_penalty = fusion.get('confidence_penalty', 0)
        if fusion.get('fundamental'):
            fe = fusion['fundamental']
            # 严重偏离时用融合后概率覆盖SPF
            oi_home = 1/ho/(1/ho+1/do_+1/ao)
            if abs(oi_home - fe['spf_probs'].get('胜', oi_home)) > 0.15:
                spf_probs.update(fusion['spf_probs'])
                plays['spf']['probs'] = spf_probs
                plays['spf']['pred'] = max(spf_probs, key=spf_probs.get)
    except Exception:
        pass
    # ── 动态置信度 ──
    odds_spread = max(ho, ao) / min(ho, ao)
    lambda_gap = abs(cal_h - cal_a)
    draw_odds_val = float(do_)
    base = 50
    if odds_spread > 2.0: base += 15
    if lambda_gap > 0.8: base += 10
    if draw_odds_val > 3.5: base += 5
    if draw_odds_val < 2.8: base -= 10
    # 历史准确率调整
    try:
        _cf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prediction')
        sys.path.insert(0, _cf_path)
        from historical_lambda import get_conn as _gconn
        _cconn = _gconn()
        _ccur = _cconn.cursor()
        _ccur.execute("""
            SELECT AVG(CASE WHEN actual_result = predicted THEN 1.0 ELSE 0.0 END)::float
            FROM prediction_match_results
            WHERE league=%s AND actual_result IS NOT NULL
            AND match_date >= (CURRENT_DATE - INTERVAL '30 days')
        """, (lg,))
        _hrow = _ccur.fetchone()
        _cconn.close()
        hist_acc = float(_hrow[0]) if _hrow and _hrow[0] else 0.5
        base = base * (0.7 + 0.3 * hist_acc / 0.5)
    except Exception:
        pass
    cf = max(20, min(95, int(base)))
    conf = cf / 100.0
    # ── 赔率轨迹置信度调整（初盘→终盘 升水/降水+变盘四分类+CLV）──
    if _traj_adj != 0.0:
        conf = max(0.05, min(0.99, conf + _traj_adj))
    v3 = None
    try:
        from ml_v3_helpers import predict_v3
        v3 = predict_v3(ho, do_, ao)
        if v3 and v3[1] < 0.55:
            conf = max(0.1, conf - 0.05)
    except Exception:
        pass
    elo = match.get('elo')
    if elo and elo.get('prediction') == ens_pred:
        conf = min(0.99, conf + 0.08)
    if ho <= 1.35 or ao <= 1.35:
        conf = min(0.99, conf + 0.10)
    elif abs(ho - ao) < 0.5 and do_ - min(ho, ao) < 0.5:
        conf = max(0.05, conf - 0.15)

    # ── 改进1: 友谊赛特殊规则 ──
    _friendly_keywords = ['友谊', '热身', 'Friendly', 'International Friendly', 'International']
    is_friendly = any(kw in lg for kw in _friendly_keywords)
    friendly_flags = []
    if is_friendly:
        conf = min(0.50, conf * 0.8)  # 置信度降低20%，上限50%
        friendly_flags.append('友谊赛置信上限50%(×0.8)')
        match['_friendly'] = True

    # ── P2: 高赔率方差惩罚（新增）──
    if _hr_pen > 0:
        conf = max(0.05, conf - _hr_pen)
        _adj_signals.append(f'🔴高风险惩罚(-{_hr_pen*100:.0f}%)')

    # ── P2-紧急回退：英冠/英超/法乙命中率崩盘降权（2026-04-07）──
    _emergency_leagues = {'英冠': 0.25, '英超': 0.25, '法乙': 0.167, '荷甲': 0.273, '韩职': 0.20}
    if lg in _emergency_leagues:
        _em_rate = _emergency_leagues[lg]
        _em_cap = min(0.55, conf)  # 强制封顶55%
        _em_pen = max(0.10, (_em_rate - 0.50) * 0.5)  # 基于偏差惩罚
        conf = min(_em_cap, conf - abs(_em_pen))
        _adj_signals.append(f'🔴{lg}紧急回退(命中率{_em_rate:.0%}, 置信封顶55%)')

    # ── 融合惩罚 ──
    if fusion_penalty > 0:
        conf = max(0.05, conf - fusion_penalty)
        friendly_flags.append(f'赔率基本面偏离→扣{fusion_penalty:.0%}')

    # ── 赔率异常检测 ──
    odds_anomaly = []
    fav_odds = min(ho, ao)
    underdog_odds = max(ho, ao)
    # 解析实际盘口（复用给后续盘口深度分析）
    actual_hc = None
    for _k in ('spread_point','asian_handicap','asian_spread_line','handicap_numeric','handicap'):
        _v = match.get(_k)
        if _v is not None:
            try: actual_hc = float(str(_v).replace('+','')); break
            except Exception: break
    # 弱方赔率>4.0但盘口≤1球 = 赔率过度反映名气差
    if underdog_odds > 4.0 and actual_hc is not None:
        if abs(actual_hc) <= 1.0:
            odds_anomaly.append(f'赔率失真嫌疑(弱方{underdog_odds:.1f}但盘口仅{actual_hc})')
            conf = max(0.05, conf - 0.10)
    # 极端赔率悬殊(>5:1)且无亚盘数据
    if underdog_odds / fav_odds > 5.0 and not sp_pt:
        odds_anomaly.append(f'极端赔率悬殊{underdog_odds/fav_odds:.1f}:1无亚盘确认')
        conf = max(0.05, conf - 0.08)

    odds_band = 'balanced'
    if ho <= 1.35: odds_band = 'home_strong'
    elif ao <= 1.35: odds_band = 'away_strong'
    elif abs(ho - ao) < 0.5: odds_band = 'balanced'

    spf = plays['spf']
    hc_play = plays['handicap']
    tg = plays['total_goals']
    bf = plays['scoreline']
    bqc = plays['half_full']
    # ── 价值投注 ──
    try:
        from prediction.value_bet import compute_value_bet
        value_bet = compute_value_bet(spf['probs'], ho, do_, ao, lg, pop, is_friendly=is_friendly)
    except Exception:
        value_bet = None

    # ── 让球独立建模 ──
    rq_pred_final = hc_play['label']  # 默认从泊松派生
    try:
        hc_val = float(match.get('handicap_numeric', 0) or match.get('asian_handicap', 0) or hc_str or 0)
    except (ValueError, TypeError):
        hc_val = 0.0
    if hc_val != 0:
        p_home_cover = sum(p for (i, j), p in matrix.items() if i + hc_val > j)
        p_draw_cover = sum(p for (i, j), p in matrix.items() if i + hc_val == j)
        p_away_cover = sum(p for (i, j), p in matrix.items() if i + hc_val < j)

        hc_odds_home = match.get('hc_win') or match.get('handicap_home_odds')
        hc_odds_away = match.get('hc_lose') or match.get('handicap_away_odds')

        if hc_odds_home and hc_odds_away:
            try:
                hc_margin = 1 / float(hc_odds_home) + 1 / float(hc_odds_away)
                ev_home = p_home_cover * float(hc_odds_home) - 1
                ev_away = p_away_cover * float(hc_odds_away) - 1

                if ev_away > ev_home and ev_away > 0.05:
                    rq_pred_final = f'让负(主让{int(abs(hc_val))}球,{int(p_away_cover * 100)}%)'
                elif ev_home > 0.05:
                    rq_pred_final = f'让胜(主让{int(abs(hc_val))}球,{int(p_home_cover * 100)}%)'
            except Exception:
                pass

            # ── P0-新增(2026-05-02): HHAD冷门区间硬限制 + EV阈值差异化 ──
            # ── P0修复(2026-05-06): 优先使用 FilterRegistry，fallback 到 post_adjust ──
            _rq_hhad_gate_pass = True
            try:
                if _FILTER_REGISTRY_AVAILABLE:
                    # 优先使用 FilterRegistry 统一过滤
                    ev_filter = get_filter('ev_filter')
                    cold_filter = get_filter('cold_filter')
                    _fr_match = {
                        'league': lg,
                        'home_odds': ho, 'draw_odds': do_, 'away_odds': ao,
                        'spf_probs': {'胜': sp_prob, '平': dp, '负': ap},
                        'ev_home': ev_home, 'ev_away': ev_away,
                    }
                    _fr_result = apply_filters(_fr_match,
                        filters=['ev_filter', 'cold_filter'], policy='all',
                        confidence=sp_confidence)
                    if not _fr_result.passed:
                        _rq_hhad_gate_pass = False
                        _adj_signals.append(f"HHAD FilterRegistry过滤: {_fr_result.reason}")
                else:
                    # Fallback 到 post_adjust
                    from prediction.post_adjust import hhad_cold_odds_gate, get_ev_threshold
                    _gate_result = hhad_cold_odds_gate(float(hc_odds_home), 'HHAD')
                    if not _gate_result.get('pass', True):
                        _adj_signals.append(f"HHAD过滤: {_gate_result.get('reason', 'cold_odds')}")
                        _rq_hhad_gate_pass = False
                    else:
                        _rq_ev_thresh = get_ev_threshold('HHAD')
                        if ev_home < _rq_ev_thresh and ev_away < _rq_ev_thresh:
                            _rq_hhad_gate_pass = False
                            _adj_signals.append(f"HHAD EV<{_rq_ev_thresh:.0%}过滤")
            except Exception:
                pass

            # 当HHAD Gate失败时，清空RQ预测
            if not _rq_hhad_gate_pass:
                rq_pred_final = '无投注'
                _rq_conf_adjusted = 0.0

    # ── P0-新增(2026-04-22): RQ预测熔断（让球盘方向与SPF矛盾时降权） ──
    _rq_conf_adjusted = conf * 0.9
    try:
        from prediction.post_adjust import rq_circuit_breaker
        _rq_raw_pred = rq_pred_final.split('(')[0] if '(' in rq_pred_final else rq_pred_final
        _rq_raw_pred = _rq_raw_pred.replace('让胜', '让胜').replace('让平', '让平').replace('让负', '让负')
        _rq_adj_pred, _rq_adj_conf, _rq_sig = rq_circuit_breaker(
            _rq_raw_pred, _rq_conf_adjusted, _spf_pred_raw, lg)
        if _rq_sig:
            _adj_signals.extend([f'RQ熔断: {s}' for s in _rq_sig])
            _rq_conf_adjusted = _rq_adj_conf
    except Exception:
        pass

    # ── Clamp confidence to [0, 1] ──
    conf = max(0.0, min(1.0, conf)) if conf is not None else 0.0

    # ── P0: 周五置信度惩罚（在 clamp 之后） ──
    try:
        from prediction.post_adjust import (
            confidence_friday_penalty, small_league_penalty,
            dow_league_adjust
        )
        conf, _conf_fri_sig = confidence_friday_penalty(
            conf, lg, match.get('match_date'), mc)
        _adj_signals.extend(_conf_fri_sig)
        # P0: 小联赛惩罚
        conf, _sl_sig = small_league_penalty(conf, lg, ho, do_, ao)
        _adj_signals.extend(_sl_sig)
        # P2: 星期×联赛交叉降权
        conf, _dow_sig = dow_league_adjust(conf, lg, match.get('match_date'), mc)
        _adj_signals.extend(_dow_sig)
        # P2: 威廉-体彩偏差熔断预警（>2.5触发）
        try:
            from prediction.post_adjust import william_deviation_alert
            mc_key = match.get('match_code') or match.get('match_id', '')
            if mc_key:
                import psycopg2 as _psycopg2
                from dotenv import load_dotenv as _ldotenv
                _ldotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.openclaw', '.env'))
                _wc = _psycopg2.connect(host=os.getenv('PGHOST','localhost'), dbname=os.getenv('PGDATABASE','myapp_db'), user=os.getenv('PGUSER','doodoo'), password=os.getenv('PGPASSWORD',''), connect_timeout=5)
                _wcur = _wc.cursor()
                _wcur.execute("SELECT william_home_open, william_draw_open, william_away_open FROM collected_match_data_2026 WHERE match_id = %s OR match_id LIKE %s LIMIT 1", (mc_key, f'%{mc_key}%'))
                _wr = _wcur.fetchone()
                _wcur.close(); _wc.close()
                if _wr and _wr[0] is not None:
                    _wdev, _wsig = william_deviation_alert(float(_wr[0]), float(_wr[1]), float(_wr[2]), ho, do_, ao)
                    if _wsig:
                        _adj_signals.append(_wsig)
                    # P1-2: 威廉希尔偏差降权（弱联赛+偏差双重降权）
                    if _wdev is not None and _wdev > 1.5:
                        conf = conf * 0.90  # 威廉希尔异常时降权
                        _adj_signals.append(f'威廉偏差{_wdev:.2f}→conf×0.90')
                        if lg in WEAK_LEAGUES:
                            conf = conf * 0.90  # 弱联赛额外降权
                            _adj_signals.append(f'弱联赛{lg}→conf×0.90')
        except Exception:
            pass
        # P1: Confidence校准（赔率悬殊时过度自信惩罚）
        try:
            from prediction.post_adjust import confidence_calibration_adjust
            conf, _cc_sig = confidence_calibration_adjust(conf, spf_probs, ho, do_, ao, league=lg)
            _adj_signals.extend(_cc_sig)
        except Exception:
            pass
        # P1-新增(2026-04-22): 周二杯赛专项
        try:
            from prediction.post_adjust import tuesday_cup_adjust
            _new_probs, _new_conf_factor, _tue_sigs = tuesday_cup_adjust(
                spf_probs, lg, match.get('match_date'), mc)
            if _new_conf_factor is not None:
                spf_probs = _new_probs
                conf *= _new_conf_factor
                _adj_signals.extend(_tue_sigs)
        except Exception:
            pass
        # P1-新增: 韩职赛季初期补丁
        try:
            from prediction.post_adjust import k_league_early_season_adjust
            spf_probs, _k_sig = k_league_early_season_adjust(
                spf_probs, lg, match.get('match_date'))
            _adj_signals.extend(_k_sig)
        except Exception:
            pass
        # P1-新增: 联赛专项降权（解放者杯/沙职/法乙/英冠）
        try:
            from prediction.post_adjust import league_confidence_penalty
            spf_probs, conf, _lp_sig = league_confidence_penalty(
                spf_probs, lg, conf)
            _adj_signals.extend(_lp_sig)
        except Exception:
            pass
        # P0-新增: 超低赔+高conf陷阱处理
        try:
            from prediction.post_adjust import ultra_low_odds_conf_cap
            conf, _ulc_sig = ultra_low_odds_conf_cap(
                spf_probs, ho, do_, ao, conf, lg)
            _adj_signals.extend(_ulc_sig)
        except Exception:
            pass
        # P2-新增(2026-05-12): 平赔预警信号 + 盘型换算值
        try:
            from prediction.post_adjust import draw_warning, handicap_conversion
            _dw_result = draw_warning(do_)
            if _dw_result.get('triggered'):
                _adj_signals.append(f"⚠️平赔预警:{_dw_result['reason']}")
            _hc_result = handicap_conversion(spf_probs)
            if _hc_result.get('ratio', 0) > 0:
                _adj_signals.append(f"📐盘型换算:{_hc_result['reason']}")
        except Exception:
            pass
    except Exception:
        pass

    # 直接从历史λ的泊松矩阵计算SPF（不受league_factor影响）
    _raw_m = _cp._build_matrix(cal_h, cal_a)
    _pw, _pd, _pa = _cp._spf_from_matrix(_raw_m)
    _ord = sorted([('胜', _pw), ('平', _pd), ('负', _pa)], key=lambda x: -x[1])
    _spf_pred_raw = _ord[0][0]

    # ── 联赛专项GBM模型（统一调度，覆盖所有已训练联赛） ──
    # ── 联赛专项GBM模型融合（泊松40% + 专项60%）──
    _league_model_info = None
    _fusion_w = 0.0  # 专项模型权重
    _poisson_w = 1.0  # 泊松权重
    try:
        from prediction.league_models.league_dispatcher import dispatch_predict
        _lm_result = dispatch_predict(
            lg, ho, do_, ao, h, a,
            poisson_pw=float(_pw), poisson_pd=float(_pd), poisson_pa=float(_pa),
            pool_poisson=True
        )
        if _lm_result:
            _lm_pred, _lm_probs, _lm_oos, _lm_conf = _lm_result
            # 只在专项模型OOS > 基准时使用融合
            _base_acc = 0.50  # 赔率基准约50%
            if _lm_oos and _lm_oos > _base_acc:
                _fusion_w = 0.6
                _poisson_w = 0.4
            elif _lm_oos:
                # OOS低于基准则低权重使用
                _fusion_w = 0.3
                _poisson_w = 0.7
            # 融合概率
            _fused = {
                'H': _poisson_w * _pw + _fusion_w * _lm_probs.get('H', 0),
                'D': _poisson_w * _pd + _fusion_w * _lm_probs.get('D', 0),
                'A': _poisson_w * _pa + _fusion_w * _lm_probs.get('A', 0),
            }
            _cn_map = {'H': '胜', 'D': '平', 'A': '负'}
            _spf_pred_raw = _cn_map[max(_fused, key=_fused.get)]
            # 融合后置信度
            _max_p = _fused[max(_fused, key=_fused.get)]
            _entropy = -sum(p * np.log(p + 1e-10) for p in _fused.values())
            _max_entropy = -np.log(1/3)
            conf = max(conf, round(_max_p * (_entropy / _max_entropy), 4))
            _spf_map_reverse = {'胜': 'H', '平': 'D', '负': 'A'}
            _league_model_info = {
                'oos': _lm_oos, 'conf': _lm_conf, 'weight': _fusion_w,
                'pred': _spf_map_reverse.get(_lm_pred),
                'probs': {k: round(v,4) for k,v in _lm_probs.items()},
                'fused': {k: round(v,4) for k,v in _fused.items()},
            }
    except Exception:
        pass

    # ── 周六庄家操盘模型融合（仅周六/周日生效）──
    if md and _league_model_info is not None:
        try:
            from datetime import datetime as _dt
            _match_dt = _dt.strptime(str(md)[:10], '%Y-%m-%d')
            _match_dow = _match_dt.weekday()  # 5=周六, 6=周日
            if _match_dow in (5, 6):
                import json as _json
                _sat_model_path = os.path.join(os.path.dirname(__file__), 'prediction', 'league_models', 'saturday_model.json')
                if os.path.exists(_sat_model_path):
                    with open(_sat_model_path) as _sf:
                        _sat_info = _json.load(_sf)
                    # C4 修复: pickle 安全 — 仅在模型来自可信来源时加载
                    # 来源验证: 文件存在 + 有 oos_score（非空）
                    if _sat_info.get('oos_score') is not None and _sat_info.get('model'):
                        import hashlib
                        _model_bytes = base64.b64decode(_sat_info['model'])
                        _expected_hash = _sat_info.get('sha256', '')
                        if _expected_hash:
                            _actual_hash = hashlib.sha256(_model_bytes).hexdigest()
                            if _actual_hash != _expected_hash:
                                raise ValueError('Saturday model SHA256 mismatch — possible tampering')
                        _sat_m = pickle.loads(_model_bytes)
                    # 构建周六特征
                    _sat_feats = _build_saturday_features(ho, do_, ao, _match_dow)
                    _sat_proba = _sat_m.predict_proba([_sat_feats])[0]
                    _sat_classes = list(_sat_m.classes_)  # [0,1,2]
                    _sat_p = {'H': 0, 'D': 0, 'A': 0}
                    for _i, _c in enumerate(_sat_classes):
                        _cn = {0: 'H', 1: 'D', 2: 'A'}.get(_c)
                        if _cn:
                            _sat_p[_cn] = _sat_proba[_i]
                    # 融合: 当前融合结果(70%) + 周六模型(30%)
                    _sat_w = 0.30 if _sat_info.get('oos_score', 0) > 0.50 else 0.15
                    _cur_w = 1.0 - _sat_w
                    _fused_sat = {
                        'H': _cur_w * _fused.get('H', _pw) + _sat_w * _sat_p['H'],
                        'D': _cur_w * _fused.get('D', _pd) + _sat_w * _sat_p['D'],
                        'A': _cur_w * _fused.get('A', _pa) + _sat_w * _sat_p['A'],
                    }
                    _cn_map = {'H': '胜', 'D': '平', 'A': '负'}
                    _spf_pred_raw = _cn_map[max(_fused_sat, key=_fused_sat.get)]
                    _fused = _fused_sat
                    _max_p = max(_fused_sat.values())
                    _entropy = -sum(p * np.log(p + 1e-10) for p in _fused_sat.values())
                    _max_entropy = -np.log(1/3)
                    conf = max(conf, round(_max_p * (_entropy / _max_entropy), 4))
                    _league_model_info['saturday_fusion'] = {
                        'weight': _sat_w, 'dow': _match_dow,
                        'sat_pred': max(_sat_p, key=_sat_p.get),
                        'sat_probs': {k: round(v, 4) for k, v in _sat_p.items()},
                    }
        except Exception:
            pass

    # ── 澳超特殊后处理（庄家低估主队+6%，高估客队-7%）──
    if lg in ('澳超', 'A-League'):
        try:
            from prediction.a_league_postprocess import adjust_a_league_probs
            import datetime as _dt
            _is_sat = False
            _md = match.get('match_date')
            if _md:
                try:
                    _dow = _dt.datetime.strptime(str(_md)[:10], '%Y-%m-%d').weekday()
                    _is_sat = (_dow == 5)
                except Exception:
                    pass
            # 周六已由saturday_asian_adjust处理，澳超后处理不再叠加weekend_factor
            _adj = adjust_a_league_probs(lg, ho, do_, ao, is_weekend=not _is_sat)
            if _adj:
                _a_probs = {_k: float(_adj[_k]) for _k in ['H', 'D', 'A']}
                _total = sum(_a_probs.values())
                for _k in _a_probs:
                    _a_probs[_k] /= _total
                _cn_map = {'H': '胜', 'D': '平', 'A': '负'}
                _adj_pred = _cn_map[max(_a_probs, key=_a_probs.get)]
                _spf_pred_raw = _adj_pred
                _max_p = max(_a_probs.values())
                _entropy = -sum(p * np.log(p + 1e-10) for p in _a_probs.values())
                _max_entropy = -np.log(1/3)
                conf = max(conf, round(_max_p * (_entropy / _max_entropy), 4))
                _league_model_info = {
                    'oos': 0.55, 'conf': _max_p, 'weight': 1.0,
                    'pred': max(_a_probs, key=_a_probs.get),
                    'probs': {k: round(v, 4) for k, v in _a_probs.items()},
                    'fused': {k: round(v, 4) for k, v in _a_probs.items()},
                    'adjustment': 'a_league_home_boost'
                }
        except Exception:
            pass

    # ── 量化增强 (趋势/高频/套利/多因子) ──
    try:
        from prediction.quant_enhancement import apply_quant_signals
        from prediction.quant_strategies_bridge import strategies_adjustment, clv_signal

        # 先用四大策略引擎对ensemble结果做二次校准
        _ens_odds = {'home': ho, 'draw': do_, 'away': ao}
        _strat = strategies_adjustment(
            match=match,
            model_probs=spf_probs,
            odds=_ens_odds,
            betfair=None,
            support=pop,
        )
        _strat_signals = _strat.get('signals', [])
        _clv = clv_signal(
            (float(match.get('home_win_odds_open', 0) or 0),
             float(match.get('draw_odds_open', 0) or 0),
             float(match.get('away_win_odds_open', 0) or 0)),
            (ho, do_, ao)
        )

        # spf_probs 是 dict → 转 list [home, draw, away]
        # 用策略引擎调整后的概率
        _adj = _strat.get('adjusted_probs', spf_probs)
        _spf_list = [_adj.get('胜', 0.33), _adj.get('平', 0.33), _adj.get('负', 0.33)]

        _qd_probs, _qd_result = apply_quant_signals(
            _spf_list, ho, do_, ao, conf, lg,
            support_data=pop,
            betfair_data=None,
            match_date=match.get('match_date'),
            init_odds=_get_william_init(match),
            final_odds=(ho, do_, ao),
            asian_handicap=match.get('asian_handicap'),
            asian_water=match.get('asian_water'),
            letball_handicap=match.get('handicap'),
            strategies_signals=_strat_signals,
            clv=_clv,
            strategies_result=_strat,
        )
        _qd_conf = _qd_result.get('quant_confidence', conf)
        # P2-1: 置信度档位重新校准（50-59上调5%，70+下调10%，弱联赛×0.85）
        try:
            from prediction.calibrated_poisson import confidence_tier_recalibrate
            _qd_conf = confidence_tier_recalibrate(_qd_conf, lg)
        except Exception: pass
        _qd_grade = _qd_result.get('quant_grade', 'D')
        _qd_signals = _qd_result.get('quant_signals', [])
        _qd_ev = _qd_result.get('ev_signal', {})
        # 用量化等级替代固定阈值
        _qd_recommend = _qd_grade in ('A', 'B')
        _qd_recommend_strong = _qd_grade == 'A'
        _quant_applied = True
        # P2-C: apply_neutral_platt_correction 接入预测链路
        # 从evolution_calibration.json读取最新Platt参数(scale=1.8317, offset=0.4773)
        _platt_applied = False
        try:
            import os as _os, json as _json
            _cal_path = _os.path.join(_os.path.dirname(__file__), 'data', 'evolution_calibration.json')
            if _os.path.exists(_cal_path):
                with open(_cal_path) as _f:
                    _cal = _json.load(_f)
                _platt_info = _cal.get('auto_review', {}).get('platt_info', {})
                _scale = float(_platt_info.get('scale', 1.0))
                _offset = float(_platt_info.get('offset', 0.0))
                if _scale != 1.0 or _offset != 0.0:
                    import math as _math
                    try:
                        _z = _scale * (_qd_conf - _offset)
                        _z = max(-500, min(500, _z))
                        _qd_conf_before = _qd_conf
                        _qd_conf = round(1 / (1 + _math.exp(-_z)), 4)
                        _qd_conf = max(0.01, min(0.99, _qd_conf))
                        _platt_applied = True
                        _qd_signals.append(f'📐Platt校准({_scale:.2f},{_offset:.3f}): {_qd_conf_before:.0%}→{_qd_conf:.0%}')
                    except Exception: pass
        except Exception: pass
    except Exception:
        _qd_conf, _qd_recommend, _qd_recommend_strong = conf, conf >= .65, conf >= .80
        _qd_signals, _qd_ev, _quant_applied = [], {}, False
        _qd_grade = 'unknown'
        _platt_applied = False

    result = {
        'match_code': mc, 'league': lg, 'home_team': h, 'away_team': a,
        'odds': f'{ho}/{do_}/{ao}', 'ml_confidence': _qd_conf, 'odds_band': odds_band,
        'recommend': _qd_recommend, 'recommend_strong': _qd_recommend_strong,
        'quant_grade': _qd_grade,
        'quant_signals': _qd_signals,
        'quant_ev': _qd_ev,
        'quant_applied': _quant_applied,
        'platt_applied': _platt_applied,
        # P2-B: 价值赌注评分信号（pred_prob vs 庄家隐含概率的EV差）
        'ev_gap_signal': _build_ev_gap_signal(value_bet),
        'balanced_hint': '建议让球玩法' if odds_band == 'balanced' else '',
        'value_bet': value_bet,
        'calibrated_lambda': {'home': round(cal_h,3), 'away': round(cal_a,3),
                               'raw_home': round(raw_h,3), 'raw_away': round(raw_a,3)},
        'home_advantage_factor': home_adj_factor,
        'play1_spf': {'name':'胜平负','prediction':_spf_pred_raw,'confidence':_qd_conf,'recommend':_qd_recommend,'quant_grade':_qd_grade},
        'play2_rqspf': {'name':'让球胜平负','handicap':hc_str,'prediction':rq_pred_final,'confidence':_rq_conf_adjusted},
        'play3_bf': {'name':'比分',
                      'predictions':[] if is_friendly else bf['top5'],
                      'top3_with_band': [] if is_friendly else [
                          bf['top'], bf.get('top2', ('', '0%')), bf.get('top3', ('', '0%')),
                          ('区间', bf.get('band', ('unknown','0%'))[1])
                      ] if bf.get('top') else [],
                      'predictions_raw':[] if is_friendly else bf.get('top5_raw',bf['top5']),
                      'expected_goals':f'{cal_h:.1f}-{cal_a:.1f}',
                      'confidence_cap': conf * 0.35,  # P1降权：命中率仅8.4%
                      'recommend': False if not is_friendly else False,  # P1:仅高置信SPF+高置信联赛推荐
                      'frozen': False,
                      'note':'友谊赛不推荐比分' if is_friendly else '比分预测仅供参考'},
        'play4_zjq': {'name':'总进球',
                       'predictions':[] if is_friendly else [(f'{t}球',f'{p*100:.1f}%') for t,p in tg['dist'][:5]],
                       'expected_total':f'{cal_h+cal_a:.1f}',
                       'recommend': False if not is_friendly else False,  # P1:仅高置信SPF+高置信联赛推荐
                       'note':'友谊赛不推荐总进球' if is_friendly else ''},
        'play5_bqc': {'name':'半全场','predictions':bqc['top3']},
    }

    # P1-3: 比分/总进球推荐逻辑 — 仅高置信SPF+高置信联赛时推荐
    _bf_recommend = (not is_friendly and _qd_conf >= 0.70 and lg in HIGH_CONF_LEAGUES_BF)
    if 'play3_bf' in result:
        result['play3_bf']['recommend'] = _bf_recommend
    if 'play4_zjq' in result:
        result['play4_zjq']['recommend'] = _bf_recommend

    # P1-2: 威廉希尔偏差字段注入result
    if '_wdev' in dir() and _wdev is not None:
        result['william_deviation'] = _wdev
        if _wdev > 2.0:
            result['william_deviation_warning'] = True

    if _league_model_info is not None:
        result['league_model'] = _league_model_info

    if _championship_cold_signal is not None:
        result['championship_cold_trap'] = {
            'detected': True,
            'cold_strength': _championship_cold_signal['cold_strength'],
            'recommendation': _championship_cold_signal['recommendation'],
            'implied_probs': _championship_cold_signal['implied_probs'],
        }

    if v3:
        result['v3'] = {'prediction': v3[0], 'max_proba': round(v3[1] * 100, 1),
                        'probas': v3[2], 'selective': v3[1] >= 0.55}
    if elo:
        result['elo'] = {'prediction': elo.get('prediction'),
                         'home_points': elo.get('home_points'),
                         'away_points': elo.get('away_points'),
                         'consensus': elo.get('prediction') == ens_pred}
    try:
        from prediction.play_selector import select_play, play_type_cn, prediction_cn
        ps = select_play(avg_home=ho, avg_draw=do_, avg_away=ao,
                         ou_line=match.get('over_line'), handicap=None, league=lg,
                         spf_prediction=_spf_pred_raw, spf_confidence=conf)
        result['smart_play'] = {'band': ps['band'], 'play_type': ps['play_type'],
                                'prediction': ps['prediction'], 'confidence': ps['confidence'],
                                'reason': ps['reason'],
                                'label': f"{play_type_cn(ps['play_type'])}→{prediction_cn(ps['prediction'])}"}
    except Exception:
        result['smart_play'] = None
    mkt_signals = []
    confidence = 'medium'
    hc_depth = 'normal'
    ou_signal = 'neutral'
    try:
        bk_h, bk_d, bk_a = implied_prob(ho, do_, ao)
        diff = abs(spf['probs'].get('胜',0)-bk_h) + abs(spf['probs'].get('平',0)-bk_d) + abs(spf['probs'].get('负',0)-bk_a)
        confidence = 'high' if diff < 0.10 else ('low' if diff >= 0.20 else 'medium')
        if confidence == 'low': mkt_signals.append(f'model_diverge:{diff:.2f}')
    except Exception: pass
    if actual_hc is not None:
        hc_diff = actual_hc - expected_handicap(ho, ao)
        if hc_diff > 0.25:
            hc_depth = 'deep'
            mkt_signals.append(f'deep_handicap:{hc_diff:+.2f}')
        elif hc_diff < -0.25:
            hc_depth = 'shallow'
            mkt_signals.append(f'shallow_handicap:{hc_diff:+.2f}')
    try:
        if ou_line is not None:
            ou_f = float(ou_line); tl = cal_h + cal_a
            if abs(ou_f - tl) > 0.25:
                ou_signal = 'over_expected' if ou_f < tl else 'under_expected'
                mkt_signals.append(f'ou_bias:{ou_f}vs{tl:.1f}({ou_signal})')
    except Exception: pass
    aw = match.get('asian_water') or match.get('spread_water')
    if aw is not None:
        try:
            aw_f = float(aw)
            mkt_signals.append(f'water:{water_level_classify(aw_f)}({aw_f:.2f})')
            if aw_f < 0.78:
                hi_ = match.get('handicap_initial') or match.get('asian_initial')
                hc_ = match.get('handicap_closing') or match.get('asian_closing')
                if hi_ is not None and hc_ is not None and hi_ == hc_:
                    mkt_signals.append('dead_water:极大概率高走')
        except Exception: pass

    # ── 二次clamp: 确保所有play的confidence在[0,1] ──
    for pk in ('play1_spf','play2_rqspf'):
        if pk in result and 'confidence' in result[pk]:
            result[pk]['confidence'] = max(0.0, min(1.0, result[pk]['confidence'] or 0))
    if result.get('smart_play') and 'confidence' in result['smart_play']:
        result['smart_play']['confidence'] = max(0.0, min(1.0, result['smart_play']['confidence'] or 0))

    # ── 宝典增强特征：返还率/平赔偏离/尾数/欧亚一致性/亚盘变化 ──
    baodian_signals = []
    try:
        rr = compute_return_rate(ho, do_, ao)
        baodian_signals.append(f'return_rate:{rr:.3f}')
        if rr < 0.89:
            baodian_signals.append('low_return:庄家操控风险大')
            conf = max(0.05, conf - 0.03)
        elif rr > 0.96:
            baodian_signals.append('high_return:市场平衡')

        dd = compute_draw_deviation(do_)
        baodian_signals.append(f'draw_deviation:{dd:+.2f}')
        # 宝典：平赔>3.40且主胜<1.95 → 平局信号
        if dd > 0.10 and ho < 1.95:
            spf_probs['平'] = min(0.40, spf_probs.get('平', 0.25) * 1.15)
            total = sum(spf_probs.values())
            for k in spf_probs: spf_probs[k] /= total
            plays['spf']['probs'] = spf_probs
            baodian_signals.append('high_draw+low_home→平局增强')

        tail = compute_odds_tail(ho)
        if tail == 1 and ho <= 1.50:
            baodian_signals.append('heavy_tail_favorite:优势方重尾→安全')
        elif tail == 0 and ho <= 1.50:
            baodian_signals.append('light_tail_favorite:优势方轻尾→警惕诱')

        ea_gap = compute_euro_asian_gap(ho, ao, actual_hc)
        if ea_gap is not None:
            baodian_signals.append(f'euro_asian_gap:{ea_gap:+.2f}')
            if ea_gap > 0.5:
                baodian_signals.append('亚盘深开:庄家高看强队但可能陷阱')
            elif ea_gap < -0.5:
                baodian_signals.append('亚盘浅开:便宜方警惕诱')

        # 亚盘变化模式
        asian_change = get_asian_change_signal(mc)
        if asian_change:
            baodian_signals.append(f'asian_pattern:{asian_change["pattern"]}')
            result['asian_change'] = asian_change  # 存原始dict供复盘闭环

        # ── 赔率轨迹信号（体彩+威廉 初盘→终盘 升水/降水/变盘四分类+CLV）──
        if _traj_signals:
            baodian_signals.extend(_traj_signals)

        # ── 亚盘深浅信号 (handicap_depth_signal) ──
        # 来自 quant_enhancement.detect_handicap_depth，完整检测死水盘/深盘/浅盘/缺口盘
        try:
            from prediction.quant_enhancement import detect_handicap_depth
            # 准备参数：盘口/水位/变化方向
            _ah = actual_hc  # 实际亚盘盘口
            _hw = match.get('home_water') or match.get('asian_home_water')
            _aw = match.get('away_water') or match.get('asian_away_water')
            _hw = float(_hw) if _hw is not None else None
            _aw = float(_aw) if _aw is not None else None
            # 盘口变化方向：降盘='down'(盘口变小)，升盘='up'(盘口变大)
            _cd = None
            if asian_change and asian_change.get('hc_change') is not None:
                _cd = 'down' if asian_change['hc_change'] < -0.01 else ('up' if asian_change['hc_change'] > 0.01 else 'steady')
            # ELO（若有）
            _home_elo = match.get('home_elo')
            _away_elo = match.get('away_elo')
            _home_elo = float(_home_elo) if _home_elo is not None else None
            _away_elo = float(_away_elo) if _away_elo is not None else None
            # 隐含盘口（从欧赔计算）
            _imp_hc = expected_handicap(ho, ao)
            hcp_signal = detect_handicap_depth(
                asian_handicap=_ah,
                implied_handicap=_imp_hc,
                water_home=_hw,
                water_away=_aw,
                change_direction=_cd,
                home_elo=_home_elo,
                away_elo=_away_elo,
            )
            if hcp_signal and hcp_signal.get('depth_type'):
                result['handicap_depth_signal'] = hcp_signal
                baodian_signals.append(f'depth:{hcp_signal["depth_type"]}:{hcp_signal["confidence"]:.2f}')
                baodian_signals.append(hcp_signal.get('signal', ''))
                # 陷阱信号调整置信度
                if hcp_signal.get('is_trap'):
                    conf = max(0.05, conf - 0.03)
        except Exception:
            pass

        # Gap-13: 必发交易偏度（体彩 vs 必发隐含概率差）
        bf_spread = compute_betfair_spread(ho, do_, ao, pop)
        if bf_spread:
            baodian_signals.append(bf_spread['signal'])
            result['bf_spread'] = bf_spread

        # Gap-10: 历史相似盘口（479k赔率库）
        hist_sim = compute_historical_similarity(lg, ho, do_, ao)
        if hist_sim:
            result['historical_similarity'] = hist_sim
            result['historical_goal_avg'] = hist_sim['historical_goal_avg']

        # Gap-11: 球队近期状态（fd_uk历史数据）
        team_form = compute_team_recent_form(h, a, lg)
        if team_form:
            result['team_recent_form'] = team_form

        # 亚盘方向置信度调整（独立于asian_change是否存在）
        if asian_change:
            if asian_change['direction'] == 1:
                conf = min(0.99, conf + 0.03)  # 升盘降水→正路→增强
            elif asian_change['direction'] == -1:
                conf = max(0.05, conf - 0.05)  # 降盘升水→诱→减弱

        # 三家公司比较信号：威廉希尔 vs 立博 vs 体彩（94体系骨架）
        try:
            import psycopg2 as _psycopg2
            from dotenv import load_dotenv as _ldotenv
            _ldotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.openclaw', '.env'))
            _tc = _psycopg2.connect(host=os.getenv('PGHOST','localhost'), dbname=os.getenv('PGDATABASE','myapp_db'), user=os.getenv('PGUSER','doodoo'), password=os.getenv('PGPASSWORD',''), connect_timeout=5)
            _tcur = _tc.cursor()
            _tcur.execute("""SELECT william_home_open, william_draw_open, william_away_open,
                                     ladbrokes_h, ladbrokes_d, ladbrokes_a
                              FROM collected_match_data_2026
                              WHERE match_id = %s OR match_id LIKE %s LIMIT 1""",
                          (mc, f'%{mc}%'))
            _tr = _tcur.fetchone()
            _tcur.close(); _tc.close()
            if _tr:
                _wh, _wd, _wa, _lh, _ld, _la = _tr
                _tc3 = compute_three_company_signal(ho, do_, ao,
                    wh=float(_wh) if _wh else None,
                    wd=float(_wd) if _wd else None,
                    wa=float(_wa) if _wa else None,
                    lh=float(_lh) if _lh else None,
                    ld=float(_ld) if _ld else None,
                    la=float(_la) if _la else None,
                    league=lg)  # 传入league用于Interwetten估算降权
                if _tc3['has_data']:
                    baodian_signals.append(f'3co:{_tc3["signal"]}')
                    result['three_company'] = _tc3
                    # 威廉立博压平时降低主胜置信（庄家控盘信号）
                    if _tc3['signal'] == '威廉立博压平':
                        spf_probs['胜'] = max(0.05, spf_probs.get('胜', 0.5) * 0.92)
                        total = sum(spf_probs.values())
                        for k in spf_probs: spf_probs[k] /= total
                        plays['spf']['probs'] = spf_probs
                        baodian_signals.append('3co压平→主胜降权')
                        conf = max(0.05, conf - 0.05)
        except Exception:
            pass
    except Exception:
        pass

    # ── P3: 置信度分级（2026-04-06）──
    try:
        if conf >= 0.70:
            result['confidence_tier'] = 'high'
        elif conf >= 0.50:
            result['confidence_tier'] = 'medium'
            result['confidence_note'] = '⚠️谨慎参考'
        else:
            result['confidence_tier'] = 'low'
            result['confidence_note'] = '🔴平局风险高，建议双选或观望'
    except Exception:
        pass

    # ── SuperFusionLayer: 11路因子超级融合（2026-04-30）──
    # 默认值（当 _skip_super_fusion=True 时保留）
    result['probs'] = {k: float(v) for k, v in spf['probs'].items()}
    result['confidence'] = float(confidence) if confidence and not isinstance(confidence, str) else 0.55
    if not _skip_super_fusion:
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prediction', 'strategies'))
            from super_fusion import super_fuse
            # 传入真实Betfair数据（供hft_arb/betfair因子使用），同时传spread供asian_hcap使用
            # bf_data 格式：{'home': float, 'away': float} (spread) + 可选 'betfair': {...} (原始)
            # 注意：compute_betfair_spread 返回的key是 spread_home/spread_draw/spread_away，不是 home/away/draw
            bf_data = None
            raw_bf = pop.get('betfair') if isinstance(pop, dict) else None
            if bf_spread and bf_spread.get('spread_home') is not None:
                bf_data = {
                    'home': float(bf_spread['spread_home']),  # 体彩-必发，主队方向差值
                    'away': float(bf_spread['spread_away']),    # 体彩-必发，客队方向差值
                }
                if raw_bf:
                    bf_data['betfair'] = raw_bf  # 原始Betfair（供compute_hft_arb_factor等使用）
                    bf_data['volume'] = {  # 成交量供compute_betfair_factor大单检测
                        'home': raw_bf.get('home_vol', 0),
                        'away': raw_bf.get('away_vol', 0),
                    }
            fusion = super_fuse(match, ho, do_, ao, raw_bf, bf_data)
            # 用融合结果更新置信度和概率
            _ci_95 = fusion.get('confidence_ci')   # (lo, hi) or (None, None)
            if fusion.get('final_probs'):
                # 以融合概率为主流参考，但保留校准矩阵的方向性
                for d in ['胜', '平', '负']:
                    if d in fusion['final_probs']:
                        result['probs'][d] = float(fusion['final_probs'][d])
            result['confidence'] = float(fusion.get('confidence', confidence)) if not isinstance(fusion.get('confidence'), str) else (float(confidence) if confidence and not isinstance(confidence, str) else 0.55)
            if _ci_95:
                result['ci_95'] = list(_ci_95)
            # ── P2-1: 存储各因子独立概率，供历史 Bootstrap CI 重算 ──
            _all_factor = fusion.get('all_factor_results', {})
            if _all_factor:
                # numpy类型清理 → JSON可序列化
                import numpy as np
                def _clean_factor(fdata):
                    if isinstance(fdata, dict):
                        return {k: _clean_factor(v) for k, v in fdata.items()}
                    if isinstance(fdata, (np.floating, np.integer)):
                        return float(fdata)
                    if isinstance(fdata, (list, tuple)):
                        return [_clean_factor(x) for x in fdata]
                    return fdata
                result['all_factor_results'] = _clean_factor(_all_factor)
            result['fusion_layer'] = {
                'factor_count': fusion.get('factor_count', 0),
                'active_factors': list(fusion.get('all_factor_results', {}).keys()),
                'dominant_direction': fusion.get('dominant_direction', ''),
                'convergence_signals': fusion.get('convergence_signals', []),
                'fusion_quality': round(fusion.get('fusion_quality', 0), 4),
                'contributions': {k: {
                    'weight': v['weight'],
                    'effective_weight': round(v['effective_weight'], 4),
                    'confidence': v['confidence']
                } for k, v in fusion.get('factor_contributions', {}).items()}
            }
            if fusion.get('dominant_direction'):
                result['baodian_signals'] = result.get('baodian_signals', [])
                result['baodian_signals'].append(f'SFL:{fusion["dominant_direction"]}({fusion.get("factor_count",0)}因子)')

            # ── P2: 写入predictions_ensemble（7因子权重+融合概率）──────────
            try:
                _fc = fusion.get('factor_contributions', {})
                _fp = fusion.get('final_probs', {})
                _fq = fusion.get('fusion_quality', 0)
                _conf = fusion.get('confidence', 0.55)

                def _pct(d, probs):
                    v = probs.get(d, 0)
                    t = sum(probs.values())
                    return round(v / t * 100, 2) if t > 0 else 0.0

                _ens_vals = {
                    'factor_ml_pct':       round((_fc.get('ml', {}) or {}).get('effective_weight', 0) * 100, 2),
                    'factor_dl_pct':       round((_fc.get('dl', {}) or {}).get('effective_weight', 0) * 100, 2),
                    'factor_kelly_pct':    round((_fc.get('kelly', {}) or {}).get('effective_weight', 0) * 100, 2),
                    'factor_trend_pct':    round((_fc.get('trend', {}) or {}).get('effective_weight', 0) * 100, 2),
                    'factor_betfair_pct':  round((_fc.get('betfair', {}) or {}).get('effective_weight', 0) * 100, 2),
                    'factor_dow_pct':      round((_fc.get('dow', {}) or {}).get('effective_weight', 0) * 100, 2),
                    'factor_league_pct':   round((_fc.get('league', {}) or {}).get('effective_weight', 0) * 100, 2),
                    'ensemble_home_pct':   _pct('胜', _fp),
                    'ensemble_draw_pct':   _pct('平', _fp),
                    'ensemble_away_pct':   _pct('负', _fp),
                    'ensemble_confidence': round(float(_conf), 2) if not isinstance(_conf, str) else 0.55,
                }
                result['_ensemble_vals'] = _ens_vals
            except Exception:
                pass  # ensemble写入失败不影响主流程

        except Exception as e:
            pass  # 融合失败不影响主线

    result['hc_depth'] = hc_depth
    result['ou_signal'] = ou_signal
    result['market_signals'] = mkt_signals
    if baodian_signals:
        result['baodian_signals'] = baodian_signals
    if _adj_signals:
        result['spf_adjust_signals'] = _adj_signals
    if friendly_flags:
        result['friendly_flags'] = friendly_flags
    if odds_anomaly:
        result['odds_anomaly'] = odds_anomaly
    if fusion_signals:
        result['fusion_signals'] = fusion_signals
    if form_adj_info:
        result['form_adjustments'] = form_adj_info
    if xg_adj_info:
        result['xg_adjustments'] = xg_adj_info
    if odds_shift:
        result['odds_shift'] = odds_shift
    news = match.get('news_context', [])
    if news:
        result['news_context'] = [
            {'title': n.get('title',''), 'signals': n.get('signals',[])}
            for n in news if any(k in ''.join(n.get('signals',[])) for k in ['伤','停','轮换','缺阵'])]
    hs, aw = match.get('home_style'), match.get('away_style')
    if hs or aw:
        parts = []
        if hs: parts.append(f"主:{hs['style']}|{hs['formation']}|攻{hs['attack']}/守{hs['defense']}")
        if aw: parts.append(f"客:{aw['style']}|{aw['formation']}|攻{aw['attack']}/守{aw['defense']}")
        result['style_context'] = ' | '.join(parts)

    # ── 方案B后叠加: 贝叶斯会诊层（football-bayes skill）──────────────
    # 注入点: return result 之前，P0-P4后处理链已完成
    try:
        from prediction.bayes_advisor import bayesian_advisor, apply_bayes_adjust
        _bayes_ho = float(match.get('home_odds') or 0)
        _bayes_do = float(match.get('draw_odds') or 0)
        _bayes_ao = float(match.get('away_odds') or 0)
        if _bayes_ho > 1.0 and _bayes_do > 1.0 and _bayes_ao > 1.0:
            _bayes_sp = spf_probs
            _bayes_pred = ens_pred
            _bayes_conf = conf
            _bayes_sig = result.get('baodian_signals', [])
            # CRON_MODE: 直接从bayes_prematch_intel构造，跳过LLM调用
            if os.environ.get('CRON_MODE'):
                from prediction.bayes_advisor import inject_bayes_advisor_from_intel
                _bayes_adv = inject_bayes_advisor_from_intel(
                    match, _bayes_sp, _bayes_pred, _bayes_conf, _bayes_sig)
            else:
                _bayes_adv = bayesian_advisor(match, _bayes_sp, _bayes_pred, _bayes_conf, _bayes_sig)
            _bayes_adj_sp = apply_bayes_adjust(_bayes_sp, _bayes_adv)
            _bt = sum(_bayes_adj_sp.values())
            if _bt > 0:
                plays['spf']['probs'] = {k: v/_bt for k, v in _bayes_adj_sp.items()}
                spf_probs = plays['spf']['probs']
            result['bayes_advisor'] = _bayes_adv
            match['_bayes_spf_probs'] = spf_probs
            match['_bayes_confidence'] = _bayes_adv.get('confidence', _bayes_conf)
            if _bayes_adv.get('warnings'):
                result.setdefault('baodian_signals', []).extend(
                    [f"贝叶斯:{w}" for w in _bayes_adv['warnings'][:3]]
                )
    except Exception:
        pass  # 贝叶斯会诊失败不影响主流程

    # ── P2: 生成 per-match Task 实例文件 ───────────────────────────────
    try:
        import json
        _mc = str(match.get('match_code', ''))
        _task_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.tasks')
        os.makedirs(_task_dir, exist_ok=True)
        _task_file = os.path.join(_task_dir, f'{_mc}.json')
        if _mc and result:
            _task = {
                'schema_version': '1.0',
                'match_code': _mc,
                'match_date': match.get('match_date'),
                'league': match.get('league'),
                'home_team': match.get('home_team'),
                'away_team': match.get('away_team'),
                'created_at': __import__('datetime').datetime.now().isoformat(),
                'prediction': {
                    'spf': result.get('spf'),
                    'rqspf': result.get('rqspf'),
                    'bf': result.get('bf'),
                    'zjq': result.get('zjq'),
                    'bqc': result.get('bqc'),
                    'confidence': float(result.get('confidence', 0)) if result.get('confidence') is not None else None,
                    'ci_95': [float(result['ci_95'][0]), float(result['ci_95'][1])] if result.get('ci_95') else None,
                    'probs': {k: round(float(v), 4) for k, v in result.get('probs', {}).items()},
                    'fusion_layer': result.get('fusion_layer'),
                    'fusion_signals': result.get('fusion_signals', []),
                    'dominant_direction': result.get('fusion_layer', {}).get('dominant_direction') if isinstance(result.get('fusion_layer'), dict) else None,
                },
                'model_type': 'v5play_fusion',
                'task_status': 'predicted',
            }
            with open(_task_file, 'w', encoding='utf-8') as _f:
                json.dump(_task, _f, ensure_ascii=False, indent=2)
    except Exception:
        pass  # Task文件生成失败不影响主流程

    return result


# ═══════════════════════════════════════════════════════════════
# 主入口: if __name__ == '__main__'
# 复刻 ml_predict_5.py 的完整逻辑，调用 predict_5play()
# model_type='v5play_fusion'（区别于旧版 ml_predict_5.py 的 poisson）
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import importlib.util, os, sys, warnings, numpy as np, json, traceback
    warnings.filterwarnings('ignore', message='X does not have valid feature names')

    from dotenv import load_dotenv
    load_dotenv(os.path.expanduser('~/.hermes/.env'))

    BASE = os.path.dirname(os.path.abspath(__file__))
    for _p in [os.path.join(BASE, 'data-collection'),
               os.path.join(BASE, 'prediction'),
               os.path.join(BASE, 'data-collection', 'quantitative')]:
        if _p not in sys.path:
            sys.path.insert(0, _p)

    import psycopg2
    from match_loader import q, _env, load_all_odds, build_match_dict, get_weekday_label, load_model_weights
    from collect_match_news import collect_match_news
    from team_style_loader import load_team_style
    from rules_engine import load_prediction_rules, apply_rules
    from betting_utils import calc_ev, calc_kelly
    from bet_tracker import BetTracker
    from quantitative.betting_utils import clamp_prob
    from quantitative.calibration_engine import (
        calibrate_prob, fusion_score, compute_clv, analyze_for_bet,
        classify_odds_band, classify_conf_band,
    )
    from data_collection.hhad_pause import is_hhad_paused

    _DL = None
    _DL_AVAILABLE = False
    try:
        from data_collection import dl_predictor as dl_mod
        _DL = dl_mod.DLPredictor()
        _DL_AVAILABLE = bool(_DL.models)
    except Exception:
        pass

    _bt = BetTracker()
    today = os.environ.get('PREDICT_DATE', __import__('datetime').date.today().isoformat())
    _is_auto_today = len(sys.argv) <= 1  # True only when today is auto-detected, not passed as arg
    if len(sys.argv) > 1:
        today = sys.argv[1]
    # P0-3: 时间门禁——预测今日须在12:00之后，防止数据未齐
    if _is_auto_today:
        _now = __import__('datetime').datetime.now()
        if _now.hour < 12:
            print(f'⏰ 时间门禁: 现在 {_now.hour}点，还不到中午12点，跳过今日预测'); sys.exit(0)
    dp = get_weekday_label(today)
    env = _env()

    odds = load_all_odds(today, env)
    model_weights = load_model_weights(BASE)

    _today_safe = today.replace("'", "''")
    _dp_safe = dp.replace("'", "''")
    # 同时用日期和周几前缀过滤，避免明日比赛混入
    rows_raw = q(
        f"SELECT match_code,home_team,away_team,"
        f"COALESCE(odds_home,0)::float,COALESCE(odds_draw,0)::float,"
        f"COALESCE(odds_away,0)::float,league "
        f"FROM predictions_from_doc WHERE match_date='{_today_safe}' "
        f"AND odds_home>0 "
        f"ORDER BY match_code", env)
    raw_lines = [l for l in rows_raw.split('\n') if l.strip()]
    if not raw_lines:
        print(f'今日无比赛({today})'); sys.exit(0)

    ms = []
    for line in raw_lines:
        c = line.split('|')
        if len(c) < 7: continue
        ms.append(build_match_dict(c[0].strip(), c, odds, model_weights, today))

    print(f'📅{today}({dp}) 5玩法预测(v5play_fusion)|{len(ms)}场')

    _conn = psycopg2.connect(host='localhost', user='myapp',
                             password=env['PGPASSWORD'], dbname='myapp_db',
                             connect_timeout=5)
    try:
        from sync_hhad_data import sync_handicap_to_spf
        _n = sync_handicap_to_spf(_conn, today)
        print(f'🔄 handicap同步: {_n}条')
    except Exception as e:
        print(f'⚠️ handicap同步失败: {e}')

    try:
        rules = load_prediction_rules()
        print(f'📋 规则模板已加载: {len(rules.get("tier_confidence", {}))}档位')
    except Exception:
        rules = None

    news_list = []
    if not os.environ.get('CRON_MODE'):
        try:
            news_list = collect_match_news(today, top_k=5)
            if news_list:
                print(f'📰 赛前资讯: {len(news_list)}条')
        except Exception as e:
            print(f'📰 资讯采集失败: {e}')

    for m in ms:
        ht, at, lg = m.get('home_team',''), m.get('away_team',''), m.get('league','')
        home_style = load_team_style(ht, lg)
        away_style = load_team_style(at, lg)
        if home_style:
            m['home_style'] = home_style
        if away_style:
            m['away_style'] = away_style

    def _parse_odds(odds_str):
        if not odds_str:
            return 0.0, 0.0, 0.0
        parts = odds_str.split('/')
        try:
            ho = float(parts[0]) if len(parts) > 0 else 0.0
            do = float(parts[1]) if len(parts) > 1 else 0.0
            ao = float(parts[2]) if len(parts) > 2 else 0.0
            return ho, do, ao
        except (ValueError, IndexError):
            return 0.0, 0.0, 0.0

    def _sanitize_val(v):
        if v is None:
            return None
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, (float, np.floating)):
            return float(v)
        if isinstance(v, dict):
            return {kk: _sanitize_val(vv) for kk, vv in v.items()}
        if isinstance(v, list):
            return [_sanitize_val(vv) for vv in v]
        return v

    def _extract_expert_signals(r):
        es = {}
        for _k in ('baodian_signals','spf_adjust_signals','market_signals',
                    'william_deviation','asian_hit','hc_depth','ou_signal',
                    'confidence_tier','quant_grade','quant_signals','quant_ev',
                    'asian_change','league_model','championship_cold_trap',
                    'bayes_advisor'):
            _v = r.get(_k)
            if _v is not None:
                es[_k] = _sanitize_val(_v)
        return es

    def db_insert(match_date, mc, ht, at, lg, odds_str,
                  spf, spfc, rqh, rqp, rqp_conf,
                  bf1, bf3, zj1, zj3, zje,
                  bq1, bq3, cf, recommend, recommend_strong, oul, ouo, ouu, oup, ouc,
                  vbet=None, model_type='v5play_fusion', time_decay=1.0,
                  spf_calibrated_prob=None, spf_calibrated_ev=None,
                  hhad_calibrated_prob=None, hhad_calibrated_ev=None,
                  expert_signals=None,
                  fusion_prediction=None, fusion_confidence=None,
                  fusion_ci_low=None, fusion_ci_high=None,
                  fusion_layer=None, fusion_quality=None,
                  fusion_probs=None, fusion_duration_ms=None,
                  factor_probs=None):
        with _conn.cursor() as cur:
            cur.execute("""
                INSERT INTO predictions_5play(match_date,match_code,home_team,away_team,
                    league,odds,spf_prediction,spf_confidence,rqspf_handicap,rqspf_prediction,
                    rqspf_confidence,
                    bf_top1,bf_top3,zjq_top1,zjq_top3,zjq_expected,bqc_top1,bqc_top3,
                    ml_confidence,recommend,recommend_strong,
                    ou_line,ou_over_odds,ou_under_odds,ou_prediction,ou_confidence,
                    value_bet,model_type,time_decay_factor,
                    spf_calibrated_prob,spf_calibrated_ev,
                    hhad_calibrated_prob,hhad_calibrated_ev,
                    expert_signals,
                    fusion_prediction,fusion_confidence,
                    fusion_ci_low,fusion_ci_high,
                    fusion_layer,fusion_quality,
                    fusion_probs,fusion_duration_ms,factor_probs)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (match_date, match_code) DO UPDATE SET
                    odds=EXCLUDED.odds, league=EXCLUDED.league,
                    spf_prediction=EXCLUDED.spf_prediction,
                    spf_confidence=EXCLUDED.spf_confidence, value_bet=EXCLUDED.value_bet,
                    bf_top1=EXCLUDED.bf_top1, bf_top3=EXCLUDED.bf_top3,
                    zjq_top1=EXCLUDED.zjq_top1, zjq_top3=EXCLUDED.zjq_top3, zjq_expected=EXCLUDED.zjq_expected,
                    bqc_top1=EXCLUDED.bqc_top1, bqc_top3=EXCLUDED.bqc_top3,
                    ml_confidence=EXCLUDED.ml_confidence, recommend=EXCLUDED.recommend,
                    recommend_strong=EXCLUDED.recommend_strong, model_type=EXCLUDED.model_type,
                    time_decay_factor=EXCLUDED.time_decay_factor,
                    rqspf_handicap=EXCLUDED.rqspf_handicap, rqspf_prediction=EXCLUDED.rqspf_prediction,
                    rqspf_confidence=EXCLUDED.rqspf_confidence,
                    spf_calibrated_prob=EXCLUDED.spf_calibrated_prob, spf_calibrated_ev=EXCLUDED.spf_calibrated_ev,
                    hhad_calibrated_prob=EXCLUDED.hhad_calibrated_prob, hhad_calibrated_ev=EXCLUDED.hhad_calibrated_ev,
                    expert_signals=EXCLUDED.expert_signals,
                    fusion_prediction=EXCLUDED.fusion_prediction,
                    fusion_confidence=EXCLUDED.fusion_confidence,
                    fusion_ci_low=EXCLUDED.fusion_ci_low, fusion_ci_high=EXCLUDED.fusion_ci_high,
                    fusion_layer=EXCLUDED.fusion_layer, fusion_quality=EXCLUDED.fusion_quality,
                    fusion_probs=EXCLUDED.fusion_probs, fusion_duration_ms=EXCLUDED.fusion_duration_ms,
                    factor_probs=EXCLUDED.factor_probs""",
                (match_date, mc, ht, at, lg, odds_str, spf,
                 float(spfc) if spfc else None,
                 rqh if rqh is not None else 0, rqp,
                 float(rqp_conf) if rqp_conf else None,
                 bf1, bf3, zj1, zj3, float(zje) if zje else None,
                 bq1, bq3, float(cf) if cf else None,
                 recommend, recommend_strong, oul, ouo, ouu, oup,
                 float(ouc) if ouc else None, vbet, model_type, time_decay,
                 spf_calibrated_prob, spf_calibrated_ev,
                 hhad_calibrated_prob, hhad_calibrated_ev,
                 json.dumps(_sanitize_val(expert_signals)) if expert_signals else None,
                 fusion_prediction, float(fusion_confidence) if fusion_confidence else None,
                 float(fusion_ci_low) if fusion_ci_low else None,
                 float(fusion_ci_high) if fusion_ci_high else None,
                 fusion_layer, round(float(fusion_quality), 4) if fusion_quality else None,
                 json.dumps(fusion_probs) if fusion_probs else None,
                 float(fusion_duration_ms) if fusion_duration_ms else None,
                 json.dumps(factor_probs) if factor_probs else None))
        _conn.commit()

    bk = odds['bk']

    _CLAUDE_ENABLED = os.environ.get('CLAUDE_ENABLED', '0') == '1'

    def _claude_single_signal(m, r):
        if not _CLAUDE_ENABLED:
            return {}
        import subprocess, re as re_mod
        ht = r.get('home_team', '?')
        at = r.get('away_team', '?')
        lg = r.get('league', '?')
        odds = r.get('odds', '?')
        sp1 = r.get('play1_spf', {}).get('prediction', '?')
        sp1c = r.get('play1_spf', {}).get('confidence', '?')
        rq1 = r.get('play2_rqspf', {}).get('prediction', '?')
        rq1h = r.get('play2_rqspf', {}).get('handicap', '?')
        bf1 = r.get('play3_bf', {}).get('predictions', [[('?', 0)]])[0][0][0]
        zjq1 = r.get('play4_zjq', {}).get('predictions', [[('?', 0)]])[0][0][0]
        news = m.get('news_context', [])
        news_str = '; '.join([f"{n.get('team','')}:{n.get('content','')[:30]}" for n in news[:3]]) if news else '无'
        home_style = m.get('home_style') or {}
        away_style = m.get('away_style') or {}
        style_str = f"主:{home_style.get('style','?')}进攻:{home_style.get('attack_mode','?')} | 客:{away_style.get('style','?')}进攻:{away_style.get('attack_mode','?')}"
        system = "你是一个足球比赛分析专家。严格只输出JSON：{\"market_trap\":0~2,\"hot_streak_bias\":\"string\",\"tactical_matchup\":\"string\",\"situational_factor\":\"string\",\"confidence_adjust\":-15~15,\"reasoning\":\"string\"}"
        prompt = f"{lg}|{ht}vs{at}\n赔率:{odds}\n胜平负:{sp1}({sp1c}) 让球:{rq1}({rq1h})\n比分:{bf1} 总进球:{zjq1}\n资讯:{news_str}\n风格:{style_str}"
        try:
            raw = subprocess.check_output(
                ['/home/doodoo/.npm-global/bin/claude', '--print', system + "\n\n" + prompt],
                timeout=45
            ).decode('utf-8', errors='replace').strip()
            mjson = re_mod.search(r'\{[^{}]*"market_trap"[^{}]*\}', raw, re_mod.DOTALL)
            if mjson:
                return json.loads(mjson.group())
        except Exception:
            pass
        return {}

    for m in ms:
        try:
            ho, do, ao = _parse_odds(m.get('odds', ''))
            if ho > 3.0:
                print(f"⏭ {m.get('match_code', '?')} cold区(ho={ho:.2f})→跳过")
                continue
        except (ValueError, IndexError, TypeError):
            pass

        m['news_context'] = news_list

        if not os.environ.get('CRON_MODE'):
            try:
                from prediction.data_enricher import inject_into_match
                m = inject_into_match(m)
            except Exception:
                pass
            try:
                from prediction.bayes_enrich import enrich_match
                m = enrich_match(m)
            except Exception:
                pass

        if os.environ.get('CRON_MODE'):
            try:
                from prediction.bayes_enrich import inject_bayes_from_db
                m = inject_bayes_from_db(m)
            except Exception:
                pass

        r = predict_5play(m)

        _hc_cur = _conn.cursor()
        _hc_cur.execute("SELECT handicap FROM sporttery_spf WHERE match_code=%s AND match_date=%s",
                        (r['match_code'], today))
        _hc_row = _hc_cur.fetchone()
        _hc_cur.close()
        _has_hc = False
        if _hc_row and _hc_row[0] and str(_hc_row[0]).strip():
            _has_hc = True
            _new_hc = str(_hc_row[0])
            r['play2_rqspf']['handicap'] = _new_hc
            try:
                from prediction.calibrated_poisson import derive_all_plays_nb
                _lo_cur = _conn.cursor()
                _lo_cur.execute("SELECT let_num,let_win,let_draw,let_lose FROM sporttery_letball WHERE match_code=%s AND match_date=%s",
                                (r['match_code'], today))
                _lo_row = _lo_cur.fetchone()
                _lo_cur.close()
                if _lo_row and float(_lo_row[1]) > 0 and float(_lo_row[3]) > 0:
                    _rq_result = derive_all_plays_nb(
                        1.0/float(_lo_row[1]), 1.0/float(_lo_row[3]),
                        handicap=float(_new_hc), apply_score_prior=False)
                    r['play2_rqspf'].update({
                        'prediction': _rq_result['pred'],
                        'confidence': _rq_result['conf'],
                        'label': _rq_result['label'],
                    })
            except Exception:
                pass

        mc = r['match_code']
        lg = r['league']
        ho, do, ao = _parse_odds(r['odds'])
        odds_str = r['odds']
        spf = r['play1_spf']['prediction']
        rq = r['play2_rqspf']['prediction']
        _rq_conf = r['play2_rqspf'].get('confidence')
        if _rq_conf is None:
            _rq_conf = r['play1_spf'].get('confidence', 0.45)
        p3, p4, p5 = r['play3_bf'], r['play4_zjq'], r['play5_bqc']
        bf = p3['predictions'][0][0]
        zjq = p4['predictions'][0][0]
        bqc = p5['predictions'][0][0]

        # ══ 强制SPF→BF/TG方向一致性修正 ══
        # 当BF预测与SPF方向矛盾时，用TG期望值反推合理比分
        _th, _ta = int(bf.split('-')[0]), int(bf.split('-')[1])
        _bf_dir_ok = (
            (spf == '胜' and _th > _ta) or
            (spf == '平' and _th == _ta) or
            (spf == '负' and _th < _ta)
        )
        if not _bf_dir_ok:
            # 用期望总进球数和SPF方向生成一致的比分数
            _zjq_val = float(zjq.replace('球','')) if zjq and '球' in zjq else 1.5
            _tg_int = max(0, min(6, round(_zjq_val)))
            if spf == '胜':
                _new_bf = f'{_tg_int}-0' if _tg_int <= 3 else f'{_tg_int}-1'
            elif spf == '负':
                _new_bf = f'0-{_tg_int}' if _tg_int <= 3 else f'1-{_tg_int}'
            else:  # 平
                _half = _tg_int // 2
                _new_bf = f'{_half}-{_half}'
            bf = _new_bf
            # TG也需重新派生
            zjq = f'{_tg_int}球'
        # TG方向也与SPF一致：主胜→TG偏高，平→TG居中，客胜→TG偏低
        _tg_int_bf = int(bf.split('-')[0]) + int(bf.split('-')[1])
        _zjq_int = max(0, min(6, _tg_int_bf))
        zjq = f'{_zjq_int}球'
        # BQC同理由BF导出
        _h = int(bf.split('-')[0])
        _a = int(bf.split('-')[1])
        if _h > _a:  # 主胜
            bqc = '胜-胜'
        elif _h < _a:  # 主负
            bqc = '负-负'
        else:  # 平
            bqc = '平-平'

        cf = max(0, min(100, round((r.get('ml_confidence') or 0) * 100)))

        _sig = _claude_single_signal(m, r)
        if _sig:
            m['extra_signals'] = _sig
            _ca = _sig.get('confidence_adjust', 0)
            if _ca:
                cf = max(0, min(100, cf + _ca))

        bn = ''
        b = bk.get(lg)
        if b and float(b[4]) > 10:
            bhh, bd2, ba3 = float(b[1]), float(b[2]), float(b[3])
            if bhh > .55: bn = '📊主胜偏高'
            elif ba3 > .45: bn = '📊客胜偏高'
            elif bd2 > .3: bn = '📊平局偏高'
        cf += 3 if '偏高' in bn else 0

        _ht_safe = r['home_team'].replace("'", "''")
        _at_safe = r['away_team'].replace("'", "''")
        mot = q(
            f"SELECT home_motivation,away_motivation,home_rank,away_rank "
            f"FROM match_info WHERE match_date='{_today_safe}' "
            f"AND home_team='{_ht_safe}' "
            f"AND away_team='{_at_safe}'", env)
        if mot:
            hm, am, hr, ar = (int(x) for x in mot.split('|'))
            md = hm - am
            if 2.0 <= ho <= 2.5 and abs(md) > 3:
                if md > 0 and spf != '胜': cf += 2
                elif md < 0 and spf != '负': cf += 2

        if rules:
            adj = apply_rules(r, rules)
            for w in adj.get('warnings', []):
                if '预期' in w and 'ML给出' in w: cf -= 3

        bh2, ba2 = int(bf.split('-')[0]), int(bf.split('-')[1])
        if spf == '胜' and bh2 <= ba2: cf -= 5
        elif spf == '负' and bh2 >= ba2: cf -= 5

        cf = max(0, min(100, cf))
        tg = '🔥' if cf >= 55 else ('✅' if cf >= 45 else '⚠️')

        spf_cal = None
        hhad_cal = None
        cal_signals = []

        clv_data = compute_clv(mc, env)

        spf_odds_map = {'胜': ho, '平': do, '负': ao}
        if spf and spf in spf_odds_map and spf_odds_map[spf] > 0:
            sel_odds = spf_odds_map[spf]
            raw_prob = clamp_prob(r['play1_spf']['confidence'])
            spf_cal = analyze_for_bet(mc, lg, 'spf', spf, raw_prob, sel_odds, env)
            cal_signals.extend([f'[{s}]' for s in spf_cal.get('signals', [])[:3]])

        rq_odds = do
        if rq and rq_odds > 0:
            rq_prob = clamp_prob(r['play2_rqspf']['confidence'])
            hhad_cal = analyze_for_bet(mc, lg, 'hhad', rq, rq_prob, rq_odds, env)
            cal_signals.extend([f'[{s}]' for s in hhad_cal.get('signals', [])[:2]])

        spf_cal_ev = spf_cal['ev'] if spf_cal else 0
        hhad_cal_ev = hhad_cal['ev'] if hhad_cal else 0

        dl_spf_prob = None
        dl_hhad_prob = None
        if _DL_AVAILABLE:
            hcap = r.get('play2_rqspf', {}).get('handicap', None)
            try:
                hcap_val = float(hcap) if hcap not in (None, '', 'NULL') else None
            except (ValueError, TypeError):
                hcap_val = None
            dl_spf = _DL.predict(mc, lg, 'spf', r['home_team'], r['away_team'],
                                  ho, do, ao, handicap=None, before_date=today)
            if 'error' not in dl_spf:
                dl_spf_prob = dl_spf['prob'][0]
                cal_signals.append(f'[DL胜{int(dl_spf_prob*100)}%]')
            if hcap_val is not None:
                dl_h = _DL.predict(mc, lg, 'hhad', r['home_team'], r['away_team'],
                                   ho, do, ao, handicap=hcap_val, before_date=today)
                if 'error' not in dl_h:
                    dl_hhad_prob = dl_h['prob'][0]
                    cal_signals.append(f'[DL让胜{int(dl_hhad_prob*100)}%]')

        ou_pred, ou_conf, ou_ls, ou_os, ou_us = '', '', '', '', ''
        ou_d = m.get('ou_odds')
        if ou_d and len(ou_d) >= 4:
            ou_ls = ou_d[1] or ''
            ou_os = ou_d[2] or ''
            ou_us = ou_d[3] or ''
            try:
                line_val = float(ou_ls) if ou_ls else 2.5
                over_w = float(ou_os) if ou_os else 0.85
                under_w = float(ou_us) if ou_us else 0.85
                exp_total = float(p4.get('expected_total', 2.5)) if p4.get('expected_total') else 2.5
                score = (2.5 - line_val) * 10 + (1.0 - over_w) * 20 + (exp_total - line_val) * 15
                if score > 0:
                    ou_pred = '大'; ou_conf = min(round(50 + score), 90)
                else:
                    ou_pred = '小'; ou_conf = min(round(50 + abs(score)), 90)
            except (ValueError, TypeError):
                pass

        sp_tag = ''
        if r.get('smart_play'):
            sp_tag = '|推荐:' + r['smart_play']['label'] + f"({round(r['smart_play']['confidence']*100)}%)"
        ou_tag = '|ou:' + ou_pred + (f'({ou_ls})' if ou_ls else '')

        vb = r.get('value_bet')
        vb_tag = ''
        if vb:
            units = vb.get('units', 1)
            vb_tag = f"|💰价值={vb['label']}@{vb['odds']}(EV+{vb['ev']}%)[{units}注×2元]"

        try:
            odds_map = {'胜': ho, '平': do, '负': ao}
            sel_odds = odds_map.get(spf, 0)
            sel_prob = r.get('play1_spf', {}).get('confidence', 0)
            if sel_odds > 0 and sel_prob > 0:
                _ev = calc_ev(sel_prob, sel_odds)
                _kelly = calc_kelly(sel_prob, sel_odds, fraction=1/6)
                cal_ev_tag = f"|校准EV={round(spf_cal_ev*100,1)}%" if spf_cal_ev else ''
                vb_tag += f"|EV={round(_ev*100,1)}%|Kelly={round(_kelly*100,1)}%{cal_ev_tag}"
        except (ValueError, TypeError, IndexError):
            pass

        signal_tag = ''
        if cal_signals:
            seen = set()
            uniq = []
            for s in cal_signals:
                key = s[:8]
                if key not in seen:
                    seen.add(key); uniq.append(s)
            signal_tag = '|'.join(uniq[:4])

        print(f"{tg}{mc}[{lg}]{r['home_team']}（主）vs（客）{r['away_team']}"
              f"|{odds_str}|{str(round(cf))}%|{spf}|{rq}|{bf}|{zjq}|{bqc}"
              f"{ou_tag}{sp_tag}{vb_tag}{signal_tag}" + (f"|{bn}" if bn else ''))

        # ── P1新增(2026-05-06): Agent Harness log_samples JSONL ──
        if _FILTER_REGISTRY_AVAILABLE:
            try:
                _log_entry = {
                    "task": f"jc_spf_{today}_{mc}",
                    "match_code": mc,
                    "date": today,
                    "home": r['home_team'],
                    "away": r['away_team'],
                    "league": lg,
                    "odds": {"home": ho, "draw": do_, "away": ao},
                    "spf_probs": {"胜": sp_prob, "平": dp, "负": ap},
                    "ev_home": ev_home, "ev_away": ev_away,
                    "prediction": {
                        "spf": spf, "spf_conf": r['play1_spf']['confidence'],
                        "rq": rq, "rq_conf": _rq_conf,
                        "bf": bf, "zjq": zjq, "bqc": bqc,
                        "ou": ou_pred, "ou_conf": ou_conf
                    },
                    "calibration": {
                        "spf_final_prob": spf_cal['final_prob'] if spf_cal else None,
                        "spf_ev": spf_cal['ev'] if spf_cal else None,
                        "hhad_final_prob": hhad_cal['final_prob'] if hhad_cal else None,
                        "hhad_ev": hhad_cal['ev'] if hhad_cal else None
                    },
                    "fusion_quality": cf,
                    "model_type": _mt,
                    "confidence_bands": {"low": cf >= 45, "high": cf >= 55},
                    "filter_passed": _rq_hhad_gate_pass,
                    "filter_signals": _adj_signals,
                    "value_bet": vb if vb else None,
                    "expert_signals": _es if _es else None,
                    "ci_95": r.get('ci_95'),
                    "all_factor_results": r.get('all_factor_results'),
                }
                _fd = _get_log_fd()
                _fd.write(json.dumps(_log_entry, ensure_ascii=False) + '\n')
                _fd.flush()
                # ── Per-match Task instance file ──
                try:
                    import time as _t
                    _ts = int(_t.time())
                    _task_dir = os.path.expanduser('~/.hermes/workspace/.tasks')
                    os.makedirs(_task_dir, exist_ok=True)
                    _task_file = os.path.join(_task_dir, f"match_{mc}_{today}.json")
                    _task = {
                        "match_id": mc,
                        "match_date": today,
                        "league": lg,
                        "home_team": r['home_team'],
                        "away_team": r['away_team'],
                        "odds": {"home": ho, "draw": do_, "away": ao},
                        "prediction": {
                            "spf": spf,
                            "spf_conf": float(r['play1_spf']['confidence']) if r['play1_spf'].get('confidence') is not None else None,
                            "rq": rq,
                            "rq_conf": float(_rq_conf) if _rq_conf else None,
                            "bf": bf, "zjq": zjq, "bqc": bqc,
                            "ou": ou_pred,
                            "ou_conf": float(ou_conf) if ou_conf else None
                        },
                        "fusion_quality": cf,
                        "ci_95": r.get('ci_95'),
                        "all_factor_results": r.get('all_factor_results'),
                        "filter_passed": _rq_hhad_gate_pass,
                        "value_bet": vb if vb else None,
                        "expert_signals": _es if _es else None,
                        "timestamp": _ts,
                        "source": "ml_predict_5play"
                    }
                    with open(_task_file, 'w', encoding='utf-8') as _tf:
                        json.dump(_task, _tf, ensure_ascii=False, indent=2)
                except Exception:
                    pass
            except Exception:
                pass

        _n_recent = getattr(_DL, '_last_seq_len', 5) if _DL else 5
        if lg in ('英超', '意甲', '西甲', '德甲', '法甲'):
            _dl_w = 0.40
        elif lg in ('法乙', '瑞超', '葡超', '挪超', '澳超'):
            _dl_w = 0.20
        else:
            _dl_w = 0.30
        if _n_recent < 5:
            _dl_w *= 0.5
        _cal_w = 1.0 - _dl_w

        try:
            match_info = {
                'match_date': today, 'match_id': mc, 'league': lg,
                'home_team': r['home_team'], 'away_team': r['away_team'],
            }
            if ho and spf_cal:
                final_prob = spf_cal['final_prob']
                if dl_spf_prob is not None:
                    final_prob = clamp_prob(_cal_w * spf_cal['final_prob'] + _dl_w * dl_spf_prob)
                _ev = calc_ev(final_prob, ho)
                _kelly = calc_kelly(final_prob, ho, fraction=1/6)
                stake_spf = round(_kelly * 10000, 2)
                _bt.record_bet(mc, 'spf', {
                    **match_info, 'selection': spf,
                    'prob': final_prob, 'ev': round(_ev, 4), 'kelly': round(_kelly, 4),
                    'calibrated_prob': spf_cal['final_prob'],
                    'calibration_signals': spf_cal.get('signals', []),
                }, ho, stake_spf,
                odds_home=ho, odds_draw=do, odds_away=ao)

            if not is_hhad_paused() and _has_hc:
                if rq_odds:
                    final_rq_prob = hhad_cal['final_prob'] if hhad_cal else clamp_prob(r['play2_rqspf']['confidence'])
                    if dl_hhad_prob is not None and hhad_cal:
                        final_rq_prob = clamp_prob(_cal_w * hhad_cal['final_prob'] + _dl_w * dl_hhad_prob)
                    _ev = calc_ev(final_rq_prob, rq_odds)
                    _kelly = calc_kelly(final_rq_prob, rq_odds, fraction=1/6)
                    stake_rq = round(_kelly * 10000, 2)
                    _bt.record_bet(mc, 'hhad', {
                        **match_info, 'selection': rq,
                        'prob': final_rq_prob, 'ev': round(_ev, 4), 'kelly': round(_kelly, 4),
                        'calibrated_prob': hhad_cal['final_prob'] if hhad_cal else final_rq_prob,
                        'calibration_signals': hhad_cal.get('signals', []) if hhad_cal else [],
                    }, rq_odds, stake_rq,
                    odds_home=ho, odds_draw=rq_odds, odds_away=ao)
        except Exception:
            pass

        bf3 = '>'.join(s[0] for s in p3['predictions'][:3])
        zj3 = '>'.join(g[0] for g in p4['predictions'][:3])
        bq3 = '>'.join(b[0] for b in p5['predictions'][:3])

        _lm = r.get('league_model')
        if _lm and _lm.get('oos', 0) > 0.50:
            _mt = 'fusion'; _td = 1.0
        elif _lm:
            _mt = 'league_weak'; _td = 0.8
        else:
            _mt = 'v5play_fusion'; _td = 1.0

        _es = _extract_expert_signals(r)
        _es['cf'] = cf
        _ba = r.get('bayes_advisor')
        if _ba:
            _es['bayes_advisor'] = {
                'key_insight': _ba.get('key_insight', ''),
                'scripts': _ba.get('scripts', []),
                'warnings': _ba.get('warnings', []),
                'confidence': _ba.get('confidence', 0),
                'model': _ba.get('model', ''),
            }
        _xs = m.get('extra_signals', {})
        if _xs:
            for _k in ('market_trap', 'hot_streak_bias', 'tactical_matchup', 'situational_factor', 'reasoning'):
                if _k in _xs:
                    _es[f'claude_{_k}'] = _xs[_k]

        # ── Fusion/CI data ─────────────────────────────────────────
        _fusion_layer = r.get('fusion_layer')
        _fusion_confidence = r.get('confidence')
        _ci_95 = r.get('ci_95')
        _fusion_quality = _fusion_layer.get('fusion_quality') if isinstance(_fusion_layer, dict) else None
        _fusion_probs = r.get('probs') if _fusion_layer else None
        try:
            # ── Bug Fix (2026-05-07): spf_confidence 归一化守卫
            # 3月数据源使用百分比格式(如"52.1"→52.1)，需归一化为小数(0.521)
            # 预防：未来任何来源传入>1的值都做此转换
            def _norm_conf(c):
                if c is None:
                    return None
                try:
                    v = float(c)
                except Exception:
                    return None
                return v / 100.0 if v > 1.0 else v
            _args = (today, mc, r['home_team'], r['away_team'], lg, odds_str,
                 spf, _norm_conf(r['play1_spf']['confidence']),
                 r['play2_rqspf']['handicap'], rq, _rq_conf,
                 bf, bf3, zjq, zj3, p4['expected_total'],
                 bqc, bq3, round(cf), cf >= 45, cf >= 55,
                 ou_ls, ou_os, ou_us, ou_pred, ou_conf,
                 f"{vb['label']}@{vb['odds']}(EV+{vb['ev']}%)" if r.get('value_bet') else None,
                 _mt, _td,
                 spf_cal['final_prob'] if spf_cal else None,
                 spf_cal['ev'] if spf_cal else None,
                 hhad_cal['final_prob'] if hhad_cal else None,
                 hhad_cal['ev'] if hhad_cal else None,
                 _es if _es else None,
                 r.get('spf'),  # fusion_prediction
                 _fusion_confidence,
                 _ci_95[0] if _ci_95 else None,
                 _ci_95[1] if _ci_95 else None,
                 json.dumps(_fusion_layer) if isinstance(_fusion_layer, dict) else str(_fusion_layer) if _fusion_layer else None,
                 _fusion_quality,
                 _fusion_probs,
                 None,  # fusion_duration_ms
                 r.get('all_factor_results'),  # factor_probs (P2-1: 各因子独立概率)
            )
            db_insert(*_args)

        except Exception as e:
            _conn.rollback()
            tb = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
            print(f"  ⚠️ db_insert失败 {mc}: {e}")
            print(f"    _args ({len(_args)} items):")
            for i, a in enumerate(_args):
                print(f"      [{i}] {repr(a)[:80]}")
            print(f"    FULL TRACEBACK:\n{tb}")
            sys.stdout.flush()

        # ── P2: 写入predictions_ensemble ────────────────────────────────
        _ens = r.get('_ensemble_vals')
        if _ens:
            try:
                with _conn.cursor() as _cur:
                    _cur.execute("""
                        INSERT INTO predictions_ensemble(
                            match_date, match_code, league, home_team, away_team,
                            factor_ml_pct, factor_dl_pct, factor_kelly_pct, factor_trend_pct,
                            factor_betfair_pct, factor_dow_pct, factor_league_pct,
                            ensemble_home_pct, ensemble_draw_pct, ensemble_away_pct, ensemble_confidence)
                        VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (match_date, match_code) DO UPDATE SET
                            factor_ml_pct=EXCLUDED.factor_ml_pct,
                            factor_dl_pct=EXCLUDED.factor_dl_pct,
                            factor_kelly_pct=EXCLUDED.factor_kelly_pct,
                            factor_trend_pct=EXCLUDED.factor_trend_pct,
                            factor_betfair_pct=EXCLUDED.factor_betfair_pct,
                            factor_dow_pct=EXCLUDED.factor_dow_pct,
                            factor_league_pct=EXCLUDED.factor_league_pct,
                            ensemble_home_pct=EXCLUDED.ensemble_home_pct,
                            ensemble_draw_pct=EXCLUDED.ensemble_draw_pct,
                            ensemble_away_pct=EXCLUDED.ensemble_away_pct,
                            ensemble_confidence=EXCLUDED.ensemble_confidence
                    """, (today, mc, lg, r['home_team'], r['away_team'],
                          _ens['factor_ml_pct'], _ens['factor_dl_pct'],
                          _ens['factor_kelly_pct'], _ens['factor_trend_pct'],
                          _ens['factor_betfair_pct'], _ens['factor_dow_pct'],
                          _ens['factor_league_pct'],
                          _ens['ensemble_home_pct'], _ens['ensemble_draw_pct'],
                          _ens['ensemble_away_pct'], _ens['ensemble_confidence']))
                _conn.commit()
            except Exception:
                _conn.rollback()

    print(f'💾入库{len(ms)}场')

    # ── P1(2026-05-06): 关闭 log_samples 文件 ──
    if _LOG_SAMPLES_FD:
        _LOG_SAMPLES_FD.close()
        _LOG_SAMPLES_FD = None

    try:
        from feishu_push import push_predictions
        push_predictions(today, os.environ.get('PREDICT_MODE', 'full'))
    except Exception as e:
        print(f'⚠️ 推送失败: {e}')
