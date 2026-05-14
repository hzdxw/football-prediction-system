#!/usr/bin/env python3
"""自动复盘校准 — 从赛果数据反向优化预测参数

每日cron运行，统计准确率，识别偏移，自动调整calibrated_poisson.py中的参数。

输出:
  1. 各玩法准确率统计（按联赛/赔率段）
  2. 参数偏移检测（哪些参数需要调整）
  3. 自动更新 evolution_calibration.json 中的校准字段
"""
import os, sys, json
from datetime import date, timedelta
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent.parent.parent  # scripts/ → football-predictor/ → skills/ → workspace/.parent  # skills/football-predictor/scripts/ → workspace
sys.path.insert(0, str(WORKSPACE / 'data-collection'))
os.environ.setdefault('PGDATABASE', 'myapp_db')

from dotenv import load_dotenv
load_dotenv(WORKSPACE / '..' / '.openclaw' / '.env')
import psycopg2

CAL_FILE = WORKSPACE / 'data' / 'evolution_calibration.json'


def get_conn():
    return psycopg2.connect(
        host=os.environ.get('PGHOST', 'localhost'),
        user=os.environ.get('PGUSER', 'myapp'),
        password=os.environ.get('PGPASSWORD', ''),
        dbname=os.environ.get('PGDATABASE', 'myapp_db'))


def fetch_matched(cur):
    """获取所有预测vs赛果匹配数据"""
    cur.execute("""
    WITH preds AS (
      SELECT DISTINCT ON (match_code, match_date) *
      FROM predictions_5play ORDER BY match_code, match_date, id DESC
    ),
    results AS (
      SELECT record_date as match_date, match_code, spf_result, full_score, half_score
      FROM match_results
    )
    SELECT p.match_date, p.match_code, p.league, p.odds,
      p.spf_prediction, p.spf_confidence,
      p.rqspf_prediction, p.rqspf_handicap,
      p.bf_top1, p.zjq_top1, p.bqc_top1, p.value_bet,
      r.spf_result, r.full_score,
      COALESCE(split_part(r.full_score,'-',1)::int, 0) as ahg,
      COALESCE(split_part(r.full_score,'-',2)::int, 0) as aag
    FROM preds p JOIN results r 
      ON p.match_code = r.match_code 
      AND p.match_date::text LIKE r.match_date::text || '%%'
    ORDER BY p.match_date, p.match_code
    """)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def calc_stats(rows):
    """计算各维度准确率"""
    n = len(rows)
    if n == 0:
        return {}

    spf_hit = sum(1 for r in rows if r['spf_prediction'] == r['spf_result'])
    bf_hit = sum(1 for r in rows if r['bf_top1'] == r['full_score'])
    zjq_hit = 0
    for r in rows:
        if r['zjq_top1']:
            try:
                pred_t = int(r['zjq_top1'].replace('球', ''))
                if r['ahg'] + r['aag'] == pred_t:
                    zjq_hit += 1
            except (ValueError, TypeError):
                pass

    # 价值投注命中
    vb_rows = [r for r in rows if r['value_bet']]
    vb_hit = 0
    for r in vb_rows:
        label = r['value_bet'].split('@')[0]
        if label == r['spf_result']:
            vb_hit += 1

    # 平局分析
    draw_actual = sum(1 for r in rows if r['spf_result'] == '平')
    draw_pred = sum(1 for r in rows if r['spf_prediction'] == '平')
    draw_hit = sum(1 for r in rows if r['spf_prediction'] == '平' and r['spf_result'] == '平')

    stats = {
        'total': n,
        'spf_hit': spf_hit, 'spf_pct': round(spf_hit / n * 100, 1),
        'bf_hit': bf_hit, 'bf_pct': round(bf_hit / n * 100, 1),
        'zjq_hit': zjq_hit, 'zjq_pct': round(zjq_hit / max(n, 1) * 100, 1),
        'draw_actual_pct': round(draw_actual / n * 100, 1),
        'draw_pred_pct': round(draw_pred / n * 100, 1),
        'draw_hit': draw_hit, 'draw_pred_total': draw_pred,
    }
    if vb_rows:
        stats['vb_total'] = len(vb_rows)
        stats['vb_hit'] = vb_hit
        stats['vb_pct'] = round(vb_hit / len(vb_rows) * 100, 1)
    return stats


def calc_by_league(rows):
    """按联赛统计"""
    leagues = {}
    for r in rows:
        lg = r['league'] or '未知'
        leagues.setdefault(lg, []).append(r)
    result = {}
    for lg, lg_rows in sorted(leagues.items(), key=lambda x: -len(x[1])):
        if len(lg_rows) >= 3:
            result[lg] = calc_stats(lg_rows)
    return result


def calc_by_odds_band(rows):
    """按赔率段统计"""
    bands = {'低赔(<1.5)': [], '中低(1.5-2.0)': [], '中赔(2.0-3.0)': [], '高赔(>3.0)': []}
    for r in rows:
        try:
            ho = float(r['odds'].split('/')[0])
        except (ValueError, AttributeError):
            continue
        if ho <= 1.5:
            bands['低赔(<1.5)'].append(r)
        elif ho <= 2.0:
            bands['中低(1.5-2.0)'].append(r)
        elif ho <= 3.0:
            bands['中赔(2.0-3.0)'].append(r)
        else:
            bands['高赔(>3.0)'].append(r)
    return {k: calc_stats(v) for k, v in bands.items() if v}


def detect_drifts(overall, by_league, by_band):
    """检测参数偏移，生成调整建议"""
    drifts = []

    # 1. 平局偏移检测
    draw_diff = overall['draw_actual_pct'] - overall['draw_pred_pct']
    if abs(draw_diff) > 5:
        drifts.append({
            'param': 'draw_boost',
            'issue': f'平局实际{overall["draw_actual_pct"]}% vs 预测{overall["draw_pred_pct"]}% (差{draw_diff:+.1f}%)',
            'suggestion': f'友谊赛draw_boost从0.06调整到{max(0.01, 0.06 + draw_diff/100):.2f}' if draw_diff > 0 else '平局被高估',
        })

    # 2. 低赔命中率检测（应>70%，否则SPF校准有问题）
    if '低赔(<1.5)' in by_band:
        low_pct = by_band['低赔(<1.5)'].get('spf_pct', 0)
        if low_pct < 70:
            drifts.append({
                'param': 'spf_calibration',
                'issue': f'低赔命中率仅{low_pct}%（应>70%）',
                'suggestion': 'SPF校准可能偏移，检查λ优化是否收敛',
            })

    # 3. 高赔段平局过多检测
    if '高赔(>3.0)' in by_band:
        high_draw = by_band['高赔(>3.0)'].get('draw_actual_pct', 0)
        if high_draw > 35:
            drifts.append({
                'param': 'balanced_draw_boost',
                'issue': f'高赔段平局率{high_draw}%（>35%）',
                'suggestion': 'balanced赔率段应额外boost平局概率',
            })

    return drifts


def update_calibration(overall, by_league, drifts):
    """更新evolution_calibration.json"""
    with open(CAL_FILE) as f:
        cal = json.load(f)

    cal['auto_review'] = {
        'updated_at': date.today().isoformat(),
        'sample_size': overall['total'],
        'spf_accuracy': overall['spf_pct'],
        'bf_accuracy': overall['bf_pct'],
        'zjq_accuracy': overall['zjq_pct'],
        'draw_actual': overall['draw_actual_pct'],
        'league_stats': {k: {'n': v['total'], 'spf': v['spf_pct']} for k, v in by_league.items()},
        'drifts': drifts,
    }

    with open(CAL_FILE, 'w') as f:
        json.dump(cal, f, ensure_ascii=False, indent=2)

    return cal.get('auto_review', {})


def _numpy_platt_fit(confs_v, hits_v, weights):
    """纯numpy实现的Platt参数最优化（无scipy依赖）"""
    import numpy as np

    def negloglik(params):
        scale, offset = params
        if scale <= 0:
            return 1e10
        z = scale * (confs_v - offset)
        z = np.clip(z, -500, 500)
        pred_prob = 1 / (1 + np.exp(-z))
        pred_prob = np.clip(pred_prob, 1e-10, 1 - 1e-10)
        loss = -weights * (hits_v * np.log(pred_prob) + (1 - hits_v) * np.log(1 - pred_prob))
        return float(loss.sum())

    # Step 1: 粗粒度网格搜索 (scale, offset)
    best = (1.0, 0.0, negloglik((1.0, 0.0)))
    for scale in np.linspace(0.5, 5.0, 20):
        for offset in np.linspace(-0.3, 0.3, 20):
            loss = negloglik((scale, offset))
            if loss < best[2]:
                best = (scale, offset, loss)

    # Step 2: 牛顿法精调（对数几率回归的解析梯度）
    scale, offset = best[0], best[1]
    for _ in range(50):
        z = scale * (confs_v - offset)
        z = np.clip(z, -500, 500)
        pred_prob = 1 / (1 + np.exp(-z))
        pred_prob = np.clip(pred_prob, 1e-10, 1 - 1e-10)
        w = weights * pred_prob * (1 - pred_prob)
        residual = hits_v - pred_prob

        # 解析梯度
        g_scale = float(-np.sum(w * residual * (confs_v - offset)))
        g_offset = float(np.sum(w * residual) * scale)
        h_scale = float(np.sum(w * (confs_v - offset)**2))
        h_offset = float(np.sum(w) * scale**2)

        # Hessian 对角近似 + 步长
        denom_scale = max(abs(h_scale), 1e-6)
        denom_offset = max(abs(h_offset), 1e-6)
        step_scale = g_scale / denom_scale
        step_offset = g_offset / denom_offset

        # 线搜索步长
        step_size = 1.0
        for _ in range(10):
            ns = scale - step_size * step_scale
            no = offset - step_size * step_offset
            if negloglik((ns, no)) < negloglik((scale, offset)):
                scale, offset = ns, no
                break
            step_size *= 0.5

        if max(abs(step_scale), abs(step_offset)) < 1e-6:
            break

    return max(0.1, float(scale)), float(offset)


def neutral_quantile_weighted_platt(confidences, hit_mask, interval=(0.42, 0.47)):
    """P1-新增: 分位数加权Platt缩放 — 修正42-47%区间的过度乐观

    根因: 42-47%置信区间gap=-11.0%，该区间样本集中，
    Platt缩放对全局优化导致该区间被"平均化"而忽视。

    方法:
    1. 识别interval区间内的样本及其分位数位置
    2. 对区间内样本赋予更高权重(1.5x)进行Platt拟合
    3. 输出调整后的scale和offset参数

    无scipy时使用纯numpy网格搜索+牛顿法。

    Args:
        confidences: 置信度数组 (0-1)
        hit_mask: 实际命中数组 (1=命中, 0=未中)
        interval: 目标区间元祖 (low, high)

    Returns:
        (scale, offset, adjustment_info)
    """
    import numpy as np

    try:
        from scipy.optimize import minimize
        HAS_SCIPY = True
    except ImportError:
        HAS_SCIPY = False

    confs = np.array(confidences, dtype=float)
    hits = np.array(hit_mask, dtype=float)

    valid = (~np.isnan(confs)) & (confs > 0) & (confs < 1)
    if valid.sum() < 10:
        return 1.0, 0.0, {'error': 'insufficient samples'}

    confs_v = confs[valid]
    hits_v = hits[valid]

    low, high = interval
    in_interval = (confs_v >= low) & (confs_v <= high)
    weights = np.where(in_interval, 1.5, 1.0)

    if HAS_SCIPY:
        def platt_negloglik(params):
            scale, offset = params
            if scale <= 0:
                return 1e10
            z = scale * (confs_v - offset)
            z = np.clip(z, -500, 500)
            pred_prob = 1 / (1 + np.exp(-z))
            pred_prob = np.clip(pred_prob, 1e-10, 1 - 1e-10)
            loss = -weights * (hits_v * np.log(pred_prob) + (1 - hits_v) * np.log(1 - pred_prob))
            return loss.sum()

        result = minimize(platt_negloglik, [10.0, 0.5], method='Nelder-Mead',
                         options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
        scale, offset = result.x
        scale = max(0.1, float(scale))
    else:
        scale, offset = _numpy_platt_fit(confs_v, hits_v, weights)

    n_in_interval = int(in_interval.sum())
    if n_in_interval > 0:
        interval_hit_rate = float(hits_v[in_interval].mean())
        interval_conf_mean = float(confs_v[in_interval].mean())
        gap = interval_hit_rate - interval_conf_mean
        adjustment_info = {
            'n_in_interval': n_in_interval,
            'interval_hit_rate': round(interval_hit_rate, 3),
            'interval_conf_mean': round(interval_conf_mean, 3),
            'gap': round(gap, 3),
            'scale': round(float(scale), 4),
            'offset': round(float(offset), 4),
            'backend': 'scipy' if HAS_SCIPY else 'numpy_grid_newton',
        }
    else:
        adjustment_info = {'n_in_interval': 0}
    return float(scale), float(offset), adjustment_info


def apply_neutral_platt_correction(confidence, scale=1.0, offset=0.0):
    """应用neutral_platt校正到单个置信度"""
    import math
    try:
        conf = float(confidence)
        z = scale * (conf - offset)
        z = max(-500, min(500, z))
        adjusted = 1 / (1 + math.exp(-z))
        return round(adjusted, 4)
    except (ValueError, TypeError, OverflowError):
        return confidence


def main():
    conn = get_conn()
    cur = conn.cursor()

    rows = fetch_matched(cur)
    conn.close()

    if not rows:
        print('⚠️ 无匹配数据')
        return

    overall = calc_stats(rows)
    by_league = calc_by_league(rows)
    by_band = calc_by_odds_band(rows)
    drifts = detect_drifts(overall, by_league, by_band)
    review = update_calibration(overall, by_league, drifts)

    # P1-新增: neutral_quantile_weighted_platt 校准
    confidences = [r['spf_confidence'] for r in rows if r['spf_confidence'] is not None]
    hit_mask = [1 if r['spf_prediction'] == r['spf_result'] else 0 for r in rows if r['spf_confidence'] is not None]
    if confidences:
        scale, offset, adj_info = neutral_quantile_weighted_platt(confidences, hit_mask)
        with open(CAL_FILE) as f:
            cal = json.load(f)
        cal.setdefault('auto_review', {})
        cal['auto_review']['platt_scale'] = scale
        cal['auto_review']['platt_offset'] = offset
        cal['auto_review']['platt_info'] = adj_info
        with open(CAL_FILE, 'w') as f:
            json.dump(cal, f, ensure_ascii=False, indent=2)
        print(f'\n📐 Platt校准: scale={scale:.4f} offset={offset:.4f}')

    # 输出报告
    print(f'📊 自动复盘报告 ({overall["total"]}场)')
    print(f'  SPF: {overall["spf_pct"]}% | 比分: {overall["bf_pct"]}% | 总进球: {overall["zjq_pct"]}%')
    print(f'  平局: 实际{overall["draw_actual_pct"]}% 预测{overall["draw_pred_pct"]}%')
    if overall.get('vb_total'):
        print(f'  价值投注: {overall["vb_pct"]}% ({overall["vb_hit"]}/{overall["vb_total"]})')

    print(f'\n📋 联赛准确率 (≥3场):')
    for lg, s in sorted(by_league.items(), key=lambda x: -x[1]['spf_pct']):
        emoji = '✅' if s['spf_pct'] >= 60 else ('⚠️' if s['spf_pct'] >= 40 else '❌')
        print(f'  {emoji} {lg}: SPF {s["spf_pct"]}% ({s["total"]}场)')

    print(f'\n📋 赔率段:')
    for band, s in by_band.items():
        print(f'  {band}: SPF {s["spf_pct"]}% 比分 {s["bf_pct"]}% ({s["total"]}场)')

    if drifts:
        print(f'\n🚨 参数偏移检测:')
        for d in drifts:
            print(f'  {d["param"]}: {d["issue"]}')
            if 'suggestion' in d:
                print(f'    → {d["suggestion"]}')
    else:
        print(f'\n✅ 无显著参数偏移')


if __name__ == '__main__':
    main()
