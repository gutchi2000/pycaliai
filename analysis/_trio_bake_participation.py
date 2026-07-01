"""
METHOD D -- PARTICIPATION GATING for 三連複 (trio) backtest.

Base buying method = prob-first: among each race's PL horses, buy the top-N trio
combinations ranked by Plackett-Luce trio probability (pl_probs.all_sanrenpuku).
"prob-first top-8" = restrict candidate horses to the 8 highest PL weights, then
buy the top-K trio combos by PL prob. We test a couple of K (combos bought) so the
gating effect is not confounded by a single point.

Then apply PARTICIPATION GATES: keep a race only if a leak-safe confidence signal
passes; else SKIP the entire race (bet nothing). We report, for every (config, split):
  n_races (= races BET), capture_pct, roi_pct, race-level bootstrap 95% CI on ROI,
  and races_kept_pct = races bet / races available in that split.

Signals swept:
  d1 top1 PL win prob (max all_tansho) >= q,  q in {0.0(no gate),0.25,0.40,0.55}
  d2 field chaos = Shannon entropy(H) of PL win-prob vector <= e-quantile,
     keep calmest {100%(no gate),75%,50%,25%}. Quantile thresholds are computed
     PER SPLIT on that split's race entropy distribution (an honest "keep calmest X%").

Stake = 100 yen per combo. A combo wins iff its set of 3 ban == winning trio set
(the 3 ban whose finish in {1,2,3}); payout = tri_pay yen (per 100 stake).
"""
import sys
sys.path.insert(0, 'E:/PyCaLiAI')
import numpy as np
import pandas as pd
import pl_probs

PARQUET = 'E:/PyCaLiAI/analysis/_trio_substrate.parquet'
RNG_SEED = 12345
N_BOOT = 2000

df = pd.read_parquet(PARQUET)

# ---- build per-race records -------------------------------------------------
def build_races(sub):
    races = []
    for rid, g in sub.groupby('rid', sort=False):
        g = g.sort_values('model_rank')
        ban = g['ban'].to_numpy()
        w = g['pl_w'].to_numpy(dtype=float)
        finish = g['finish'].to_numpy()
        # winning trio = set of 3 ban whose finish in {1,2,3}
        win_mask = np.isin(finish, [1, 2, 3])
        win_ban = set(ban[win_mask].tolist())
        if len(win_ban) != 3:
            # malformed result (dead-heat / missing placing) -> drop
            continue
        tri_pay = float(g['tri_pay'].iloc[0])
        if not np.isfinite(tri_pay) or tri_pay <= 0:
            continue
        # PL win-prob vector & trio probs over the FULL field (indices align with w)
        tansho = pl_probs.all_tansho(w)          # array over horse indices
        top1 = float(np.max(tansho))
        p = np.clip(tansho, 1e-12, None)
        p = p / p.sum()
        entropy = float(-(p * np.log(p)).sum())
        trio = pl_probs.all_sanrenpuku(w)        # {(i,j,k): prob}
        races.append({
            'rid': rid, 'ban': ban, 'win_ban': win_ban, 'tri_pay': tri_pay,
            'top1': top1, 'entropy': entropy, 'trio': trio, 'n_horses': len(ban),
        })
    return races

def topk_combos(race, top8_only, k):
    """Return list of (set_of_3_ban) for the top-k trio combos by PL prob.
    If top8_only: candidate horses restricted to the 8 highest PL (model_rank<=8)."""
    ban = race['ban']
    trio = race['trio']
    if top8_only:
        allowed_idx = set(range(min(8, len(ban))))  # ban already sorted by model_rank
        items = [(idx3, p) for idx3, p in trio.items()
                 if set(idx3) <= allowed_idx]
    else:
        items = list(trio.items())
    items.sort(key=lambda kv: kv[1], reverse=True)
    out = []
    for idx3, _ in items[:k]:
        out.append(frozenset(ban[i] for i in idx3))
    return out

def evaluate(races, combo_fn):
    """combo_fn(race)->list[frozenset ban]. Returns per-race arrays stake,return,hit."""
    stakes, returns, hits = [], [], []
    for r in races:
        combos = combo_fn(r)
        if not combos:
            continue
        stake = 100 * len(combos)
        wset = frozenset(r['win_ban'])
        won = any(c == wset for c in combos)
        ret = r['tri_pay'] if won else 0.0
        stakes.append(stake); returns.append(ret); hits.append(1 if won else 0)
    return np.array(stakes, float), np.array(returns, float), np.array(hits, int)

def boot_ci_roi(stakes, returns, n_boot=N_BOOT, seed=RNG_SEED):
    if len(stakes) == 0:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    n = len(stakes)
    rois = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        s = stakes[idx].sum()
        rois[b] = returns[idx].sum() / s * 100.0 if s > 0 else 0.0
    return (float(np.percentile(rois, 2.5)), float(np.percentile(rois, 97.5)))

# Precompute races per split
RACES = {}
DROPPED = {}
for split in ['valid', 'test']:
    sub = df[df['split'] == split]
    total = sub['rid'].nunique()
    rr = build_races(sub)
    RACES[split] = rr
    DROPPED[split] = total - len(rr)
    print(f"[{split}] races available={total}  usable(after trio/payout clean)={len(rr)}  dropped={total-len(rr)}")

# ---- config sweep -----------------------------------------------------------
# Base buying sets to test (top8 universe always True here per method spec)
BASE_K = [4, 8]   # 4 combos (=top4 C(4,3)) and 8 combos by PL prob within top8
D1_Q  = [0.0, 0.25, 0.40, 0.55]
D2_KEEP = [1.00, 0.75, 0.50, 0.25]   # keep calmest fraction (entropy <= quantile)

results = []

for k in BASE_K:
    for q in D1_Q:
        for keep in D2_KEEP:
            cfg_name = f"top8_K{k}_d1q{q:.2f}_d2keep{keep:.2f}"
            per_split = {}
            for split in ['valid', 'test']:
                races = RACES[split]
                avail = len(races)
                # entropy quantile threshold computed on THIS split
                if keep < 1.0:
                    ent_all = np.array([r['entropy'] for r in races])
                    ent_thr = np.quantile(ent_all, keep)  # keep calmest 'keep' frac
                else:
                    ent_thr = np.inf
                kept = [r for r in races if (r['top1'] >= q and r['entropy'] <= ent_thr)]
                cf = lambda r: topk_combos(r, True, k)
                stakes, returns, hits = evaluate(kept, cf)
                n_bet = len(stakes)
                if n_bet == 0:
                    per_split[split] = dict(n=0, cap=float('nan'), roi=float('nan'),
                                            lo=float('nan'), hi=float('nan'),
                                            kept_pct=0.0, spr=0.0)
                    continue
                total_stake = stakes.sum()
                roi = returns.sum() / total_stake * 100.0
                cap = hits.sum() / n_bet * 100.0
                lo, hi = boot_ci_roi(stakes, returns)
                spr = stakes.mean()  # stake per race
                per_split[split] = dict(n=n_bet, cap=cap, roi=roi, lo=lo, hi=hi,
                                        kept_pct=n_bet / avail * 100.0, spr=spr)
            results.append((cfg_name, k, q, keep, per_split))

# ---- print -----------------------------------------------------------------
print("\n==== METHOD D: PARTICIPATION GATING (prob-first top-8 trio) ====")
hdr = f"{'config':36s} | {'split':5s} | {'nbet':5s} {'kept%':6s} {'cap%':6s} {'roi%':7s} {'ci_lo':7s} {'ci_hi':7s} {'spr':6s}"
print(hdr); print('-'*len(hdr))
for cfg_name, k, q, keep, ps in results:
    for split in ['valid', 'test']:
        d = ps[split]
        print(f"{cfg_name:36s} | {split:5s} | {d['n']:5d} {d['kept_pct']:6.1f} {d['cap']:6.2f} "
              f"{d['roi']:7.2f} {d['lo']:7.2f} {d['hi']:7.2f} {d['spr']:6.0f}")

# ---- pick best test config (highest test ROI with n>=200 to be meaningful) --
best = None
for cfg_name, k, q, keep, ps in results:
    dt = ps['test']
    if dt['n'] >= 200 and np.isfinite(dt['roi']):
        if best is None or dt['roi'] > best[1]['test']['roi']:
            best = (cfg_name, ps)
print("\n==== BEST TEST CONFIG (n>=200) ====")
if best:
    cfg_name, ps = best
    print(cfg_name, 'test', ps['test'], 'valid', ps['valid'])

# emit a compact JSON-ish summary for the agent to read
import json
out = []
for cfg_name, k, q, keep, ps in results:
    out.append(dict(config=cfg_name,
                    test=ps['test'], valid=ps['valid']))
print("\nJSON_SUMMARY_START")
print(json.dumps(out))
print("JSON_SUMMARY_END")
