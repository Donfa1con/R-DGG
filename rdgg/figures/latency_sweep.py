"""Regenerate figures/latency_curves.json: recompute the R-DGG gain G_{a|m} with the speaker's driver
displaced by a constant offset over the grid, delta in {-5..+5} slots (25 fps -> 40 ms/slot, so -200..+200 ms).
net(delta) = (aud|mot(delta) - floor) * 1e4, same timeshift floor as the reported number; delta=0 reproduces Table 1.
Writes {system: [[offset_ms, G_{a|m} x1e4], ...]} for figures/plot_latency.py.

aud|mot(off) = gain(both) - gain(motion) = log(sum e_mot / sum e_both) (E_own cancels), pooled over all rows --
the same quantity capture_energies() reports at offset 0. We hook capture_energies on the LEADERBOARD (common-mask)
capture of each system and sweep the offsets there, reusing score_source's exact setup (operator, target-norm,
masks, surrogate wrapping).

Needs the metric env + a GPU (it re-scores every system):

    cd rdgg && pixi run python figures/latency_sweep.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import compute as m  # noqa: E402

OFFSETS = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]          # grid slots; 40 ms each -> +-200 ms
LABELS = [lbl for lbl, _ in m.MODELS_ALL]                 # score_source loop order: GT, MODELS..., surrogate
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "latency_curves.json")
CURVES = {}

_common = [True]
_ocd = m.collect_design
def _cd(mcache, ctx, mask=None):                          # is the design just built the COMMON (leaderboard) mask?
    _common[0] = mask is None
    return _ocd(mcache, ctx, mask)
m.collect_design = _cd

_ocap = m.capture_energies
def _cap(per, ops, ctx, timeshifts=None):
    if _common[0] and timeshifts is None:            # the per-model leaderboard capture
        p2 = dict(per)                                    # sweep on a copy; pool_design(consume) nulls the copy, not the original
        order, XO, Y, tids, depths = m.pool_design(p2, consume=True)
        pred = m.DriverPredictor(ops, ctx, order, tids, depths)
        Xz_o = m.own_block_standardised(ctx, order, tids, XO, pred.lag_is_in_segment)
        fit = m.ControlFit(Xz_o, Y, order, pred.rows_per_listener,
                           target_scale_inv=ctx.target_scale_inv)

        def aud_mot(off):                                 # = log(sum e_mot / sum e_both), pooled over all rows
            shat = pred.predict(ctx.corpus.pairs, off)
            g_both = float(fit.gain(fit.residual_with([shat["audio"], shat["motion"]])))
            g_mot = float(fit.gain(fit.residual_with([shat["motion"]])))
            return g_both - g_mot

        floor = float(np.mean([aud_mot(ts) for ts in ctx.corpus.timeshifts]))
        curve = [[d * 40, (aud_mot({ls: d for ls in order} if d else None) - floor) * 1e4] for d in OFFSETS]
        CURVES[LABELS[len(CURVES)]] = curve
        print(f"    swept {LABELS[len(CURVES) - 1]}: net@0 {dict((int(p[0]), p[1]) for p in curve)[0]:+.2f}, "
              f"peak {max(c[1] for c in curve):+.2f} at {max(curve, key=lambda c: c[1])[0]:+d} ms", flush=True)
    return _ocap(per, ops, ctx, timeshifts)
m.capture_energies = _cap

m.report_matrix = lambda *a, **k: None                    # only the leaderboard-capture sweep is wanted

if __name__ == "__main__":
    m.main()
    json.dump(CURVES, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}: {list(CURVES)}")
