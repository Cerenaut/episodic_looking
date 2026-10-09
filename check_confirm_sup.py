"""Continual confirmation criterion for the supervised CLS/STM (as runs_v2_pilot_confirm/check_confirm.py for the heads):
the confirmation run's value of the selection curve at the budget (the last epoch of each phase, averaged over phases
and orders) against the selection run's smoothed value at the budget; passes within 0.03. Validation files only.
Usage (code repo root): python check_confirm_sup.py <tree, e.g. sup_lr0.01> <budget> [model: sup (default) or sup2]"""
import sys

import metrics_v2 as mv

tree, b = sys.argv[1], int(sys.argv[2])
model = sys.argv[3] if len(sys.argv) > 3 else "sup"
sel = mv.validation_curve(f"runs_v2_pilot_{model}/{tree}", model, "continual", None, "pair0_1")
sel_s = sel.rolling(mv.SMOOTH_POINTS, center=True, min_periods=1).mean()
con = mv.validation_curve(f"runs_v2_pilot_{model}_confirm/{tree}", model, "continual", None, "pair0_1")
if con is None:
    sys.exit("confirmation runs missing or unfinished")
assert int(con.index.max()) == b, (tree, con.index.max())
con_s = con.rolling(mv.SMOOTH_POINTS, center=True, min_periods=1).mean()
s, r, cs = float(sel_s.loc[b]), float(con.loc[b]), float(con_s.loc[b])
print("model & budget & selection (smoothed) & confirmation (raw) & confirmation (smoothed) & difference (raw) & pass")
print(f"{tree} & {b} & {s:.3f} & {r:.3f} & {cs:.3f} & {r - s:+.3f} & {'yes' if abs(r - s) <= 0.03 else 'NO'}")
