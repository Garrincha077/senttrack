# Positioning Proxy Backtest v1

Research only. No production weight, Composite, history, or paper state was changed.

- Exact CFTC contract: 13874A
- CFTC weekly rows: 1048 (2006-06-13 to 2026-07-07)
- Valid rolling 156-week Asset Manager percentile scores: 893
- Official NAAIM rows parsed: 0
- Aligned NAAIM/CFTC rows: 0
- Walk-forward champion by OOS MAE: NONE

## Controls
- CFTC is aligned by publication availability; no future row is used.
- NAAIM raw exposure uses the locked v1.0 anchor curve.
- Walk-forward uses 104 prior observations and 26 unseen observations.
- Raw, inverted, linear-calibrated and inverted-linear variants are compared.
- RIA blends remain a seven-observation pilot and cannot select production weights.

## Decision gate
No production adoption without complete exact data, materially larger RIA overlap and stable out-of-sample evidence.
