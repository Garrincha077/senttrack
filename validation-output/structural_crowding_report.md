# Structural/Crowding v1.2 — research backtest

Window: **2024-07-05 → 2026-04-30**, weekly snapshots: **96**.

Bucket Coverage: mean **100.0%**, min **100.0%**.

## Forward SPY price-return diagnostics

| Horizon | N | Spearman rho | p | Low Q mean | High Q mean | Low−High | 95% block-bootstrap CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1W | 96 | -0.073 | 0.482 | 0.02% | 0.06% | -0.04% | [-0.99%, +1.25%] |
| 4W | 96 | -0.132 | 0.200 | 2.17% | 0.88% | +1.29% | [-2.47%, +4.62%] |
| 8W | 96 | -0.349 | 0.000 | 7.10% | -0.03% | +7.12% | [+2.35%, +10.97%] |
| 13W | 96 | -0.392 | 0.000 | 10.28% | 1.62% | +8.65% | [+3.26%, +12.66%] |

Contrarian expectation: higher Structural/Crowding should be associated with weaker subsequent SPY returns (negative rho; positive Low−High spread).

No Structural/Crowding threshold is treated as a trading rule. Production Composite/history/paper state are untouched.
