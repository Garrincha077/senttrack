# COT TFF v2 frozen-formula validation

**Verdict: NOT_SUPPORTED_FOR_COMPOSITE**

CFTC source: CFTC annual TFF ZIP archives  
Price source: Yahoo Finance SPY adjusted close  
Valid observations: 883 (2009-09-01 to 2026-07-28)

|Horizon|N|rho|p|Low 0-20|High 80-100|Difference|Check|
|---|---:|---:|---:|---:|---:|---:|:---:|
|1W|882|0.0348|0.3019|0.09%|1.48%|-1.39%|FAIL|
|4W|879|0.0306|0.3648|0.30%|2.76%|-2.47%|FAIL|
|8W|875|0.0340|0.3154|-0.72%|4.21%|-4.93%|FAIL|
|13W|870|0.0800|0.0183|-2.33%|4.55%|-6.88%|FAIL|

## Guardrails
- Formula was frozen before the run; no weights or thresholds were tuned.
- COT dated Tuesday was aligned to the first SPY close on/after Friday.
- 13874A and later 13874+ consolidated rows are separately identifiable in output.
- This validates a positioning filter, not a standalone trading strategy.
