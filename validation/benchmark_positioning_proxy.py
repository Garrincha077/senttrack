from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_positioning_proxy import calc_metrics, fit_linear, json_safe

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation-output"
ALIGNED = OUT / "positioning_proxy_robust_aligned.csv"


def walk_forward_with_benchmarks(aligned: pd.DataFrame, train_weeks: int = 104, test_weeks: int = 26) -> pd.DataFrame:
    x = aligned.dropna(subset=["naaim_score", "cftc_score"]).sort_values("naaim_date").reset_index(drop=True).copy()
    x["pred_naaim_lag1"] = x["naaim_score"].shift(1)
    x["pred_naaim_lag4"] = x["naaim_score"].shift(1).rolling(4).mean()
    rows = []
    start = train_weeks
    fold = 0
    while start < len(x):
        stop = min(start + test_weeks, len(x))
        train = x.iloc[start - train_weeks : start]
        test = x.iloc[start:stop].copy()
        if test.empty:
            break
        fold += 1
        test["fold"] = fold
        test["pred_cftc_raw"] = test["cftc_score"]
        test["pred_cftc_linear"] = fit_linear(train["cftc_score"], train["naaim_score"], test["cftc_score"])
        test["pred_train_mean"] = float(train["naaim_score"].mean())
        test["pred_train_median"] = float(train["naaim_score"].median())
        test["train_start"] = train["naaim_date"].min()
        test["train_end"] = train["naaim_date"].max()
        rows.append(test)
        start = stop
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> None:
    aligned = pd.read_csv(ALIGNED, parse_dates=["naaim_date", "available", "date"])
    wf = walk_forward_with_benchmarks(aligned)
    columns = [
        "pred_cftc_raw",
        "pred_cftc_linear",
        "pred_train_mean",
        "pred_train_median",
        "pred_naaim_lag1",
        "pred_naaim_lag4",
    ]
    metrics = {column: calc_metrics(wf["naaim_score"], wf[column]).__dict__ for column in columns}
    eligible = {key: value for key, value in metrics.items() if value["n"] and value["mae"] is not None}
    winner = min(eligible, key=lambda key: eligible[key]["mae"]) if eligible else None
    cftc_mae = metrics["pred_cftc_linear"]["mae"]
    median_mae = metrics["pred_train_median"]["mae"]
    lag1_mae = metrics["pred_naaim_lag1"]["mae"]
    cftc_beats_median = bool(cftc_mae is not None and median_mae is not None and cftc_mae < median_mae)
    cftc_beats_lag1 = bool(cftc_mae is not None and lag1_mae is not None and cftc_mae < lag1_mae)
    summary = json_safe(
        {
            "version": "POSITIONING-PROXY-BENCHMARK-v1",
            "production_impact": "NONE",
            "walk_forward_rows": len(wf),
            "walk_forward_folds": int(wf["fold"].nunique()) if not wf.empty else 0,
            "metrics": metrics,
            "winner_by_mae": winner,
            "cftc_linear_beats_train_median": cftc_beats_median,
            "cftc_linear_beats_naaim_lag1": cftc_beats_lag1,
            "replication_gate": "FAIL" if not (cftc_beats_median and cftc_beats_lag1) else "PASS",
            "conclusion": "The calibrated CFTC score does not add enough out-of-sample NAAIM replication value versus simple benchmarks." if not (cftc_beats_median and cftc_beats_lag1) else "The calibrated CFTC score beats both simple benchmarks in this out-of-sample window.",
        }
    )
    wf.to_csv(OUT / "positioning_proxy_benchmark_walk_forward.csv", index=False)
    (OUT / "positioning_proxy_benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
