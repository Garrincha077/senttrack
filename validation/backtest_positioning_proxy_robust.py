from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from backtest_positioning_proxy import (
    RIA_OBSERVATIONS,
    asof,
    calc_metrics,
    fit_linear,
    json_safe,
    naaim_score,
    rolling_percentile,
    safe_corr,
)

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "validation-output" / "cot_v2_weekly_scores.csv"
OUT = ROOT / "validation-output"
NAAIM_TABLE_URL = "https://index.naaim.org/embeddable/table"
SEED = 20260806
BOOTSTRAP_REPS = 5000


def fetch_official_naaim() -> pd.DataFrame:
    response = requests.get(
        NAAIM_TABLE_URL,
        headers={"User-Agent": "Mozilla/5.0 (compatible; ContrarianGreedResearch/1.0)"},
        timeout=45,
    )
    response.raise_for_status()
    tables = pd.read_html(io.StringIO(response.text))
    for table in tables:
        columns = {str(c).strip().lower(): c for c in table.columns}
        date_col = next((original for key, original in columns.items() if key == "date"), None)
        value_col = next((original for key, original in columns.items() if "naaim number" in key), None)
        if date_col is None or value_col is None:
            continue
        result = pd.DataFrame(
            {
                "naaim_date": pd.to_datetime(table[date_col], errors="coerce"),
                "naaim_raw": pd.to_numeric(table[value_col], errors="coerce"),
            }
        ).dropna()
        result = result[(result["naaim_raw"] >= -100) & (result["naaim_raw"] <= 250)]
        result = result.drop_duplicates("naaim_date").sort_values("naaim_date").reset_index(drop=True)
        if len(result) >= 100:
            result["naaim_score"] = naaim_score(result["naaim_raw"])
            return result
    raise RuntimeError("Official NAAIM embeddable table was reachable but no valid Date/NAAIM Number table was parsed")


def circular_block_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=blocks)
    pieces = [(start + np.arange(block_length)) % n for start in starts]
    return np.concatenate(pieces)[:n]


def block_bootstrap_spread(z: pd.DataFrame, score_col: str, return_col: str, block_length: int) -> dict:
    z = z[[score_col, return_col]].dropna().reset_index(drop=True)
    q20 = float(z[score_col].quantile(0.2))
    q80 = float(z[score_col].quantile(0.8))
    low = z.loc[z[score_col] <= q20, return_col]
    high = z.loc[z[score_col] >= q80, return_col]
    observed = float(low.mean() - high.mean())
    rng = np.random.default_rng(SEED + block_length)
    samples = []
    for _ in range(BOOTSTRAP_REPS):
        boot = z.iloc[circular_block_indices(len(z), block_length, rng)]
        boot_low = boot.loc[boot[score_col] <= q20, return_col]
        boot_high = boot.loc[boot[score_col] >= q80, return_col]
        if not boot_low.empty and not boot_high.empty:
            samples.append(float(boot_low.mean() - boot_high.mean()))
    arr = np.asarray(samples, dtype=float)
    return {
        "n": int(len(z)),
        "block_length": int(block_length),
        "bootstrap_reps": int(len(arr)),
        "q20": q20,
        "q80": q80,
        "low_n": int(len(low)),
        "high_n": int(len(high)),
        "low_mean": float(low.mean()),
        "high_mean": float(high.mean()),
        "low_minus_high": observed,
        "ci_2_5": float(np.quantile(arr, 0.025)),
        "ci_97_5": float(np.quantile(arr, 0.975)),
        "probability_positive": float(np.mean(arr > 0)),
        "one_sided_p_le_zero": float(np.mean(arr <= 0)),
        "spearman": safe_corr(z[score_col], z[return_col], "spearman"),
    }


def threshold_audit(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    rows = []
    for horizon in ["fwd_1W", "fwd_4W", "fwd_8W", "fwd_13W"]:
        z = df[[score_col, horizon]].dropna()
        for label, mask in [("LOW_LE_20", z[score_col] <= 20), ("HIGH_GE_80", z[score_col] >= 80)]:
            sample = z.loc[mask, horizon]
            rows.append(
                {
                    "score": score_col,
                    "horizon": horizon,
                    "signal": label,
                    "n": int(len(sample)),
                    "mean_return": float(sample.mean()) if len(sample) else None,
                    "median_return": float(sample.median()) if len(sample) else None,
                    "positive_rate": float((sample > 0).mean()) if len(sample) else None,
                }
            )
    return pd.DataFrame(rows)


def walk_forward(aligned: pd.DataFrame, train_weeks: int = 104, test_weeks: int = 26) -> pd.DataFrame:
    x = aligned.dropna(subset=["naaim_score", "cftc_score"]).reset_index(drop=True).copy()
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
        test["pred_raw"] = test["cftc_score"]
        test["pred_inverted"] = 100 - test["cftc_score"]
        test["pred_linear"] = fit_linear(train["cftc_score"], train["naaim_score"], test["cftc_score"])
        test["pred_linear_inverted"] = fit_linear(
            100 - train["cftc_score"],
            train["naaim_score"],
            100 - test["cftc_score"],
        )
        test["train_start"] = train["naaim_date"].min()
        test["train_end"] = train["naaim_date"].max()
        rows.append(test)
        start = stop
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def ria_pilot(cftc: pd.DataFrame, naaim: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    ria = pd.DataFrame(RIA_OBSERVATIONS, columns=["ria_date", "ria_score", "ria_url"])
    ria["ria_date"] = pd.to_datetime(ria["ria_date"])
    joined = asof(ria, cftc[["available", "cftc_score"]].dropna(), "ria_date", "available")
    joined = asof(
        joined,
        naaim[["naaim_date", "naaim_raw", "naaim_score"]],
        "ria_date",
        "naaim_date",
        direction="nearest",
        tolerance=pd.Timedelta(days=7),
    )
    for weight in [0.25, 0.50, 0.75]:
        joined[f"proxy_ria_{int(weight * 100)}"] = weight * joined["ria_score"] + (1 - weight) * joined["cftc_score"]
    metrics = {
        column: calc_metrics(joined["naaim_score"], joined[column]).__dict__
        for column in ["proxy_ria_25", "proxy_ria_50", "proxy_ria_75"]
    }
    return joined, metrics


def main() -> None:
    OUT.mkdir(exist_ok=True)
    cftc = pd.read_csv(INPUT, parse_dates=["date", "available", "price_date"])
    cftc = cftc[cftc["code"].astype(str).eq("13874A")].sort_values("date").drop_duplicates("date").copy()
    cftc["cftc_score"] = rolling_percentile(pd.to_numeric(cftc["asset_net"], errors="coerce"), 156)
    cftc["cftc_score_inverted"] = 100 - cftc["cftc_score"]
    valid = cftc.dropna(subset=["cftc_score"]).copy()

    bootstrap_rows = []
    blocks = {"fwd_1W": 1, "fwd_4W": 4, "fwd_8W": 8, "fwd_13W": 13}
    for horizon, block_length in blocks.items():
        result = block_bootstrap_spread(valid, "cftc_score", horizon, block_length)
        result["horizon"] = horizon
        bootstrap_rows.append(result)
    bootstrap = pd.DataFrame(bootstrap_rows)
    thresholds = threshold_audit(valid, "cftc_score")

    naaim = fetch_official_naaim()
    aligned = asof(
        naaim,
        valid[["available", "date", "cftc_score", "cftc_score_inverted", "asset_net"]],
        "naaim_date",
        "available",
    )
    aligned["cftc_age_days"] = (aligned["naaim_date"] - aligned["available"]).dt.days
    aligned = aligned[(aligned["cftc_age_days"] >= 0) & (aligned["cftc_age_days"] <= 14)].copy()
    aligned["pred_raw"] = aligned["cftc_score"]
    aligned["pred_inverted"] = aligned["cftc_score_inverted"]

    full_metrics = {
        "raw": calc_metrics(aligned["naaim_score"], aligned["pred_raw"]).__dict__,
        "inverted": calc_metrics(aligned["naaim_score"], aligned["pred_inverted"]).__dict__,
    }
    wf = walk_forward(aligned)
    wf_metrics = {
        column: calc_metrics(wf["naaim_score"], wf[column]).__dict__
        for column in ["pred_raw", "pred_inverted", "pred_linear", "pred_linear_inverted"]
    }
    eligible = {key: value for key, value in wf_metrics.items() if value["n"] and value["mae"] is not None}
    champion = min(eligible, key=lambda key: eligible[key]["mae"]) if eligible else None

    ria, ria_metrics = ria_pilot(valid, naaim)
    significant_horizons = bootstrap[(bootstrap["ci_2_5"] > 0) & (bootstrap["probability_positive"] >= 0.95)]["horizon"].tolist()
    stable_market_edge = len(significant_horizons) >= 2
    oos_usable = champion is not None and wf_metrics[champion]["n"] >= 20

    summary = json_safe(
        {
            "version": "POSITIONING-PROXY-BACKTEST-v2-ROBUST",
            "production_impact": "NONE",
            "exact_contract": "13874A",
            "cftc_rows": len(cftc),
            "valid_156w_scores": len(valid),
            "cftc_start": cftc["date"].min(),
            "cftc_end": cftc["date"].max(),
            "naaim_source": NAAIM_TABLE_URL,
            "naaim_rows": len(naaim),
            "naaim_start": naaim["naaim_date"].min(),
            "naaim_end": naaim["naaim_date"].max(),
            "aligned_rows": len(aligned),
            "full_sample_naaim_replication": full_metrics,
            "walk_forward_design": {"train_weeks": 104, "test_weeks": 26, "no_lookahead": True},
            "walk_forward_rows": len(wf),
            "walk_forward_folds": int(wf["fold"].nunique()) if not wf.empty else 0,
            "walk_forward_metrics": wf_metrics,
            "walk_forward_champion_by_mae": champion,
            "market_block_bootstrap": bootstrap.to_dict(orient="records"),
            "significant_contrarian_horizons": significant_horizons,
            "stable_market_edge": stable_market_edge,
            "ria_pilot_n": len(ria),
            "ria_pilot_metrics": ria_metrics,
            "decision": "KEEP_RESEARCH_ONLY",
            "decision_reason": "CFTC market edge and NAAIM replication are evaluated, but RIA overlap remains only seven observations and cannot justify production weight selection.",
            "activation_gate": "A new canonical version is required; v1.0 production history and paper state remain unchanged.",
        }
    )

    naaim.to_csv(OUT / "positioning_proxy_robust_naaim.csv", index=False)
    aligned.to_csv(OUT / "positioning_proxy_robust_aligned.csv", index=False)
    wf.to_csv(OUT / "positioning_proxy_robust_walk_forward.csv", index=False)
    bootstrap.to_csv(OUT / "positioning_proxy_robust_bootstrap.csv", index=False)
    thresholds.to_csv(OUT / "positioning_proxy_robust_thresholds.csv", index=False)
    ria.to_csv(OUT / "positioning_proxy_robust_ria.csv", index=False)
    (OUT / "positioning_proxy_robust_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
