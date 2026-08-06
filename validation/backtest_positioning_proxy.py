from __future__ import annotations

import io
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "validation-output" / "cot_v2_weekly_scores.csv"
OUT = ROOT / "validation-output"
NAAIM_URL = "https://www.naaim.org/programs/naaim-exposure-index/"

RIA_OBSERVATIONS = [
    ("2026-01-10", 82.92, "https://realinvestmentadvice.com/resources/blog/market-outlook-for-2026-copy/"),
    ("2026-02-14", 72.14, "https://realinvestmentadvice.com/resources/blog/the-weak-dollar-narrative/"),
    ("2026-02-28", 72.80, "https://realinvestmentadvice.com/resources/blog/market-topping-process/"),
    ("2026-04-18", 74.37, "https://realinvestmentadvice.com/resources/blog/short-covering-rally-or-correction-over/"),
    ("2026-05-02", 76.11, "https://realinvestmentadvice.com/resources/blog/sp-earnings-record-may-be-a-warning/"),
    ("2026-06-13", 74.15, "https://realinvestmentadvice.com/resources/blog/may-inflation-print-why-the-4-2-headline-is-an-oil-story/"),
    ("2026-07-25", 69.05, "https://realinvestmentadvice.com/resources/blog/the-ai-capex-bill-comes-due/"),
]


@dataclass
class Metrics:
    n: int
    mae: float | None
    rmse: float | None
    pearson: float | None
    spearman: float | None
    direction_accuracy: float | None
    regime_accuracy: float | None
    bias: float | None


def finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return finite_or_none(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def safe_corr(x: pd.Series, y: pd.Series, kind: str) -> float | None:
    z = pd.concat([x, y], axis=1).dropna()
    if len(z) < 3 or z.iloc[:, 0].nunique() < 2 or z.iloc[:, 1].nunique() < 2:
        return None
    try:
        result = pearsonr(z.iloc[:, 0], z.iloc[:, 1]).statistic if kind == "pearson" else spearmanr(z.iloc[:, 0], z.iloc[:, 1]).statistic
    except Exception:  # noqa: BLE001
        return None
    return finite_or_none(result)


def regime(s: pd.Series) -> pd.Series:
    return pd.cut(s, [-np.inf, 35, 65, np.inf], labels=["LOW", "MID", "HIGH"], right=False).astype("string")


def calc_metrics(actual: pd.Series, pred: pd.Series) -> Metrics:
    z = pd.DataFrame({"actual": actual, "pred": pred}).dropna()
    if z.empty:
        return Metrics(0, None, None, None, None, None, None, None)
    err = z["pred"] - z["actual"]
    a_delta = z["actual"].diff()
    p_delta = z["pred"].diff()
    dmask = a_delta.notna() & p_delta.notna() & (a_delta != 0)
    direction = float((np.sign(a_delta[dmask]) == np.sign(p_delta[dmask])).mean()) if dmask.any() else None
    return Metrics(
        n=int(len(z)),
        mae=finite_or_none(err.abs().mean()),
        rmse=finite_or_none(np.sqrt(np.mean(err**2))),
        pearson=safe_corr(z["actual"], z["pred"], "pearson"),
        spearman=safe_corr(z["actual"], z["pred"], "spearman"),
        direction_accuracy=finite_or_none(direction),
        regime_accuracy=finite_or_none((regime(z["actual"]) == regime(z["pred"])).mean()),
        bias=finite_or_none(err.mean()),
    )


def naaim_score(raw: pd.Series) -> pd.Series:
    x = pd.to_numeric(raw, errors="coerce").astype(float)
    return pd.Series(
        np.interp(x, [-20.0, 0.0, 50.0, 100.0, 150.0], [0.0, 20.0, 50.0, 80.0, 100.0]),
        index=raw.index,
    ).clip(0, 100)


def rolling_percentile(series: pd.Series, window: int = 156) -> pd.Series:
    def pct(a: np.ndarray) -> float:
        if len(a) != window or np.isnan(a).any():
            return np.nan
        return float(100.0 * np.count_nonzero(a <= a[-1]) / window)

    return series.rolling(window=window, min_periods=window).apply(pct, raw=True)


def normalize_table(df: pd.DataFrame) -> pd.DataFrame | None:
    d = df.copy()
    d.columns = [" ".join(map(str, c)).strip() if isinstance(c, tuple) else str(c).strip() for c in d.columns]
    lower = {c: re.sub(r"\s+", " ", c.lower()) for c in d.columns}
    date_cols = [c for c, lc in lower.items() if "date" in lc]
    value_cols = [c for c, lc in lower.items() if any(k in lc for k in ["average", "mean", "exposure index", "naaim exposure", "index value"]) and not any(k in lc for k in ["most", "minimum", "maximum", "median", "quartile"])]
    if not date_cols:
        return None
    date_col = date_cols[0]
    if value_cols:
        value_col = value_cols[0]
    else:
        candidates = []
        for c in d.columns:
            if c == date_col:
                continue
            converted = pd.to_numeric(d[c].astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False), errors="coerce")
            if converted.notna().mean() >= 0.5:
                candidates.append((c, int(converted.notna().sum())))
        if not candidates:
            return None
        value_col = max(candidates, key=lambda item: item[1])[0]
    out = pd.DataFrame({
        "naaim_date": pd.to_datetime(d[date_col], errors="coerce"),
        "naaim_raw": pd.to_numeric(d[value_col].astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False), errors="coerce"),
    }).dropna()
    out = out[(out["naaim_raw"] >= -100) & (out["naaim_raw"] <= 250)]
    if len(out) < 20:
        return None
    return out.drop_duplicates("naaim_date").sort_values("naaim_date")


def fetch_naaim_history() -> tuple[pd.DataFrame | None, list[str]]:
    notes: list[str] = []
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ContrarianGreedResearch/1.0)"}
    try:
        response = requests.get(NAAIM_URL, headers=headers, timeout=45)
        notes.append(f"official_page_status={response.status_code}")
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        notes.append(f"official_page_error={type(exc).__name__}: {exc}")
        return None, notes

    candidates: list[pd.DataFrame] = []
    try:
        for table in pd.read_html(io.StringIO(response.text)):
            normalized = normalize_table(table)
            if normalized is not None:
                candidates.append(normalized)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"html_table_error={type(exc).__name__}: {exc}")

    try:
        soup = BeautifulSoup(response.text, "html.parser")
        links = [requests.compat.urljoin(NAAIM_URL, a["href"]) for a in soup.find_all("a", href=True)]
    except Exception as exc:  # noqa: BLE001
        notes.append(f"html_parse_error={type(exc).__name__}: {exc}")
        links = []

    for href in links:
        if not re.search(r"\.(csv|xlsx?|xls)(?:\?|$)", href, re.I):
            continue
        try:
            linked = requests.get(href, headers=headers, timeout=45)
            notes.append(f"linked_file={href} status={linked.status_code}")
            linked.raise_for_status()
            frames = [pd.read_csv(io.BytesIO(linked.content))] if re.search(r"\.csv(?:\?|$)", href, re.I) else [pd.read_excel(io.BytesIO(linked.content), sheet_name=None)]
            expanded: list[pd.DataFrame] = []
            for frame in frames:
                expanded.extend(frame.values() if isinstance(frame, dict) else [frame])
            for frame in expanded:
                normalized = normalize_table(frame)
                if normalized is not None:
                    candidates.append(normalized)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"linked_file_error={href} {type(exc).__name__}: {exc}")

    if not candidates:
        notes.append("no_official_history_table_parsed")
        return None, notes
    best = max(candidates, key=len)
    notes.append(f"selected_rows={len(best)} start={best.naaim_date.min().date()} end={best.naaim_date.max().date()}")
    return best, notes


def asof(left: pd.DataFrame, right: pd.DataFrame, left_on: str, right_on: str, direction: str = "backward", tolerance: pd.Timedelta | None = None) -> pd.DataFrame:
    l = left.copy().sort_values(left_on)
    r = right.copy().sort_values(right_on)
    l[left_on] = pd.to_datetime(l[left_on])
    r[right_on] = pd.to_datetime(r[right_on])
    return pd.merge_asof(l, r, left_on=left_on, right_on=right_on, direction=direction, tolerance=tolerance, allow_exact_matches=True)


def fit_linear(train_x: pd.Series, train_y: pd.Series, test_x: pd.Series) -> pd.Series:
    z = pd.DataFrame({"x": train_x, "y": train_y}).dropna()
    if len(z) < 30 or z["x"].nunique() < 2:
        return pd.Series(np.nan, index=test_x.index)
    slope, intercept = np.polyfit(z["x"].to_numpy(), z["y"].to_numpy(), 1)
    return (intercept + slope * test_x).clip(0, 100)


def walk_forward(aligned: pd.DataFrame, train_weeks: int = 104, test_weeks: int = 26) -> pd.DataFrame:
    x = aligned.dropna(subset=["naaim_score", "cftc_score"]).reset_index(drop=True).copy()
    rows: list[pd.DataFrame] = []
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
        test["pred_linear_inverted"] = fit_linear(100 - train["cftc_score"], train["naaim_score"], 100 - test["cftc_score"])
        rows.append(test)
        start = stop
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def market_audit(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    rows = []
    for horizon in ["fwd_1W", "fwd_4W", "fwd_8W", "fwd_13W"]:
        z = df[[score_col, horizon]].dropna()
        if len(z) < 20:
            continue
        rho = safe_corr(z[score_col], z[horizon], "spearman")
        q20, q80 = z[score_col].quantile([0.2, 0.8])
        low = z.loc[z[score_col] <= q20, horizon]
        high = z.loc[z[score_col] >= q80, horizon]
        low_mean, high_mean = finite_or_none(low.mean()), finite_or_none(high.mean())
        rows.append({
            "score": score_col,
            "horizon": horizon,
            "n": len(z),
            "spearman": rho,
            "low_quintile_mean": low_mean,
            "high_quintile_mean": high_mean,
            "low_minus_high": finite_or_none((low_mean or 0) - (high_mean or 0)),
            "contrarian_pass": bool(rho is not None and rho < 0 and low_mean is not None and high_mean is not None and low_mean > high_mean),
        })
    return pd.DataFrame(rows)


def subperiod_audit(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    periods = [("2009-2014", "2009-01-01", "2014-12-31"), ("2015-2019", "2015-01-01", "2019-12-31"), ("2020-2026", "2020-01-01", "2026-12-31")]
    rows = []
    for label, start, end in periods:
        sub = df[(df["available"] >= pd.Timestamp(start)) & (df["available"] <= pd.Timestamp(end))]
        for horizon in ["fwd_4W", "fwd_8W", "fwd_13W"]:
            z = sub[[score_col, horizon]].dropna()
            rows.append({"period": label, "score": score_col, "horizon": horizon, "n": len(z), "spearman": safe_corr(z[score_col], z[horizon], "spearman")})
    return pd.DataFrame(rows)


def pilot_ria(cftc: pd.DataFrame, naaim: pd.DataFrame | None) -> tuple[pd.DataFrame, dict[str, dict]]:
    ria = pd.DataFrame(RIA_OBSERVATIONS, columns=["ria_date", "ria_score", "ria_url"])
    ria["ria_date"] = pd.to_datetime(ria["ria_date"])
    out = asof(ria, cftc[["available", "cftc_score"]].dropna(), "ria_date", "available")
    if naaim is not None:
        n = naaim.copy()
        n["naaim_score"] = naaim_score(n["naaim_raw"])
        out = asof(out, n, "ria_date", "naaim_date", direction="nearest", tolerance=pd.Timedelta(days=7))
    else:
        out["naaim_date"] = pd.NaT
        out["naaim_raw"] = np.nan
        out["naaim_score"] = np.nan
    for weight in [0.25, 0.50, 0.75]:
        out[f"proxy_ria_{int(weight * 100)}"] = weight * out["ria_score"] + (1 - weight) * out["cftc_score"]
    metrics = {column: asdict(calc_metrics(out["naaim_score"], out[column])) for column in ["proxy_ria_25", "proxy_ria_50", "proxy_ria_75"]}
    return out, metrics


def main() -> None:
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(INPUT, parse_dates=["date", "available", "price_date"])
    df = df[df["code"].astype(str).eq("13874A")].sort_values("date").drop_duplicates("date").copy()
    if len(df) < 156:
        raise RuntimeError(f"Insufficient exact 13874A history: {len(df)} rows")
    df["cftc_score"] = rolling_percentile(pd.to_numeric(df["asset_net"], errors="coerce"), 156)
    df["cftc_score_inverted"] = 100 - df["cftc_score"]

    market = pd.concat([market_audit(df, "cftc_score"), market_audit(df, "cftc_score_inverted")], ignore_index=True)
    subperiod = pd.concat([subperiod_audit(df, "cftc_score"), subperiod_audit(df, "cftc_score_inverted")], ignore_index=True)

    naaim, naaim_notes = fetch_naaim_history()
    aligned = pd.DataFrame()
    wf = pd.DataFrame()
    full_metrics: dict[str, dict] = {}
    wf_metrics: dict[str, dict] = {}
    if naaim is not None:
        naaim = naaim.copy()
        naaim["naaim_score"] = naaim_score(naaim["naaim_raw"])
        aligned = asof(naaim, df[["available", "date", "cftc_score", "cftc_score_inverted", "asset_net"]].dropna(subset=["cftc_score"]), "naaim_date", "available")
        aligned["cftc_age_days"] = (aligned["naaim_date"] - aligned["available"]).dt.days
        aligned = aligned[(aligned["cftc_age_days"] >= 0) & (aligned["cftc_age_days"] <= 14)].copy()
        aligned["pred_raw"] = aligned["cftc_score"]
        aligned["pred_inverted"] = aligned["cftc_score_inverted"]
        full_metrics = {"raw": asdict(calc_metrics(aligned["naaim_score"], aligned["pred_raw"])), "inverted": asdict(calc_metrics(aligned["naaim_score"], aligned["pred_inverted"]))}
        wf = walk_forward(aligned)
        if not wf.empty:
            wf_metrics = {column: asdict(calc_metrics(wf["naaim_score"], wf[column])) for column in ["pred_raw", "pred_inverted", "pred_linear", "pred_linear_inverted"]}

    ria_pilot, ria_metrics = pilot_ria(df, naaim)
    eligible = {k: v for k, v in wf_metrics.items() if v.get("n") and v.get("mae") is not None and v.get("spearman") is not None}
    champion = min(eligible, key=lambda key: eligible[key]["mae"]) if eligible else None

    summary = json_safe({
        "version": "POSITIONING-PROXY-BACKTEST-v1",
        "production_impact": "NONE",
        "cftc_exact_code": "13874A",
        "cftc_rows": len(df),
        "cftc_start": str(df["date"].min().date()),
        "cftc_end": str(df["date"].max().date()),
        "cftc_valid_156w_scores": int(df["cftc_score"].notna().sum()),
        "naaim_fetch_notes": naaim_notes,
        "naaim_rows": len(naaim) if naaim is not None else 0,
        "aligned_rows": len(aligned),
        "full_sample_replication_metrics": full_metrics,
        "walk_forward_design": {"train_weeks": 104, "test_weeks": 26, "no_lookahead": True},
        "walk_forward_metrics": wf_metrics,
        "walk_forward_champion_by_mae": champion,
        "ria_pilot_n": len(ria_pilot),
        "ria_pilot_metrics": ria_metrics,
        "activation_gate": "BLOCKED unless exact data, sufficient RIA overlap, and stable out-of-sample evidence support a new canonical model version",
    })

    df.to_csv(OUT / "positioning_proxy_cftc_weekly.csv", index=False)
    market.to_csv(OUT / "positioning_proxy_market_audit.csv", index=False)
    subperiod.to_csv(OUT / "positioning_proxy_subperiod_audit.csv", index=False)
    ria_pilot.to_csv(OUT / "positioning_proxy_ria_pilot.csv", index=False)
    if naaim is not None:
        naaim.to_csv(OUT / "positioning_proxy_naaim_official.csv", index=False)
    if not aligned.empty:
        aligned.to_csv(OUT / "positioning_proxy_aligned.csv", index=False)
    if not wf.empty:
        wf.to_csv(OUT / "positioning_proxy_walk_forward.csv", index=False)
    (OUT / "positioning_proxy_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")

    report = [
        "# Positioning Proxy Backtest v1",
        "",
        "Research only. No production weight, Composite, history, or paper state was changed.",
        "",
        f"- Exact CFTC contract: 13874A",
        f"- CFTC weekly rows: {len(df)} ({df['date'].min().date()} to {df['date'].max().date()})",
        f"- Valid rolling 156-week Asset Manager percentile scores: {df['cftc_score'].notna().sum()}",
        f"- Official NAAIM rows parsed: {len(naaim) if naaim is not None else 0}",
        f"- Aligned NAAIM/CFTC rows: {len(aligned)}",
        f"- Walk-forward champion by OOS MAE: {champion or 'NONE'}",
        "",
        "## Controls",
        "- CFTC is aligned by publication availability; no future row is used.",
        "- NAAIM raw exposure uses the locked v1.0 anchor curve.",
        "- Walk-forward uses 104 prior observations and 26 unseen observations.",
        "- Raw, inverted, linear-calibrated and inverted-linear variants are compared.",
        "- RIA blends remain a seven-observation pilot and cannot select production weights.",
        "",
        "## Decision gate",
        "No production adoption without complete exact data, materially larger RIA overlap and stable out-of-sample evidence.",
    ]
    (OUT / "positioning_proxy_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
