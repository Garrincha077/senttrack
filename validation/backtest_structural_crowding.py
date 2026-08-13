#!/usr/bin/env python3
"""Research-only backtest of Contrarian Greed v1.2 Structural/Crowding bucket.

The production Composite/history is not modified. The test uses the canonical v1.2
Structural/Crowding definition (nominal 50% of the model) and evaluates whether
higher structural greed/crowding predicts weaker subsequent SPY price returns.

Point-in-time rules used here:
- NAAIM: official weekly survey date, available from the next U.S. trading session.
- AAII: official sentiment.xls week-ending date, available from the next U.S. trading session.
- HY OAS: FRED observation shifted to the next U.S. trading session, matching the
  existing audited Contrarian Greed backtest convention.
- Market-derived leaves: same completed session close, exact 20-session windows.

The public NAAIM table is currently three months delayed, so the sample deliberately
ends at the last official row exposed by the public table. No proxy, interpolation,
or future NAAIM value is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
import json
import math
import re
import sys

import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr
import yfinance as yf

OUT = Path("validation-output")
OUT.mkdir(parents=True, exist_ok=True)

NAAIM_URL = "https://index.naaim.org/embeddable/table"
AAII_URL = "https://www.aaii.com/files/surveys/sentiment.xls"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAMLH0A0HYM2"
SYMBOLS = [
    "SPY", "RSP", "IWM", "XLY", "XLI", "XLF", "XLK", "XLP", "XLU", "XLV",
    "SPHB", "SPLV", "HYG", "IEF", "UUP",
]
START = "2024-06-01"  # warmup before first public-NAAIM snapshot used
END = "2026-08-15"    # enough room for 13W forward returns after Apr-2026 snapshots
HORIZONS = {"1W": 5, "4W": 20, "8W": 40, "13W": 65}
WEIGHTS = {
    "NAAIM": 0.12,
    "AAII": 0.09,
    "RSP_SPY": 0.025,
    "OFF_DEF": 0.025,
    "SPHB_SPLV": 0.025,
    "IWM_SPY": 0.05,
    "HY_OAS": 0.075,
    "HYG_IEF": 0.04,
    "UUP": 0.05,
}
assert abs(sum(WEIGHTS.values()) - 0.50) < 1e-12


def piecewise(x: float, anchors: list[tuple[float, float]], low_cap=None, high_cap=None) -> float:
    if pd.isna(x):
        return np.nan
    anchors = sorted(anchors)
    if x <= anchors[0][0]:
        return anchors[0][1] if low_cap is None else low_cap
    if x >= anchors[-1][0]:
        return anchors[-1][1] if high_cap is None else high_cap
    for (x0, y0), (x1, y1) in zip(anchors[:-1], anchors[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    raise RuntimeError("piecewise interpolation failure")


def score_naaim(x):
    return piecewise(x, [(-20, 0), (0, 20), (50, 50), (100, 80), (150, 100)], 0, 100)


def score_aaii(x):
    return piecewise(x, [(-40, 0), (-20, 20), (6.5, 50), (20, 80), (40, 100)], 0, 100)


def score_pm8(x):
    return piecewise(x, [(-0.08, 0), (-0.04, 20), (0, 50), (0.04, 80), (0.08, 100)], 0, 100)


def score_pm12(x):
    return piecewise(x, [(-0.12, 0), (-0.06, 20), (0, 50), (0.06, 80), (0.12, 100)], 0, 100)


def score_hy_oas(x):
    return piecewise(x, [(2.0, 110), (2.5, 100), (3.0, 80), (4.0, 50), (5.5, 20), (8.0, 0), (12.0, -5), (20.0, -10)], 110, -10)


def score_hyg_ief(x):
    return piecewise(x, [(-0.04, 0), (-0.02, 20), (0, 50), (0.02, 80), (0.04, 100)], 0, 100)


def score_uup(x):
    return piecewise(x, [(-0.03, 100), (-0.015, 80), (0, 50), (0.015, 20), (0.03, 0)], 100, 0)


def download_market() -> pd.DataFrame:
    data = yf.download(
        SYMBOLS,
        start=START,
        end=END,
        auto_adjust=False,
        actions=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    if data.empty:
        raise RuntimeError("Yahoo/yfinance returned no market data")
    if isinstance(data.columns, pd.MultiIndex):
        # yfinance current shape: level 0 OHLC field, level 1 ticker
        if "Close" not in data.columns.get_level_values(0):
            raise RuntimeError(f"Unexpected yfinance columns: {data.columns}")
        close = data["Close"].copy()
    else:
        if len(SYMBOLS) != 1:
            raise RuntimeError("Unexpected non-MultiIndex market frame")
        close = data[["Close"]].rename(columns={"Close": SYMBOLS[0]})
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    close = close.sort_index()
    missing = [s for s in SYMBOLS if s not in close.columns]
    if missing:
        raise RuntimeError(f"Missing symbols from market data: {missing}")
    # Market-data QC: no impossible nonpositive closes; require high coverage.
    for s in SYMBOLS:
        if (close[s].dropna() <= 0).any():
            raise RuntimeError(f"Nonpositive close for {s}")
    return close


def _flatten_cols(cols):
    out = []
    for c in cols:
        if isinstance(c, tuple):
            c = " ".join(str(x) for x in c if str(x) != "nan")
        out.append(re.sub(r"\s+", " ", str(c)).strip())
    return out


def download_naaim() -> pd.DataFrame:
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(NAAIM_URL, headers=headers, timeout=30)
    r.raise_for_status()
    tables = pd.read_html(StringIO(r.text))
    chosen = None
    for t in tables:
        t = t.copy()
        t.columns = _flatten_cols(t.columns)
        cols_lower = [c.lower() for c in t.columns]
        date_candidates = [i for i, c in enumerate(cols_lower) if c == "date" or c.startswith("date ")]
        val_candidates = [i for i, c in enumerate(cols_lower) if "naaim" in c and ("number" in c or "mean" in c)]
        if date_candidates and val_candidates:
            chosen = t.iloc[:, [date_candidates[0], val_candidates[0]]].copy()
            chosen.columns = ["survey_date", "naaim"]
            break
    if chosen is None:
        raise RuntimeError("Could not find official NAAIM table")
    chosen["survey_date"] = pd.to_datetime(chosen["survey_date"], errors="coerce")
    chosen["naaim"] = pd.to_numeric(chosen["naaim"], errors="coerce")
    chosen = chosen.dropna().drop_duplicates("survey_date").sort_values("survey_date")
    if len(chosen) < 80:
        raise RuntimeError(f"Public NAAIM table unexpectedly short: {len(chosen)} rows")
    return chosen


def _to_excel_date(v):
    if isinstance(v, pd.Timestamp):
        return v.tz_localize(None).normalize() if v.tzinfo else v.normalize()
    if hasattr(v, "to_pydatetime"):
        return pd.Timestamp(v).tz_localize(None).normalize()
    if isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(v):
        if v > 20000:
            return (pd.Timestamp("1899-12-30") + pd.to_timedelta(float(v), unit="D")).normalize()
    d = pd.to_datetime(v, errors="coerce")
    if pd.isna(d):
        return pd.NaT
    if getattr(d, "tzinfo", None) is not None:
        d = d.tz_localize(None)
    return pd.Timestamp(d).normalize()


def download_aaii() -> pd.DataFrame:
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/vnd.ms-excel,*/*"}
    r = requests.get(AAII_URL, headers=headers, timeout=30)
    r.raise_for_status()
    if len(r.content) < 5000:
        raise RuntimeError(f"AAII workbook response too small ({len(r.content)} bytes)")
    book = pd.ExcelFile(BytesIO(r.content))
    best = None
    for sheet in book.sheet_names:
        raw = pd.read_excel(book, sheet_name=sheet, header=None)
        if raw.empty:
            continue
        header_row = None
        bull_col = bear_col = date_col = None
        for i in range(min(60, len(raw))):
            vals = [str(v).strip().lower() for v in raw.iloc[i].tolist()]
            b = [j for j, v in enumerate(vals) if v == "bullish" or v.startswith("bullish ")]
            br = [j for j, v in enumerate(vals) if v == "bearish" or v.startswith("bearish ")]
            if b and br:
                header_row, bull_col, bear_col = i, b[0], br[0]
                ds = [j for j, v in enumerate(vals) if v == "date" or "survey date" in v]
                date_col = ds[0] if ds else 0
                break
        if header_row is None:
            continue
        rows = []
        for i in range(header_row + 1, len(raw)):
            d = _to_excel_date(raw.iat[i, date_col])
            bull = pd.to_numeric(pd.Series([raw.iat[i, bull_col]]), errors="coerce").iloc[0]
            bear = pd.to_numeric(pd.Series([raw.iat[i, bear_col]]), errors="coerce").iloc[0]
            if pd.isna(d) or pd.isna(bull) or pd.isna(bear):
                continue
            bull = float(bull)
            bear = float(bear)
            if abs(bull) <= 1.5 and abs(bear) <= 1.5:
                bull *= 100.0
                bear *= 100.0
            if not (-5 <= bull <= 105 and -5 <= bear <= 105):
                continue
            rows.append((d, bull, bear, bull - bear))
        candidate = pd.DataFrame(rows, columns=["survey_date", "bullish", "bearish", "aaii_spread"])
        candidate = candidate.drop_duplicates("survey_date").sort_values("survey_date")
        if best is None or len(candidate) > len(best):
            best = candidate
    if best is None or len(best) < 100:
        raise RuntimeError(f"Could not parse official AAII historical workbook; rows={0 if best is None else len(best)}")
    return best


def download_hy_oas() -> pd.DataFrame:
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(FRED_URL, headers=headers, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    # FRED currently labels date column observation_date.
    date_col = df.columns[0]
    value_col = [c for c in df.columns if c != date_col][0]
    df = df.rename(columns={date_col: "observation_date", value_col: "hy_oas"})
    df["observation_date"] = pd.to_datetime(df["observation_date"], errors="coerce")
    df["hy_oas"] = pd.to_numeric(df["hy_oas"], errors="coerce")
    df = df.dropna().sort_values("observation_date")
    df = df[(df["observation_date"] >= pd.Timestamp(START)) & (df["observation_date"] <= pd.Timestamp(END))]
    if len(df) < 300:
        raise RuntimeError(f"FRED HY OAS history unexpectedly short: {len(df)}")
    return df


def next_trading_day(calendar: pd.DatetimeIndex, date: pd.Timestamp) -> pd.Timestamp | None:
    pos = calendar.searchsorted(pd.Timestamp(date).normalize(), side="left")
    if pos >= len(calendar):
        return None
    return calendar[pos]


def asof_value(df: pd.DataFrame, date_col: str, value_col: str, target: pd.Timestamp):
    q = df[df[date_col] <= target]
    if q.empty:
        return np.nan, pd.NaT
    row = q.iloc[-1]
    return float(row[value_col]), pd.Timestamp(row[date_col])


def add_forward_returns(rows: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    cal = close.index
    loc = {d: i for i, d in enumerate(cal)}
    for label, h in HORIZONS.items():
        vals = []
        for d in rows["snapshot_date"]:
            i = loc.get(d)
            if i is None or i + h >= len(cal):
                vals.append(np.nan)
            else:
                vals.append(float(close.iloc[i + h]["SPY"] / close.iloc[i]["SPY"] - 1.0))
        rows[f"fwd_{label}"] = vals
    return rows


def bootstrap_spread(df: pd.DataFrame, ret_col: str, score_col="structural_score", block=8, reps=5000, seed=20260813):
    x = df[[score_col, ret_col]].dropna().reset_index(drop=True)
    n = len(x)
    if n < 30:
        return (np.nan, np.nan)
    q20, q80 = x[score_col].quantile([0.2, 0.8])
    x["low"] = x[score_col] <= q20
    x["high"] = x[score_col] >= q80
    rng = np.random.default_rng(seed)
    vals = []
    starts = np.arange(n)
    for _ in range(reps):
        idx = []
        while len(idx) < n:
            s = int(rng.choice(starts))
            idx.extend([(s + k) % n for k in range(block)])
        samp = x.iloc[idx[:n]]
        lo = samp.loc[samp["low"], ret_col]
        hi = samp.loc[samp["high"], ret_col]
        if len(lo) and len(hi):
            vals.append(float(lo.mean() - hi.mean()))
    if len(vals) < reps * 0.8:
        return (np.nan, np.nan)
    return tuple(np.quantile(vals, [0.025, 0.975]).tolist())


def horizon_metrics(df: pd.DataFrame, score_col: str, ret_col: str):
    x = df[[score_col, ret_col]].dropna().copy()
    if len(x) < 20:
        return {"n": int(len(x))}
    rho, p = spearmanr(x[score_col], x[ret_col])
    q20, q80 = x[score_col].quantile([0.2, 0.8])
    low = x[x[score_col] <= q20][ret_col]
    high = x[x[score_col] >= q80][ret_col]
    return {
        "n": int(len(x)),
        "spearman_rho": float(rho),
        "spearman_p_two_sided": float(p),
        "q20_score": float(q20),
        "q80_score": float(q80),
        "low_q_mean_return": float(low.mean()),
        "high_q_mean_return": float(high.mean()),
        "low_minus_high": float(low.mean() - high.mean()),
        "low_q_positive_rate": float((low > 0).mean()),
        "high_q_positive_rate": float((high > 0).mean()),
        "low_n": int(len(low)),
        "high_n": int(len(high)),
    }


def main():
    close = download_market()
    naaim = download_naaim()
    aaii = download_aaii()
    hy = download_hy_oas()
    cal = close.index

    # Point-in-time release dates: both surveys are week-ending Wednesday data that
    # become available on the following U.S. trading session. FRED observations are
    # shifted to the next U.S. trading session per the audited project convention.
    naaim["release_date"] = [next_trading_day(cal, d + pd.Timedelta(days=1)) for d in naaim["survey_date"]]
    aaii["release_date"] = [next_trading_day(cal, d + pd.Timedelta(days=1)) for d in aaii["survey_date"]]
    hy["release_date"] = [next_trading_day(cal, d + pd.Timedelta(days=1)) for d in hy["observation_date"]]
    naaim = naaim.dropna(subset=["release_date"])
    aaii = aaii.dropna(subset=["release_date"])
    hy = hy.dropna(subset=["release_date"])

    # One weekly snapshot per official NAAIM release. This avoids pseudo-replicating
    # a weekly positioning observation across five daily rows.
    rows = []
    for _, nr in naaim.iterrows():
        snap = pd.Timestamp(nr["release_date"])
        if snap < pd.Timestamp("2024-07-01") or snap > pd.Timestamp("2026-05-15"):
            continue
        if snap not in close.index:
            continue
        i = close.index.get_loc(snap)
        if isinstance(i, slice) or i < 20:
            continue
        now = close.iloc[i]
        prev = close.iloc[i - 20]
        ret20 = now / prev - 1.0

        aaii_raw, aaii_release = asof_value(aaii, "release_date", "aaii_spread", snap)
        hy_raw, hy_release = asof_value(hy, "release_date", "hy_oas", snap)

        rsp_spy = float(ret20["RSP"] - ret20["SPY"])
        iwm_spy = float(ret20["IWM"] - ret20["SPY"])
        off = float(np.mean([ret20["XLY"], ret20["XLI"], ret20["XLF"], ret20["XLK"]]))
        deff = float(np.mean([ret20["XLP"], ret20["XLU"], ret20["XLV"]]))
        off_def = off - deff
        sphb_splv = float(ret20["SPHB"] - ret20["SPLV"])
        hyg_ief = float(ret20["HYG"] - ret20["IEF"])
        uup = float(ret20["UUP"])

        raw = {
            "NAAIM": float(nr["naaim"]),
            "AAII": aaii_raw,
            "RSP_SPY": rsp_spy,
            "OFF_DEF": off_def,
            "SPHB_SPLV": sphb_splv,
            "IWM_SPY": iwm_spy,
            "HY_OAS": hy_raw,
            "HYG_IEF": hyg_ief,
            "UUP": uup,
        }
        scores = {
            "NAAIM": score_naaim(raw["NAAIM"]),
            "AAII": score_aaii(raw["AAII"]),
            "RSP_SPY": score_pm8(raw["RSP_SPY"]),
            "OFF_DEF": score_pm8(raw["OFF_DEF"]),
            "SPHB_SPLV": score_pm12(raw["SPHB_SPLV"]),
            "IWM_SPY": score_pm8(raw["IWM_SPY"]),
            "HY_OAS": score_hy_oas(raw["HY_OAS"]),
            "HYG_IEF": score_hyg_ief(raw["HYG_IEF"]),
            "UUP": score_uup(raw["UUP"]),
        }
        contribution = 0.0
        valid_weight = 0.0
        provisional = []
        used_scores = {}
        for k, w in WEIGHTS.items():
            s = scores[k]
            if pd.isna(s):
                s = 50.0
                provisional.append(k)
            else:
                valid_weight += w
            used_scores[k] = float(s)
            contribution += float(s) * w
        structural = contribution / 0.50
        coverage = valid_weight / 0.50

        position = (used_scores["NAAIM"] * 0.12 + used_scores["AAII"] * 0.09) / 0.21
        participation = (
            used_scores["RSP_SPY"] * 0.025 + used_scores["OFF_DEF"] * 0.025 +
            used_scores["SPHB_SPLV"] * 0.025 + used_scores["IWM_SPY"] * 0.05
        ) / 0.125
        credit = (used_scores["HY_OAS"] * 0.075 + used_scores["HYG_IEF"] * 0.04) / 0.115

        row = {
            "snapshot_date": snap,
            "naaim_survey_date": pd.Timestamp(nr["survey_date"]),
            "naaim_raw": raw["NAAIM"],
            "aaii_spread": raw["AAII"],
            "aaii_release_date": aaii_release,
            "hy_oas": raw["HY_OAS"],
            "hy_release_date": hy_release,
            "rsp_spy_20d": rsp_spy,
            "off_def_20d": off_def,
            "sphb_splv_20d": sphb_splv,
            "iwm_spy_20d": iwm_spy,
            "hyg_ief_20d": hyg_ief,
            "uup_20d": uup,
            "spy_close": float(now["SPY"]),
            "structural_score": structural,
            "bucket_coverage": coverage,
            "positioning_score": position,
            "participation_score": participation,
            "credit_score": credit,
            "dollar_score": used_scores["UUP"],
            "provisional_leaves": ",".join(provisional),
        }
        for k, s in used_scores.items():
            row[f"score_{k.lower()}"] = s
        rows.append(row)

    bt = pd.DataFrame(rows).sort_values("snapshot_date").reset_index(drop=True)
    if len(bt) < 70:
        raise RuntimeError(f"Structural backtest sample too short: {len(bt)}")
    bt = add_forward_returns(bt, close)

    # Publication-quality research sample: canonical bucket gate >=80%.
    pub = bt[bt["bucket_coverage"] >= 0.80].copy()
    if len(pub) < 70:
        raise RuntimeError(f"Too few >=80% coverage snapshots: {len(pub)}")

    summary = {
        "status": "RESEARCH_ONLY",
        "model": "v1.2-PARETO-AI Structural/Crowding",
        "frequency": "weekly, one snapshot per NAAIM release",
        "first_snapshot": pub["snapshot_date"].min().date().isoformat(),
        "last_snapshot": pub["snapshot_date"].max().date().isoformat(),
        "snapshots": int(len(pub)),
        "mean_bucket_coverage": float(pub["bucket_coverage"].mean()),
        "min_bucket_coverage": float(pub["bucket_coverage"].min()),
        "score_mean": float(pub["structural_score"].mean()),
        "score_std": float(pub["structural_score"].std()),
        "score_min": float(pub["structural_score"].min()),
        "score_max": float(pub["structural_score"].max()),
        "score_ge_65_count": int((pub["structural_score"] >= 65).sum()),
        "score_le_35_count": int((pub["structural_score"] <= 35).sum()),
        "note_thresholds": "Counts only; Structural/Crowding has no canonical trade thresholds.",
        "data": {
            "naaim_rows_public": int(len(naaim)),
            "naaim_first": naaim["survey_date"].min().date().isoformat(),
            "naaim_last": naaim["survey_date"].max().date().isoformat(),
            "aaii_rows_official_workbook": int(len(aaii)),
            "aaii_first": aaii["survey_date"].min().date().isoformat(),
            "aaii_last": aaii["survey_date"].max().date().isoformat(),
            "fred_rows": int(len(hy)),
            "market_sessions": int(len(close)),
        },
        "horizons": {},
        "block_diagnostics_4W": {},
        "stability_4W": {},
    }

    for label in HORIZONS:
        ret_col = f"fwd_{label}"
        m = horizon_metrics(pub, "structural_score", ret_col)
        if m.get("n", 0) >= 30:
            lo, hi = bootstrap_spread(pub, ret_col, block=8, reps=5000)
            m["block_bootstrap_95ci_low_minus_high"] = [float(lo), float(hi)]
        summary["horizons"][label] = m

    for col in ["positioning_score", "participation_score", "credit_score", "dollar_score"]:
        summary["block_diagnostics_4W"][col] = horizon_metrics(pub, col, "fwd_4W")

    # Chronological halves + calendar years, focused on the central 4W horizon.
    mid = len(pub) // 2
    partitions = {
        "first_half": pub.iloc[:mid],
        "second_half": pub.iloc[mid:],
    }
    for yr in sorted(pub["snapshot_date"].dt.year.unique()):
        partitions[str(int(yr))] = pub[pub["snapshot_date"].dt.year == yr]
    for name, frame in partitions.items():
        summary["stability_4W"][name] = horizon_metrics(frame, "structural_score", "fwd_4W")

    # Quintile table across horizons for easy inspection.
    ranked = pub.copy()
    ranked["quintile"] = pd.qcut(ranked["structural_score"], 5, labels=["Q1_low", "Q2", "Q3", "Q4", "Q5_high"], duplicates="drop")
    qrows = []
    for q, g in ranked.groupby("quintile", observed=True):
        rec = {"quintile": str(q), "n": int(len(g)), "score_mean": float(g["structural_score"].mean())}
        for label in HORIZONS:
            rec[f"mean_fwd_{label}"] = float(g[f"fwd_{label}"].mean())
            rec[f"positive_fwd_{label}"] = float((g[f"fwd_{label}"] > 0).mean())
        qrows.append(rec)
    qtable = pd.DataFrame(qrows)

    bt.to_csv(OUT / "structural_crowding_weekly_scores.csv", index=False)
    qtable.to_csv(OUT / "structural_crowding_quintiles.csv", index=False)
    (OUT / "structural_crowding_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = []
    md.append("# Structural/Crowding v1.2 — research backtest\n")
    md.append(f"Window: **{summary['first_snapshot']} → {summary['last_snapshot']}**, weekly snapshots: **{summary['snapshots']}**.\n")
    md.append(f"Bucket Coverage: mean **{summary['mean_bucket_coverage']:.1%}**, min **{summary['min_bucket_coverage']:.1%}**.\n")
    md.append("## Forward SPY price-return diagnostics\n")
    md.append("| Horizon | N | Spearman rho | p | Low Q mean | High Q mean | Low−High | 95% block-bootstrap CI |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for label, m in summary["horizons"].items():
        ci = m.get("block_bootstrap_95ci_low_minus_high", [np.nan, np.nan])
        md.append(
            f"| {label} | {m.get('n',0)} | {m.get('spearman_rho',np.nan):.3f} | {m.get('spearman_p_two_sided',np.nan):.3f} | "
            f"{m.get('low_q_mean_return',np.nan):.2%} | {m.get('high_q_mean_return',np.nan):.2%} | "
            f"{m.get('low_minus_high',np.nan):+.2%} | [{ci[0]:+.2%}, {ci[1]:+.2%}] |"
        )
    md.append("\nContrarian expectation: higher Structural/Crowding should be associated with weaker subsequent SPY returns (negative rho; positive Low−High spread).")
    md.append("\nNo Structural/Crowding threshold is treated as a trading rule. Production Composite/history/paper state are untouched.\n")
    (OUT / "structural_crowding_report.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
