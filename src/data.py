"""Load the World Bank Enterprise Survey files for Nigeria and build harmonised variables.

Survey codes: 1 = Yes, 2 = No, negatives (-9 don't know, -8 refusal, -7 n/a, ...) = missing.
"""
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import pyreadstat

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def _read(name):
    """Read a Stata file from data/raw. World Bank downloads are sometimes a ZIP saved with a .dta
    extension (or a .zip); both are unpacked automatically into data/interim/."""
    path = RAW / name
    if not path.exists():
        found = sorted(p.name for p in RAW.glob("*")) if RAW.exists() else []
        raise FileNotFoundError(
            f"Missing {path}.\nExpected files in data/raw/: Nigeria-2014-full-data.dta, Nigeria-2025-full-data.dta, "
            f"Nigeria-2026-AI-followup.dta (see data/raw/README.md).\nFound: {found}")
    if zipfile.is_zipfile(path):
        cache = RAW.parent / "interim" / path.stem
        cache.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path) as z:
            dta = [n for n in z.namelist() if n.lower().endswith(".dta")]
            if not dta:
                raise ValueError(f"{path} is a ZIP with no .dta file inside")
            z.extract(dta[0], cache)
        path = cache / dta[0]
    df, _ = pyreadstat.read_dta(str(path), apply_value_formats=False)
    return df


def _yes(s):
    """1/2 survey answer -> 1.0/0.0, anything else -> NaN."""
    s = s.where(s.isin([1, 2]))
    return (s == 1).astype(float).where(s.notna())


def _clean(s):
    s = pd.to_numeric(s, errors="coerce")
    return s.where(s >= 0)


def _legal_form(code_map, s):
    return s.map(code_map)


def load_2014():
    d = _read("Nigeria-2014-full-data.dta")
    o = pd.DataFrame({"idstd": d.idstd, "year": 2014})
    # innovation types (last three years)
    o["product"] = _yes(d.h1)
    o["process"] = _yes(d.h3)
    o["organizational"] = _yes(d.h5)
    o["marketing"] = _yes(d.h6)
    o["rnd"] = _yes(d.h7)
    # 2018 paper's definition reproduces 1,897 innovators exactly:
    # any of h1, h3, h4a, h4b, h5, h6, h7 == 1
    o["innov_paper"] = np.any([d[c] == 1 for c in ["h1", "h3", "h4a", "h4b", "h5", "h6", "h7"]], axis=0).astype(float)
    # exports: any direct or indirect export sales
    o["export_any"] = ((d.d3b > 0) | (d.d3c > 0)).astype(float)
    o["export_direct"] = (d.d3c > 0).astype(float)
    o["export_share"] = _clean(d.d3b).fillna(0) + _clean(d.d3c).fillna(0)
    o.loc[(d.d3b < 0) & (d.d3c < 0), "export_share"] = np.nan
    # controls
    o["training"] = _yes(d.l10)
    o["legal"] = d.b1.map({3: "sole", 1: "shareholder", 2: "shareholder", 4: "partnership", 5: "partnership"})
    o["log_emp"] = np.log(_clean(d.l1).where(lambda s: s > 0))
    o["age"] = 2014 - _clean(d.b5).where(lambda s: s > 1800)
    o["female_owner"] = _yes(d.b4)
    o["mgr_exp"] = _clean(d.b7)
    o["foreign_owned"] = (_clean(d.b2b) > 0).astype(float).where(_clean(d.b2b).notna()) if "b2b" in d else np.nan
    o["quality_cert"] = _yes(d.b8)
    o["foreign_tech"] = _yes(d.e6)   # asked of a subset of firms in 2014
    o["export_zone"] = d.ngb1.where(d.ngb1.isin([1, 2, 3])).map({3: 1, 2: 2, 1: 3})  # 1 neither, 2 industrial, 3 EPZ
    o["region"] = d.a2
    o["sector"] = d.a4a
    o["weight"] = d.wmedian
    return o


def load_2025():
    d = _read("Nigeria-2025-full-data.dta")
    o = pd.DataFrame({"idstd": d.idstd, "year": 2025})
    o["product"] = _yes(d.h1)
    o["process"] = _yes(d.h5)
    o["rnd"] = _yes(d.h8)  # R&D spending in last fiscal year
    o["export_any"] = ((d.d3b > 0) | (d.d3c > 0)).astype(float)
    o["export_direct"] = (d.d3c > 0).astype(float)
    o["export_share"] = _clean(d.d3b).fillna(0) + _clean(d.d3c).fillna(0)
    o.loc[(d.d3b < 0) & (d.d3c < 0), "export_share"] = np.nan
    o["training"] = _yes(d.l10)
    o["legal"] = d.b1.map({3: "sole", 1: "shareholder", 2: "shareholder", 4: "partnership", 5: "partnership"})
    o["log_emp"] = np.log(_clean(d.l1).where(lambda s: s > 0))
    o["age"] = 2025 - _clean(d.b5).where(lambda s: s > 1800)
    o["female_owner"] = _yes(d.b4)
    o["mgr_exp"] = _clean(d.b7)
    o["foreign_owned"] = (_clean(d.b2b) > 0).astype(float).where(_clean(d.b2b).notna())
    o["quality_cert"] = _yes(d.b8)
    o["foreign_tech"] = _yes(d.e6)
    o["owner_manager"] = _yes(d.b3a)
    o["panel_2025"] = d.panel
    o["region"] = d.a2
    o["sector"] = d.a4a
    o["weight"] = d.wmedian
    return o


def add_innovation_indices(df, types):
    """innov_any / n_types / combined (>=2 types) from the listed 0/1 columns.
    Firms missing every listed answer stay missing."""
    x = df[types]
    df["n_types"] = x.sum(axis=1, min_count=1)
    df["innov_any"] = (df.n_types > 0).astype(float).where(df.n_types.notna())
    df["single_only"] = (df.n_types == 1).astype(float).where(df.n_types.notna())
    df["combined"] = (df.n_types >= 2).astype(float).where(df.n_types.notna())
    return df


def comparable(df14, df25):
    """Cross-year comparable innovation measure: product, process or R&D (asked in both surveys)."""
    types = ["product", "process", "rnd"]
    return add_innovation_indices(df14.copy(), types), add_innovation_indices(df25.copy(), types)


def load_ai_followup(baseline25):
    """2026 AI follow-up (CATI, Jan-Mar 2026, 777 of the 1,043 baseline firms), merged on idstd to the 2025 baseline.
    Weight `weight` is the follow-up weight `wt` (baseline weight x stratum non-response adjustment)."""
    a = _read("Nigeria-2026-AI-followup.dta")
    base = baseline25.drop(columns=["weight"])
    # baseline interview month, for checking whether AI adoption pre-dates the baseline survey
    raw = _read("Nigeria-2025-full-data.dta")[["idstd", "a14y", "a14m"]]
    base = base.merge(raw, on="idstd", how="left")
    out = a[["idstd", "wt", "ai_adopt1", "ai_b2a1", "ai_b2a2", "ai_b2a3", "ai_b2a4", "ai_b2a5", "ai_b2a6",
             "ai_b4a1", "ai_b5a", "ai_b1b", "ai_b1a3", "ai_b6f"]].rename(columns={"wt": "weight"})
    m = out.merge(base, on="idstd", how="left", validate="1:1")
    m["ai_adopt"] = m.ai_adopt1.astype(float)
    for c in ["ai_b2a1", "ai_b2a2", "ai_b2a3", "ai_b2a4", "ai_b2a5", "ai_b2a6", "ai_b6f", "ai_b1a3"]:
        m[c] = _yes(m[c])
    m["ai_rnd_use"] = _yes(m.ai_b4a1)                      # asked of adopters only
    m["computer_share"] = _clean(m.ai_b1b)
    m["ai_first_yyyymm"] = _clean(m.ai_b5a)
    m["baseline_yyyymm"] = m.a14y * 100 + m.a14m
    m["ai_before_baseline"] = (m.ai_first_yyyymm < m.baseline_yyyymm).astype(float).where(m.ai_first_yyyymm.notna())
    return data_add_innov(m)


def data_add_innov(m):
    return add_innovation_indices(m, ["product", "process", "rnd"])
