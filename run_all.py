"""Rebuild: Innovation and exports, Nigeria firm-level (WBES 2014 vs 2025).  Run:  python run_all.py"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import data, models, ai_extension

OUT = Path(__file__).parent / "output"
(OUT / "tables").mkdir(parents=True, exist_ok=True); (OUT / "figures").mkdir(parents=True, exist_ok=True)
CTRL = "C(legal) + training + log_emp + age + female_owner + mgr_exp + quality_cert + foreign_owned + C(sector) + C(region)"
log = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); log.append(s)
def save(df, name):
    df.to_csv(OUT / "tables" / f"{name}.csv"); return df

d14, d25 = data.load_2014(), data.load_2025()
c14, c25 = data.comparable(d14, d25)        # innovation = product | process | R&D in both years

# ---------------------------------------------------------------- T1 descriptives
def wshare(df, col):
    m = df[col].notna(); return np.average(df.loc[m, col], weights=df.loc[m, "weight"])
rows = []
for lab, df, cols in [("2014", c14, ["innov_paper", "innov_any", "product", "process", "organizational", "marketing", "rnd", "export_any", "export_direct", "training"]),
                      ("2025", c25, ["innov_any", "product", "process", "rnd", "export_any", "export_direct", "training"])]:
    for c in cols:
        rows.append({"year": lab, "variable": c, "n_valid": int(df[c].notna().sum()), "unweighted_share": df[c].mean(), "weighted_share": wshare(df, c)})
t1 = save(pd.DataFrame(rows), "T1_descriptives"); say("\n== T1 descriptives\n", t1.round(3).to_string(index=False))
say(f"\n2014 paper-definition innovators: {int(c14.innov_paper.sum())} of {len(c14)} (2018 paper: 1,897 of 2,676)")
say(f"2014 exporters (any direct/indirect sales): {int(c14.export_any.sum())} (2018 paper: 651)")

# ---------------------------------------------------------------- T2 replication of 2018 Table 4.1
c14p = c14.assign(innov=c14.innov_paper, export=c14.export_any)
f = models.fit_probit(c14p, "export", "innov + C(legal) + training + C(export_zone)")
t2 = save(models.fmt_ame(f["ame"]), "T2_replication_2014")
say(f"\n== T2 replication (2014, paper spec: innov + firm type + training + export zone), N={f['n']}\n", t2[["ame", "se", "p"]].round(4).to_string())
say("   2018 paper reported innovation ME = +0.1034")

# ---------------------------------------------------------------- T3 baseline both years
rows = []
for yr, df in [(2014, c14), (2025, c25)]:
    for yv in ["export_any", "export_direct"]:
        for wlab, w in [("unweighted", None), ("weighted", "weight")]:
            f = models.fit_probit(df, yv, f"innov_any + {CTRL}", weight=w)
            a = f["ame"].loc["innov_any"]
            rows.append({"year": yr, "outcome": yv, "weights": wlab, "n": f["n"], "ame": a.ame, "se": a.se, "p": a.p})
t3 = save(pd.DataFrame(rows), "T3_innovation_effect_by_year"); say("\n== T3 effect of innovation (product|process|R&D) on export probability\n", t3.round(4).to_string(index=False))

# ---------------------------------------------------------------- T4 type of innovation (Objective 1)
rows = []
for yr, df, types in [(2014, c14, ["product", "process", "organizational", "marketing", "rnd"]), (2025, c25, ["product", "process", "rnd"])]:
    f = models.fit_probit(df, "export_any", f"{' + '.join(types)} + {CTRL}")
    for t in types:
        a = f["ame"].loc[t]; rows.append({"year": yr, "type": t, "n": f["n"], "ame": a.ame, "se": a.se, "p": a.p})
    wt = f["res"].wald_test(" = ".join(types[:1]) + "".join(f" , {types[0]} = {t}" for t in types[1:]) if False else
                            ", ".join(f"{types[0]} = {t}" for t in types[1:]), scalar=True)
    rows.append({"year": yr, "type": "Wald: all types equal (coef)", "n": f["n"], "ame": np.nan, "se": np.nan, "p": float(wt.pvalue)})
t4 = save(pd.DataFrame(rows), "T4_innovation_types"); say("\n== T4 innovation types -> export_any\n", t4.round(4).to_string(index=False))

# ---------------------------------------------------------------- T5 combined vs single (Objective 2)
rows = []
for yr, df in [(2014, c14), (2025, c25)]:
    f = models.fit_probit(df, "export_any", f"single_only + combined + {CTRL}")
    for t in ["single_only", "combined"]:
        a = f["ame"].loc[t]; rows.append({"year": yr, "term": t + " (vs none)", "n": f["n"], "ame": a.ame, "se": a.se, "p": a.p})
    wt = f["res"].wald_test("single_only = combined", scalar=True)
    rows.append({"year": yr, "term": "Wald: combined = single (coef)", "n": f["n"], "ame": np.nan, "se": np.nan, "p": float(wt.pvalue)})
t5 = save(pd.DataFrame(rows), "T5_combined_vs_single"); say("\n== T5 combined vs single innovation -> export_any\n", t5.round(4).to_string(index=False))

# ---------------------------------------------------------------- T6 recursive bivariate probit
BI_X = "log_emp + age + female_owner + mgr_exp + quality_cert + foreign_owned"
rows = []
for yr, df in [(2014, c14), (2025, c25)]:
    # innovation equation adds two excluded instruments: foreign-licensed technology, formal training
    r = models.fit_biprobit(df, "innov_any", "export_any", f"{BI_X} + foreign_tech + training", BI_X)
    simple = models.fit_probit(df.loc[df[["foreign_tech", "training"]].notna().all(axis=1)], "export_any", f"innov_any + {BI_X}")
    rows.append({"year": yr, "n": r["n"], "converged": r["converged"], "biprobit_ATE_innov": r["ate"], "se": r["ate_se"],
                 "rho": r["rho"], "p_rho=0": r["rho_p"], "probit_AME_same_sample": simple["ame"].loc["innov_any", "ame"]})
    save(r["table"], f"T6_biprobit_coefficients_{yr}")
t6 = save(pd.DataFrame(rows), "T6_biprobit_summary"); say("\n== T6 recursive bivariate probit (innovation endogenous)\n", t6.round(4).to_string(index=False))

# ---------------------------------------------------------------- T7 pooled 2014 vs 2025 difference
pool = pd.concat([c14.assign(y25=0.0), c25.assign(y25=1.0)], ignore_index=True)
pool["legal"] = pool["legal"].astype(object)
f = models.fit_probit(pool.assign(inn_y25=pool.innov_any * pool.y25), "export_any",
                      "innov_any + inn_y25 + y25 + C(legal) + training + log_emp + age + female_owner + mgr_exp + quality_cert + foreign_owned")
t7 = save(models.fmt_ame(f["ame"].loc[["innov_any", "inn_y25", "y25"]]), "T7_pooled_difference")
say(f"\n== T7 pooled model, N={f['n']}  (inn_y25 = change in innovation effect in 2025 vs 2014)\n", t7[["ame", "se", "p"]].round(4).to_string())

# ---------------------------------------------------------------- AI follow-up extension (T8-T11)
ai_df = ai_extension.run(c25, say, save)

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(figsize=(6.4, 3.4))
sub = t3[(t3.outcome == "export_any")]
labels, est, lo, hi = [], [], [], []
for _, r in sub.iterrows():
    labels.append(f"{int(r.year)} {r.weights}"); est.append(r.ame); lo.append(r.ame - 1.96 * r.se); hi.append(r.ame + 1.96 * r.se)
y = np.arange(len(labels))[::-1]
ax.errorbar(est, y, xerr=[np.array(est) - np.array(lo), np.array(hi) - np.array(est)], fmt="o", color="#1f5f8b", capsize=3)
ax.axvline(0, color="grey", lw=0.8); ax.set_yticks(y); ax.set_yticklabels(labels)
ax.set_xlabel("Effect of innovating on P(export): average marginal effect, 95% CI", fontsize=9)
ax.set_title("Innovation and export participation, Nigeria", fontsize=10, loc="left")
for s in ("top", "right"): ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(OUT / "figures" / "innovation_effect_by_year.png", dpi=160)
(OUT / "run_log.txt").write_text("\n".join(log))
