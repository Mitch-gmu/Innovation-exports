"""Extension: AI adoption (2026 follow-up) versus innovation and export status measured in the 2025 baseline."""
import numpy as np, pandas as pd
import models, data

CTRL = "C(legal) + training + log_emp + age + female_owner + mgr_exp + quality_cert + foreign_owned + C(sector) + C(region)"


def wmean(df, col):
    m = df[col].notna()
    return np.average(df.loc[m, col], weights=df.loc[m, "weight"])


def run(c25, say, save):
    ai = data.load_ai_followup(c25)
    say(f"\n== AI extension: {len(ai)} firms, merged to 2025 baseline on idstd (all {ai.idstd.isin(c25.idstd).sum()} matched)")

    # T8 adoption prevalence (weighted and unweighted)
    rows = []
    for c, lab in [("ai_adopt", "any AI"), ("ai_b2a2", "AI chatbot"), ("ai_b2a4", "generative AI (media/text)"),
                   ("ai_b2a1", "machine learning"), ("ai_b2a3", "AI agent"), ("ai_b2a5", "workflow automation / decision support"),
                   ("ai_b2a6", "autonomous machines"), ("ai_rnd_use", "uses AI in R&D (adopters only)"), ("ai_b6f", "sells own AI technology")]:
        rows.append({"measure": lab, "n_valid": int(ai[c].notna().sum()), "unweighted": ai[c].mean(), "weighted": wmean(ai, c)})
    t8 = save(pd.DataFrame(rows), "T8_ai_prevalence"); say("\n== T8 AI use (2026)\n", t8.round(3).to_string(index=False))

    # T9 adoption by baseline innovation / export status
    rows = []
    for var, lab in [("innov_any", "innovated (product|process|R&D), 2025"), ("product", "product innovation"), ("process", "process innovation"),
                     ("rnd", "R&D spending"), ("export_any", "exporter, 2025")]:
        for v in (0.0, 1.0):
            s = ai[ai[var] == v]
            rows.append({"group": lab, "value": int(v), "n": len(s), "ai_adoption_weighted": wmean(s, "ai_adopt"), "ai_adoption_unweighted": s.ai_adopt.mean()})
    t9 = save(pd.DataFrame(rows), "T9_ai_adoption_by_baseline_status"); say("\n== T9 AI adoption by baseline status\n", t9.round(3).to_string(index=False))

    # T10 probit: AI adoption ~ baseline innovation and export status (timing consistent: baseline 2025 -> AI 2026)
    rows = []
    for spec, rhs in [("innovation only", f"innov_any + {CTRL}"),
                      ("innovation types", f"product + process + rnd + {CTRL}"),
                      ("innovation + exporter", f"innov_any + export_any + {CTRL}")]:
        for wlab, w in [("unweighted", None), ("weighted (wt)", "weight")]:
            f = models.fit_probit(ai, "ai_adopt", rhs, weight=w)
            for t in [x for x in ["innov_any", "product", "process", "rnd", "export_any"] if x in f["ame"].index]:
                a = f["ame"].loc[t]
                rows.append({"spec": spec, "weights": wlab, "term": t, "n": f["n"], "ame": a.ame, "se": a.se, "p": a.p})
    t10 = save(pd.DataFrame(rows), "T10_ai_adoption_probit"); say("\n== T10 effect of baseline innovation / exporting on P(AI adoption)\n", t10.round(4).to_string(index=False))

    # T11 is AI adoption associated with exporting once innovation is controlled? (association only: exports are FY2024, AI use is 2026)
    rows = []
    for yv in ["export_any", "export_direct"]:
        for wlab, w in [("unweighted", None), ("weighted (wt)", "weight")]:
            f = models.fit_probit(ai, yv, f"innov_any + ai_adopt + {CTRL}", weight=w)
            for t in ["innov_any", "ai_adopt"]:
                a = f["ame"].loc[t]
                rows.append({"outcome": yv, "weights": wlab, "term": t, "n": f["n"], "ame": a.ame, "se": a.se, "p": a.p})
    t11 = save(pd.DataFrame(rows), "T11_export_on_innov_and_ai"); say("\n== T11 export ~ innovation + AI adoption (association only)\n", t11.round(4).to_string(index=False))

    # timing check
    adopters = ai[(ai.ai_adopt == 1) & ai.ai_before_baseline.notna()]
    say(f"\nTiming: among {len(adopters)} adopters with a first-use date, {adopters.ai_before_baseline.mean():.0%} first used AI BEFORE the baseline interview "
        f"(so AI adoption often precedes the 2025 innovation/export measures).")
    return ai
