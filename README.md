# Innovation and Exports: Nigerian Firms, 2014 to 2026

Python rebuild of a 2018 B.Sc. study (originally Stata, World Bank Enterprise Survey 2014), extended with the
2025 Enterprise Survey and the 2026 "ES follow-up on AI". It estimates how innovation, quality certification and
AI adoption relate to export participation, with weighted and unweighted probit models, average marginal effects,
and a recursive bivariate probit written from scratch.

## Headline results (associations, not causal effects)
| Question | Finding |
|---|---|
| Does innovating raise the chance of exporting? | 2014: +3 to +4 points (p<0.05). 2025: about 0. No significant change between years (p=0.45). |
| Which innovation type? | 2014: only formal R&D stands out (+8.6 points). 2025: none. Combining types does not help. |
| What else goes with exporting? | Quality certification: +9 points (2014), +16 points (2025). |
| AI (2026 follow-up, 777 firms) | 43% use AI, mostly chatbots. No link to exporting. Product innovators adopt more (+12 points weighted). |

## Quick start (conda)
```bash
git clone https://github.com/<your-username>/innovation-exports.git
cd innovation-exports
conda env create -f environment.yml
conda activate innovation-exports

# 1. put the survey files in data/raw/ (see data/raw/README.md for names and where to download)
# 2. run the tests (no data needed)
pytest -q
# 3. run the full analysis (about 35 seconds)
python run_all.py
```
Outputs: `output/tables/*.csv` (T1 to T11), `output/figures/innovation_effect_by_year.png`, `output/run_log.txt`
(the printed results). Committed copies of the tables from the last run are included for reference.

## Data
Enterprise Survey microdata is free but requires registration at <https://www.enterprisesurveys.org> and cannot be
redistributed, so it is not in this repo. Expected files in `data/raw/`:

- `Nigeria-2014-full-data.dta`
- `Nigeria-2025-full-data.dta`
- `Nigeria-2026-AI-followup.dta`

If a download is a ZIP (even one named `.dta`), leave it as is; the code unpacks it automatically.

## What each table is
| Table | Content |
|---|---|
| T1 | Descriptive shares by year (weighted and unweighted) |
| T2 | Replication of the 2018 headline model on 2014 data |
| T3 | Effect of innovation (product, process or R&D) on exporting, by year |
| T4 | Effect by innovation type, with a test that all types are equal |
| T5 | Combined versus single innovation |
| T6 | Recursive bivariate probit (innovation treated as endogenous). **Exploratory**: instruments are weak-by-design, estimates unstable |
| T7 | Pooled 2014/2025 model testing whether the innovation effect changed |
| T8 to T11 | 2026 AI follow-up: prevalence, adoption by baseline status, adoption models, exports on innovation and AI |

## Layout
```
run_all.py            runs everything
environment.yml       conda environment (Python 3.11)
requirements.txt      pip alternative
src/data.py           loads .dta files, builds harmonised variables
src/models.py         probit + marginal effects (delta-method SEs, survey weights), bivariate probit
src/ai_extension.py   2026 AI follow-up analysis
tests/                unit tests (bivariate CDF, marginal effects, bivariate probit)
data/raw/             put survey files here (git-ignored)
output/               tables, figure, log
```

## Notes and caveats
- 2014 and 2025 are different samples and cannot be linked at firm level, so this is a comparison of two cross-sections.
- The 2018 paper's 1,897 innovators is reproduced exactly; its export count (651 vs 639 here) and 10.3-point effect
  (8.4 here) are not, and the exact original export definition could not be recovered.
- Innovation = product, process or R&D, the only types asked in both surveys. The R&D question covers three years in
  2014 and the last fiscal year in 2025.
- 55% of AI adopters started before the 2025 baseline interview, so AI results are associations only.

## Citation
Anyasor, M. E. (2026). *Innovation and exports in Nigerian firms* [Computer software]. Code and replication files.
Survey data: World Bank Enterprise Surveys, Nigeria 2014 and 2025; ES follow-up on AI, Nigeria 2026.

## License
MIT (code only). Survey data remain subject to World Bank terms of use.
