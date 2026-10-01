# Raw data (not included in this repository)

The World Bank Enterprise Surveys microdata are free but require registration and may not be redistributed,
so the files are **not** in this repo. Download them from https://www.enterprisesurveys.org (Data -> Nigeria)
and save them here with exactly these names:

| File name in this folder | What it is |
|---|---|
| `Nigeria-2014-full-data.dta` | Nigeria Enterprise Survey 2014 (2,676 firms) |
| `Nigeria-2025-full-data.dta` | Nigeria Enterprise Survey 2025 (1,043 firms) |
| `Nigeria-2026-AI-followup.dta` | 2026 "ES follow-up on AI" phone survey (777 firms), merges to 2025 on `idstd` |

Notes
- Downloads often arrive as `.zip`. If a `.dta` is really a ZIP (some are), leave it as is: the code unpacks it
  automatically into `data/interim/`. You can also unzip it yourself.
- The documentation PDFs (questionnaires, implementation reports) are on the same download pages; they are not needed to run the code.
