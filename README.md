# HTVE Global Futures Model

Replication repository for the multi-jurisdiction scenario experiments associated with the manuscript:

**Beyond money? Global diffusion, crisis islanding, and abundance futures under the Homayoon Theory of Value Exchange**

## Purpose

This repository provides the executable model, synthetic jurisdiction definitions, run-level simulation outputs, summary tables, and generated figures used in the manuscript. The study uses **no empirical country, participant, clinical, or proprietary data**. All jurisdiction characteristics are normalized scenario assumptions used for mechanism testing rather than estimates of real countries.

## Research questions

The model tests five families of questions:

1. **Global diffusion** — whether adoption depends on both monetary need and technological/institutional capability.
2. **Exchange islanding** — whether a pre-positioned closed-loop exchange network can preserve local exchange when external settlement connectivity is disrupted but local productive capacity remains.
3. **Monetary competition** — when an internal unit remains complementary to conventional money and when its settlement share becomes substantial under stress.
4. **Governance and elite capture** — how concentration, safeguards, and a transaction-funded social-access pool affect access, diversion risk, platform viability, and completion.
5. **AI/robotics abundance** — whether abundant productive capacity eliminates allocation needs when access to productive infrastructure remains concentrated.

The complete experimental design contains **21,830 condition-replication evaluations**. Experiment A additionally records period-by-jurisdiction diffusion trajectories.

## Repository structure

```text
HTVE-Global-Futures-Model/
├── README.md
├── CITATION.cff
├── LICENSE
├── requirements.txt
├── code/
│   └── htve_global_futures_replication.py
├── docs/
│   └── MODEL_SPECIFICATION.md
├── data/
│   ├── jurisdiction_archetypes.csv
│   ├── key_results.csv
│   ├── experiment_*_runs.csv
│   └── summary_*.csv
├── figures/
│   └── Fig1_...png through Fig6_...png
└── .github/workflows/
    └── generate-replication-data.yml
```

## Reproduction

Python 3 is required. Install dependencies:

```bash
pip install -r requirements.txt
```

Run the model:

```bash
python code/htve_global_futures_replication.py
```

The script initially writes CSV outputs beside the script and figures to `FIGURES/`. The included GitHub Actions workflow executes the model in a clean environment and organizes the generated artifacts into `data/` and `figures/`, providing a machine-executed reproducibility check.

Fixed random-seed ranges are used. Comparisons of architectures or policy conditions use paired random seeds where applicable.

## Experiments

- **A — Global diffusion:** 30 synthetic jurisdictions in six archetypes, 30 trajectories, 80 periods.
- **B — Exchange islanding:** 100 paired replications across three settlement architectures.
- **C — Monetary competition:** 100 paired replications across 20 stress/breadth conditions.
- **D1 — Elite capture:** 100 paired replications across 15 concentration/safeguard conditions.
- **D2 — Social-access levy:** 100 paired replications across 60 fee/allocation/capacity conditions.
- **E — AI/robotics abundance:** 100 paired replications across 120 automation/resource/concentration/safeguard conditions.

## Interpretation boundary

This is a synthetic scenario model, not a forecast of any actual country, government, currency, conflict, or future technological trajectory. Numerical thresholds are parameter-dependent. The model cannot create absent physical resources and does not imply that any internal exchange unit is exempt from applicable monetary, tax, consumer-protection, AML/CFT, data-protection, or professional regulation.

The intended use is mechanism testing, transparent criticism, robustness analysis, and design of future empirical or pilot studies.

## Reproducibility

The workflow in `.github/workflows/generate-replication-data.yml` reruns the complete model and commits regenerated run-level data and figures when the model or dependency specification changes. A successful workflow run therefore provides an independent machine execution of the public code.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff).

**Author:** Homayoon Kazemy  
**ORCID:** 0000-0003-1929-5999

## License

The code and repository materials are released under the MIT License. This license permits reuse with attribution; it does not imply endorsement of any implementation or waive compliance with applicable law or regulation.
