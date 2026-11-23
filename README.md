# Human-Centred Explainable AI for Enterprise Data Quality and Anomaly Investigation

MSc Computing (Human-Centred AI), Technological University Dublin, 2026.

Enterprise transaction data contains data-quality errors (missing values, duplicates, wrong balances, typos). ML models can flag suspicious records, but an investigator also needs to know **why**. This project injects known data-quality errors into clean transaction data, compares how well five models detect them, and tests whether SHAP and LIME explanations stay stable and point to the real cause.

## Research questions

1. **RQ1.** How do statistical ML models perform in detecting data-quality anomalies in enterprise transactional data?
2. **RQ2.** How do different data-quality anomalies affect the stability and consistency of XAI explanations?
3. **RQ3.** How can XAI provide meaningful explanations of model decisions for data-quality anomaly detection?

## Method in brief

| Step | What happens | Code |
| --- | --- | --- |
| 1. Prepare | Remove the 1,411 money-laundering rows, take a reproducible 50,000-row sample, unpack the nested metadata, screen fields with stated rules, verify the sample is clean and representative | `src/prepare.py` |
| 2. Inject | Six error types (missing, duplicate, balance, numerical, categorical, cross-field) at 1%, 5% and 10% severity, three random seeds each, with an exact answer key | `src/inject.py` |
| 3. Features | Data-quality check features (e.g. balance difference, invalid-category flags) plus encoded transaction fields | `src/features.py` |
| 4. Models | Isolation Forest, LOF, One-Class SVM (unsupervised); Random Forest, XGBoost (supervised); with and without check features | `src/models.py` |
| 5. Explain | SHAP and LIME; stability via Spearman rank correlation and top-5 Jaccard overlap; does the top feature match the injected error? | `src/explain.py` |

## Project structure

```
config.py              settings: paths, sample size, severities, seeds, screening thresholds
src/                   all project logic (imported by notebooks and scripts)
tests/                 unit tests for preparation, injection, features and stability measures
scripts/run_pipeline.py  runs the steps end to end from the command line
notebooks/             step-by-step runs with outputs, plots and interpretation
data/raw/              AMLNet_v2_transactions.csv (download from Zenodo; not committed)
data/processed/        generated samples and dirty datasets (not committed)
results/               tables and figures produced by the notebooks and pipeline
```

## How to run

```bash
pip install -r requirements.txt
# download AMLNet_v2_transactions.csv from https://zenodo.org/records/21237971 into data/raw/
pytest                                   # 15 tests, about 2 seconds, no data needed
python scripts/run_pipeline.py           # prepare -> inject -> baseline
```

Or open the notebooks in `notebooks/` in order (01, 02, 03, …).

## Data

AMLNet v2 synthetic Australian transaction dataset, Zenodo record [21237971](https://zenodo.org/records/21237971). The data is synthetic; the EDA notes artefacts (identical median amounts across transaction types, activity spread evenly across all hours) that limit how far results transfer to real banking data.

## Status

- [x] Data preparation and EDA
- [x] Data-quality injection and check features (with tests)
- [x] Baseline of five models, two feature sets, timing test
- [ ] Hyperparameter tuning, three severities × three seeds, full-data check
- [ ] SHAP and LIME explanations and stability analysis
- [ ] Paper (Sage format)

## Acknowledgements

Code was developed with assistance from Claude (Anthropic); design decisions, analysis and interpretation are the author's.
