# Model card: credit-default risk score

| | |
|---|---|
| **Purpose** | Estimate the probability that a personal-loan applicant defaults, to *support* underwriters. |
| **Not for** | Fully automated decline decisions. Under UK GDPR Article 22 and FCA Consumer Duty, significant automated decisions need human review and an explanation. |
| **Model** | Gradient-boosted trees (scikit-learn HistGradientBoosting) inside one preprocessing pipeline. |
| **Inputs** | Application data: income, loan amount and term, credit score, debt-to-income, credit lines, recent credit searches, employment years, channel, purpose, home ownership, age. |
| **Excluded** | Protected characteristics other than age are not collected. **Age** is included only because it is permitted in UK credit decisions when justified. Teams should test whether it is needed (see bias checks). |
| **Training data** | Rolling 21-day window of accepted batches. The exact data version is fingerprinted in the registry. |
| **Metrics** | AUROC (ranking), PR-AUC (imbalanced positives), KS (credit-risk standard), Brier score (calibration). |
| **Monitoring** | Daily PSI per feature, prediction PSI, 3-day live AUROC, an absolute AUROC floor, and automatic champion/challenger retraining. |
| **Known limitations** | Synthetic data. Labels are assumed available the same day (real defaults take months, so a production version would monitor proxy metrics and use delayed labels). After concept drift, performance stays reduced until enough post-drift data accumulates. |

## Checks to run before deployment

1. **Bias:** compare approval rates, AUROC and calibration by age band and (if collected for monitoring only, with consent) sex and ethnicity. Investigate proxies such as postcode.
2. **Explainability:** provide applicant-level reasons (e.g. SHAP top factors) to meet adverse-action and Consumer Duty expectations.
3. **Privacy:** raw application data stays in the landing zone with access controls. Models are trained on pseudonymised IDs, and quarantined batches are retained only as long as investigation requires.
4. **Human oversight:** risk bands guide underwriters and never auto-decline.
