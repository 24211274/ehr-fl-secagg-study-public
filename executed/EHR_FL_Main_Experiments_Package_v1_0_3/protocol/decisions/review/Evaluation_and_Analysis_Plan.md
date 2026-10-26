# Adopted evaluation and analysis plan
Version 1.1 | 9 September 2026 | Decisions set under researcher delegation; evaluator implementation still pending.

## What is unchanged
Retain A (pooled centralised), B (Flower FedAvg) and C (FedAvg with the pinned locally patched SecAgg+), the Reference and controlled Non-IID conditions, the three simulated clients, and training seeds 11, 29, 47, 71 and 97. The fixed training budget is 30 pooled epochs or 30 federated rounds with one local epoch. Do not use trained pilot checkpoints to initialise main runs. Retain the existing optimiser-state reset policy and partial final batches; equal data exposure is not identical optimisation trajectories.

The primary test remains `test_held_out`; `test_known` is a separate secondary scenario analysis. Threat is the positive class. Primary detection outcome is session-level Average Precision (AP), not accuracy and not trapezoidal PR area. Report AUROC, precision, recall, F1, FPR, FNR, confusion counts and false alerts per 1,000 legitimate sessions with denominators. Do not rebalance test prevalence. [S05/S10/S24; W3]

## Checkpoints, scores and alert calibration
Choose the lowest session-count-weighted routine-validation loss among rounds/epochs 1-30; earliest tie within 1e-12. Round 0 is not eligible. Retain the validation-only, per-client recipe separately for each configuration, condition and seed: median; 1.482602218505602*MAD; fallback IQR/1.3489795003921634, then 1.0 at epsilon 1e-12, with flags. Quantiles use the inherited linear method.

For pooled global AP, concatenate the calibrated client scores; pooled raw-score AP is a prespecified secondary diagnostic. Per-client AP uses raw scores. Do not choose whichever representation produces a higher test result. Alert when raw score is strictly above the client's full routine-validation 0.99 quantile. The existing 0.98 and 0.995 quantiles are fixed sensitivity results, not a threshold search. No numerical threshold is invented before training. Selection and calibration use development data and do not justify claiming an unbiased validation performance estimate. Final tests must not influence these choices. [S10; W4]

Missing-class AP/AUROC are recorded as undefined for the corresponding endpoint. With actual positives but no predicted positives, record undefined precision and F1=0; preserve counts. A threat-only subgroup is not a standalone AUROC comparison. Use only supported scenario labels and disclose any legitimate comparison background.

## Explicit simplification of the previous P2 proposal
The previous actor/campaign connected-component hierarchical bootstrap was a proposal, not an executed or accepted analysis. It is superseded now by paired seed-level analysis to avoid a new custom resampling subsystem. This changes the uncertainty target: the study estimates algorithmic run-to-run variability conditional on the fixed benchmark, not uncertainty from sampling new patients, actors, campaigns, organisations or datasets. It does NOT assume sessions or events are independent.

For each fixed condition and test set, report all five seed results, mean, sample SD (ddof=1), median and range. Compute paired differences using the same seed: B-A for RQ1 and C-B for the detection component of RQ2. Global AP is not the mean of three client APs. For worst-client reporting, take each method's minimum client AP within each seed and retain the client identity; differences of minima need not concern the same client, so also show every matched per-client difference.

For the primary global and worst-client mean AP contrasts only, report an exploratory model-based interval:

    mean(d) +/- t.ppf(0.975, 4) * sd(d, ddof=1) / sqrt(5)

This is the usual Student-t interval applied to five paired differences. [W1, W2] Its nominal coverage relies on independent run differences and approximate normality; five preselected seeds cannot establish these assumptions. Do not claim validated 95% coverage or uncertainty over real hospitals. Always present all five differences so the reader can see heterogeneity and direction.

Require all five finite matched pairs. If any endpoint or pair is missing, suppress its interval and report the available values, counts and reason; do not silently drop failed seeds and call the study complete. With zero sample SD, report zero observed variability and suppress a degenerate interval rather than asserting certainty. Do not clip an interval to the feasible AP-difference range or switch to a more favourable interval after seeing results. Other metrics and per-client contrasts are descriptive. No formal p-value, equivalence, non-inferiority or multiplicity-based superiority claim is planned. The inherited 0.02 AP margin remains a practical reference, not a pass threshold or established clinical importance.

## System cost
Keep run-level instrumented fit time, evaluation time, startup, total wall time, sampled process-tree RSS, prepared-input inference latency and application record-content volume separate. Main cost comparisons are paired by seed and condition; use one prespecified run summary, not rounds as independent replicates. For positive B denominators, report 100*(C-B)/B per pair and summarise those ratios without conflating mean ratio with ratio of means. Preserve slow results and failures. Three earlier small-Reference timing pairs cannot substitute for full-size or Non-IID measurements. B/C diagnostic differences remain a limitation; do not call the difference pure encryption cost. [S20]

## Test history and release
Prior structural inspection of test scenario names, labels, counts and relationships is documented. Reviewed pilot executions report no final-test detection scoring; a complete declaration about unreported external scoring has not been supplied. Do not call the test blind, pristine or fully independent. This omission is disclosed without inventing a negative declaration or concluding that contamination occurred. If prior score-driven development is later confirmed, affected findings are exploratory and must be described accordingly.

Prepare/check the main evaluator on fabricated numerical fixtures before any final test scoring. Store all planned training-run outcomes and selected checkpoint hashes, then validation-only calibration values and hashes, before authorising the comparative final-test evaluation. No test data, scores, confidence intervals or calibration values were computed in this addendum.

## References
S01-S26 refer to the unchanged prior readiness archive identified in `review/authority_and_provenance.json`.
W1. NIST/SEMATECH e-Handbook, Confidence Limits for the Mean: https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm (consulted 9 September 2026; formula, not validation of these seeds).
W2. NIST/SEMATECH e-Handbook, Two-Sample t-Test for Equal Means: https://www.itl.nist.gov/div898/handbook/eda/section3/eda353.htm (consulted 9 September 2026; paired differences definition).
W3. scikit-learn, average_precision_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html (consulted 9 September 2026; AP definition, not a runtime upgrade).
W4. scikit-learn, Common pitfalls and recommended practices: https://scikit-learn.org/stable/common_pitfalls.html (consulted 9 September 2026; development/test separation, not a claim that this benchmark is leak-free).
