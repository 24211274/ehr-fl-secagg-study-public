# Source basis

1. EHR_FL_Decision_Closure_v1_1: adopted experiment order, candidate30-round configuration, validation-only selection/calibration, paired-seed analysis, bounded claims and failure policy. Retained under `protocol/decisions/` without rewriting missing information as fact.
2. EHR_FL_Timing_BC_Pilot_Package_v1_0_2: unchanged successful runtime, CPU training/model, communication-counter empty-value repair, build identity, local-service lifecycle, native FAB and preflight tests. Historical documents are retained under `reference/inherited_docs/`, not current main-run commands.
3. EHR_FL_FullSize_BC_Pilot_Package_v1_0: complete routine membership and full-size client counts/first-five-round manifests.
4. EHR_FL_Inference_Seed_Pilot_Package_v1_0 and accepted report20260908T164243Z_526bf9b5: accepted environment/data chain and fixed validation inference selection.
5. Original approved encoded training/validation preparation: schema119 and training-only scaler. Source arrays were used in local engineering tests but are not bundled in this release.

Primary technical documentation inspected during implementation:
- scikit-learn1.8, Average Precision: https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.average_precision_score.html
- PyTorch2.10, reproducibility notes: https://docs.pytorch.org/docs/2.10/notes/randomness.html

These sources specify APIs/limitations; they do not establish this project's test performance, dataset publicity, licence or privacy certification. The Flower PRG source and provenance in the inherited package remain pinned; no claim of an updated upstream release or newly merged PR is made here.
