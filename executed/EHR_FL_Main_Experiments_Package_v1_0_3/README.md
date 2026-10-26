# Main experiments — measurement-scope code repair 1.0.3

Use `README_AR.md` for the current Windows/WSL command and
`MEASUREMENT_SCOPE_FIX.md` for the exact cause, scope and validation limits.

Current action is **PrepareRepair**, not TrainNextBlock or RetryBlock. The reviewed
parent is `20260909T071824Z_83a874db`. Preparation preserves that parent and copies
its sealed inputs, initial states and schedules into a newly sealed successor.
No main training or final-test access is authorised by this package alone.

This correction binds measurement to each job's frozen rounds/condition and adds
complete-horizon native-message envelope probes before any main-block training.
It also retains sanitised error frames when a worker fails. No model, loss,
training, aggregation mathematics, PRG, calibration or evaluation decision is
changed. Historical repair notes are retained; their old parent paths are not
current instructions.

A successful successor preparation still needs a new campaign-specific reviewed
TRAIN receipt. Incomplete parent results remain archived; do not select a better
A result across attempts or mix partial runs between campaigns.

All target-host tests must run without failures or skips. Local testing does not
claim native Flower execution where that dependency is absent. Native envelope
probes use native objects but a local fixture transport; they are not a full
training/deployment or security proof.
