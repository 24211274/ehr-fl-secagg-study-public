# Session Check application source

Open [the hosted demonstration](https://session-check-demo.onrender.com) to score a new session without installing a local model. Read [the English user guide](../docs/user_guides/Session_Check_User_Guide_EN.pdf) and use [synthetic CSV samples](samples/).

This public folder contains reviewed application/preprocessing/model code and synthetic examples. It excludes `assets/` and the working runtime configuration. For a local installation, obtain the complete inference bundle from [the controlled archive](https://github.com/24211274/ehr-fl-secagg-study/tree/study-artifacts-v1.30/inference), including its original checkpoint, schema, scaler, calibration and runtime configuration. Follow that bundle's quick start.

The live demonstration uses the original C / Reference / seed 11 / selected round 30 checkpoint. The anomaly score is reconstruction error; review recommendations require human interpretation. Synthetic sample decisions are functional examples with no verified threat ground truth. This publication performs no training or recalibration.
