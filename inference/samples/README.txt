SESSION CHECK — TEST SAMPLES

1. Open http://127.0.0.1:8766
2. Choose client C1.
3. Upload one CSV file and select Run inference.

01_below_threshold.csv: 12 events. Expected Below threshold; score approximately 0.198235.
02_above_threshold.csv: 1 event. Expected Review recommended; score approximately 1.707739.
03_invalid_sequence.csv: Expected rejection because positions start at 1 instead of 0.
04_missing_columns.csv: Expected rejection because required columns are missing.

C1 threshold: 0.21659636572003307. The comparison is strictly greater than.
These are hand-constructed synthetic examples for checking the application.
Above/below threshold describes model output; no verified threat or legitimate ground truth is assigned.
Different sequence lengths are used to exercise both decision branches; this is not an accuracy evaluation.
The model, preprocessing and thresholds were not changed.
