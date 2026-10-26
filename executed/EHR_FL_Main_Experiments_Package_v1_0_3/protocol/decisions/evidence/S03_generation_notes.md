# Source and generation notes

## What is established
The researcher explicitly reported supervisor approval to use the project's own synthetic benchmark and to publish it on Kaggle. The eleven uploaded CSV files were used as supplied. `metadata/source_mapping.csv` maps upload filenames to release filenames. No rows, values, labels or partitions were edited.

Internal hospital metadata identify the profile as `full_reference_balanced` and its version as `2.0.0-full-reference_seed_01`. That version is not the package version 1.0 and is not an automatically assigned Kaggle version.

## What is not established
The original generator code, original exact generation seed, dependency lockfile and generator configuration were not supplied. The preparation scripts can repeat packaging and structural auditing, but cannot regenerate the original synthetic CSVs from first principles. The original authoring/generation date is unknown. File-system copy timestamps are not acquisition or publication dates.

Every `generator_seed` field in labels.csv is a scientific-notation string. Its full original integer cannot be recovered reliably from this CSV alone. Preserve the string and report the limitation rather than inventing missing digits.

## Relationship to the earlier Kaggle source
No verified deterministic lineage from the two-file dataset published by Kashish Mudliar to this eleven-table benchmark is available. Do not attribute this benchmark to that uploader, reuse the earlier dataset's MIT licence, or describe this release as ordinary cleaning of those files. Any actual conceptual or code dependency should be separately documented by the researcher.

## Release policy
Use the existing snapshot for v1.0 documentation, record omissions publicly, and attach the original generator later only if authentic records are located. Do not construct a retrospective generator and claim it was the one originally used. Any change to CSV values requires a new version and a clear change log.

Public data access supports file-based reproduction, but publication alone does not prove realism, independence, privacy, or generator reproducibility.
