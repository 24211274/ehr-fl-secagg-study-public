# EHR federated anomaly detection study

Public code, aggregate results and documentation for the thesis comparing pooled centralised training, Flower FedAvg and FedAvg with SecAgg+.

- [Code, data and artifact availability](docs/AVAILABILITY.md)
- [Study results](results/tables/) and [figures](results/figures/)
- [Thesis and concise appendices](manuscript/README.md)
- [Use the hosted session demonstration](https://session-check-demo.onrender.com) and [English user guide](docs/user_guides/Session_Check_User_Guide_EN.pdf)
- [Versioned public download](https://github.com/24211274/ehr-fl-secagg-study-public/releases/tag/study-artifacts-v1.30)

## What is included

The public archive contains a reviewed selection of implementation and analysis code, original aggregate result tables and figures, synthetic demonstration examples, author documentation, and publication copies of the thesis and appendices. [The selection record](evidence/PUBLIC_SELECTION.json) identifies original source files and exclusions. Third-party attribution and applicable licences are retained.

The complete 215-file executed package, raw benchmark, selected models, fitted preprocessing assets, record-level scores and full replay materials are in the [controlled-access archive](https://github.com/24211274/ehr-fl-secagg-study). That repository requires read permission. Dataset origin and redistribution rights remain unverified; those materials are not included in this public repository.

## Verify and try the public files

```console
python -I -B tools/verify_public_archive.py
```

The public arithmetic tests use synthetic arrays and do not need the benchmark. See [reproduction instructions](docs/REPRODUCIBILITY.md). For new-session inference, use the hosted demonstration. The public application source requires the separately authorised model bundle to run locally.

The study reports poor separation of threats from unusual legitimate activity. A high score is reconstruction error, not a threat probability. Publication adds access and documentation; it does not change the experimental results or perform new training.
