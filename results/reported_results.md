# Reported Results from the Original University Submission

These values summarize the final runs reported in the submitted course report.

| Method | Validation Accuracy | Train Accuracy | Test Accuracy | Runtime |
|---|---:|---:|---:|---:|
| MLP — Full gradient | 94.71% | 97.12% | 96.54% | 0.1622 s |
| MLP — Two-block | 94.52% | 96.15% | 96.54% | 2.5850 s |
| SVM — CVXOPT | 95.29% | 96.15% | 95.77% | 0.4153 s |
| SVM — MVP | — | 96.15% | 95.77% | 0.0579 s |

## Optimization diagnostics

- MLP full gradient: 609 iterations, 635 function evaluations.
- MLP two-block: 442 outer iterations, 9,787 function evaluations.
- SVM CVXOPT: 11 iterations, dual objective `-174.209004`, KKT quantity `5.42e-03`.
- SVM MVP: 1,015 pair updates, dual objective `-174.208975`, KKT gap `9.97e-04`.

The public repository uses a neutral random seed and does not include the original dataset, so reproduced values may differ from these course-report numbers.
