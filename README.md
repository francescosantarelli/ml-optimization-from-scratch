# ML Optimization from Scratch

Implementation and comparison of optimization methods for binary classification using a shallow neural network and a nonlinear SVM.

This project was originally developed for the **Optimization Methods for Machine Learning** course at Sapienza University of Rome. The focus is not only on predictive accuracy, but also on the optimization procedures used to train each model.

## What is implemented

- **Shallow MLP** with one hidden layer and `tanh` activation
- Analytical gradient via backpropagation
- Full-parameter training with **L-BFGS-B**
- **Two-block MLP optimization** with an exact update for the output layer and L-BFGS-B for hidden parameters
- **Nonlinear SVM** trained in the dual with a Gaussian kernel
- Dual quadratic-programming solution with **CVXOPT**
- Custom **Most Violating Pair (MVP)** decomposition with analytical `q = 2` updates
- K-fold cross-validation for hyperparameter selection
- Train-only feature scaling to avoid data leakage
- Comparison of predictive performance and computational cost

> The core model logic is implemented directly with NumPy/SciPy/CVXOPT rather than high-level model APIs such as PyTorch, TensorFlow, or `sklearn.neural_network`.

## Reported results

The following results come from the original university experiment and are included here as a summary of the final evaluation.

| Method | Validation Accuracy | Train Accuracy | Test Accuracy | Notes |
|---|---:|---:|---:|---|
| MLP — Full gradient | 94.71% | 97.12% | 96.54% | Analytical gradient + L-BFGS-B |
| MLP — Two-block | 94.52% | 96.15% | 96.54% | Exact output-layer solve + L-BFGS-B |
| SVM — CVXOPT | 95.29% | 96.15% | 95.77% | Dual QP with Gaussian kernel |
| SVM — MVP | — | 96.15% | 95.77% | Custom pairwise decomposition |

Additional optimization results:

- Full-gradient MLP: 635 function evaluations, about 0.16 s in the submitted run.
- Two-block MLP: 9,787 function evaluations, about 2.59 s.
- SVM with CVXOPT: 11 solver iterations, dual objective about -174.2090.
- SVM with MVP: 1,015 pair updates, dual objective about -174.2090, about 0.058 s in the submitted run.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── mlp.py
│   ├── mlp_block.py
│   ├── svm_common.py
│   ├── svm_dual.py
│   └── svm_mvp.py
├── experiments/
│   ├── run_mlp_full.py
│   ├── run_mlp_block.py
│   ├── run_svm_dual.py
│   └── run_svm_mvp.py
└── results/
    └── reported_results.md
```

## Dataset

The original dataset was provided for a university assignment and is **not included in this public repository** because redistribution rights are unclear.

To run the experiments, place the CSV file locally and pass its path with `--data`. The expected format is:

- numerical feature columns
- one target column named `label`
- binary labels encoded as `-1` and `+1`

Example:

```bash
python experiments/run_mlp_full.py --data /path/to/dataset_final_proj.csv
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the experiments

```bash
python experiments/run_mlp_full.py --data /path/to/dataset.csv
python experiments/run_mlp_block.py --data /path/to/dataset.csv
python experiments/run_svm_dual.py --data /path/to/dataset.csv
python experiments/run_svm_mvp.py --data /path/to/dataset.csv
```

The public scripts use a neutral default random seed (`42`). Therefore, results may differ from the original course report, which used the course-specific split required by the assignment.

## Main takeaways

The experiments show that different optimization strategies can reach very similar predictive performance while having very different computational profiles. In particular:

- the full-gradient MLP matched the two-block method on test accuracy with substantially lower computational cost;
- the custom MVP decomposition reached essentially the same SVM solution as the general-purpose QP solver while using simple analytical pair updates.

## Author

**Francesco Santarelli**  
MSc Management Engineering — Business Intelligence & Analytics  
Sapienza University of Rome
