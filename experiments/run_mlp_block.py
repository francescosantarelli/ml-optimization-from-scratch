from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_dataset, scale_train_test, split_train_test
from src.mlp import accuracy, plain_mse, predict
from src.mlp_block import full_objective, train_block_mlp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X, y = load_dataset(args.data)
    X_train_raw, X_test_raw, y_train, y_test = split_train_test(X, y, seed=args.seed)
    X_train, X_test, _ = scale_train_test(X_train_raw, X_test_raw)

    n_hidden, sigma, rho = 10, 1.0, 1e-3

    start = time.time()
    params, history, info = train_block_mlp(
        X_train, y_train, n_hidden=n_hidden, sigma=sigma, rho=rho, seed=args.seed
    )
    elapsed = time.time() - start

    train_acc = accuracy(y_train, predict(X_train, params, n_hidden, sigma))
    test_acc = accuracy(y_test, predict(X_test, params, n_hidden, sigma))

    print("MLP — two-block optimization")
    print(f"converged: {info['converged']}")
    print(f"outer iterations: {info['outer_iterations']}")
    print(f"function evaluations: {info['total_function_evaluations']}")
    print(f"final objective: {full_objective(params, X_train, y_train, n_hidden, sigma, rho):.6f}")
    print(f"train MSE: {plain_mse(params, X_train, y_train, n_hidden, sigma):.6f}")
    print(f"test MSE: {plain_mse(params, X_test, y_test, n_hidden, sigma):.6f}")
    print(f"train accuracy: {100*train_acc:.2f}%")
    print(f"test accuracy: {100*test_acc:.2f}%")
    print(f"runtime: {elapsed:.4f}s")


if __name__ == "__main__":
    main()
