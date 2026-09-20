from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_dataset, scale_train_test, split_train_test
from src.mlp import accuracy, objective_and_grad, plain_mse, predict, train_full_mlp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to the project CSV dataset")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X, y = load_dataset(args.data)
    X_train_raw, X_test_raw, y_train, y_test = split_train_test(X, y, seed=args.seed)
    X_train, X_test, _ = scale_train_test(X_train_raw, X_test_raw)

    n_hidden, sigma, rho = 10, 1.0, 1e-3

    start = time.time()
    params0, result = train_full_mlp(
        X_train, y_train, n_hidden=n_hidden, sigma=sigma, rho=rho, seed=args.seed
    )
    elapsed = time.time() - start

    initial_obj, _ = objective_and_grad(params0, X_train, y_train, n_hidden, sigma, rho)
    final_obj, _ = objective_and_grad(result.x, X_train, y_train, n_hidden, sigma, rho)

    train_acc = accuracy(y_train, predict(X_train, result.x, n_hidden, sigma))
    test_acc = accuracy(y_test, predict(X_test, result.x, n_hidden, sigma))

    print("MLP — full-parameter optimization")
    print(f"status: {result.message}")
    print(f"objective: {initial_obj:.6f} -> {final_obj:.6f}")
    print(f"iterations: {result.nit}")
    print(f"function evaluations: {result.nfev}")
    print(f"train MSE: {plain_mse(result.x, X_train, y_train, n_hidden, sigma):.6f}")
    print(f"test MSE: {plain_mse(result.x, X_test, y_test, n_hidden, sigma):.6f}")
    print(f"train accuracy: {100*train_acc:.2f}%")
    print(f"test accuracy: {100*test_acc:.2f}%")
    print(f"runtime: {elapsed:.4f}s")


if __name__ == "__main__":
    main()
