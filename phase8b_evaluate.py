"""
phase8b_evaluate.py
=====================
Combined accuracy + GPU-synchronized latency evaluation.
Does NOT trust any framework's built-in speed mode — reuses the same
GPU-synchronized methodology already validated for MambaVision_S.
"""

import argparse
import json
import os

import torch
from ultralytics import YOLO

RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

WARMUP_ITERS = 20
BENCHMARK_ITERS = 100


def benchmark_latency(model, imgsz=640, device="cuda:0"):
    torch_model = model.model
    torch_model.eval()
    torch_model.to(device)

    dummy_input = torch.randn(1, 3, imgsz, imgsz, device=device)

    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            _ = torch_model(dummy_input)
    torch.cuda.synchronize()

    timings_ms = []
    with torch.no_grad():
        for _ in range(BENCHMARK_ITERS):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            torch.cuda.synchronize()
            start_event.record()
            _ = torch_model(dummy_input)
            end_event.record()
            torch.cuda.synchronize()

            timings_ms.append(start_event.elapsed_time(end_event))

    timings_ms.sort()
    mean_ms = sum(timings_ms) / len(timings_ms)
    min_ms = timings_ms[0]
    p95_ms = timings_ms[int(len(timings_ms) * 0.95)]

    return {
        "mean_ms": round(mean_ms, 3),
        "min_ms": round(min_ms, 3),
        "p95_ms": round(p95_ms, 3),
        "warmup_iters": WARMUP_ITERS,
        "benchmark_iters": BENCHMARK_ITERS,
        "batch_size": 1,
        "imgsz": imgsz,
        "methodology": "GPU-synchronized CUDA events",
    }


def evaluate_accuracy(model, data_yaml):
    metrics = model.val(data=data_yaml, split="test")
    return {
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }


def get_param_count(model):
    return sum(p.numel() for p in model.model.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    print("=" * 70)
    print(f"  Phase 8B Evaluation — {args.model}")
    print("=" * 70)

    model = YOLO(args.weights)

    print("Running accuracy evaluation...")
    accuracy_results = evaluate_accuracy(model, args.data)
    print(f"  mAP50: {accuracy_results['mAP50']:.4f}")

    print("Running GPU-synchronized latency benchmark...")
    latency_results = benchmark_latency(model, imgsz=args.imgsz)
    print(f"  Mean latency: {latency_results['mean_ms']} ms")
    print(f"  Min latency:  {latency_results['min_ms']} ms")
    print(f"  P95 latency:  {latency_results['p95_ms']} ms")

    param_count = get_param_count(model)
    print(f"Parameters: {param_count / 1e6:.2f}M")

    combined = {
        "model": args.model,
        "weights_path": args.weights,
        "parameters_M": round(param_count / 1e6, 2),
        "accuracy": accuracy_results,
        "latency": latency_results,
        "gpu": torch.cuda.get_device_name(0),
    }

    out_path = os.path.join(RESULTS_DIR, f"phase8b_{args.model}_results.json")
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()