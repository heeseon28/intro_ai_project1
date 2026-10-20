"""Measure synthetic ResNet-18 CPU kernels to compare thread counts.

This excludes image decoding, preprocessing and DataLoader overhead. Results are
throughput measurements, not model evaluation or an end-to-end ETA.
"""

import argparse
import gc
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", nargs="+", type=int, default=[4, 8, 12, 16])
    parser.add_argument("--skip-eval", action="store_true", help="Only time training steps")
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "results/performance/cpu_benchmark.json",
    )
    args = parser.parse_args()
    if any(count < 1 for count in args.threads):
        parser.error("Thread counts must be positive")
    args.threads = list(dict.fromkeys(args.threads))
    return args


def write_results(path, results):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main():
    args = arguments()
    import torch
    from torchvision.models import resnet18

    batch_size = 64
    # All candidates reuse these exact tensors. No image files are loaded.
    generator = torch.Generator(device="cpu").manual_seed(0)
    inputs = torch.randn(batch_size, 3, 224, 224, generator=generator, dtype=torch.float32)
    targets = torch.randint(0, 50, (batch_size,), generator=generator)
    results = {
        "benchmark": "synthetic CPU kernel benchmark; excludes data I/O and preprocessing",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "state": "running",
        "configuration": {
            "model": "torchvision.models.resnet18",
            "pretrained_weights": None,
            "classes": 50,
            "seed": 0,
            "dtype": "float32",
            "device": "cpu",
            "input_shape": [batch_size, 3, 224, 224],
            "optimizer": "Adam",
            "learning_rate": 0.0003,
            "loss": "unweighted CrossEntropyLoss",
            "training_warmup_steps": 1,
            "training_measured_steps": 2,
            "evaluation_warmup_steps": 0 if args.skip_eval else 1,
            "evaluation_measured_steps": 0 if args.skip_eval else 2,
            "thread_candidates": args.threads,
        },
        "system": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "machine": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_interop_threads": torch.get_num_interop_threads(),
            "torch_parallel_info": torch.__config__.parallel_info(),
        },
        "candidates": [],
        "fastest_thread_count": None,
    }
    write_results(args.output, results)
    for count in args.threads:
        torch.set_num_threads(count)
        torch.manual_seed(0)
        model = resnet18(weights=None, num_classes=50).to(device="cpu", dtype=torch.float32)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
        criterion = torch.nn.CrossEntropyLoss()
        model.train()

        def training_step():
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs)
            loss = criterion(prediction, targets)
            loss.backward()
            optimizer.step()

        print(f"threads={count}: warming up one training step...", flush=True)
        started = time.perf_counter()
        training_step()
        warmup_seconds = time.perf_counter() - started
        training_seconds = []
        for _ in range(2):
            started = time.perf_counter()
            training_step()
            training_seconds.append(time.perf_counter() - started)
        entry = {
            "thread_count": count,
            "actual_torch_threads": torch.get_num_threads(),
            "training_warmup_seconds": warmup_seconds,
            "training_step_seconds": training_seconds,
            "training_images_per_second": 2 * batch_size / sum(training_seconds),
        }
        if not args.skip_eval:
            model.eval()
            evaluation_seconds = []
            with torch.inference_mode():
                model(inputs)
                for _ in range(2):
                    started = time.perf_counter()
                    model(inputs)
                    evaluation_seconds.append(time.perf_counter() - started)
            entry["evaluation_step_seconds"] = evaluation_seconds
            entry["evaluation_images_per_second"] = 2 * batch_size / sum(evaluation_seconds)
        results["candidates"].append(entry)
        results["fastest_thread_count"] = max(
            results["candidates"], key=lambda item: item["training_images_per_second"]
        )["thread_count"]
        write_results(args.output, results)
        print(json.dumps(entry, ensure_ascii=False), flush=True)
        del model, optimizer, criterion, training_step
        gc.collect()
    results["state"] = "completed"
    results["completed_at"] = datetime.now(timezone.utc).isoformat()
    write_results(args.output, results)
    print(f"Fastest measured training thread count: {results['fastest_thread_count']}", flush=True)
    print(f"Saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
