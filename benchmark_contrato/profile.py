import argparse
import statistics
from typing import List, Optional, Tuple

import torch

from common import (
    OUTPUTS_DIR,
    RESULTS_DIR,
    collect_environment_snapshot,
    count_encoder_parameters,
    detect_checkpoint_variant,
    ensure_contract_dirs,
    load_contract_configs,
    load_encoder_for_checkpoint,
    percentile,
    require_cuda,
    resolve_checkpoint_path,
    seed_everything,
    write_json,
)


def try_compute_flops(
    encoder: torch.nn.Module,
    image1: torch.Tensor,
    image2: torch.Tensor,
    flow_iterations: int,
) -> Tuple[Optional[float], Optional[str]]:
    try:
        from fvcore.nn import FlopCountAnalysis

        class EncoderWrapper(torch.nn.Module):
            def __init__(self, model: torch.nn.Module, iters: int) -> None:
                super().__init__()
                self.model = model
                self.iters = iters

            def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
                return self.model(image1=x1, image2=x2, iters=self.iters, test_mode=True)[1]

        wrapper = EncoderWrapper(encoder, flow_iterations).eval()
        flops = FlopCountAnalysis(wrapper, (image1, image2)).total() / 1e9
        return float(flops), None
    except Exception as exc:
        return None, str(exc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    args = parser.parse_args()

    ensure_contract_dirs()
    manifest, _, runtime_cfg, experiment_cfg = load_contract_configs()
    seed_everything(int(runtime_cfg.get("seed", 360)))
    device = require_cuda(runtime_cfg)

    profile_cfg = runtime_cfg.get("standardized_efficiency", {})
    checkpoint_path = resolve_checkpoint_path(experiment_cfg, manifest)
    checkpoint_variant = detect_checkpoint_variant(checkpoint_path, experiment_cfg)
    encoder = load_encoder_for_checkpoint(
        checkpoint_path=checkpoint_path,
        checkpoint_variant=checkpoint_variant,
        device=device,
    )

    batch_size = int(runtime_cfg.get("batch_size", 1))
    height = int(profile_cfg.get("input_height", 320))
    width = int(profile_cfg.get("input_width", 640))
    flow_iterations = int(profile_cfg.get("flow_iterations", 12))
    warmup_runs = int(runtime_cfg.get("warmup_runs", 10))
    measured_runs = int(runtime_cfg.get("measured_runs", 50))

    image1 = torch.rand(batch_size, 3, height, width, device=device)
    image2 = torch.rand(batch_size, 3, height, width, device=device)

    timings_ms: List[float] = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for _ in range(warmup_runs):
            encoder(image1=image1, image2=image2, iters=flow_iterations, test_mode=True)
        torch.cuda.synchronize(device)

        for _ in range(measured_runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            encoder(image1=image1, image2=image2, iters=flow_iterations, test_mode=True)
            end.record()
            torch.cuda.synchronize(device)
            timings_ms.append(float(start.elapsed_time(end)))

    flops_g, flops_error = try_compute_flops(encoder, image1, image2, flow_iterations)
    checkpoint_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
    max_gpu_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    latency_mean_ms = sum(timings_ms) / len(timings_ms) if timings_ms else None
    latency_median_ms = statistics.median(timings_ms) if timings_ms else None
    latency_p95_ms = percentile(timings_ms, 0.95)
    fps = (1000.0 / latency_mean_ms) if latency_mean_ms else None

    notes = [
        "Latency measured with synthetic random tensors at the official benchmark resolution.",
        "Parameter count refers to the deployed encoder path used for inference.",
    ]
    if flops_error:
        notes.append(f"FLOPs unavailable: {flops_error}")

    payload = {
        "parameters": count_encoder_parameters(encoder),
        "checkpoint_size_mb": checkpoint_size_mb,
        "flops_g": flops_g,
        "latency_mean_ms": latency_mean_ms,
        "latency_median_ms": latency_median_ms,
        "latency_p95_ms": latency_p95_ms,
        "max_gpu_memory_mb": max_gpu_memory_mb,
        "fps": fps,
        "scenario": args.scenario,
        "checkpoint": str(checkpoint_path),
        "checkpoint_variant": checkpoint_variant,
        "batch_size": batch_size,
        "input_shape": [batch_size, 3, height, width],
        "flow_iterations": flow_iterations,
        "environment_snapshot": collect_environment_snapshot(),
        "notes": " ".join(notes),
    }

    write_json(RESULTS_DIR / "efficiency_metrics.json", payload)
    write_json(OUTPUTS_DIR / "profile_summary.json", payload)
    write_json(
        RESULTS_DIR / "metadata.json",
        {
            "method_name": manifest.get("method_name", "slof"),
            "method_family": manifest.get("method_family"),
            "paper_year": manifest.get("paper_year"),
            "framework": manifest.get("framework"),
            "scenario": args.scenario,
            "checkpoint": str(checkpoint_path),
            "checkpoint_variant": checkpoint_variant,
            "method_root": str(experiment_cfg.get("method_root", "/app")),
            "status": "completed",
            "notes": (
                "Metadata emitted by the standardized efficiency path. "
                "Quality metrics remain untouched in this scenario."
            ),
        },
    )


if __name__ == "__main__":
    main()
