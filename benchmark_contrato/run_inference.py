import argparse
from typing import Any, Dict, List, Tuple

import torch

from common import (
    OUTPUTS_DIR,
    RESULTS_DIR,
    RunningMean,
    build_dataloader,
    collect_environment_snapshot,
    count_encoder_parameters,
    detect_checkpoint_variant,
    ensure_contract_dirs,
    load_contract_configs,
    load_density_mask,
    load_encoder_for_checkpoint,
    merge_json,
    require_cuda,
    resolve_checkpoint_path,
    resolve_dataset_root,
    seed_everything,
    tensor_shape_list,
    write_json,
    angular_error_map,
)


OFFICIAL_GROUPS = {
    "all": None,
    "lt5": lambda magnitude: magnitude < 5.0,
    "lt10": lambda magnitude: magnitude < 10.0,
    "lt20": lambda magnitude: magnitude < 20.0,
    "gte20": lambda magnitude: magnitude >= 20.0,
}


def make_mean_grid(keys: List[str]) -> Dict[str, Dict[str, RunningMean]]:
    return {
        key: {
            "epe": RunningMean(),
            "ae": RunningMean(),
        }
        for key in keys
    }


def latitude_masks(
    height: int,
    latitude_bins: List[List[float]],
    polar_threshold_deg: float,
    device: torch.device,
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    latitudes = 90.0 - ((torch.arange(height, device=device).float() + 0.5) * 180.0 / height)
    latitudes = latitudes.view(1, height, 1)

    bin_masks = []
    for lower, upper in latitude_bins:
        upper_inclusive = upper == latitude_bins[-1][1]
        if upper_inclusive:
            mask = (latitudes >= lower) & (latitudes <= upper)
        else:
            mask = (latitudes >= lower) & (latitudes < upper)
        bin_masks.append(mask)

    polar_mask = latitudes.abs() >= polar_threshold_deg
    equatorial_mask = latitudes.abs() < polar_threshold_deg
    return bin_masks, polar_mask, equatorial_mask


def update_running_mean(target: RunningMean, values: torch.Tensor, mask: torch.Tensor) -> None:
    selected = values[mask]
    target.update(selected)


def update_official_breakdown(
    target: Dict[str, Dict[str, RunningMean]],
    epe_map: torch.Tensor,
    ae_map: torch.Tensor,
    magnitude: torch.Tensor,
) -> None:
    for group_name, predicate in OFFICIAL_GROUPS.items():
        if predicate is None:
            mask = torch.ones_like(epe_map, dtype=torch.bool)
        else:
            mask = predicate(magnitude)
        update_running_mean(target[group_name]["epe"], epe_map, mask)
        update_running_mean(target[group_name]["ae"], ae_map, mask)


def serialize_mean_grid(grid: Dict[str, Dict[str, RunningMean]]) -> Dict[str, Dict[str, Any]]:
    return {
        group_name: {
            metric_name: accumulator.mean()
            for metric_name, accumulator in metrics.items()
        }
        for group_name, metrics in grid.items()
    }


def save_sample_prediction(
    frame1_path: str,
    frame2_path: str,
    flow_path: str,
    pred_flow: torch.Tensor,
    gt_flow: torch.Tensor,
) -> Dict[str, Any]:
    sample_dir = OUTPUTS_DIR / "sample_prediction"
    sample_dir.mkdir(parents=True, exist_ok=True)

    pred_path = sample_dir / "pred_flow.npy"
    gt_path = sample_dir / "gt_flow.npy"
    pred_np = pred_flow.detach().cpu().float().numpy()
    gt_np = gt_flow.detach().cpu().float().numpy()

    import numpy as np

    np.save(pred_path, pred_np)
    np.save(gt_path, gt_np)

    info = {
        "frame1_path": frame1_path,
        "frame2_path": frame2_path,
        "ground_truth_flow_path": flow_path,
        "predicted_flow_npy": str(pred_path),
        "ground_truth_flow_npy": str(gt_path),
        "predicted_flow_shape": list(pred_np.shape),
    }
    write_json(sample_dir / "sample_info.json", info)
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    args = parser.parse_args()

    ensure_contract_dirs()
    manifest, datasets_cfg, runtime_cfg, experiment_cfg = load_contract_configs()
    seed_everything(int(runtime_cfg.get("seed", 360)))

    device = require_cuda(runtime_cfg)
    dataset_name = experiment_cfg.get("dataset", datasets_cfg.get("default_dataset"))
    dataset_cfg = datasets_cfg["datasets"][dataset_name]
    dataset_root = resolve_dataset_root(dataset_cfg)
    checkpoint_path = resolve_checkpoint_path(experiment_cfg, manifest)
    checkpoint_variant = detect_checkpoint_variant(checkpoint_path, experiment_cfg)

    official_cfg = runtime_cfg.get("official_reproduction", {})
    regional_cfg = runtime_cfg.get("regional_robustness", {})
    resize = tuple(official_cfg.get("resize", [320, 640]))
    split = experiment_cfg.get("official_eval", {}).get("split", dataset_cfg.get("split", "test"))
    flow_iterations = int(
        experiment_cfg.get("official_eval", {}).get(
            "flow_iterations", official_cfg.get("flow_iterations", 12)
        )
    )

    dataset, loader = build_dataloader(
        dataset_root=dataset_root,
        dataset_cfg=dataset_cfg,
        split=split,
        resize=resize,
        batch_size=int(runtime_cfg.get("batch_size", 1)),
        num_workers=int(runtime_cfg.get("num_workers", 2)),
    )

    encoder = load_encoder_for_checkpoint(
        checkpoint_path=checkpoint_path,
        checkpoint_variant=checkpoint_variant,
        device=device,
    )
    density_mask = load_density_mask(device)

    raw_grid = make_mean_grid(list(OFFICIAL_GROUPS.keys()))
    weighted_grid = make_mean_grid(list(OFFICIAL_GROUPS.keys()))
    latitude_bins = regional_cfg.get(
        "latitude_bins",
        [[-90, -60], [-60, -30], [-30, 0], [0, 30], [30, 60], [60, 90]],
    )
    latitude_raw = [RunningMean() for _ in latitude_bins]
    latitude_weighted = [RunningMean() for _ in latitude_bins]
    polar = RunningMean()
    equatorial = RunningMean()
    valid_ratio = RunningMean()

    sample_prediction_info = None
    bin_masks = None
    polar_mask = None
    equatorial_mask = None
    use_valid_mask = bool(regional_cfg.get("use_valid_mask", True))

    with torch.inference_mode():
        for batch in loader:
            frame1 = batch["frame1"].to(device, non_blocking=True)
            frame2 = batch["frame2"].to(device, non_blocking=True)
            flow_gt = batch["fflow"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True) >= 0.5

            _, flow_pred = encoder(
                image1=frame1,
                image2=frame2,
                iters=flow_iterations,
                test_mode=True,
            )

            if sample_prediction_info is None and bool(
                runtime_cfg.get("save_predictions", True)
            ) and bool(official_cfg.get("save_sample_prediction", True)):
                sample_prediction_info = save_sample_prediction(
                    frame1_path=batch["frame1_path"][0],
                    frame2_path=batch["frame2_path"][0],
                    flow_path=batch["flow_path"][0],
                    pred_flow=flow_pred[0],
                    gt_flow=flow_gt[0],
                )

            epe_map = torch.sum((flow_gt - flow_pred) ** 2, dim=1).sqrt()
            ae_map = angular_error_map(flow_gt, flow_pred)
            weighted_epe_map = epe_map / density_mask
            weighted_ae_map = ae_map / density_mask
            magnitude = torch.sum(flow_gt**2, dim=1).sqrt()

            update_official_breakdown(raw_grid, epe_map, ae_map, magnitude)
            update_official_breakdown(weighted_grid, weighted_epe_map, weighted_ae_map, magnitude)

            if bin_masks is None:
                bin_masks, polar_mask, equatorial_mask = latitude_masks(
                    height=epe_map.shape[1],
                    latitude_bins=latitude_bins,
                    polar_threshold_deg=float(regional_cfg.get("polar_threshold_deg", 60)),
                    device=device,
                )

            valid_ratio.update(valid.float())
            mask_base = valid if use_valid_mask else torch.ones_like(valid, dtype=torch.bool)

            for idx, region_mask in enumerate(bin_masks):
                region_mask_batch = mask_base & region_mask.expand_as(mask_base)
                update_running_mean(latitude_raw[idx], epe_map, region_mask_batch)
                update_running_mean(latitude_weighted[idx], weighted_epe_map, region_mask_batch)

            update_running_mean(polar, epe_map, mask_base & polar_mask.expand_as(mask_base))
            update_running_mean(
                equatorial, epe_map, mask_base & equatorial_mask.expand_as(mask_base)
            )

    official_breakdown = {
        "raw": serialize_mean_grid(raw_grid),
        "distortion_aware": serialize_mean_grid(weighted_grid),
    }
    latitude_payload = [
        {
            "range_deg": latitude_bins[idx],
            "epe": latitude_raw[idx].mean(),
        }
        for idx in range(len(latitude_bins))
    ]
    latitude_weighted_payload = [
        {
            "range_deg": latitude_bins[idx],
            "eped": latitude_weighted[idx].mean(),
        }
        for idx in range(len(latitude_bins))
    ]

    quality_payload: Dict[str, Any] = {
        "epe_global": official_breakdown["raw"]["all"]["epe"],
        "ae_global": official_breakdown["raw"]["all"]["ae"],
        "eped_global": official_breakdown["distortion_aware"]["all"]["epe"],
        "aed_global": official_breakdown["distortion_aware"]["all"]["ae"],
        "epe_polar": polar.mean(),
        "epe_equatorial": equatorial.mean(),
        "epe_by_latitude": latitude_payload,
        "eped_by_latitude": latitude_weighted_payload,
        "valid_pixels_ratio": valid_ratio.mean(),
        "official_breakdown": official_breakdown,
        "sample_prediction": sample_prediction_info,
        "notes": (
            "Metric formulas follow the repository evaluation logic in evaluate_raft.py. "
            "Regional metrics are added on top of the official global breakdown."
        ),
    }
    write_json(RESULTS_DIR / "quality_metrics.json", quality_payload)
    write_json(OUTPUTS_DIR / "official_metrics.json", official_breakdown)

    metadata_payload = {
        "method_name": manifest.get("method_name", "slof"),
        "method_family": manifest.get("method_family"),
        "paper_year": manifest.get("paper_year"),
        "framework": manifest.get("framework"),
        "scenario": args.scenario,
        "dataset": dataset_name,
        "dataset_root": str(dataset_root),
        "split": split,
        "checkpoint": str(checkpoint_path),
        "checkpoint_variant": checkpoint_variant,
        "method_root": str(experiment_cfg.get("method_root", "/app")),
        "flow_iterations": flow_iterations,
        "resize": list(resize),
        "dataset_samples": len(dataset),
        "scene_count": len(dataset.scene_dirs),
        "encoder_parameters": count_encoder_parameters(encoder),
        "output_flow_shape": tensor_shape_list(flow_pred),
        "sample_prediction": sample_prediction_info,
        "source_files_used": [
            "RAFT/load_raft.py",
            "simsiam.py",
            "KernelTransformerNetwork/KernelTransformer/KTNLayer.py",
            "evaluate_raft.py",
            "dataloader.py",
            "utils.py",
        ],
        "environment_snapshot": collect_environment_snapshot(),
        "status": "completed",
        "notes": (
            "The contract loader reproduces the repository's FLOW360 data layout and "
            "the same metric formulas used by the official evaluation script."
        ),
    }
    write_json(RESULTS_DIR / "metadata.json", metadata_payload)

    merge_json(
        RESULTS_DIR / "run_config.json",
        {
            "scenario": args.scenario,
            "dataset": dataset_name,
            "dataset_root": str(dataset_root),
            "checkpoint": str(checkpoint_path),
            "checkpoint_variant": checkpoint_variant,
            "flow_iterations": flow_iterations,
            "resize": list(resize),
        },
    )


if __name__ == "__main__":
    main()
