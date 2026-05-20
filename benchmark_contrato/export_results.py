import argparse

from common import (
    CONTRACT_ROOT,
    RESULTS_DIR,
    collect_environment_snapshot,
    default_efficiency_payload,
    default_metadata_payload,
    default_quality_payload,
    ensure_contract_dirs,
    ensure_json,
    load_contract_configs,
    merge_json,
    resolve_checkpoint_path,
    resolve_dataset_root,
    write_json,
)


def preflight(scenario: str) -> None:
    ensure_contract_dirs()
    manifest, datasets_cfg, runtime_cfg, experiment_cfg = load_contract_configs()
    dataset_name = experiment_cfg.get("dataset", datasets_cfg.get("default_dataset"))
    dataset_cfg = datasets_cfg["datasets"][dataset_name]

    dataset_root = None
    dataset_error = None
    checkpoint = None
    checkpoint_error = None

    try:
        dataset_root = str(resolve_dataset_root(dataset_cfg))
    except Exception as exc:
        dataset_error = str(exc)

    try:
        checkpoint = str(resolve_checkpoint_path(experiment_cfg, manifest))
    except Exception as exc:
        checkpoint_error = str(exc)

    run_config = {
        "scenario": scenario,
        "dataset": dataset_name,
        "dataset_root": dataset_root,
        "dataset_resolution": runtime_cfg.get("official_reproduction", {}).get("resize"),
        "batch_size": runtime_cfg.get("batch_size"),
        "precision": runtime_cfg.get("precision"),
        "warmup_runs": runtime_cfg.get("warmup_runs"),
        "measured_runs": runtime_cfg.get("measured_runs"),
        "checkpoint": checkpoint,
        "checkpoint_variant": experiment_cfg.get("checkpoint_variant"),
        "method_root": experiment_cfg.get("method_root"),
        "dataset_root_error": dataset_error,
        "checkpoint_error": checkpoint_error,
    }
    write_json(RESULTS_DIR / "run_config.json", run_config)
    environment = collect_environment_snapshot()
    environment["framework"] = manifest.get("framework")
    environment["method_name"] = manifest.get("method_name")
    write_json(RESULTS_DIR / "environment.json", environment)

    ensure_json(RESULTS_DIR / "metadata.json", default_metadata_payload(scenario))
    ensure_json(RESULTS_DIR / "quality_metrics.json", default_quality_payload())
    ensure_json(RESULTS_DIR / "efficiency_metrics.json", default_efficiency_payload())


def finalize(scenario: str) -> None:
    ensure_contract_dirs()
    ensure_json(RESULTS_DIR / "metadata.json", default_metadata_payload(scenario))
    ensure_json(RESULTS_DIR / "quality_metrics.json", default_quality_payload())
    ensure_json(RESULTS_DIR / "efficiency_metrics.json", default_efficiency_payload())
    merge_json(
        RESULTS_DIR / "metadata.json",
        {
            "scenario": scenario,
            "contract_root": str(CONTRACT_ROOT),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["preflight", "finalize"])
    parser.add_argument("--scenario", required=True)
    args = parser.parse_args()

    if args.mode == "preflight":
        preflight(args.scenario)
    else:
        finalize(args.scenario)


if __name__ == "__main__":
    main()
